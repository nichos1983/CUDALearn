#pragma once

#include "../utils/constants.cuh"
#include "../utils/cuda_utils.cuh"

/**
 * @brief Tiled SGEMM Kernel using shared memory
 * A: M x K
 * B: K x N
 * C: M x N
 * CTA output tile: BM x BN
 * K tile: BK
 * CTA threads (Block dimension): THREADS_M x BN
 * Each thread computes ROWS_PER_THREAD = BM / THREADS_M output elements.
 *
 */
template <int BM, int BN, int BK, int THREADS_M>
__global__ void tiled_sgemm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C, int M, int N, int K,
                                   int lda, int ldb, int ldc) {
    static_assert(BM > 0 && BN > 0 && BK > 0,
                  "Tile dimensions must be positive");
    static_assert(THREADS_M > 0, "THREADS_M must be positive");
    static_assert(BM % THREADS_M == 0, "BM must be a multiple of THREADS_M");
    static_assert(THREADS_M * BN <= constants::MAX_THREADS_PER_BLOCK,
                  "CTA thread count not exceed the limitation");

    constexpr int ROWS_PER_THREAD = BM / THREADS_M;
    constexpr int NUM_THREADS = THREADS_M * BN;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    // One dimensional thread index in the CTA
    const int tid = ty * BN + tx;

    const int block_row_start = blockIdx.y * BM;
    const int block_col_start = blockIdx.x * BN;

    // Each thread computes ROWS_PER_THREAD output elements in the same column.
    // The accumulator keeps the sum along num_k_tiles iterations.
    float accumulator[ROWS_PER_THREAD] = {};

    const int num_k_tiles = ceil_div(K, BK);

    // Each iteration computes a BM x BN output tile along the K dimension.
    for (int tile = 0; tile < num_k_tiles; tile++) {
        // The start k index of the tile in the global matrix
        // k is A's column and B's row
        const int tile_k_start = tile * BK;

        // 1. Load A's BM x BN tile into As.
        // Each thread loads multiple elements.
        for (int idx = tid; idx < BM * BK; idx += NUM_THREADS) {
            // Calculate the local indices in the local matrix laid in the
            // shared memory.
            const int local_row = idx / BK;
            const int local_k = idx % BK;

            // The row index in the global matrix A
            const int global_row = block_row_start + local_row;
            // The column(k) index in the global matrix A
            const int global_k = tile_k_start + local_k;

            As[local_row][local_k] = (global_row < M && global_k < K)
                                         ? A[global_row * lda + global_k]
                                         : 0.0f;
        }

        // 2. Load B's BK x BN tile into Bs.
        // Each thread loads multiple elements.
        for (int idx = tid; idx < BK * BN; idx += NUM_THREADS) {
            // Calculate the local indices in the local matrix laid in the
            // shared memory.
            const int local_k = idx / BN;
            const int local_col = idx % BN;

            // The k(row) index in the global matrix B
            const int global_k = tile_k_start + local_k;
            // The column index in the global matrix B
            const int global_col = block_col_start + local_col;

            Bs[local_k][local_col] = (global_k < K && global_col < N)
                                         ? B[global_k * ldb + global_col]
                                         : 0.0f;
        }

        // All tile values must be loaded before the computation.
        __syncthreads();

        // Each iteration computes ROWS_PER_THREAD elements in the same column
        // in the output tile.
#pragma unroll
        for (int k = 0; k < BK; k++) {
            const float b = Bs[k][tx];

#pragma unroll
            for (int r = 0; r < ROWS_PER_THREAD; r++) {
                const int local_row = r * THREADS_M + ty;
                accumulator[r] += As[local_row][k] * b;
            }
        }

        // When the next iteration along k starts, the values in As and Bs will
        // be overwritten. So synchronize to make sure the computation of As and
        // Bs in the current iteration is complete.
        __syncthreads();
    }

    // Write the output tile to global memory.
#pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; r++) {
        const int local_row = r * THREADS_M + ty;
        const int global_row = block_row_start + local_row;
        const int global_col = block_col_start + tx;

        if (global_row < M && global_col < N) {
            C[global_row * ldc + global_col] = accumulator[r];
        }
    }
}
