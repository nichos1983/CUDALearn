#pragma once

#include "../utils/constants.cuh"
#include "../utils/cuda_utils.cuh"

/**
 * @brief SGEMM Kernel using shared memory tiling.
 * A: M x K
 * B: K x N
 * C: M x N
 * CTA output tile: BM x BN
 * K tile: BK
 * Each CTA computes a BM x BN output tile. It processes the K dimension
 * in chunks of BK, staging the corresponding A and B tiles in shared memory.
 *
 * Each thread computes ROWS_PER_THREAD = BM / THREADS_M output elements
 * in one output column.
 *
 */
template <int BM, int BN, int BK, int THREADS_M>
__global__ void tiled_sgemm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C, int M, int K, int N,
                                   int lda, int ldb, int ldc) {
    static_assert(BM > 0 && BN > 0 && BK > 0,
                  "Tile dimensions must be positive");
    static_assert(THREADS_M > 0, "THREADS_M must be positive");
    static_assert(BM % THREADS_M == 0, "BM must be divisible by THREADS_M");
    static_assert(THREADS_M * BN <= constants::MAX_THREADS_PER_BLOCK,
                  "The CTA thread count exceeds the device limit");

    constexpr int ROWS_PER_THREAD = BM / THREADS_M;
    constexpr int NUM_THREADS = THREADS_M * BN;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    // Flatten the 2D thread coordinates into a linear CTA thread ID.
    const int tid = ty * BN + tx;

    const int block_row_start = blockIdx.y * BM;
    const int block_col_start = blockIdx.x * BN;

    // Each thread owns ROWS_PER_THREAD output elements in the same column.
    // The accumulator remains live across all K tiles.
    float accumulator[ROWS_PER_THREAD] = {};

    const int num_k_tiles = ceil_div(K, BK);

    // Each iteration accumulates one BM x BN output tile
    // over a BK-wide slice of the reduction dimension.
    for (int tile = 0; tile < num_k_tiles; tile++) {
        // The start k index of the tile in the global matrix.
        const int tile_k_start = tile * BK;

        // Stage the BM x BK tile of A in As.
        // Threads cooperatively load the tile; each thread may load multiple
        // elements when the tile contains more elements than the CTA has
        // threads.
        for (int idx = tid; idx < BM * BK; idx += NUM_THREADS) {
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

        // Stage the BK x BN tile of B into Bs.
        // Each thread loads multiple elements.
        for (int idx = tid; idx < BK * BN; idx += NUM_THREADS) {
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

        // Ensure the entire shared-memory tile is ready before it is consumed.
        __syncthreads();

        // Accumulate this K tile. For each k, a B value is reused across
        // ROWS_PER_THREAD output rows owned by the same thread.
#pragma unroll
        for (int k = 0; k < BK; k++) {
            const float b = Bs[k][tx];

#pragma unroll
            for (int r = 0; r < ROWS_PER_THREAD; r++) {
                const int local_row = r * THREADS_M + ty;
                accumulator[r] += As[local_row][k] * b;
            }
        }

        // Do not overwrite the shared-memory buffers for the next K tile until
        // every thread has finished reading the current tile.
        __syncthreads();
    }

    // Store the output tile. Threads write their owned rows in the same column.
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

/**
 * @brief Launch the tiled SGEMM kernel for row-major matrices with explicit
 * leading dimensions.
 *
 * lda, ldb, and ldc are physical row strides measured in elements. They may
 * be larger than the corresponding logical column counts to support padded
 * matrices and submatrix views.
 *
 * Preconditions:
 *   M, N, K >= 0
 *   lda >= K, ldb >= N, ldc >= N
 *   Device pointers refer to valid allocations whenever the corresponding
 *   matrix is accessed.
 *
 * If K == 0, the kernel writes zeros to C, consistent with C = A * B.
 */
template <int BM = 32, int BN = 32, int BK = 32, int THREADS_M = 8>
void launch_tiled_sgemm(const float* __restrict__ A,
                        const float* __restrict__ B, float* __restrict__ C,
                        int M, int K, int N, int lda, int ldb, int ldc,
                        cudaStream_t stream = nullptr) {
    if (M == 0 || N == 0) {
        return;
    }

    const dim3 block(BN, THREADS_M);
    const dim3 grid(ceil_div(N, BN), ceil_div(M, BM));

    tiled_sgemm_kernel<BM, BN, BK, THREADS_M>
        <<<grid, block, 0, stream>>>(A, B, C, M, K, N, lda, ldb, ldc);

    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Convenience launcher for compact row-major matrices.
 * The physical row strides are set to the logical column counts.
 */
template <int BM = 32, int BN = 32, int BK = 32, int THREADS_M = 8>
void launch_tiled_sgemm_compact(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C, int M, int K, int N,
                                cudaStream_t stream = nullptr) {
    launch_tiled_sgemm<BM, BN, BK, THREADS_M>(A, B, C, M, K, N, K, N, N,
                                              stream);
}
