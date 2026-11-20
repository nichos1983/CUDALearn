#pragma once

#include "../utils/cuda_utils.cuh"

/**
 * @brief Naive SGEMM kernel
 */
__global__ void naive_sgemm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            // A[row, k] * B[k, col]
            // Within a warp, threadIdx.x varies -> col varies, row fixed.
            // B[k * N + col] and C[row * N + col] are coalesced along columns.
            // A[row * K + k] is the same address for all lanes -> broadcast.
            sum += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

/**
 * @brief Launch wrapper for naive sgemm kernel
 *
 * @tparam BLOCK_SIZE block size
 * @param A Device pointer to matrix A [M, K]
 * @param B Device pointer to matrix B [K, N]
 * @param C Device pointer to matrix C [M, N]
 * @param M Number of rows of A and C
 * @param K Number of columns of A and rows of B
 * @param N Number of columns of B and C
 * @param stream Stream to launch kernel on
 */
template <int BLOCK_SIZE = 32>
void launch_naive_sgemm(const float* A, const float* B, float* C, int M, int K,
                        int N, cudaStream_t stream = 0) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(ceil_div(N, block.x), ceil_div(M, block.y));

    naive_sgemm_kernel<<<grid, block, 0, stream>>>(A, B, C, M, K, N);

    CUDA_CHECK(cudaGetLastError());
}
