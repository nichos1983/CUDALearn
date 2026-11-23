#include <stdio.h>
#include <stdlib.h>

#include "../common/common.cuh"

#define SHARED_BLOCK_DIM_X 32
#define SHARED_BLOCK_DIM_Y 16
#define SHARED_BLOCK_COUNT (1 << 8)
#define LOAD_STORE_ITER_COUNT 16
#define PADDING_1 1
#define PADDING_2 2

#define BLOCK_SIZE 128

__global__ void setRowReadRow(int* out) {
    __shared__ int tile[SHARED_BLOCK_DIM_Y][SHARED_BLOCK_DIM_X];

    unsigned int globalIdx = blockIdx.x * blockDim.x * blockDim.y +
                             threadIdx.y * blockDim.x + threadIdx.x;

    for (int i = 0; i < LOAD_STORE_ITER_COUNT; i++) {
        tile[threadIdx.y][threadIdx.x] = globalIdx;
        __syncthreads();
        out[globalIdx] = tile[threadIdx.y][threadIdx.x];
    }
}

__global__ void setColReadCol(int* out) {
    __shared__ int tile[SHARED_BLOCK_DIM_X][SHARED_BLOCK_DIM_Y];

    unsigned int globalIdx = blockIdx.x * blockDim.x * blockDim.y +
                             threadIdx.y * blockDim.x + threadIdx.x;
    for (int i = 0; i < LOAD_STORE_ITER_COUNT; i++) {
        tile[threadIdx.x][threadIdx.y] = globalIdx;
        __syncthreads();
        out[globalIdx] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void setRowReadCol(int* out) {
    __shared__ int tile[SHARED_BLOCK_DIM_Y][SHARED_BLOCK_DIM_X];

    unsigned int globalIdx = blockIdx.x * blockDim.x * blockDim.y +
                             threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int localIdx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int transRow = localIdx / blockDim.y;
    unsigned int transCol = localIdx % blockDim.y;
    for (int i = 0; i < LOAD_STORE_ITER_COUNT; i++) {
        tile[threadIdx.y][threadIdx.x] = globalIdx;
        __syncthreads();
        out[globalIdx] = tile[transCol][transRow];
    }
}

__global__ void setRowReadColDynamic(int* out) {
    extern __shared__ int tile[];

    unsigned int globalIdx = blockIdx.x * blockDim.x * blockDim.y +
                             threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int localIdx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int transRow = localIdx / blockDim.y;
    unsigned int transCol = localIdx % blockDim.y;

    unsigned int transLocalIdx = transCol * blockDim.x + transRow;

    for (int i = 0; i < LOAD_STORE_ITER_COUNT; i++) {
        tile[localIdx] = globalIdx;
        __syncthreads();
        out[globalIdx] = tile[transLocalIdx];
    }
}

// Note:
// When PADDING_NUM = 1, there is 2.0 - way bank conflict.
// When PADDING_NUM = 2, there is no bank conflict.
template <int PADDING_NUM>
__global__ void setRowReadColPad(int* out) {
    __shared__ int tile[SHARED_BLOCK_DIM_Y][SHARED_BLOCK_DIM_X + PADDING_NUM];

    unsigned int globalIdx = blockIdx.x * blockDim.x * blockDim.y +
                             threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int localIdx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int transRow = localIdx / blockDim.y;
    unsigned int transCol = localIdx % blockDim.y;
    for (int i = 0; i < LOAD_STORE_ITER_COUNT; i++) {
        tile[threadIdx.y][threadIdx.x] = globalIdx;
        __syncthreads();
        out[globalIdx] = tile[transCol][transRow];
    }
}

// Note:
// The same as setRowReadColPad.
// When PADDING_NUM = 1, there is 2.0 - way bank conflict.
// When PADDING_NUM = 2, there is no bank conflict.
template <int PADDING_NUM>
__global__ void setRowReadColDynamicPad(int* out) {
    extern __shared__ int tile[];

    unsigned int globalIdx = blockIdx.x * blockDim.x * blockDim.y +
                             threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int localIdx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int transRow = localIdx / blockDim.y;
    unsigned int transCol = localIdx % blockDim.y;

    unsigned int localPadIdx =
        threadIdx.y * (blockDim.x + PADDING_NUM) + threadIdx.x;
    unsigned int transLocalPadIdx =
        transCol * (blockDim.x + PADDING_NUM) + transRow;

    for (int i = 0; i < LOAD_STORE_ITER_COUNT; i++) {
        tile[localPadIdx] = globalIdx;
        __syncthreads();
        out[globalIdx] = tile[transLocalPadIdx];
    }
}

void printResult(int* out) {
    printf("* result start: *\n");
    for (int k = 0; k < 2; k++) {
        printf("--- loop %d ---\n", k);
        int offset = k * SHARED_BLOCK_DIM_X * SHARED_BLOCK_DIM_Y;
        for (int i = 0; i < SHARED_BLOCK_DIM_X * 4; i++) {
            printf("%d ", out[offset + i]);
        }
        printf("\n");
    }
    printf("\n* result end *\n");
}

int runTest() {
    const size_t bytes = (size_t)SHARED_BLOCK_COUNT * SHARED_BLOCK_DIM_X *
                         SHARED_BLOCK_DIM_Y * sizeof(int);

    int* h_out = (int*)malloc(bytes);
    if (!h_out) {
        fprintf(stderr, "Not enough memory\n");
        return 1;
    }

    int* d_out;
    CHECK(cudaMalloc((void**)&d_out, bytes));

    dim3 blockDim(SHARED_BLOCK_DIM_X, SHARED_BLOCK_DIM_Y);

    printf("=== For warmup. setRowReadRow ===\n");
    setRowReadRow<<<SHARED_BLOCK_COUNT, blockDim>>>(d_out);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    printResult(h_out);

    printf("=== setRowReadRow ===\n");
    setRowReadRow<<<SHARED_BLOCK_COUNT, blockDim>>>(d_out);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    printResult(h_out);

    printf("=== setColReadCol ===\n");
    setColReadCol<<<SHARED_BLOCK_COUNT, blockDim>>>(d_out);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    printResult(h_out);

    printf("=== setRowReadCol ===\n");
    setRowReadCol<<<SHARED_BLOCK_COUNT, blockDim>>>(d_out);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    printResult(h_out);

    printf("=== setRowReadColDynamic ===\n");
    unsigned int dynamicBytes =
        SHARED_BLOCK_DIM_X * SHARED_BLOCK_DIM_Y * sizeof(int);
    setRowReadColDynamic<<<SHARED_BLOCK_COUNT, blockDim, dynamicBytes>>>(d_out);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    printResult(h_out);

    printf("=== setRowReadCol pad 1 ===\n");
    setRowReadColPad<PADDING_1><<<SHARED_BLOCK_COUNT, blockDim>>>(d_out);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    printResult(h_out);

    printf("=== setRowReadCol pad 2 ===\n");
    setRowReadColPad<PADDING_2><<<SHARED_BLOCK_COUNT, blockDim>>>(d_out);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    printResult(h_out);

    printf("=== setRowReadColDynamicPad pad 1 ===\n");
    unsigned int dynamicPadBytes1 =
        (SHARED_BLOCK_DIM_X + PADDING_1) * SHARED_BLOCK_DIM_Y * sizeof(int);
    setRowReadColDynamicPad<PADDING_1>
        <<<SHARED_BLOCK_COUNT, blockDim, dynamicPadBytes1>>>(d_out);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    printResult(h_out);

    printf("=== setRowReadColDynamicPad pad 2 ===\n");
    unsigned int dynamicPadBytes2 =
        (SHARED_BLOCK_DIM_X + PADDING_2) * SHARED_BLOCK_DIM_Y * sizeof(int);
    setRowReadColDynamicPad<PADDING_2>
        <<<SHARED_BLOCK_COUNT, blockDim, dynamicPadBytes2>>>(d_out);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    printResult(h_out);

    cudaFree(d_out);
    free(h_out);

    return 0;
}

int main() {
    runTest();
    return 0;
}
