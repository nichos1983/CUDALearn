#include <stdio.h>
#include <stdlib.h>

#define MASK_ALL_32 0xFFFFFFFF
#define WARP_SIZE 32

#define GRID_SIZE 2
#define BLOCK_SIZE (32 * 4)
#define TOTAL_SIZE (GRID_SIZE * BLOCK_SIZE)

#define TEST_WIDTH_32 32
#define TEST_WIDTH_16 16

#define TEST_SOURCE_LANE 2
#define TEST_LANE_DELTA 3
#define TEST_LANE_OFFSET 4
#define TEST_XOR_MASK_1 1
#define TEST_XOR_MASK_3 3

#define CHECK(x)                                                                                    \
    do                                                                                              \
    {                                                                                               \
        cudaError_t e_ = (x);                                                                       \
        if (e_ != cudaSuccess)                                                                      \
        {                                                                                           \
            fprintf(stderr, "CUDA error %s: %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
            exit(1);                                                                                \
        }                                                                                           \
    } while (0)

__global__ void shuffleBroadcat(int *out, int *in, int const srcLane, int width)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int value = in[idx];
    value = __shfl_sync(MASK_ALL_32, value, srcLane, width);
    out[idx] = value;
}

__global__ void shuffleUp(int *out, int *in, int const delta, int width)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int value = in[idx];
    value = __shfl_up_sync(MASK_ALL_32, value, delta, width);
    out[idx] = value;
}

__global__ void shuffleDown(int *out, int *in, int const delta, int width)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int value = in[idx];
    value = __shfl_down_sync(MASK_ALL_32, value, delta, width);
    out[idx] = value;
}

__global__ void shuffleWrap(int *out, int *in, int const offset, int width)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int warpIdx = threadIdx.x % WARP_SIZE;
    int value = in[idx];
    value = __shfl_sync(MASK_ALL_32, value, warpIdx + offset, width);
    out[idx] = value;
}

__global__ void shuffleXor(int *out, int *in, int const mask, int width)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int value = in[idx];
    value = __shfl_xor_sync(MASK_ALL_32, value, mask, width);
    out[idx] = value;
}

void printResult(int *out, unsigned int count)
{
    printf("* result start: *\n");
    for (int i = 0; i < count; i++)
    {
        printf("%d ", out[i]);
    }
    printf("\n* result end *\n");
}

int runTest()
{
    const size_t bytes = (size_t)TOTAL_SIZE * sizeof(int);

    int *h_in = (int *)malloc(bytes);
    int *h_out = (int *)malloc(bytes);
    if (!h_in || !h_out)
    {
        fprintf(stderr, "Not enough memory\n");
        return 1;
    }

    for (int i = 0; i < TOTAL_SIZE; i++)
    {
        h_in[i] = i % BLOCK_SIZE;
    }

    int *d_in;
    CHECK(cudaMalloc((void **)&d_in, bytes));
    int *d_out;
    CHECK(cudaMalloc((void **)&d_out, bytes));

    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    int testWidth = TEST_WIDTH_32;
    printf("=== shuffleBroadcat width:%d ===\n", testWidth);
    shuffleBroadcat<<<GRID_SIZE, BLOCK_SIZE>>>(d_out, d_in, TEST_SOURCE_LANE, testWidth);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    printResult(h_out, TOTAL_SIZE);

    testWidth = TEST_WIDTH_16;
    printf("=== shuffleBroadcat width:%d ===\n", testWidth);
    shuffleBroadcat<<<GRID_SIZE, BLOCK_SIZE>>>(d_out, d_in, TEST_SOURCE_LANE, testWidth);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    printResult(h_out, TOTAL_SIZE);

    printf("=== shuffleUp width:%d ===\n", testWidth);
    shuffleUp<<<GRID_SIZE, BLOCK_SIZE>>>(d_out, d_in, TEST_LANE_DELTA, testWidth);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    printResult(h_out, TOTAL_SIZE);

    printf("=== shuffleDown width:%d ===\n", testWidth);
    shuffleDown<<<GRID_SIZE, BLOCK_SIZE>>>(d_out, d_in, TEST_LANE_DELTA, testWidth);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    printResult(h_out, TOTAL_SIZE);

    printf("=== shuffleWrap width:%d ===\n", testWidth);
    shuffleWrap<<<GRID_SIZE, BLOCK_SIZE>>>(d_out, d_in, TEST_LANE_OFFSET, testWidth);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    printResult(h_out, TOTAL_SIZE);

    int xor_mask = TEST_XOR_MASK_1;
    printf("=== shuffleXor width:%d xor_mask:%d ===\n", testWidth, xor_mask);
    shuffleXor<<<GRID_SIZE, BLOCK_SIZE>>>(d_out, d_in, xor_mask, testWidth);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    printResult(h_out, TOTAL_SIZE);

    xor_mask = TEST_XOR_MASK_3;
    printf("=== shuffleXor width:%d xor_mask:%d ===\n", testWidth, xor_mask);
    shuffleXor<<<GRID_SIZE, BLOCK_SIZE>>>(d_out, d_in, xor_mask, testWidth);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    printResult(h_out, TOTAL_SIZE);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}

int main()
{
    runTest();
    return 0;
}
