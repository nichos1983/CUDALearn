#include <stdio.h>
#include <stdlib.h>

#define SHARED_BLOCK_DIM_X 32
#define SHARED_BLOCK_DIM_Y 32
#define SHARED_BLOCK_COUNT (1 << 8)
#define LOAD_STORE_ITER_COUNT 16

#define BLOCK_SIZE 128

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

__global__ void setRowReadRow(int *out)
{
    __shared__ int tile[SHARED_BLOCK_DIM_Y][SHARED_BLOCK_DIM_X];

    unsigned int idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

    for (int i = 0; i < LOAD_STORE_ITER_COUNT; i++)
    {
        tile[threadIdx.y][threadIdx.x] = idx;
        __syncthreads();
        out[idx] = tile[threadIdx.y][threadIdx.x];
    }
}

__global__ void setColReadCol(int *out)
{
    __shared__ int tile[SHARED_BLOCK_DIM_X][SHARED_BLOCK_DIM_Y];

    unsigned int idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    for (int i = 0; i < LOAD_STORE_ITER_COUNT; i++)
    {
        tile[threadIdx.x][threadIdx.y] = idx;
        __syncthreads();
        out[idx] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void setRowReadCol(int *out)
{
    __shared__ int tile[SHARED_BLOCK_DIM_X][SHARED_BLOCK_DIM_Y];

    unsigned int idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    for (int i = 0; i < LOAD_STORE_ITER_COUNT; i++)
    {
        tile[threadIdx.y][threadIdx.x] = idx;
        __syncthreads();
        out[idx] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void setRowReadColDynamic(int *out)
{
    extern __shared__ int tile[];
    unsigned int globalRowIdx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int sharedRowIdx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int sharedColIdx = threadIdx.x * blockDim.x + threadIdx.y;

    for (int i = 0; i < LOAD_STORE_ITER_COUNT; i++)
    {
        tile[sharedRowIdx] = sharedRowIdx;
        __syncthreads();
        out[globalRowIdx] = tile[sharedColIdx];
    }
}

template<int PADDING_NUM>
__global__ void setRowReadColPad(int *out)
{
    __shared__ int tile[SHARED_BLOCK_DIM_X][SHARED_BLOCK_DIM_Y + PADDING_NUM];

    unsigned int idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    for (int i = 0; i < LOAD_STORE_ITER_COUNT; i++)
    {
        tile[threadIdx.y][threadIdx.x] = idx;
        __syncthreads();
        out[idx] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void setRowReadColDynamicPad(int *out)
{

}

void printResult(int *out)
{
    printf("* result start: *\n");
    for (int k = 0; k < 2; k++)
    {
        printf("--- loop %d ---\n", k);
        for (int i = 0; i < SHARED_BLOCK_DIM_X * 4; i++)
        {
            printf("%d ", out[i]);
        }
        printf("\n");
    }
    printf("\n* result end *\n");
}

int runTest()
{
    const size_t bytes = (size_t)SHARED_BLOCK_COUNT * SHARED_BLOCK_DIM_X * SHARED_BLOCK_DIM_Y * sizeof(int);

    int *h_out = (int *)malloc(bytes);
    if (!h_out)
    {
        fprintf(stderr, "Not enough memory\n");
        return 1;
    }

    int *d_out;
    CHECK(cudaMalloc((void **)&d_out, bytes));

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
    unsigned int sharedMemBytes = SHARED_BLOCK_DIM_X * SHARED_BLOCK_DIM_Y * sizeof(int);
    setRowReadColDynamic<<<SHARED_BLOCK_COUNT, blockDim, sharedMemBytes>>>(d_out);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    printResult(h_out);

    printf("=== setRowReadCol pad 1 ===\n");
    setRowReadColPad<1><<<SHARED_BLOCK_COUNT, blockDim>>>(d_out);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    printResult(h_out);

    printf("=== setRowReadCol pad 2 ===\n");
    setRowReadColPad<2><<<SHARED_BLOCK_COUNT, blockDim>>>(d_out);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    printResult(h_out);

    cudaFree(d_out);
    free(h_out);

    return 0;
}

int main()
{
    runTest();
    return 0;
}
