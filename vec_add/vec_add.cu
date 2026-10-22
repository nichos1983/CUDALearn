#include <stdio.h>
#include <stdlib.h>

#define ARRAY_NUM_LOG 26
#define ARRAY_NUM (1 << ARRAY_NUM_LOG)
#define BLOCK_SIZE 128
#define TEST_KERNEL_VERSION 0

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

template <class K>
static int persistentGrid(K k, int sms, int n)
{
    int blockPerSM = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blockPerSM, k, BLOCK_SIZE, 0));
    int blockTotal = blockPerSM * sms;
    int need = (n + BLOCK_SIZE - 1) / BLOCK_SIZE; // 小规模时不要启动比需要更多的 block
    return std::max(1, std::min(blockTotal, need));
}

__global__ void vecAddKernel_v0(const float *a, const float *b,
                                float *c, int n)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n)
    {
        c[i] = 4.0f * a[i] + b[i];
    }
}

__global__ void vecAddKernel_v1(const float *a, const float *b,
                                float *c, int n)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n)
    {
        c[i] = 4.0f * a[i] + b[i];
    }
}

int getGridDim(int version, int sms, int n)
{
    int result = 0;
    switch (version)
    {
        case 0:
            result = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            break;
        case 1:
            result = persistentGrid(vecAddKernel_v1, sms, n);
            break;
    }
    return result;
}

void vecAdd(float *a_h, float *b_h, float *c_h, int bytes, int sms, int version)
{
    float *a_d, *b_d, *c_d;
    int gridDim;

    CHECK(cudaMalloc((void **)&a_d, bytes));
    CHECK(cudaMalloc((void **)&b_d, bytes));
    CHECK(cudaMalloc((void **)&c_d, bytes));

    cudaMemcpy(a_d, a_h, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, bytes, cudaMemcpyHostToDevice);

    printf("Before call kernel\n");

    switch (version)
    {
        case 0:
            gridDim = (ARRAY_NUM + BLOCK_SIZE - 1) / BLOCK_SIZE;
            vecAddKernel_v0<<<gridDim, BLOCK_SIZE>>>(a_d, b_d, c_d, ARRAY_NUM);
            break;
        case 1:
            gridDim = persistentGrid(vecAddKernel_v1, sms, ARRAY_NUM);
            vecAddKernel_v1<<<gridDim, BLOCK_SIZE>>>(a_d, b_d, c_d, ARRAY_NUM);
            break;
    }

    printf("gridDim: %d\n", gridDim);

    cudaDeviceSynchronize();
    printf("After call kernel\n");

    cudaMemcpy(c_h, c_d, bytes, cudaMemcpyDeviceToHost);

    printf("After call memcpy\n");

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}

int runTest()
{
    const size_t bytes = (size_t)ARRAY_NUM * sizeof(float);

    int dev = 0, sms = 0, memClockKHz = 0, busBits = 0;
    CHECK(cudaGetDevice(&dev));
    CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    CHECK(cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, dev));
    CHECK(cudaDeviceGetAttribute(&busBits, cudaDevAttrGlobalMemoryBusWidth, dev));

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, dev));
    double peak = 2.0 * memClockKHz * 1e3 * (busBits / 8.0) / 1e9; // 理论峰值带宽GB/s（DDR 因子 2）
    printf("GPU: %s | SM数: %d | 理论显存带宽: %.0f GB/s | n = 2^%d = %d (每个数组 %.2f MB)\n",
           prop.name, sms, peak, ARRAY_NUM_LOG, ARRAY_NUM, bytes / 1048576.0);

    float *ha = (float *)malloc(bytes);
    float *hb = (float *)malloc(bytes);
    float *hc = (float *)malloc(bytes);
    if (!ha || !hb || !hc)
    {
        fprintf(stderr, "Not enough memory\n");
        return 1;
    }

    for (int i = 0; i < ARRAY_NUM; i++)
    {
        ha[i] = (float)(i % 1000);
        hb[i] = (float)(i % 500) * 0.5f;
    }

    vecAdd(ha, hb, hc, bytes, sms, TEST_KERNEL_VERSION);

    printf("Result start:\n");
    for (int i = 0; i < ARRAY_NUM; i += ARRAY_NUM / 16)
    {
        printf("i: %d, value: %.2f ", i, hc[i]);
    }
    printf("\nResult end\n");

    free(ha);
    free(hb);
    free(hc);

    return 0;
}

int main()
{
    runTest();
    return 0;
}
