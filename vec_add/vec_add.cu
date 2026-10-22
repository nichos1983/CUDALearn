#include <stdio.h>
#include <stdlib.h>

#define TEST_KERNEL_VERSION 1

#define ARRAY_NUM_LOG 26
#define ARRAY_NUM (1 << ARRAY_NUM_LOG)
#define BLOCK_SIZE 128
#define WARM_UP_COUNT 5
#define TEST_KERNEL_COUNT 20

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
    printf("blockPerSM:%d\n", blockPerSM);
    int blockTotal = blockPerSM * sms;
    int need = (n + BLOCK_SIZE - 1) / BLOCK_SIZE; // 小规模时不要启动比需要更多的 block
    return std::max(1, std::min(blockTotal, need));
}

__global__ void vecAddKernel_v0(const float* __restrict__ a, const float* __restrict__ b,
                                float *c, int n)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n)
    {
        c[i] = 4.0f * a[i] + b[i];
    }
}

__global__ void vecAddKernel_v1(const float* __restrict__ a, const float* __restrict__ b,
                                float *c, int n)
{
    int stride = gridDim.x * blockDim.x;
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < n; i += stride)
    {
        c[i] = 4.0f * a[i] + b[i];
    }
}

void vecAddVersion(float * __restrict__ a_d, float * __restrict__ b_d, float *c_d, int version, int myGridDim)
{
    switch (version)
    {
        case 0:
            vecAddKernel_v0<<<myGridDim, BLOCK_SIZE>>>(a_d, b_d, c_d, ARRAY_NUM);
            break;
        case 1:
            vecAddKernel_v1<<<myGridDim, BLOCK_SIZE>>>(a_d, b_d, c_d, ARRAY_NUM);
            break;
        default:
            break;
    }
}

void vecAdd(float *a_h, float *b_h, float *c_h, int bytes, int sms, int version)
{
    float *a_d, *b_d, *c_d;
    int myGridDim;

    CHECK(cudaMalloc((void **)&a_d, bytes));
    CHECK(cudaMalloc((void **)&b_d, bytes));
    CHECK(cudaMalloc((void **)&c_d, bytes));

    cudaMemcpy(a_d, a_h, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, bytes, cudaMemcpyHostToDevice);

    printf("Before call kernel version: %d\n", version);

    switch (version)
    {
        case 0:
            myGridDim = (ARRAY_NUM + BLOCK_SIZE - 1) / BLOCK_SIZE;
            break;
        case 1:
            myGridDim = persistentGrid(vecAddKernel_v1, sms, ARRAY_NUM);
            break;
        default:
            fprintf(stderr, "version error:%d\n", version);
            break;
    }
    printf("myGridDim: %d\n", myGridDim);

    printf("warm up start\n");
    for (int i = 0; i < WARM_UP_COUNT; i++)
    {
        vecAddVersion(a_d, b_d, c_d, version, myGridDim);
    }
    cudaDeviceSynchronize();
    printf("warm up end\n");

    printf("test kernel start, count:%d\n", TEST_KERNEL_COUNT);
    for (int i = 0; i < TEST_KERNEL_COUNT; i++)
    {
        vecAddVersion(a_d, b_d, c_d, version, myGridDim);
    }
    cudaDeviceSynchronize();
    printf("test kernel end\n");

    cudaMemcpy(c_h, c_d, bytes, cudaMemcpyDeviceToHost);

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
