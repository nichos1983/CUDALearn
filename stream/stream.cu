#include <stdio.h>
#include <stdlib.h>
#include <string>

#define TEST_BASIC_REPEAT_COUNT 10000

#define TEST_GRID_DIM_X 128
#define TEST_BLOCK_DIM_X 128

#define TEST_NUM_KERNELS 64
#define TEST_NUM_STREAMS 12
#define TEST_REPEAT_COUNT 100000

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

int printBasicInfo()
{
    int deviceCount = 0;
    CHECK(cudaGetDeviceCount(&deviceCount));

    for (int i = 0; i < deviceCount; i++)
    {
        cudaDeviceProp prop;
        CHECK(cudaGetDeviceProperties(&prop, i));
        printf("Device: %d, name: %s, asyncEngineCount:%d\n", i, prop.name, prop.asyncEngineCount);
        printf("streamPrioritiesSupported: %s\n", prop.streamPrioritiesSupported ? "Yes" : "No");

        if (prop.streamPrioritiesSupported)
        {
            int leastPriority, greatestPriority;
            CHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
            printf("stream priority：[%d, %d]\n", greatestPriority, leastPriority);
        }
    }

    return 0;
}

__global__ void workKernel(float* data, int repeat) {
    // printf("Device: workKernel called.\n");

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float val = static_cast<float>(idx);
    for (int i = 0; i < repeat; ++i) {
        val = sinf(val) * 0.99f + cosf(val) * 0.01f;
    }

    // avoid compile optimization
    if (idx == 0) {
        data[0] = val;
    }
}

int testBasicStream()
{
    printf("=== testBasicStream start. ===\n");

    float *d_data;
    CHECK(cudaMalloc(&d_data, sizeof(float)));

    cudaStream_t myStream;
    CHECK(cudaStreamCreate(&myStream));
    printf("CPU: succeed to create stream.\n");

    workKernel<<<1, 1, 0, myStream>>>(d_data, TEST_BASIC_REPEAT_COUNT);
    printf("CPU: called simpleKernel.\n");

    printf("CPU: waiting stream.\n");
    CHECK(cudaStreamSynchronize(myStream));
    printf("CPU: synced stream.\n");

    CHECK(cudaStreamDestroy(myStream));
    printf("CPU: destroyed stream.\n");

    CHECK(cudaFree(d_data));

    printf("=== testBasicStream end. ===\n");

    return 0;
}

int testMultiStream(bool useMultiStream)
{
    printf("=== testMultiStream start. ===\n");

    float* d_data;
    CHECK(cudaMalloc(&d_data, sizeof(float)));

    if (!useMultiStream)
    {
        printf("Default stream mode.\n");
        for (int i = 0; i < TEST_NUM_KERNELS; i++)
        {
            workKernel<<<TEST_GRID_DIM_X, TEST_BLOCK_DIM_X, 0, 0>>>(d_data, TEST_REPEAT_COUNT);
        }
    }
    else
    {
        printf("Multi stream mode.\n");

        cudaStream_t streams[TEST_NUM_STREAMS];

        for (int i = 0; i < TEST_NUM_STREAMS; i++)
        {
            CHECK(cudaStreamCreate(&streams[i]));
        }

        printf("created streams.\n");

        for (int i = 0; i < TEST_NUM_KERNELS; i++)
        {
            workKernel<<<TEST_GRID_DIM_X, TEST_BLOCK_DIM_X, 0, streams[i % TEST_NUM_STREAMS]>>>(d_data, TEST_REPEAT_COUNT);
        }

        printf("launched kernels.\n");

        for (int i = 0; i < TEST_NUM_STREAMS; i++)
        {
            CHECK(cudaStreamDestroy(streams[i]));
        }

        printf("destroyed streams.\n");
    }
    
    CHECK(cudaDeviceSynchronize());
    printf("Device synced end.\n");

    CHECK(cudaFree(d_data));

    printf("=== testMultiStream end. ===\n");

    return 0;
}

int main(int argc, char *argv[])
{
    bool useMultiStream = false;
    if (argc > 1 && std::string(argv[1]) == "multi")
    {
        useMultiStream = true;
    }
    printf("main: useMultiStream: %d\n", useMultiStream);

    printBasicInfo();
    testBasicStream();
    testMultiStream(useMultiStream);

    return 0;
}
