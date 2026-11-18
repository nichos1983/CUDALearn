#include <stdio.h>
#include <stdlib.h>

#include <string>

#include "../common/common.cuh"

#define TEST_BASIC_REPEAT_COUNT 10000

#define TEST_GRID_DIM_X 128
#define TEST_BLOCK_DIM_X 128

#define TEST_NUM_KERNELS 64
#define TEST_NUM_STREAMS 12
#define TEST_REPEAT_COUNT 100000

#define TEST_MULTI_ADD_MAX_INIT_VALUE 65536
#define TEST_MULTI_ADD_REPEAT_COUNT 128
#define TEST_MULTI_ADD_NUM_ELEMENTS (1 << 24)
#define TEST_MULTI_ADD_BLOCK_DIM_X 128
#define TEST_MULTI_ADD_STREAM_NUM 4

int printBasicInfo() {
    int deviceCount = 0;
    CHECK(cudaGetDeviceCount(&deviceCount));

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        CHECK(cudaGetDeviceProperties(&prop, i));
        printf("Device: %d, name: %s, asyncEngineCount:%d\n", i, prop.name, prop.asyncEngineCount);
        printf("streamPrioritiesSupported: %s\n", prop.streamPrioritiesSupported ? "Yes" : "No");

        if (prop.streamPrioritiesSupported) {
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

int testBasicStream() {
    printf("=== testBasicStream start. ===\n");

    float* d_data;
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

int testMultiStream(bool useMultiStream) {
    printf("=== testMultiStream start. ===\n");

    float* d_data;
    CHECK(cudaMalloc(&d_data, sizeof(float)));

    if (!useMultiStream) {
        printf("Default stream mode.\n");
        for (int i = 0; i < TEST_NUM_KERNELS; i++) {
            workKernel<<<TEST_GRID_DIM_X, TEST_BLOCK_DIM_X, 0, 0>>>(d_data, TEST_REPEAT_COUNT);
        }
    } else {
        printf("Multi stream mode.\n");

        cudaStream_t streams[TEST_NUM_STREAMS];

        for (int i = 0; i < TEST_NUM_STREAMS; i++) {
            CHECK(cudaStreamCreate(&streams[i]));
        }

        printf("created streams.\n");

        for (int i = 0; i < TEST_NUM_KERNELS; i++) {
            workKernel<<<TEST_GRID_DIM_X, TEST_BLOCK_DIM_X, 0, streams[i % TEST_NUM_STREAMS]>>>(
                d_data, TEST_REPEAT_COUNT);
        }

        printf("launched kernels.\n");

        for (int i = 0; i < TEST_NUM_STREAMS; i++) {
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

__global__ void sumArrays(const float* a, const float* b, float* c, const int num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num) {
        for (int i = 0; i < TEST_MULTI_ADD_REPEAT_COUNT; i++) {
            c[idx] = a[idx] + b[idx];
        }
    }
}

void initArray(float* array, unsigned int num) {
    for (int i = 0; i < num; i++) {
        array[i] = (float)(i % TEST_MULTI_ADD_MAX_INIT_VALUE);
    }
}

int testMultiAddStream(bool useNonBlocking) {
    printf("=== testMultiAddStream start. ===\n");

    int elementCount = TEST_MULTI_ADD_NUM_ELEMENTS;
    int bytes = elementCount * sizeof(float);

    float *h_a, *h_b, *h_c;
    CHECK(cudaHostAlloc((void**)&h_a, bytes, cudaHostAllocDefault));
    CHECK(cudaHostAlloc((void**)&h_b, bytes, cudaHostAllocDefault));
    CHECK(cudaHostAlloc((void**)&h_c, bytes, cudaHostAllocDefault));

    initArray(h_a, elementCount);
    initArray(h_b, elementCount);

    float *d_a, *d_b, *d_c;
    CHECK(cudaMalloc((float**)&d_a, bytes));
    CHECK(cudaMalloc((float**)&d_b, bytes));
    CHECK(cudaMalloc((float**)&d_c, bytes));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    dim3 block(TEST_MULTI_ADD_BLOCK_DIM_X);
    dim3 grid((elementCount - block.x + 1) / block.x);
    printf("grid: (%d, %d) block: (%d, %d)\n", grid.x, grid.y, block.x, block.y);

    // sequential test
    CHECK(cudaEventRecord(start, 0));
    CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));

    float memcpyH2DTime;
    CHECK(cudaEventElapsedTime(&memcpyH2DTime, start, stop));

    CHECK(cudaEventRecord(start, 0));
    sumArrays<<<grid, block>>>(d_a, d_b, d_c, elementCount);
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));

    float kernelTime;
    CHECK(cudaEventElapsedTime(&kernelTime, start, stop));

    CHECK(cudaEventRecord(start, 0));
    CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));

    float memcpyD2HTime;
    CHECK(cudaEventElapsedTime(&memcpyD2HTime, start, stop));
    float sequentialTotal = kernelTime + memcpyH2DTime + memcpyD2HTime;

    printf("\nMeasured timings (throughput):\n");
    printf(" Memcpy H2D\t\t: %f ms (%f GB/s)\n", memcpyH2DTime, (bytes * 1e-6) / memcpyH2DTime);
    printf(" Memcpy D2H\t\t: %f ms (%f GB/s)\n", memcpyD2HTime, (bytes * 1e-6) / memcpyD2HTime);
    printf(" Kernel\t\t\t: %f ms (%f GB/s)\n", kernelTime, (bytes * 2e-6) / kernelTime);
    printf(" Total\t\t\t: %f ms (%f GB/s)\n", sequentialTotal, (bytes * 2e-6) / sequentialTotal);

    // parallel test
    int elementCountPerStream = elementCount / TEST_MULTI_ADD_STREAM_NUM;
    size_t bytesPerStream = elementCountPerStream * sizeof(float);
    grid.x = (elementCountPerStream + block.x - 1) / block.x;

    cudaStream_t stream[TEST_MULTI_ADD_STREAM_NUM];

    for (int i = 0; i < TEST_MULTI_ADD_STREAM_NUM; i++) {
        if (useNonBlocking) {
            CHECK(cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking));
        } else {
            CHECK(cudaStreamCreate(&stream[i]));
        }
    }

    CHECK(cudaEventRecord(start, 0));

    for (int i = 0; i < TEST_MULTI_ADD_STREAM_NUM; i++) {
        int ioffset = i * elementCountPerStream;
        CHECK(cudaMemcpyAsync(&d_a[ioffset], &h_a[ioffset], bytesPerStream, cudaMemcpyHostToDevice,
                              stream[i]));
        CHECK(cudaMemcpyAsync(&d_b[ioffset], &h_b[ioffset], bytesPerStream, cudaMemcpyHostToDevice,
                              stream[i]));
        sumArrays<<<grid, block, 0, stream[i]>>>(&d_a[ioffset], &d_b[ioffset], &d_c[ioffset],
                                                 bytesPerStream);
        CHECK(cudaMemcpyAsync(&h_c[ioffset], &d_c[ioffset], bytesPerStream, cudaMemcpyDeviceToHost,
                              stream[i]));
    }

    // Add call cudaDeviceSynchronize() comparing to original code on book.
    // Otherwise, the stop event triggers when the 1st stream ends...
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));

    float executionTime;
    CHECK(cudaEventElapsedTime(&executionTime, start, stop));

    printf("\nActual results from overlapped data transfers:\n");
    printf(" overlap with %d streams : %f ms (%f GB/s)\n", TEST_MULTI_ADD_STREAM_NUM, executionTime,
           (bytes * 2e-6) / executionTime);
    printf(" speedup\t\t: %f%% \n", ((sequentialTotal - executionTime) * 100.0f) / sequentialTotal);

    CHECK(cudaGetLastError());

    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));

    CHECK(cudaFreeHost(h_a));
    CHECK(cudaFreeHost(h_b));
    CHECK(cudaFreeHost(h_c));

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    for (int i = 0; i < TEST_MULTI_ADD_STREAM_NUM; ++i) {
        CHECK(cudaStreamDestroy(stream[i]));
    }

    CHECK(cudaDeviceReset());

    printf("=== testMultiAddStream end. ===\n");
    return 0;
}

int main(int argc, char* argv[]) {
    // bool useMultiStream = false;
    // if (argc > 1 && std::string(argv[1]) == "multi")
    // {
    //     useMultiStream = true;
    // }
    // printf("main: useMultiStream: %d\n", useMultiStream);

    // printBasicInfo();
    // testBasicStream();
    // testMultiStream(useMultiStream);

    bool useNonBlocking = false;
    if (argc > 1 && std::string(argv[1]) == "nonblocking") {
        useNonBlocking = true;
    }
    printf("main: useNonBlocking: %d\n", useNonBlocking);

    testMultiAddStream(useNonBlocking);

    return 0;
}
