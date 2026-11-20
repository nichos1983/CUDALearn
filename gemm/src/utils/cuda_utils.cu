#include <cstdio>
#include <cstdlib>
#include <random>

#include "cuda_utils.cuh"

void print_gpu_info() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("\nGPU Device: %s\n", prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  SM Count: %d\n", prop.multiProcessorCount);
    printf("  Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max Thread Dimension: (%d, %d, %d)\n", prop.maxThreadsDim[0],
           prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  Max Grid Size: (%d, %d, %d)\n", prop.maxGridSize[0],
           prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  Shared Memory Per Block: %zu KB\n",
           prop.sharedMemPerBlock / 1024);
    printf("  Global Memory: %.2f GB\n",
           static_cast<float>(prop.totalGlobalMem) / (1024 * 1024 * 1024));
    printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    printf("  L2 Cache Size: %d KB\n", prop.l2CacheSize / 1024);
    printf("  Registers Per Block: %d\n", prop.regsPerBlock);
    printf("\n");
}

void print_matrix(const float *data, int rows, int cols, const char *name) {
    printf("Matrix %s (%d x %d):\n", name, rows, cols);
    for (int i = 0; i < rows && i < 8; ++i) {
        for (int j = 0; j < cols && j < 8; ++j) {
            printf("%8.4f ", data[i * cols + j]);
        }
        if (cols > 8)
            printf("...");
        printf("\n");
    }

    if (rows > 8)
        printf("...\n");
    printf("\n");
}

void init_random_matrix(float *data, int rows, int cols, float min_val,
                        float max_val, unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(min_val, max_val);

    for (int i = 0; i < rows * cols; ++i) {
        data[i] = dist(gen);
    }
}

void init_zero_matrix(float *data, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        data[i] = 0.0f;
    }
}

void init_identity_matrix(float *data, int n) {
    for (int i = 0; i < n * n; ++i) {
        data[i] = 0.0f;
    }
    for (int i = 0; i < n; ++i) {
        data[i * n + i] = 1.0f;
    }
}

void init_constant_matrix(float *data, int rows, int cols, float val) {
    for (int i = 0; i < rows * cols; ++i) {
        data[i] = val;
    }
}
