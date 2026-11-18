#pragma once

#define CHECK(x)                                                              \
    do {                                                                      \
        cudaError_t err = (x);                                                \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s: %d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                 \
            exit(1);                                                          \
        }                                                                     \
    } while (0)
