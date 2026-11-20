#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <type_traits>

#define CUDA_CHECK(x)                                                         \
    do {                                                                      \
        cudaError_t err = (x);                                                \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s: %d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                 \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

#define CUBLAS_CHECK(x)                                               \
    do {                                                              \
        cublasStatus_t status = (x);                                  \
        if (status != CUBLAS_STATUS_SUCCESS) {                        \
            fprintf(stderr, "cuBLAS error at %s: %d: %d\n", __FILE__, \
                    __LINE__, static_cast<int>(status));              \
            exit(1);                                                  \
        }                                                             \
    } while (0)

void print_gpu_info();
void print_matrix(const float* data, int rows, int cols, const char* name);

void init_random_matrix(float* data, int rows, int cols, float min_val = -1.0f,
                        float max_val = 1.0f, unsigned int seed = 42);
void init_zero_matrix(float* data, int rows, int cols);
void init_identity_matrix(float* data, int n);
void init_constant_matrix(float* data, int rows, int cols, float val);

/**
 * @brief Calculate ceiling division. The common case is to calculate the number
 * of blocks needed to process a number of elements.
 *
 * @tparam A Type of the numerator
 * @tparam B Type of the denominator
 * @param a The numerator, e.g. number of elements.
 * @param b The denominator, e.g. block size.
 * @return __host__ constexpr The ceiling division, e.g. number of blocks.
 */
template <typename A, typename B>
__host__ __device__ constexpr auto ceil_div(A a, B b) {
    static_assert(std::is_integral_v<A>, "A must be integral");
    static_assert(std::is_integral_v<B>, "B must be integral");
    return (a + b - 1) / b;
}
