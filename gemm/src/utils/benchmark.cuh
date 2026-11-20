#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <string>
#include <vector>

#include "constants.cuh"
#include "cuda_utils.cuh"
#include "verifier.cuh"

struct BenchmarkResult {
    std::string kernel_name;
    int M, K, N;
    float time_ms;
    float gflops;
    float bandwidth_gbps;
    bool is_reference;
    bool is_correct;
    int error_count;
    float max_error;

    void print() const {
        printf(
            "| %-14s | %4d %4d %4d | time=%8.3f ms | %8.2f GFLOPS | %6.2f "
            "GB/s | %-4s | %4d | %8.2e |\n",
            kernel_name.c_str(), M, K, N, time_ms, gflops, bandwidth_gbps,
            is_reference ? "REF" : (is_correct ? "PASS" : "FAIL"), error_count,
            max_error);
    }
};

/**
 * @brief SGEMM Benchmark Class
 *
 */
class SGEMMBenchmark {
public:
    SGEMMBenchmark() {
        CUDA_CHECK(cudaEventCreate(&_start));
        CUDA_CHECK(cudaEventCreate(&_stop));
        CUBLAS_CHECK(cublasCreate(&_cublas_handle));
    }

    ~SGEMMBenchmark() {
        CUDA_CHECK(cudaEventDestroy(_start));
        CUDA_CHECK(cudaEventDestroy(_stop));
        CUBLAS_CHECK(cublasDestroy(_cublas_handle));
    }

    /**
     * @brief Run a kernel and verify the result.
     */
    template <typename KernelFunc>
    BenchmarkResult run_kernel(const std::string& kernel_name,
                               KernelFunc kernel_func, int M, int K, int N,
                               int warmup_runs, int benchmark_runs,
                               float rtol = constants::RTOL,
                               float atol = constants::ATOL) {
        return run_internal(kernel_name, kernel_func, M, K, N, warmup_runs,
                            benchmark_runs, true, rtol, atol);
    }

    /**
     * @brief Run cuBLAS for reference, without verification.
     */
    inline BenchmarkResult run_cublas(int M, int K, int N, int warmup_runs,
                                      int benchmark_runs) {
        auto cublas_func = [this](const float* A, const float* B, float* C,
                                  int m, int k, int n) {
            float alpha = constants::SGEMM_ALPHA, beta = constants::SGEMM_BETA;
            CUBLAS_CHECK(cublasSgemm(_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     n, m, k, &alpha, B, n, A, k, &beta, C, n));
        };
        return run_internal("cuBLAS (ref)", cublas_func, M, K, N, warmup_runs,
                            benchmark_runs, false);
    }

    void print_results() const;

private:
    template <typename KernelFunc>
    BenchmarkResult run_internal(const std::string& kernel_name,
                                 KernelFunc kernel_func, int M, int K, int N,
                                 int warmup_runs, int benchmark_runs,
                                 bool is_verify = true,
                                 float rtol = constants::RTOL,
                                 float atol = constants::ATOL) {
        BenchmarkResult result;
        result.kernel_name = kernel_name;
        result.M = M;
        result.K = K;
        result.N = N;

        // Allocate host memory
        std::vector<float> h_A(M * K), h_B(K * N);
        // used when is_verify is true
        std::vector<float> h_C, h_C_ref;
        if (is_verify) {
            h_C.resize(M * N);
            h_C_ref.resize(M * N);
        }

        // Allocate device memory
        float *d_A, *d_B, *d_C;
        // used when is_verify is true
        float* d_C_ref;

        CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

        if (is_verify)
            CUDA_CHECK(cudaMalloc(&d_C_ref, M * N * sizeof(float)));

        init_random_matrix(h_A.data(), M, K, -1.0f, 1.0f,
                           constants::RANDOM_SEED1);
        init_random_matrix(h_B.data(), K, N, -1.0f, 1.0f,
                           constants::RANDOM_SEED2);

        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float),
                              cudaMemcpyHostToDevice));

        float alpha = constants::SGEMM_ALPHA, beta = constants::SGEMM_BETA;

        // When verification is needed, calculate reference result with cuBLAS
        if (is_verify) {
            CUBLAS_CHECK(cublasSgemm(_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     N, M, K, &alpha, d_B, N, d_A, K, &beta,
                                     d_C_ref, N));
        }

        // Warmup runs
        for (int i = 0; i < warmup_runs; ++i) {
            CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));
            kernel_func(d_A, d_B, d_C, M, K, N);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Benchmark runs
        CUDA_CHECK(cudaEventRecord(_start));
        for (int i = 0; i < benchmark_runs; ++i) {
            // Not call cudaMemset here, since beta is 0. Otherwise, cudaMemset
            // is necessary.
            kernel_func(d_A, d_B, d_C, M, K, N);
        }
        CUDA_CHECK(cudaEventRecord(_stop));
        CUDA_CHECK(cudaEventSynchronize(_stop));

        float time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&time_ms, _start, _stop));
        result.time_ms = time_ms / static_cast<float>(benchmark_runs);

        // 2.0f is for multiply and add
        double flops = 2.0 * M * N * K;
        result.gflops =
            static_cast<float>(flops / (result.time_ms * 1e-3) / 1e9);

        // Read A (M * K) and B (K * N), Write C (M * N)
        // It's just theoretical estimation.
        double bytes =
            static_cast<double>(M * K + K * N + M * N) * sizeof(float);
        result.bandwidth_gbps =
            static_cast<float>(bytes / (result.time_ms * 1e-3) / 1e9);

        if (is_verify) {
            CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_C_ref.data(), d_C_ref,
                                  M * N * sizeof(float),
                                  cudaMemcpyDeviceToHost));

            VerifyResult verify_result = SGEMMVerifier::verify(
                h_C.data(), h_C_ref.data(), M, N, rtol, atol);

            result.is_reference = false;
            result.is_correct = verify_result.is_passed();
            result.error_count = verify_result.error_count;
            result.max_error = verify_result.max_abs_error;
        } else {
            result.is_reference = true;
            result.is_correct = true;
            result.error_count = 0;
            result.max_error = 0.0f;
        }

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));

        if (is_verify)
            CUDA_CHECK(cudaFree(d_C_ref));

        _results.push_back(result);
        return result;
    }

    cudaEvent_t _start, _stop;
    cublasHandle_t _cublas_handle;
    std::vector<BenchmarkResult> _results;
};
