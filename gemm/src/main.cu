/**
 * SGEMM Optimization Benchmark.
 *
 * Benchmarks:
 * 1. Naive
 * 2. Tiled
 * 3. Bank conflict free
 * 4. Double buffer
 * 5. Tensor core
 * 6. cuBLAS
 */

#include <array>
#include <cstdio>
// #include <cmath>
// #include <cstdlib>
// #include <vector>

#include "kernels/naive_sgemm.cuh"
#include "utils/benchmark.cuh"

inline constexpr int WARMUP_RUNS = 5;
inline constexpr int BENCHMARK_RUNS = 10;

struct BenchmarkMatrixSize {
    int M;
    int K;
    int N;
};

inline constexpr std::array<BenchmarkMatrixSize, 6> DEFAULT_SIZES = {
    // Square matrices
    {{512, 512, 512},
     {1024, 1024, 1024},
     {2048, 2048, 2048},
     {4096, 4096, 4096},
     // Non-square matrices
     {1024, 512, 2048},
     {2048, 4096, 1024}}};

void naive_kernel(const float* A, const float* B, float* C, int M, int K,
                  int N) {
    launch_naive_sgemm<32>(A, B, C, M, K, N);
}

void run_benchmark(int M, int K, int N) {
    printf("\n");
    printf("============================================================\n");
    printf("                Benchmarking %d x %d x %d\n", M, K, N);
    printf("============================================================\n");

    SGEMMBenchmark benchmark;

    printf("\nRunning cuBLAS for reference...\n");
    benchmark.run_cublas(M, K, N, WARMUP_RUNS, BENCHMARK_RUNS);

    printf("Running naive SGEMM...\n");
    benchmark.run_kernel("Naive", naive_kernel, M, K, N, WARMUP_RUNS,
                         BENCHMARK_RUNS);

    benchmark.print_results();
}

int main() {
    printf("\n");
    printf("============================================================\n");
    printf("                SGEMM Benchmark Start...\n");
    printf("============================================================\n");

    print_gpu_info();

    for (const auto& size : DEFAULT_SIZES) {
        run_benchmark(size.M, size.K, size.N);
    }

    printf("\n");
    printf("============================================================\n");
    printf("                SGEMM Benchmark Complete!\n");
    printf("============================================================\n");
    return 0;
}
