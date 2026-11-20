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

struct BenchmarkMatrixSize {
    int M;
    int K;
    int N;
};

inline constexpr int WARMUP_RUNS_OF_DEFAULT_MODE = 5;
inline constexpr int BENCHMARK_RUNS_OF_DEFAULT_MODE = 10;

constexpr std::array<BenchmarkMatrixSize, 6> MATRIX_SIZES_OF_DEFAULT_MODE = {
    // Square matrices
    {{512, 512, 512},
     {1024, 1024, 1024},
     {2048, 2048, 2048},
     {4096, 4096, 4096},
     // Non-square matrices
     {1024, 512, 2048},
     {2048, 4096, 1024}}};

inline constexpr int WARMUP_RUNS_OF_PROFILE_MODE = 3;
inline constexpr int BENCHMARK_RUNS_OF_PROFILE_MODE = 5;

constexpr std::array<BenchmarkMatrixSize, 6> MATRIX_SIZES_OF_PROFILE_MODE = {
    // Square matrices
    {{512, 512, 512}, {1024, 1024, 1024}, {2048, 2048, 2048}}};

void naive_kernel(const float* A, const float* B, float* C, int M, int K,
                  int N) {
    launch_naive_sgemm<32>(A, B, C, M, K, N);
}

void run_benchmark(int M, int K, int N, int warmup_runs, int benchmark_runs) {
    printf("\n");
    printf("============================================================\n");
    printf("                Benchmarking %d x %d x %d\n", M, K, N);
    printf("============================================================\n");

    SGEMMBenchmark benchmark;

    printf("\nRunning cuBLAS for reference...\n");
    benchmark.run_cublas(M, K, N, warmup_runs, benchmark_runs);

    printf("Running naive SGEMM...\n");
    benchmark.run_kernel("Naive", naive_kernel, M, K, N, warmup_runs,
                         benchmark_runs);

    benchmark.print_results();
}

int main(int argc, char* argv[]) {
    printf("\n");
    printf("============================================================\n");
    printf("                SGEMM Benchmark Start...\n");
    printf("============================================================\n");

    print_gpu_info();

    // When profile mode is enabled, run less iterations and test smaller
    // matrices to save time.
    bool use_profile_mode = false;
    if (argc > 1 && std::string(argv[1]) == "-profile") {
        use_profile_mode = true;
    }

    printf("Using %s mode...\n", use_profile_mode ? "profile" : "default");

    int warmup_runs = use_profile_mode ? WARMUP_RUNS_OF_PROFILE_MODE
                                       : WARMUP_RUNS_OF_DEFAULT_MODE;
    int benchmark_runs = use_profile_mode ? BENCHMARK_RUNS_OF_PROFILE_MODE
                                          : BENCHMARK_RUNS_OF_DEFAULT_MODE;
    const std::array<BenchmarkMatrixSize, 6>& matrix_sizes =
        use_profile_mode ? MATRIX_SIZES_OF_PROFILE_MODE
                         : MATRIX_SIZES_OF_DEFAULT_MODE;

    for (const auto& size : matrix_sizes) {
        run_benchmark(size.M, size.K, size.N, warmup_runs, benchmark_runs);
    }

    printf("\n");
    printf("============================================================\n");
    printf("                SGEMM Benchmark Complete!\n");
    printf("============================================================\n");
    return 0;
}
