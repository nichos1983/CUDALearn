#include "benchmark.cuh"

void SGEMMBenchmark::print_results() const {
    printf("\n");
    printf("============================================================\n");
    printf("                SGEMM Benchmark Results");
    printf("============================================================\n");

    printf("| %-20s | %-18s | %11s | %15s | %13s | %4s | %15s |\n", "Kernel",
           "Dimensions", "Time", "GFLOPS", "Bandwidth", "Pass", "Max Error");

    for (const auto& result : _results) {
        result.print();
    }

    printf("============================================================\n");
}