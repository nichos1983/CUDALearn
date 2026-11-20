#include "benchmark.cuh"

void SGEMMBenchmark::print_results() const {
    printf("\n");
    printf("============================================================\n");
    printf("                SGEMM Benchmark Results\n");
    printf("============================================================\n");

    printf("| %-14s | %-14s | %-16s | %-15s | %-11s | %-4s | %-4s | %-8s |\n",
           "Kernel", "Dimensions", "Time", "GFLOPS", "Bandwidth", "Pass",
           "#Err", "Max Err");

    for (const auto& result : _results) {
        result.print();
    }

    printf("============================================================\n");
}