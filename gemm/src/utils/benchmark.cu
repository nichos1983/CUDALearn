#include "benchmark.cuh"

void SGEMMBenchmark::print_results() const {
    printf("\n");
    printf("============================================================\n");
    printf("                SGEMM Benchmark Results\n");
    printf("============================================================\n");

    printf("| %-18s | %-14s | %-16s | %-15s | %-13s | %-4s | %-9s | %-8s |\n",
           "Kernel", "Dimensions", "Time", "GFLOPS", "Bandwidth", "Pass",
           "Err Count", "Max Err");

    for (const auto &result : _results) {
        result.print();
    }

    printf("============================================================\n");
}