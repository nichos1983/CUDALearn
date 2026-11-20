#!/bin/sh
set -e
mkdir -p debug
nvcc -O3 -arch=sm_86 -lineinfo -Wno-deprecated-gpu-targets \
    ./src/main.cu \
    ./src/utils/benchmark.cu \
    ./src/utils/verifier.cu \
    ./src/utils/cuda_utils.cu \
    -lcublas \
    -o ./debug/main

echo 'Build done.'
