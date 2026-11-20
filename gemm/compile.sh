#!/bin/sh
mkdir -p debug
nvcc -O3 -arch=sm_86 -lineinfo -Wno-deprecated-gpu-targets ./src/main.cu -o ./debug/main
