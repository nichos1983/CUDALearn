#!/bin/sh
mkdir -p debug
nvcc -O3 -arch=sm_86 -lineinfo -Wno-deprecated-gpu-targets warp_shuffle.cu -o ./debug/warp_shuffle
