#!/bin/sh
nvcc -O3 -lineinfo -Wno-deprecated-gpu-targets shared_mem.cu -o ./debug/shared_mem
