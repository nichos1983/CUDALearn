#!/bin/sh
nvcc -O3 -lineinfo -Wno-deprecated-gpu-targets vec_add.cu -o ./debug/vec_add
