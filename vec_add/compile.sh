#!/bin/sh
nvcc -O2 -lineinfo -Wno-deprecated-gpu-targets vec_add.cu -o ./debug/vec_add
