#!/bin/sh
mkdir -p profile
ncu --set detailed --import-source no -f -o ./profile/gemm_ncu ./debug/main -profile
# --set full will generate big size profile file
# ncu --set full -f -o ./profile/gemm_ncu ./debug/main -profile
