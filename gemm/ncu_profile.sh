#!/bin/sh
mkdir -p profile
ncu --set full -f -o ./profile/gemm_ncu ./debug/main
