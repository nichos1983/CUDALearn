#!/bin/sh
mkdir -p profile
nsys profile --force-overwrite=true -o ./profile/gemm_nsys ./debug/main -profile
