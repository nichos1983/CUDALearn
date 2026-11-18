// Minimal CUDA launch syntax stub for clangd on machines without a CUDA toolkit.
// Inert under real CUDA toolchains.
// Declarations are compatible with __cudaPushCallConfiguration in crt/device_functions.h.
#pragma once

struct dim3;
struct CUstream_st;
extern "C" int cudaConfigureCall(dim3 gridDim, dim3 blockDim, unsigned long sharedMem = 0,
                                 struct CUstream_st* stream = 0);
