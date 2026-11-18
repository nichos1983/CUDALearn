#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../common/common.cuh"

#define MASK_ALL_32 0xFFFFFFFF

// Basic tests
#define GRID_SIZE 2
#define BLOCK_SIZE (32 * 4)
#define TOTAL_SIZE (GRID_SIZE * BLOCK_SIZE)

#define TEST_WIDTH_32 32
#define TEST_WIDTH_16 16

#define TEST_SOURCE_LANE 2
#define TEST_LANE_DELTA 3
#define TEST_LANE_OFFSET 4
#define TEST_XOR_MASK_1 1
#define TEST_XOR_MASK_3 3
#define TEST_XOR_MASK_7 7

// Reduce tests
#define REDUCE_SHARED_MEM_SIZE 32

#define REDUCE_GRID_SIZE 32
#define REDUCE_BLOCK_SIZE (32 * 4)
#define REDUCE_NUMBER_OF_GRID 6
#define REDUCE_TOTAL_SIZE (REDUCE_NUMBER_OF_GRID * REDUCE_GRID_SIZE * REDUCE_BLOCK_SIZE)

// Warp multiply reduce tests(float)
// grid 内一共有 MUL_TOTAL_WARPS 个 warp，每个 warp 输出 1 个部分积
#define MUL_GRID_SIZE 16
#define MUL_BLOCK_SIZE (32 * 4)
#define MUL_WARPS_PER_BLOCK (MUL_BLOCK_SIZE / 32)
#define MUL_TOTAL_WARPS (MUL_GRID_SIZE * MUL_WARPS_PER_BLOCK)
// 每个 warp 用 stride 循环处理 MUL_CHUNKS_PER_WARP 段连续元素，覆盖整个缓冲区
#define MUL_CHUNKS_PER_WARP 4
#define MUL_TOTAL_SIZE (MUL_TOTAL_WARPS * 32 * MUL_CHUNKS_PER_WARP)

// 输入数据的取值：base + step * (i % modulus)，保持在 1.0 附近防止连乘溢出
#define MUL_INPUT_BASE 1.0f
#define MUL_INPUT_STEP 0.0005f
#define MUL_INPUT_MODULUS 20

// #define DEBUG_REDUCE_PRINT
// #define DEBUG_MUL_PRINT

__global__ void shuffleBroadcat(int* out, int* in, int const srcLane, int width) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int value = in[idx];
    value = __shfl_sync(MASK_ALL_32, value, srcLane, width);
    out[idx] = value;
}

__global__ void shuffleUp(int* out, int* in, int const delta, int width) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int value = in[idx];
    value = __shfl_up_sync(MASK_ALL_32, value, delta, width);
    out[idx] = value;
}

__global__ void shuffleDown(int* out, int* in, int const delta, int width) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int value = in[idx];
    value = __shfl_down_sync(MASK_ALL_32, value, delta, width);
    out[idx] = value;
}

__global__ void shuffleWrap(int* out, int* in, int const offset, int width) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int warpIdx = threadIdx.x % warpSize;
    int value = in[idx];
    value = __shfl_sync(MASK_ALL_32, value, warpIdx + offset, width);
    out[idx] = value;
}

__global__ void shuffleXor(int* out, int* in, int const mask, int width) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int value = in[idx];
    value = __shfl_xor_sync(MASK_ALL_32, value, mask, width);
    out[idx] = value;
}

template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
#pragma unroll
    for (int offset = (warpSize >> 1); offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(MASK_ALL_32, val, offset);
        // val += __shfl_down_sync(MASK_ALL_32, val, offset);
    }

#ifdef DEBUG_REDUCE_PRINT
    if (blockIdx.x == 0) {
        printf("warpReduceSum, threadIdx.x: %d, val: %d\n", threadIdx.x, (int)val);
    }
#endif
    return val;
}

// warp 内所有元素相乘(butterfly / 异或归约)，结果落在每条 warp 的 lane 0 上
template <typename T>
__device__ __forceinline__ T warpReduceMul(T val) {
#pragma unroll
    for (int offset = (warpSize >> 1); offset > 0; offset >>= 1) {
        val *= __shfl_xor_sync(MASK_ALL_32, val, offset);
    }

#ifdef DEBUG_MUL_PRINT
    if (blockIdx.x == 0) {
        printf("warpReduceMul, threadIdx.x: %d, val: %f\n", threadIdx.x, (float)val);
    }
#endif
    return val;
}

template <typename T>
__device__ inline T blockReduceSum(T val) {
    static __shared__ T shared[REDUCE_SHARED_MEM_SIZE];

    const int laneIdx = threadIdx.x % warpSize;
    const int warpIdx = threadIdx.x / warpSize;

    val = warpReduceSum(val);
    __syncthreads();

    if (laneIdx == 0) {
        shared[warpIdx] = val;
    }
    __syncthreads();

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[laneIdx] : T(0);
    if (warpIdx == 0) {
        val = warpReduceSum(val);
    }

    return val;
}

template <typename T>
__global__ void deviceReduceSum(T* input, T* output, unsigned int num) {
    T sum = T(0);

#ifdef DEBUG_REDUCE_PRINT
    bool check = false;
    int checkNum = 0;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        check = true;
        printf("deviceReduceSum start, gridDim.x:%d, blockDim.x:%d, num:%d\n", gridDim.x,
               blockDim.x, num);
    }
#endif

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num;
         idx += blockDim.x * gridDim.x) {
        sum += input[idx];

#ifdef DEBUG_REDUCE_PRINT
        if (check) {
            checkNum++;
        }
#endif
    }

#ifdef DEBUG_REDUCE_PRINT
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("deviceReduceSum end, checkNum:%d\n", checkNum);
    }
#endif

    sum = blockReduceSum(sum);

    if (threadIdx.x == 0) {
        output[blockIdx.x] = sum;
    }
}

// 核函数：做 warp 内所有 float 元素相乘
// 每个 warp 负责若干段连续的 32 个元素(stride 循环覆盖整个缓冲区)，
// 段内用 warpReduceMul 求积，各段累乘后由 lane 0 写出到 out[globalWarpId]
__global__ void deviceWarpMulReduce(float* out, float* in, unsigned int num) {
    const unsigned int globalWarpId = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    const unsigned int numWarps = (gridDim.x * blockDim.x) / warpSize;
    const unsigned int laneIdx = threadIdx.x % warpSize;

    float prod = 1.0f;

    // warp 粒度的 grid-stride 循环：循环次数只与 warp 有关，
    // warp 内 32 个 lane 的迭代次数一致，因此全 mask 的 shuffle 是安全的
    for (unsigned int base = globalWarpId * warpSize; base < num; base += numWarps * warpSize) {
        unsigned int idx = base + laneIdx;
        // 越界的 lane 用乘法单位元 1.0f 填充，不能像求和那样填 0
        float value = (idx < num) ? in[idx] : 1.0f;
        prod *= warpReduceMul(value);
    }

    if (laneIdx == 0) {
        out[globalWarpId] = prod;
    }
}

void printResult(int* out, unsigned int count) {
    printf("* result start: *\n");
    for (int i = 0; i < count; i++) {
        printf("%d ", out[i]);
    }
    printf("\n* result end *\n");
}

void printResultFloat(float* out, unsigned int count) {
    printf("* result start: *\n");
    for (int i = 0; i < count; i++) {
        printf("%.6f ", out[i]);
    }
    printf("\n* result end *\n");
}

int runBasicTests() {
    const size_t bytes = (size_t)TOTAL_SIZE * sizeof(int);

    int* h_in = (int*)malloc(bytes);
    int* h_out = (int*)malloc(bytes);
    if (!h_in || !h_out) {
        fprintf(stderr, "Not enough memory\n");
        return 1;
    }

    for (int i = 0; i < TOTAL_SIZE; i++) {
        h_in[i] = i % BLOCK_SIZE;
    }

    int* d_in;
    CHECK(cudaMalloc((void**)&d_in, bytes));
    int* d_out;
    CHECK(cudaMalloc((void**)&d_out, bytes));

    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    int testWidth = TEST_WIDTH_32;
    printf("=== shuffleBroadcat width:%d ===\n", testWidth);
    shuffleBroadcat<<<GRID_SIZE, BLOCK_SIZE>>>(d_out, d_in, TEST_SOURCE_LANE, testWidth);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    printResult(h_out, TOTAL_SIZE);

    testWidth = TEST_WIDTH_16;
    printf("=== shuffleBroadcat width:%d ===\n", testWidth);
    shuffleBroadcat<<<GRID_SIZE, BLOCK_SIZE>>>(d_out, d_in, TEST_SOURCE_LANE, testWidth);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    printResult(h_out, TOTAL_SIZE);

    printf("=== shuffleUp width:%d ===\n", testWidth);
    shuffleUp<<<GRID_SIZE, BLOCK_SIZE>>>(d_out, d_in, TEST_LANE_DELTA, testWidth);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    printResult(h_out, TOTAL_SIZE);

    printf("=== shuffleDown width:%d ===\n", testWidth);
    shuffleDown<<<GRID_SIZE, BLOCK_SIZE>>>(d_out, d_in, TEST_LANE_DELTA, testWidth);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    printResult(h_out, TOTAL_SIZE);

    printf("=== shuffleWrap width:%d ===\n", testWidth);
    shuffleWrap<<<GRID_SIZE, BLOCK_SIZE>>>(d_out, d_in, TEST_LANE_OFFSET, testWidth);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    printResult(h_out, TOTAL_SIZE);

    int xor_mask = TEST_XOR_MASK_1;
    printf("=== shuffleXor width:%d xor_mask:%d ===\n", testWidth, xor_mask);
    shuffleXor<<<GRID_SIZE, BLOCK_SIZE>>>(d_out, d_in, xor_mask, testWidth);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    printResult(h_out, TOTAL_SIZE);

    xor_mask = TEST_XOR_MASK_3;
    printf("=== shuffleXor width:%d xor_mask:%d ===\n", testWidth, xor_mask);
    shuffleXor<<<GRID_SIZE, BLOCK_SIZE>>>(d_out, d_in, xor_mask, testWidth);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    printResult(h_out, TOTAL_SIZE);

    xor_mask = TEST_XOR_MASK_7;
    printf("=== shuffleXor width:%d xor_mask:%d ===\n", testWidth, xor_mask);
    shuffleXor<<<GRID_SIZE, BLOCK_SIZE>>>(d_out, d_in, xor_mask, testWidth);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    printResult(h_out, TOTAL_SIZE);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}

int runReduceTests() {
    const size_t bytes = (size_t)REDUCE_TOTAL_SIZE * sizeof(int);

    int* h_in = (int*)malloc(bytes);
    int* h_out = (int*)malloc(bytes);
    if (!h_in || !h_out) {
        fprintf(stderr, "Not enough memory\n");
        return 1;
    }

    for (int i = 0; i < REDUCE_TOTAL_SIZE; i++) {
        h_in[i] = i % BLOCK_SIZE;
    }

    int* d_in;
    CHECK(cudaMalloc((void**)&d_in, bytes));
    int* d_out;
    CHECK(cudaMalloc((void**)&d_out, bytes));

    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    printf("=== deviceReduceSum ===\n");
    deviceReduceSum<<<REDUCE_GRID_SIZE, REDUCE_BLOCK_SIZE>>>(d_in, d_out, REDUCE_TOTAL_SIZE);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    printResult(h_out, REDUCE_GRID_SIZE);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}

// 初始化 float 乘法测试的输入缓冲区
// 取值保持在 1.0 附近，避免大量元素连乘时 float 溢出成 inf
void fillMulInputData(float* buf, unsigned int count) {
    for (unsigned int i = 0; i < count; i++) {
        buf[i] = MUL_INPUT_BASE + MUL_INPUT_STEP * (float)(i % MUL_INPUT_MODULUS);
    }
}

// 调用 deviceWarpMulReduce，做整个缓冲区所有 float 元素相乘
int runWarpMulReduceTests() {
    const unsigned int num = MUL_TOTAL_SIZE;
    /* 每个 warp 输出 1 个部分积，所以输出数组长度为 warp 总数 */
    const unsigned int numWarps = MUL_TOTAL_WARPS;

    const size_t inBytes = (size_t)num * sizeof(float);
    const size_t outBytes = (size_t)numWarps * sizeof(float);

    float* h_in = (float*)malloc(inBytes);
    float* h_out = (float*)malloc(outBytes);
    if (!h_in || !h_out) {
        fprintf(stderr, "Not enough memory\n");
        return 1;
    }

    fillMulInputData(h_in, num);

    float* d_in;
    CHECK(cudaMalloc((void**)&d_in, inBytes));
    float* d_out;
    CHECK(cudaMalloc((void**)&d_out, outBytes));

    CHECK(cudaMemcpy(d_in, h_in, inBytes, cudaMemcpyHostToDevice));

    printf("=== deviceWarpMulReduce (num:%u, warps:%u) ===\n", num, numWarps);
    deviceWarpMulReduce<<<MUL_GRID_SIZE, MUL_BLOCK_SIZE>>>(d_out, d_in, num);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_out, d_out, outBytes, cudaMemcpyDeviceToHost));
    printf("--- per-warp partial products ---\n");
    printResultFloat(h_out, numWarps);

    // 把每个 warp 的部分积再乘起来，得到整个缓冲区的乘积
    double wholeProduct = 1.0;
    for (unsigned int i = 0; i < numWarps; i++) {
        wholeProduct *= (double)h_out[i];
    }

    // CPU 端参考值：用 log/exp 累加，同样是为了避免连乘溢出
    double logRef = 0.0;
    for (unsigned int i = 0; i < num; i++) {
        logRef += log((double)h_in[i]);
    }
    double refProduct = exp(logRef);

    printf("* whole buffer product (gpu): %.8e\n", wholeProduct);
    printf("* whole buffer product (cpu reference): %.8e\n", refProduct);
    printf("* relative diff: %.3e\n", fabs(wholeProduct - refProduct) / refProduct);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}

int main() {
    runBasicTests();
    runReduceTests();
    runWarpMulReduceTests();
    return 0;
}
