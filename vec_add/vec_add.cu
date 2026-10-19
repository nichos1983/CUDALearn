#include <stdio.h>

#define ARRAY_NUM (1 << 16)
#define BLOCK_SIZE 128.0f

__device__ void vecAddKernelImpl(float *a, float *b, float *c, int i, int n)
{
    if (i < n)
    {
        c[i] = 4.0f * a[i] + b[i];
    }
    else
    {
        printf("Device: i>=n, i:%d, n:%d\n", i, n);
    }
}

__global__ void vecAddKernel(float *a, float *b, float *c, int n)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    vecAddKernelImpl(a, b, c, i, n);
}

void vecAdd(float *a_h, float *b_h, float *c_h, int n)
{
    int size = n * sizeof(float);
    float *a_d, *b_d, *c_d;

    cudaMalloc((void **)&a_d, size);
    cudaMalloc((void **)&b_d, size);
    cudaMalloc((void **)&c_d, size);

    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);

    printf("Before call kernel\n");

    vecAddKernel<<<ceil(n / BLOCK_SIZE), BLOCK_SIZE>>>(a_d, b_d, c_d, n);

    cudaDeviceSynchronize();
    printf("After call kernel\n");

    cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);

    printf("After call memcpy\n");

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}

void runTest()
{
    float a[ARRAY_NUM], b[ARRAY_NUM], c[ARRAY_NUM];
    for (int i = 0; i < ARRAY_NUM; i++)
    {
        a[i] = (float)i;
        b[i] = (float)i * 0.5;
    }

    vecAdd(a, b, c, ARRAY_NUM);

    printf("Result start:\n");
    for (int i = 0; i < ARRAY_NUM; i += ARRAY_NUM / 16)
    {
        printf("i: %d, value: %.2f ", i, c[i]);
    }
    printf("\nResult end\n");
}

int main()
{
    runTest();
    return 0;
}
