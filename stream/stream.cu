#include <stdio.h>
#include <stdlib.h>

#define CHECK(x)                                                                                    \
    do                                                                                              \
    {                                                                                               \
        cudaError_t e_ = (x);                                                                       \
        if (e_ != cudaSuccess)                                                                      \
        {                                                                                           \
            fprintf(stderr, "CUDA error %s: %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
            exit(1);                                                                                \
        }                                                                                           \
    } while (0)

int printAsyncEngineCount()
{
    int deviceCount = 0;
    CHECK(cudaGetDeviceCount(&deviceCount));

    for (int i = 0; i < deviceCount; i++)
    {
        cudaDeviceProp prop;
        CHECK(cudaGetDeviceProperties(&prop, i));
        printf("Device prop name: %s, asyncEngineCount:%d\n", prop.name, prop.asyncEngineCount);
    }

    return 0;
}

int main()
{
    printAsyncEngineCount();
    return 0;
}
