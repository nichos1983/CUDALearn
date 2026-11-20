# CUDALearn
记录CUDA学习历程

SGEMM运行有两种模式：
1. 默认模式，不加参数
    ./main
    更多的测试迭代次数，更大的测试矩阵。以保证测试数据的准确性。
2. profile模式，加 -profile参数
    ./main -profile
    更少的测试迭代次数，更小的测试矩阵。以节省用nsys和ncu做性能分析的运行时间。
