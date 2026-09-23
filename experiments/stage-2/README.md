# 阶段 2：WMMA 与 Tensor Core

## 阶段目标

理解 warp 级矩阵运算的协作方式，而不是只调用一个更快的库函数。先保留阶段 1 的 CUDA Core GEMM 作为 baseline，再比较 WMMA、MMA 和 cuBLASLt。

## 推荐 task 顺序

1. 从 `resources/tensor-cores/LeetCUDA/` 或 CUDA Samples 找到最小 WMMA GEMM。
2. 验证 FP16 输入、FP32 累加和结果容差。
3. 记录 fragment/layout、warp 协作和 `mma.sync` 约束。
4. 在同一输入规模下比较 CUDA Core、WMMA 和 cuBLASLt。

硬件不支持目标 Tensor Core 时，task 可以停在代码阅读和编译分析，但结果必须明确标注“未实测”。