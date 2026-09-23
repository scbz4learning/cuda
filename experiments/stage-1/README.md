# 阶段 1：CUDA Core 性能工程

## 阶段目标

从正确的并行程序进入性能工程：保留 baseline，改变一个性能变量，用计时和 profiler 解释 global memory、shared memory、寄存器、occupancy 与算术强度之间的关系。

## 推荐 task 顺序

1. transpose：连续访问与 shared memory。
2. reduction：naive、shared memory、warp shuffle。
3. scan：先阅读官方 sample，再实现自己的版本。
4. tiled FP32 GEMM：先正确，再测量，再优化。

每个 task 都应使用 `resources/foundations/cuda-samples` 或 `resources/performance/` 中的具体入口，并在自己的目录保留 baseline、命令和结果。