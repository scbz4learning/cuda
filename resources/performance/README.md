# CUDA 性能工程

## 本地入口

当前性能任务主要使用 `foundations/cuda-samples` 的基础 kernel 和自写实验；后续可在此补充性能专项 submodule。

## 学习范围

阶段 1 关注 tiled GEMM、内存合并访问、shared memory、warp shuffle、arithmetic intensity、roofline、寄存器和 occupancy。性能结论必须配套重复计时和 profiler 指标。

## 推荐参考

- CUDA Best Practices Guide
- Nsight Compute Documentation
- `experiments/log-template.md`
