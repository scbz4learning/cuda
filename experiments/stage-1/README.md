# 阶段 1：基础并行模式

建议按以下顺序实现，每一步都保留 baseline：

1. matrix transpose：连续访问与 shared memory
2. reduction：naive、shared memory、warp shuffle
3. scan：先读 `cuda-samples`，再实现自己的版本
4. tiled FP32 GEMM：先正确，再测量，再优化

推荐参考：

- [cuda-samples](https://github.com/NVIDIA/cuda-samples)
- [cuda-practice-tutorial](https://github.com/YouXam/cuda-practice-tutorial)
- [OpenlabLecture](https://github.com/hageboeck/OpenlabLecture)

每个实验复制 `../log-template.md`，并记录输入规模、正确性和 ncu 指标。