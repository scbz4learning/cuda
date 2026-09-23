# 阶段 2：性能工程

从阶段 1 的 tiled GEMM 继续，不要直接跳到 CUTLASS。

实验顺序：

- [ ] 画出输入规模对应的算术强度和 roofline 位置
- [ ] 改变 tile 尺寸，观察寄存器、shared memory、occupancy
- [ ] 人为制造并修复 shared memory bank conflict
- [ ] 比较 coalesced 与 uncoalesced global memory access
- [ ] 实验不同 pipeline stages
- [ ] 阅读并尝试 persistent kernel

参考工具：`ncu`、CUDA Best Practices Guide、cuBLASLt benchmark。