# 资料索引

资料按"中文优先、官方为准、代码可执行"整理。核心代码库通过 git submodule 管理，更新时使用 `git submodule update --remote`。

## 本地仓库（submodule）

- `cutlass/`：NVIDIA CUTLASS，Blackwell TCGen05 / TMEM / 2-SM MMA 最全实现，阅读 `examples/78_*` 和 `examples/79_*`
- `learn-cuda/`：gau-nernst 的 Blackwell 裸 PTX 实战，含 tcgen05 for dummies 博客，适合理解硬件行为
- `LeetCUDA/`：xlite-dev 的现代 CUDA 自助训练营，WMMA → MMA → CuTe → Blackwell TCGen05，含大量练习和 benchmark
- `cuda-samples/`：NVIDIA 官方 CUDA 样例库，适合基础 CUDA、WMMA、MMA 入门

## 在线资料

- [中文优先清单](zh-first.md)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)
- [PTX ISA Reference](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [Nsight Compute Documentation](https://docs.nvidia.com/cuda/nsight-compute/)

## 推荐使用方式

1. 先看中文导览，形成关键词列表。
2. 立刻打开 submodule 仓库或在线文档，找到可以编译运行的例子。
3. 修改一个变量、tile、stage 或数据类型。
4. 用 `ncu` 或计时结果记录变化。
5. 在 `experiments/` 下保存命令、结果和结论。
