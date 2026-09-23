# 资料索引

资料按“中文优先、官方为准、代码可执行”整理。中文资料主要负责建立上下文；涉及 ISA、硬件限制和 API 语义时，以 NVIDIA 官方文档和 CUTLASS 源码为准。

- [中文优先清单](zh-first.md)
- [实验代码库](https://github.com/NVIDIA/cuda-samples)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass)
- [CUTLASS examples](https://github.com/NVIDIA/cutlass/tree/main/examples)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)
- [PTX ISA Reference](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)

## 推荐使用方式

1. 先看中文导览，形成关键词列表。
2. 立刻打开对应仓库或官方文档，找到可以编译运行的例子。
3. 修改一个变量、tile、stage 或数据类型。
4. 用 `ncu` 或计时结果记录变化。
5. 在 `experiments/` 下保存命令、结果和结论。
