# Hopper 与 Blackwell

## 本地入口

- `cutlass/`：Hopper 与 Blackwell 的高性能实现，重点阅读 `examples/70_*`、`examples/78_*` 和 `examples/79_*`。
- `learn-cuda/`：面向 Blackwell 裸 PTX 和 `tcgen05` 的解释与练习。

## 硬件限制

Hopper、SM100 和 SM120 的指令、布局和可用资源不同。任务必须先确认 GPU 的 compute capability；没有目标硬件时可做阅读和编译分析，但不能把它们写成实测性能结论。

## 阅读方法

从一个最小示例开始，标出数据从 global memory、shared memory、TMA、WGMMA 或 TMEM 的流动，再修改一个参数并验证。具体语义最终以 PTX ISA、CUDA 文档和 CUTLASS 源码为准。
