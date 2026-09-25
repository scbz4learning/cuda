# 从 0 到 Blackwell：能力路线

路线按“能运行 -> 能测量 -> 能解释 -> 能修改”推进。方括号是学习进度：`[ ]` 未开始，`[-]` 进行中，`[x]` 已完成。当前尚未开始，第一项任务由助教放在 `experiments/stage-0/task-001-*`。

## 前置条件

- CUDA Toolkit >= 12.8，推荐使用与目标 GPU 匹配的版本。
- 能运行 `nvcc`；有 NVIDIA GPU 时再进行真实性能和 `ncu` 实验。
- 没有目标硬件时可以先完成代码阅读和正确性任务，但必须在结果中写明限制。

## 阶段 0：基础 CUDA 与内存层级

资源：`resources/foundations/`、CUDA Programming Guide。

- [ ] 能解释 kernel、thread、block、grid 和 host/device 内存的关系。
- [ ] 完成 vector add、transpose、reduction、scan、histogram 的正确性实验。
- [ ] 能通过访问模式和 profiler 结果定位 coalescing 问题与 shared memory bank conflict。
- 完成标准：能独立读懂一个基础 kernel，提出一个可验证的改动，并用结果解释变化。

## 阶段 1：CUDA Core 性能工程

资源：`resources/performance/`、CUDA Best Practices Guide。

- [ ] 从 naive 到 tiled 实现 FP32 GEMM，并保留每个版本。
- [ ] 使用重复计时、arithmetic intensity、roofline 和 occupancy 解释性能。
- [ ] 处理至少一个 occupancy cliff，并将自写 SGEMM 与 cuBLAS 对比。
- 完成标准：SGEMM 达到 cuBLAS 约 50% 以上，且能用数据说明瓶颈和优化原因。

## 阶段 2：WMMA 与 Tensor Core

资源：`resources/tensor-cores/`、CUDA Samples、LeetCUDA。

- [ ] 跑通 FP16 WMMA GEMM。
- [ ] 解释 fragment、warp 协作、布局和 `mma.sync` 的基本约束。
- [ ] 对比 CUDA Core、WMMA 和 cuBLASLt 的正确性与性能。
- 完成标准：能解释一次 fragment/layout 选择如何影响数据流。

## 阶段 3：Hopper MMA 与 CuTe

资源：`resources/modern-architectures/`、CUTLASS `examples/70_*`。

- [ ] 标出 CUTLASS Hopper GEMM 中的 TMA producer、WGMMA consumer 和 pipeline stages。
- [ ] 修改 stages 为 2/3/4，记录 shared memory、寄存器和 kernel 时间变化。
- [ ] 阅读 FlashAttention-3 Hopper mainloop，并画出数据路径。
- 完成标准：能沿着 mainloop 解释数据从 global memory 到 WGMMA 的流动。

## 阶段 4：Blackwell tcgen05 与 TMEM

资源：`resources/modern-architectures/`、CUTLASS `examples/78_*` 和 `79_*`。

- [ ] 阅读 `tcgen05 for dummies`，完成一个最小可验证的 TCGen05 练习。
- [ ] 跑通适配硬件的 CUTLASS Blackwell GEMM 示例。
- [ ] 修改一个数据布局或 pipeline 参数，并重新验证正确性和性能。
- 完成标准：能读懂并修改一个 Blackwell 示例，说明改动对数据流或性能的影响。

## 阶段 5：Blackwell 特性拓展

资源：CUTLASS、Blackwell Tuning Guide、PTX ISA。

- [ ] 探索 NVFP4/MXFP block scaling、TMEM 中的 scale factor。
- [ ] 了解 2-SM 联合 MMA、CTA-pair、TMA 新 API 和可变 cluster 形状。
- 完成标准：能说明这些特性解决的硬件瓶颈，以及当前实验的硬件限制。

## 进度记录规则

阶段完成必须同时满足：对应 task 的 `result.md` 已填写、正确性有输出证据、测量条件完整、学习者能用自己的话解释结果。助教只在这些证据出现后更新本文件；单纯读完链接不算完成。
