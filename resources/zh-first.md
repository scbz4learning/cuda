# 中文优先资料

中文资料优先用于快速建立术语和架构演进的上下文；涉及具体指令时必须回到英文官方文档核对。

## 入门与架构总览

- [CUDA 编程指南中文镜像（英伟达开发者社区）](https://developer.nvidia.com/zh-cn/cuda-toolkit)
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [NVIDIA GPU Architectures for LLMs](https://jinseok-moon.github.io/p/archs)：A100、H100、B200、RTX 50 等架构对比，英文为主但适合作为图表索引。
- [Blackwell UMMA 架构导览](https://shawrong.github.io/posts/nvidia-blackwell-umma-architecture-guide---part-one)：中文 Blackwell / TMEM / UMMA 导览。

## CUDA 基础与实验

- [NVIDIA cuda-samples](https://github.com/NVIDIA/cuda-samples)：官方可编译示例，优先做 transpose、reduction、scan、histogram。
- [cuda-practice-tutorial](https://github.com/YouXam/cuda-practice-tutorial)：带练习和答案，适合阶段 1。
- [Openlab GPU Lecture](https://github.com/hageboeck/OpenlabLecture)：代码和练习优先，视频不是必需项。
- [Stanford CS149 assignments](https://github.com/stanford-cs149)：并行编程作业，可作为额外练习。

## Tensor Core、Hopper、Blackwell

- [Programming Tensor Cores in CUDA 9](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/)
- [PMPP 4e](https://github.com/parallel-forall): 配套材料和代码入口，书籍本身需单独获取。
- [CUTLASS](https://github.com/NVIDIA/cutlass)：阅读 `examples/70_*` 到 `examples/84_*`，以本地 CUDA 版本支持范围为准。
- [CUTLASS tcgen05 MMA Programming Guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/mma/tcgen05_programming.html)
- [Colfax Hopper GEMM 系列](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-with-wgmma-for-hopper/)
- [Colfax Blackwell GEMM 系列](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/)
- [FlashAttention Hopper 代码](https://github.com/Dao-AILab/flash-attention/tree/main/hopper)
- [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)
- [Blackwell Compatibility Guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/)

## 中文资料的边界

目前 Blackwell 的 `tcgen05`、TMEM、CTA-pair、block scaling 等资料更新很快，中文材料数量和时效性有限。中文文章用于导航，最终以 PTX ISA、CUDA Programming Guide、CUTLASS 源码和可重复实验为准。
