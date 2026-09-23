# 学习路线与阶段清单

阶段顺序不是按课程观看顺序，而是按“能跑通、能测量、能解释、能修改”推进。

## 阶段 0：建立心智模型

目标：理解 thread、warp、block、grid、内存层次和基本 kernel 调度。

- [ ] 安装 CUDA Toolkit，确认 `nvcc --version`
- [ ] 跑通一个 vector add
- [ ] 用 Nsight Compute (`ncu`) profile 一次，不做优化
- [ ] 阅读 CUDA C++ Programming Guide 的编程模型、用 CUDA 编程、进阶部分
- [ ] 完成一次实验记录

进入下一阶段：能解释 thread / warp / block / grid，以及寄存器、shared memory、L1/L2、HBM 的关系。

## 阶段 1：写对并自己优化

- [ ] 阅读 `cuda-samples` 中 transpose、reduction、scan、histogram
- [ ] 完成 vector add 的 grid-stride loop
- [ ] matrix transpose：naive、coalesced、shared memory、消除 bank conflict
- [ ] reduction：naive、shared memory、warp shuffle 三版
- [ ] tiled FP32 GEMM
- [ ] 每个版本记录正确性、吞吐、关键 ncu counter

进入下一阶段：能独立定位一次 uncoalesced access、同步错误或 bank conflict。

## 阶段 2：性能工程

- [ ] 阅读 CUDA C++ Best Practices Guide
- [ ] 理解 roofline、occupancy、算术强度和 pipeline
- [ ] tiled FP32 GEMM 达到 cuBLAS SGEMM 的 50% 以上
- [ ] 至少修复一次 occupancy cliff 或 bank conflict
- [ ] 实验 grid-stride loop、L2 access policy、persistent kernel

进入下一阶段：能用数据说明一个优化为什么有效，而不是只比较总时间。

## 阶段 3：WMMA / Tensor Core

- [ ] 阅读 Tensor Core 编程入门和 PMPP 4e Tensor Core 章节
- [ ] 跑通 FP16 WMMA GEMM
- [ ] 阅读并理解 `mma.sync` 的 fragment 分布
- [ ] 对比 CUDA Core GEMM、WMMA、cuBLASLt
- [ ] 记录数据类型、累加类型、矩阵布局和硬件架构

进入下一阶段：能解释 fragment、warp 协作和 `mma.sync` 的基本约束。

## 阶段 4：Hopper

硬件前提：H100/H200，或可用的云 GPU。

- [ ] 阅读 Hopper Tuning Guide 与架构资料
- [ ] 阅读 CUTLASS `70_hopper_gemm*`
- [ ] 标出 TMA producer、WGMMA consumer、pipeline stages
- [ ] 修改 stages 为 2/3/4 并记录 shared memory 与 kernel 时间
- [ ] 阅读 FlashAttention-3 Hopper mainloop

进入下一阶段：能沿着 CUTLASS mainloop 说清楚数据从 global memory 到 WGMMA 的路径。

## 阶段 5：Blackwell

硬件前提：完整 TMEM / `SM100` 实验需要 B200/GB200 等数据中心 Blackwell。RTX 50 的 `SM120` 只覆盖部分内容。

- [ ] 阅读 Blackwell Tuning Guide、Compatibility Guide、PTX ISA `tcgen05`
- [ ] 跑通 CUTLASS `71_blackwell_gemm_with_collective_builder`
- [ ] 说明 A/B 经 SMEM、C/D 经 TMEM 的数据路径
- [ ] 阅读 `77_blackwell_fmha`，标注 `tcgen05`、TMA、mbarrier
- [ ] 对比 narrow precision、cluster、grouped GEMM 示例
- [ ] 在一个 Blackwell 示例中做小改动并重新验证正确性

完成标准：能读懂并修改一个 CUTLASS Blackwell 示例，并能解释改动对数据流或性能的影响。

## 全程实验记录字段

GPU、compute capability、驱动、CUDA、编译命令、输入规模、数据类型、正确性结果、kernel 时间、有效带宽或 TFLOP/s、ncu 关键指标、结论。
