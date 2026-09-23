# 从 0 到 Blackwell 实验路线

顺序不是看课顺序，而是按“能跑通、能测量、能解释、能改”推进。

## 前置
- CUDA Toolkit >= 12.8，推荐 12.9
- 能跑 `vector add`，会 `ncu`
- 有 NVIDIA GPU；没有 B200/GB200 时用云 GPU（Modal / Lambda Labs），`gau-nernst` 的教程支持 Modal

## 阶段 0：基础 CUDA + 内存层级
资源：`cuda-samples` submodule / 官方仓库

- [ ] 跑通 transpose、reduction、scan、histogram
- [ ] 每个 kernel 至少改一个参数并记录 `ncu` 指标
- [ ] 每个 kernel 至少修复一次 bank conflict / uncoalesced access
- 完成标准：能独立定位 shared memory bank conflict 和 uncoalesced global memory access

## 阶段 1：CUDA Core GEMM 性能工程
资源：`cuda-samples`、CUDA Best Practices Guide

- [ ] tiled FP32 GEMM（naive → coalesced → shared memory → warp shuffle）
- [ ] 至少 5 次 benchmark，记录 effective bandwidth / TFLOP/s
- [ ] 理解 roofline、occupancy、arithmetic intensity
- [ ] 修复至少 1 次 occupancy cliff
- 完成标准：sgemm 达到 cuBLAS 50% 以上，能用数据解释优化有效的原因

## 阶段 2：WMMA / Ampere Tensor Core
资源：`cuda-samples`、LeetCUDA

- [ ] 跑通 FP16 WMMA GEMM
- [ ] 理解 fragment 分布、warp 协作、`mma.sync` 约束
- [ ] 对比 CUDA Core / WMMA / cuBLASLt 三份结果
- 完成标准：能解释 fragment 布局和 `mma.sync` 的基本约束

## 阶段 3：Hopper MMA + CuTe 基础
资源：`LeetCUDA`、CUTLASS `examples/70_*`

- [ ] 读 CUTLASS `70_hopper_gemm*`，标出 TMA producer / WGMMA consumer / pipeline stages
- [ ] 修改 stages 为 2/3/4，记录 smem 使用和 kernel 时间
- [ ] 读 FlashAttention-3 Hopper mainloop
- 完成标准：能沿着 CUTLASS mainloop 说清数据从 global memory 到 WGMMA 的路径

## 阶段 4：Blackwell tcgen05 + TMEM
资源：`learn-cuda`（裸PTX）、CUTLASS `examples/78_*`、`examples/79_*`（SM120）

- [ ] 读 `tcgen05 for dummies` 博客，手写 1 个最小 SM100 TCGen05 MMA kernel
- [ ] 跑通 CUTLASS `78_blackwell_gemm`（SM100）或 `79_blackwell_geforce_gemm`（SM120）
- [ ] 在示例中做 1 处小改动并重新验证正确性
- 完成标准：能读懂并修改一个 CUTLASS Blackwell 示例，并解释改动对数据流或性能的影响

## 阶段 5：Blackwell 新特性拓展
资源：CUTLASS ≥ 4.0、Blackwell Tuning Guide

- [ ] NVFP4 / MXFP4/6/8 block scaled GEMM，scale factor 放 TMEM
- [ ] 2-SM 联合 MMA（CTA-pair）
- [ ] TMA 新 API、runtime 可变 cluster 形状
- 完成标准：能解释为什么 scale factor 必须放 TMEM 而不是 shared memory

## 全程实验记录字段
GPU、compute capability、驱动、CUDA、编译命令、输入规模、数据类型、正确性结果、kernel 时间、有效带宽或 TFLOP/s、ncu 关键指标、结论。
