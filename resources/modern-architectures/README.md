# Hopper 与 Blackwell

本目录的 submodule 已就位：

| submodule | commit | 用途 |
| --- | --- | --- |
| `cutlass/` | `0b55a2f691d69981583568fd9eb69687b1f0de8a` | Hopper 与 Blackwell 的高性能实现，以及官方 CuTe/Blackwell 文档 |
| `learn-cuda/` | `8c4d1b887a25727b320bc3ace19b63e2db6f8b44` | 逐级 matmul 阶梯与裸 PTX 解释 |

## 硬件限制

| 特性 | T4（`sm_75`，当前） | B200（`sm_100`，目标） |
| --- | :---: | :---: |
| `cp.async` | 不可用（需 sm_80+） | 可用 |
| TMA `cp.async.bulk.tensor` | 不可用（需 sm_90+） | 可用 |
| thread block cluster / 分布式 shared memory | 不可用（需 sm_90+） | 可用 |
| `wgmma` | 不可用 | 可用 |
| `tcgen05` / TMEM | 不可用（需 sm_100+） | 可用 |

T4 上对应阶段 11~13 的任务用交叉编译与静态分析完成，**不得写实测性能数字**：

```bash
nvcc -arch=sm_80  -c example.cu -o /dev/null   # Ampere
nvcc -arch=sm_90  -c example.cu -o /dev/null   # Hopper
nvcc -arch=sm_100 -c example.cu -o /dev/null   # Blackwell datacenter
nvcc -arch=sm_100a -c example.cu -o /dev/null  # Blackwell 架构特定（B200 的 tcgen05 通常需要）
```

完整的架构能力矩阵见 [`ROADMAP.md`](../../ROADMAP.md#架构能力矩阵)。

## CUTLASS 官方文档（在 repo 内）

`cutlass/media/docs/` 下有 CUTLASS 自带的设计文档，比博客可靠，任务书优先引用这里。

### CuTe 教程（阶段 11）

| 路径 | 内容 |
| --- | --- |
| `cutlass/media/docs/cpp/cute/00_quickstart.md` | CuTe 起步 |
| `cutlass/media/docs/cpp/cute/01_layout.md` | Layout 代数基础 |
| `cutlass/media/docs/cpp/cute/02_layout_algebra.md` | Layout 组合与化简 |
| `cutlass/media/docs/cpp/cute/03_tensor.md` | Tensor 与 tiling |
| `cutlass/media/docs/cpp/cute/04_algorithms.md` | 在 Tensor 上组合算法 |
| `cutlass/media/docs/cpp/cute/0t_mma_atom.md` | MMA atom（阶段 11、12） |
| `cutlass/media/docs/cpp/cute/0z_tma_tensors.md` | TMA tensor（阶段 12） |
| `cutlass/media/docs/cpp/cute/0x_gemm_tutorial.md` | 用 CuTe 写 GEMM |

### CUTLASS 3.x 与 Blackwell

| 路径 | 内容 |
| --- | --- |
| `cutlass/media/docs/cpp/cutlass_3x_design.md` | 3.x 的 mainloop 结构（阶段 11、12） |
| `cutlass/media/docs/cpp/pipeline.md` | 多级流水（阶段 11） |
| `cutlass/media/docs/cpp/blackwell_functionality.md` | Blackwell 特性总览（阶段 13） |
| `cutlass/media/docs/cpp/blackwell_cluster_launch_control.md` | cluster launch control（阶段 13） |
| `cutlass/media/docs/cpp/gemm_performance_measurement_methodology_guidelines.md` | **官方 GEMM 测量方法学**（阶段 07、8） |
| `cutlass/media/docs/cpp/terminology.md` | 术语表，读源码前的对照表 |
| `cutlass/media/docs/cpp/programming_guidelines.md` | CUTLASS 编码约定 |

### 指令头文件

按 compute capability 分文件，阶段 10~13 查指令语义时直接读这些：

```text
cutlass/include/cutlass/arch/wmma_sm70.h   wmma_sm72.h   wmma_sm75.h
cutlass/include/cutlass/arch/mma_sm50.h    mma_sm60.h    mma_sm61.h
cutlass/include/cutlass/arch/mma_sm70.h    mma_sm75.h    mma_sm80.h
cutlass/include/cutlass/arch/mma_sm89.h    mma_sm90.h    mma_sm100.h
cutlass/include/cutlass/arch/memory_sm75.h memory_sm80.h
cutlass/include/cutlass/arch/barrier.h     cluster_launch.hpp
cutlass/include/cute/atom/                # MMA atom 定义
```

`mma_sm75.h` 正是 T4 能用的那一条路径，阶段 10 读它；`mma_sm100.h` 是阶段 13 的核心参考。

## CUTLASS 示例路径

`examples/` 的编号不按架构连续分布，容易引错。实测确认的对应关系：

| 架构 | 路径 | 说明 |
| --- | --- | --- |
| Hopper | `examples/48_hopper_warp_specialized_gemm` | warp specialization 的主参考 |
| Hopper | `examples/54_hopper_fp8_warp_specialized_gemm` | FP8 + block scaling |
| Hopper | `examples/88_hopper_fmha` | FlashAttention mainloop |
| Hopper | `examples/113_hopper_gemm_activation_fusion` | epilogue 融合 |
| Blackwell sm_100 | `examples/70_blackwell_gemm` | `tcgen05` GEMM 入门 |
| Blackwell sm_100 | `examples/71_blackwell_gemm_with_collective_builder` | CollectiveBuilder |
| Blackwell sm_100 | `examples/72_blackwell_narrow_precision_gemm` | `72a` nvfp4+bf16、`72b` nvfp4+nvfp4、`72c` mxfp8+bf16 |
| Blackwell sm_100 | `examples/73_blackwell_gemm_preferred_cluster` | cluster 形状 |
| Blackwell sm_100 | `examples/75_blackwell_grouped_gemm` | `75_blackwell_grouped_gemm_block_scaled.cu` 覆盖 block scaling |
| Blackwell sm_100 | `examples/77_blackwell_fmha` | Blackwell FMHA / MLA |
| Blackwell sm_120 | `examples/79_blackwell_geforce_gemm` | GeForce 变体，**不是** sm_100 |
| Blackwell sm_120 | `examples/78_blackwell_emulated_bf16x9_gemm` | bf16x9 模拟路径，与 `tcgen05` 练习无关 |

## learn-cuda 路径

`learn-cuda/` 提供一条按架构递进的 matmul 阶梯，比 CUTLASS 更适合逐版本阅读：

| 路径 | 内容 |
| --- | --- |
| `learn-cuda/02_matmul_simt/` | CUDA Core：block tiling、thread tiling、warp tiling |
| `learn-cuda/02_matmul_sm80/` | inline PTX：`cvta`、`ldmatrix`、`mma` |
| `learn-cuda/02_matmul_sm100/` | `tcgen05`。含 `matmul_v0.cu` 到 `matmul_v7.cu` 的演进链与 `_matmul_tmem.cu` |
| `learn-cuda/02_matmul_sm120/` | sm_120 路径 |
| `learn-cuda/03_sum/` | 通用 reduction，为 softmax 做准备 |
| `learn-cuda/04_softmax/` | naive softmax、online softmax、`atomicCAS` |
| `learn-cuda/07_attention/` | FlashAttention |
| `learn-cuda/08_row_scaled_mm/` | 简单的 epilogue |
| `learn-cuda/09_block_scaled_mm_sm120/` | MXFP8 |

`02_matmul_sm100/` 还自带 `trace_v5.json.gz` 与 `trace_v6.json.gz`（Nsight 导出的 trace），可用于在没有 Hopper/Blackwell 硬件时分析数据流，但不能当作本机实测结果。

## 阅读方法

从一个最小示例开始，标出数据从 global memory、shared memory、TMA、WGMMA 或 TMEM 的流动，再修改一个参数并验证。具体语义最终以 PTX ISA、CUDA 文档和 CUTLASS 源码为准；不要用博客或本目录的二手说明替代官方定义。
