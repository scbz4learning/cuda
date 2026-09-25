# CUDA 性能工程

## 本地入口

`cuda-kernel-academy/`（commit `e95bcd3ca2f9928dfa75d3994fb5b91ae29f1b53`，上游仓库 `AICL-Lab/cuda-kernel-academy`，也称 `cuda-foundations`）。中英双语文档在 `docs/en/` 与 `docs/zh/`，有 CMake preset 和 ctest 测试框架，适合用“跑通测试”代替肉眼检查。

| 路径 | 内容 | 对应阶段 |
| --- | --- | --- |
| `01-sgemm-tutorial/src/kernels/naive_sgemm.cuh` | 三重循环 baseline | 阶段 08 |
| `01-sgemm-tutorial/src/kernels/tiled_sgemm.cuh` | shared memory 分块 | 阶段 08 |
| `01-sgemm-tutorial/src/kernels/bank_conflict_free_sgemm.cuh` | 消除 bank conflict | 阶段 03、8 |
| `01-sgemm-tutorial/src/kernels/double_buffer_sgemm.cuh` | 双缓冲流水 | 阶段 11 |
| `01-sgemm-tutorial/src/kernels/tensor_core_sgemm.cuh` | Tensor Core 版本 | 阶段 10 |
| `01-sgemm-tutorial/tests/test_sgemm.cu` | 正确性测试写法参考 | 阶段 08 |
| `02-tensorcraft-core/` | header-only 算子库组织方式 | 阶段 09 |
| `03-hpc-advanced/src/02_reduction/` | reduction 实验 | 阶段 04 |
| `03-hpc-advanced/src/03_gemm/` | GEMM 实验 | 阶段 08 |
| `03-hpc-advanced/src/04_convolution/` | 卷积 | 阶段 06 之后 |
| `03-hpc-advanced/src/05_attention/flash_attention.cu` | FlashAttention | 阶段 12、14 |
| `03-hpc-advanced/src/05_attention/rope.cu` | RoPE | 阶段 14 |
| `03-hpc-advanced/src/06_quantization/` | INT8/FP8 量化与反量化 | 阶段 13 |
| `03-hpc-advanced/src/07_cuda13_features/tma.cu` | TMA 示例 | 阶段 12 |
| `03-hpc-advanced/src/07_cuda13_features/cluster.cu` | thread block cluster 示例 | 阶段 12 |
| `03-hpc-advanced/src/07_cuda13_features/fp8_gemm.cu` | FP8 GEMM | 阶段 11、13 |
| `04-inference-engine/` | 小型推理系统把 kernel 串起来 | 阶段 14 |

构建与测试：

```bash
cmake --list-presets
cmake --preset default
cmake --build --preset default
ctest --preset default
```

## 学习范围

阶段 07~ 08 关注 tiled GEMM、合并访问、shared memory、warp shuffle、算术强度、roofline、寄存器和 occupancy。阶段 13 之后关注 FP8/FP4 量化与推理组件。性能结论必须配套重复计时和 profiler 指标。

## 硬件说明

上游推荐 Volta（sm_70）起、Ampere/Ada/Hopper 更好。T4（sm_75）可运行 01、02、03 模块；`03-hpc-advanced/src/07_cuda13_features/` 中的 TMA 与 cluster 示例需要 sm_90+，在 T4 上只能阅读或交叉编译。目标机 B200（sm_100）可全部运行。
