# Tensor Core 与 CuTe

## 本地入口

`LeetCUDA/`（commit `f7abcd75c9fd1245f6251b715f3c4e7a6220d508`）按练习推进，覆盖面比原说明更广：

| 路径 | 内容 | 对应阶段 |
| --- | --- | --- |
| `LeetCUDA/kernels/hgemm/wmma/` | WMMA API 的 FP16 路径 | 阶段 10（**本机可实测**） |
| `LeetCUDA/kernels/hgemm/mma/` | 更低层的 `mma` 指令路径 | 阶段 10~11 |
| `LeetCUDA/kernels/hgemm/cutlass/` | `cute` / `cute_dsl` / `cutlass-3.x` | 阶段 11 |
| `LeetCUDA/kernels/hgemm/cublas/`、`bench/`、`naive/` | 基线与对比 | 阶段 10 |
| `LeetCUDA/kernels/ws-hgemm/` | warp specialization | 阶段 12 |
| `LeetCUDA/kernels/sgemm/`、`sgemv/` | CUDA Core 基线 | 阶段 8 |
| `LeetCUDA/kernels/softmax/`、`layer-norm/`、`rms-norm/`、`rope/` | AI 算子 | 阶段 4~6 之后 |
| `LeetCUDA/kernels/flash-attn/` | FlashAttention | 阶段 12 |
| `LeetCUDA/kernels/reduce/`、`histogram/`、`mat-transpose/` | 并行模式 | 阶段 3~6 |

## 硬件限制

本机是 Tesla T4（`sm_75`）。实测结论：

- **可用**：`wmma` API（FP16 16x16x16）、`mma.sync.m16n8k8.f16`。
- **不可用**：`mma.sync.m16n8k16.f32.bf16.bf16.f32`（需 sm_80+）、`mma.sync.m16n8k32.s32.s8.s8.s32`（需 sm_80+）、`cp.async`（需 sm_80+）、`wgmma`（需 sm_90+）。

因此阶段 10 只走 FP16 路线。看到 `bf16` 或 `cp.async` 的教程不要照抄，那是 sm_80+ 的内容。`wgmma` 相关练习（`LeetCUDA/kernels/hgemm/wgmma/`）在本机只能阅读和交叉编译。

## 学习范围

先建立 FP16 WMMA 的 fragment、warp 协作和布局模型，再进入更底层的 MMA/CuTe。不要跳过 CUDA Core baseline，也不要只根据最终 TFLOP/s 判断是否理解了数据流。

## 使用方式

任务书应指定具体练习路径或关键词，并要求保存上游 commit、编译参数和正确性结果。抄练习之前先确认该练习依赖的指令在本机能否编译。
