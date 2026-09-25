# 实验记录：转置、合并访问与 shared memory bank conflict

- 日期：
- 阶段：stage-0 / task-002
- 代码位置：`experiments/stage-0/task-002-transpose-coalescing/transpose.cu`、`bank_probe.cu`
- 上游 commit / 参考链接：`resources/foundations/cuda-samples` @ `5443602d89ed99aede2e4b7bf329daddeadb320e`，参考 `cpp/6_Performance/transpose/transpose.cu` 的 `transposeNaive` / `transposeCoalesced` / `transposeNoBankConflicts`

## 环境

- GPU（型号 + device index）：
- Compute Capability：
- Driver：
- CUDA Toolkit：
- 编译器：`nvcc --version`
- 编译命令：`./run.sh` 中的 `ARCH_FLAGS`，实际生效的架构（`nvcc -arch=native` 解析结果或报错原文）：
- L2 大小、SM 数量、memoryClockRate、memoryBusWidth（harness 已打印）：

## 约定与实现说明

- 转置定义与索引约定是否与任务书一致：
- `transpose_tiled` 的实现思路（阶段 1 / 同步 / 阶段 2 / 同步，循环结构）：
- 为什么 `N` 是 32 的倍数时不需要边界判断；如果不是 32 的倍数会发生什么：
- `pitch` 如何参与 shared memory 寻址：

## 第 1 步：naive 访问模式手算

warp 内 32 个 lane 的索引（`threadIdx.x = 0..31`，`threadIdx.y` 固定）：

| 量 | 读 `in` | 写 `out` |
| --- | --- | --- |
| 相邻 lane 的地址间隔（字节） |  |  |
| 覆盖的 32 字节 sector 数 |  |  |
| 是否合并 |  |  |

- 每个 block 处理多少元素、循环几次、每线程几个元素：
- 为什么读和写不能同时合并（自己的判断和理由）：

## 第 3 步：阶段 2 的 bank 号手算

记 `tx = threadIdx.x`（0..31），`ty = threadIdx.y`（0..15）。

`pitch = 32`：

| lane | tx | ty | shared 地址（4 字节单位） | bank = 地址 % 32 | 是否同 bank |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0 |  |  |  |  |
| 1 | 1 |  |  |  |  |
| 2 | 2 |  |  |  |  |
| 7 | 7 |  |  |  |  |
| 15 | 15 |  |  |  |  |
| 16 | 16 |  |  |  |  |
| 30 | 30 |  |  |  |  |
| 31 | 31 |  |  |  |  |

- 是否 32-way conflict：冲突路数：
- 同一 warp 的 8 个不同 `ty` 各自落在哪个 bank：
- 读同一地址是否算 conflict（广播）：

`pitch = 33`：

| lane | tx | ty | shared 地址（4 字节单位） | bank = 地址 % 32 | 是否同 bank |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0 |  |  |  |  |
| 1 | 1 |  |  |  |  |
| 2 | 2 |  |  |  |  |
| 7 | 7 |  |  |  |  |
| 15 | 15 |  |  |  |  |
| 16 | 16 |  |  |  |  |
| 30 | 30 |  |  |  |  |
| 31 | 31 |  |  |  |  |

- 用自己的话说明“为什么列距取 32 会让整个 warp 撞在同一个 bank，取 33 就不会”：

## 输入与正确性

- N：2048 / 4096 / 8192
- 数据类型与填充方式：
- 对照实现：host 端 `out[y * N + x] = in[x * N + y]`
- 容差：精确 `==`
- 正确性结果（三个尺寸 × 三个变体的 PASS/FAIL，FAIL 时贴首个不匹配下标）：

## 测量

warm-up 5 次，计时 20 次取 min / median / mean；有效带宽 = `2 * N * N * sizeof(float)` / median 时间。

| N | 变体 | min (us) | median (us) | mean (us) | GB/s | 相对峰值的比例 | 备注 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2048 | naive |  |  |  |  |  |  |
| 2048 | tiled_pitch32 |  |  |  |  |  |  |
| 2048 | tiled_pitch33 |  |  |  |  |  |  |
| 4096 | naive |  |  |  |  |  |  |
| 4096 | tiled_pitch32 |  |  |  |  |  |  |
| 4096 | tiled_pitch33 |  |  |  |  |  |  |
| 8192 | naive |  |  |  |  |  |  |
| 8192 | tiled_pitch32 |  |  |  |  |  |  |
| 8192 | tiled_pitch33 |  |  |  |  |  |  |

- `tiled_pitch32` 与 `tiled_pitch33` 的差距（绝对值和百分比，4096 上）：
- 如果差距小于 5%，你的解释：

## bank probe

命令：`./bank_probe <blocks> <iterations>`（默认值）：

| 变体 | PASS/FAIL | min (us) | median (us) | mean (us) |
| --- | --- | ---: | ---: | ---: |
| probe_no_pad |  |  |  |  |
| probe_padded |  |  |  |  |

- 时间比值：
- 两个 kernel 的索引表达式（原文贴出）：
- 两个索引如何满足 / 不满足“同一 bank 不同地址”和“32 个不同 bank”：
- `acc_total` 与期望值是否一致；如果 FAIL，怀疑了什么：

## Profiler

```text
nsys profile -o transpose_trace ./transpose 4096
nsys stats transpose_trace.nsys-rep
```

kernel 列表与时间：

```text
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum ./transpose 4096
```

- 实际输出（成功则填三个计数器数值与单位；失败则贴错误原文）：

## 结论

- 观察到（计时测到的）：
- 原因假设（手算推出来的）：
- 未能验证（需要 ncu 才能确认的）：
- 概念区分：哪些是测量、哪些是推断、哪些是待验证：
- 下一次只改：

## 附：与上游实现的差异

- 和上游 `transposeCoalesced` / `transposeNoBankConflicts` 在索引约定、block 形状、循环结构上的区别：
- 上游还有哪些变体（`transposeDiagonal`、`transposeFineGrained`、`transposeCoarseGrained`、partition camping）没有做，理由：
