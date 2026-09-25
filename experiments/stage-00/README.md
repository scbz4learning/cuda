# 阶段 00：工具链与可测量性

## 阶段目标

建立 CUDA 实验闭环：能编译 host/device 混合程序，能解释线程索引和边界判断，能用输出确认结果，能稳定计时，并如实记录环境。本阶段专门建立“可测量性”，它不会随第一个 kernel 自动获得。

## Task 队列

| 进度 | Task | 只引入 |
| --- | --- | --- |
| [x] | `task-001-vector-add` | 第一个 kernel、1D 索引、边界判断 |
| [-] | `task-002-device-query-and-error-check` | 设备查询与 `cudaGetLastError` / 错误检查宏 |
| [ ] | `task-003-compile-flags-and-resource-usage` | `nvcc` 选项与 `cuobjdump -res-usage` 看到的寄存器/共享内存 |
| [ ] | `task-004-timing-protocol` | `cudaEvent` 计时、warm-up、重复、min/median/mean、空 kernel 的 launch 开销 |
| [ ] | `task-005-巩固-harness` | **巩固**：把计时 harness 固化成可复用文件 |

本阶段 task 编号 001–005。下一个 task 是 `task-006`，属于阶段 01。后续 task 由助教根据 `ROADMAP.md`、基础资源和上一个 task 的结果生成，不提前伪造完成进度。目录在本阶段开始时创建。

## 阶段验收

- [ ] 能用 `cudaGetDeviceProperties` 打印并解释 CC、SM 数、L2 大小、峰值带宽四个字段的来源。
- [ ] 能说出一个“编译通过但运行报错”的例子，并用 `cudaGetLastError` 与 `compute-sanitizer` 区分是启动失败还是运行中出错。
- [ ] 能用 `cuobjdump -res-usage` 读出某个 kernel 的寄存器数和静态共享内存数，并说明它受哪个编译选项影响。
- [ ] 给出空 kernel 的 launch 开销中位数，后续所有 kernel 计时都以此为噪声下限。

## 注意

原 `task-002-transpose-coalescing/` 一次引入二维索引、shared memory、动态共享内存、bank 模型、计时协议共五项，已删除。内容按新路线拆到 `task-014` 与 `task-019`~`021`，计时协议移到本阶段的 `task-004`。说明见 [`ROADMAP.md`](../../ROADMAP.md) 的“现状映射”。
