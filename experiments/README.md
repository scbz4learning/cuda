# 实验与 task

`experiments/` 保存学习过程，而不是只保存漂亮的最终答案。每次助教布置一个独立 task，目录格式为 `stage-x/task-y-topic/`，任务书和结果记录跟着代码一起保存。

## 一个 task 的内容

- `README.md`：目标、前置知识、具体步骤、提示、交付物和验收标准。
- `result.md`：复制 [记录模板](log-template.md)，填写实际环境、命令、输出、数据和结论。
- `src/`：本 task 的源码；可以是从上游复制的最小版本。
- `run.sh`：命令较多或需要固定顺序时使用。

每个 task 至少要有 `README.md` 和 `result.md`。新 task 不覆盖旧 task；优化实验保留 baseline，让结果可比较。

## task 的大小约定

学习跨度失控通常来自 task 太大。布置 task 时遵守：

1. **一个 task 只引入一个新概念。** 任务书里写明“本 task 只引入 X”，其余概念必须在前一个 task 已经教过。
2. **不得把未教过的知识写成“前置条件”。** shared memory、bank、warp、`__shfl_*_sync`、计时协议、`float4` 这些都需要专门的 task 引入。
3. **一个 task 一个主要变量。** 只改 block 大小、只改 pitch、只改线程数，其余保持不变，并在 `result.md` 记录改了什么。
4. **一次不超过两个待实现的函数。** 需要更多就拆。
5. **工具链失败如实记录。** 编译错误、profiler 报错贴原文，不要用猜测的数值代替。

`ROADMAP.md` 的阶段总览里，每个 task 都有“只引入”一列；布置时以那一列为准。

## 巩固任务

每个阶段的 `task-9xx` 是巩固任务，规则见 `ROADMAP.md` 的“巩固任务”一节。核心约束：

- **不引入新概念**，只复用本阶段教过的内容。
- **换一个题型**，算子或索引模式必须与阶段内的教学任务不同。
- **先独立完成再对答案。** 做之前不要打开 [`resources/practice/`](../resources/practice/)，对照之后把差异写进 `result.md`。
- **允许跳过**，标 `[ ]` 由学习者决定。

## 阶段目录

`x` 对应 `ROADMAP.md` 的阶段编号，`y` 是三位阶段内序号（不是全局序号）。`9xx` 保留给该阶段的巩固任务。

| 目录 | 阶段 | T4 | B200 |
| --- | --- | :---: | :---: |
| `stage-0/` | 工具链与可测量性 | 实测 | — |
| `stage-1/` | 执行模型与线程组织 | 实测 | — |
| `stage-2/` | 合并访问与内存事务 | 实测 | — |
| `stage-3/` | shared memory 与 bank conflict | 实测 | — |
| `stage-4/` | Reduction 与 warp 原语 | 实测 | — |
| `stage-5/` | Prefix sum / scan | 实测 | — |
| `stage-6/` | Histogram 与原子操作 | 实测 | — |
| `stage-7/` | 性能分析方法论 | 实测 | — |
| `stage-8/` | SGEMM 优化阶梯 | 实测 | — |
| `stage-9/` | CUDA 库与并发执行 | 实测 | — |
| `stage-10/` | Tensor Core 与 WMMA（FP16） | 实测 | 可选补测 |
| `stage-11/` | CuTe 与 sm_80+ 低层 MMA | 交叉编译 | 实测 |
| `stage-12/` | Hopper | 交叉编译 | 实测 |
| `stage-13/` | Blackwell | 阅读 | 实测 |
| `stage-14/` | 综合项目 | 实测 | 实测 |

阶段目录只在开始该阶段时创建，不预先建立空目录。

## 最低记录要求

源码或上游 commit、编译和运行命令、GPU/驱动/CUDA/compute capability、输入尺寸和数据类型、正确性结果、warm-up 策略与重复次数、min/median/mean 计时、`nsys` 输出、`ncu` 命令与关键指标（不可用时贴错误原文和待验证的期望量级）、结论属于“实测 / 交叉编译 / 静态分析 / 待验证”中的哪一级、以及与 `resources/practice/` 参考实现的差异（如适用）。
