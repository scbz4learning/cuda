# 实验与 task

`experiments/` 保存学习过程，而不是只保存漂亮的最终答案。每次助教布置一个独立 task，目录格式为 `stage-x/task-y-topic/`，任务书和结果记录跟着代码一起保存。

## 一个 task 的内容

- `README.md`：目标、前置知识、具体步骤、提示、交付物和验收标准。
- `result.md`：复制 [记录模板](log-template.md)，填写实际环境、命令、输出、数据和结论。
- `src/`：本 task 的源码；可以是从上游复制的最小版本。
- `run.sh`：命令较多或需要固定顺序时使用。

每个 task 至少要有 `README.md` 和 `result.md`。新 task 不覆盖旧 task；优化实验保留 baseline，让结果可比较。

## 阶段目录

- `stage-0/`：基础 CUDA 与内存层级，第一项任务是环境和 vector add。
- `stage-1/`：CUDA Core GEMM、性能工程、roofline、occupancy。
- `stage-2/`：WMMA / Ampere Tensor Core。
- `stage-3/`：Hopper MMA + CuTe。
- `stage-4/`：Blackwell `tcgen05` + TMEM。
- `stage-5/`：Blackwell 新特性拓展。

## 最低记录要求

源码或上游 commit、编译和运行命令、GPU/驱动/CUDA/compute capability、输入尺寸和数据类型、正确性结果、至少三次计时或 warm-up 策略、`ncu` 命令与关键指标、观察和下一次只改什么。
