# 实验与 task

`experiments/` 保存 task 的代码和任务书。实验记录可以放在这里、放在学习者自己的笔记中，或只在对话中提供；不要求为了完成 task 新建记录文件。

## 一个 task 的内容

- `README.md`：目标、前置知识、具体步骤、提示、交付物和验收标准。
- `result.md`：可选；需要整理实验记录时，可复制[记录模板](log-template.md)。
- `src/`：本 task 的源码；可以是从上游复制的最小版本。
- `run.sh`：命令较多或需要固定顺序时使用。

每个 task 至少要有 `README.md`。新 task 不覆盖旧 task；优化实验保留 baseline，让结果可比较。学习者可按需记录环境、命令、输出和结论，不限记录位置或格式。

## 阶段目录

- `stage-0/`：基础 CUDA 与内存层级，第一项任务是环境和 vector add。
- `stage-1/`：CUDA Core GEMM、性能工程、roofline、occupancy。
- `stage-2/`：WMMA / Ampere Tensor Core。
- `stage-3/`：Hopper MMA + CuTe。
- `stage-4/`：Blackwell `tcgen05` + TMEM。
- `stage-5/`：Blackwell 新特性拓展。

## 性能结论的证据

正确性 task 以代码、运行输出和学习者自己的解释作为验收证据，不要求持久化记录。若要提出性能结论，应提供足以复核的环境、输入、编译参数、计时方法和 profiler 证据；可在对话或学习者选择的位置提供。工具或权限不可用时，明确说明限制，不要伪造指标。
