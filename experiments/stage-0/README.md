# 阶段 0：基础 CUDA 与内存层级

## 阶段目标

建立 CUDA 的执行模型和内存模型，能够从一个最小 kernel 开始，逐步读懂线程映射、访问模式、正确性检查和 profiler 输出。

## Task 队列

- `task-001-vector-add/`：编译并解释第一个 CUDA kernel。
- 后续 task：由助教根据 `ROADMAP.md`、基础资源和上一个 task 的结果生成，不提前伪造完成进度。

## 阶段验收

至少完成 vector add、transpose、reduction、scan、histogram 的 task，并能指出一个 coalescing 或 shared memory 访问问题。每项都必须有对应 `result.md`，而不是只在聊天中报告“跑过了”。
