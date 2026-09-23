# 基础 CUDA

## 本地入口

- `cuda-samples/`：NVIDIA 官方样例，优先查找 transpose、reduction、scan、histogram 和基础 runtime API 示例。

## 使用边界

这里用于阶段 0 的 CUDA 编程模型、线程组织、global/shared memory 和基础并行模式。先运行上游样例，再把最小可读版本复制到自己的 task 中；不要直接修改 submodule。

## 推荐顺序

1. 先完成 `experiments/stage-0/task-001-vector-add`。
2. 搜索对应 sample，记录输入输出和线程映射。
3. 每次只改一个 block size、访问模式或 shared memory 布局。
4. 用正确性输出和 `ncu` 结果验证假设。
