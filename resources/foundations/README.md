# 基础 CUDA

## 本地入口

- `cuda-samples/`：NVIDIA 官方样例，优先查找 transpose、reduction、scan、histogram 和基础 runtime API 示例。
  当前已 checkout 到 commit `5443602d89ed99aede2e4b7bf329daddeadb320e`，样例在 `cpp/` 下（不是旧文档里的 `Samples/`）。
  未初始化时用 `git submodule update --init --depth 1 resources/foundations/cuda-samples`；GitHub 直连可用时不需要镜像，镜像站容易返回 403。

| 主题 | 路径 |
| --- | --- |
| 转置与 bank conflict | `cuda-samples/cpp/6_Performance/transpose/transpose.cu` |
| Tile 版转置 | `cuda-samples/cpp/9_CUDA_Tile/tileTranspose/tileTranspose.cu` |
| reduction | `cuda-samples/cpp/2_Concepts_and_Techniques/reduction` |
| scan | `cuda-samples/cpp/2_Concepts_and_Techniques/scan` |
| histogram | `cuda-samples/cpp/2_Concepts_and_Techniques/histogram` |

## 使用边界

这里用于阶段 0 的 CUDA 编程模型、线程组织、global/shared memory 和基础并行模式。先运行上游样例，再把最小可读版本复制到自己的 task 中；不要直接修改 submodule。

## 推荐顺序

1. 先完成 `experiments/stage-0/task-001-vector-add`。
2. 搜索对应 sample，记录输入输出和线程映射。
3. 每次只改一个 block size、访问模式或 shared memory 布局。
4. 用正确性输出和 `ncu` 结果验证假设。

`ncu` 硬件计数器如果报 `ERR_NVGPUCTRPERM`（容器内 `RmProfilingAdminOnly: 1` 无法修改），就用“只改一个变量的重复计时”代替，并把需要的计数器名称写进 `result.md` 的待验证清单。
