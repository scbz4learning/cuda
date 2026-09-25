# 基础 CUDA

## 本地入口

`cuda-samples/`（commit `5443602d89ed99aede2e4b7bf329daddeadb320e`，v13.4）：NVIDIA 官方样例，代码在 `cpp/` 下（不是旧文档里的 `Samples/`）。未初始化时用 `git submodule update --init --depth 1 resources/foundations/cuda-samples`。

| 阶段 | 主题 | 路径 |
| --- | --- | --- |
| 阶段 0 | 设备查询、编译选项 | `cpp/1_Utilities/` |
| 阶段 1 | SIMT、二维线程、线程迁移 | `cpp/2_Concepts_and_Techniques/threadMigration/` |
| 阶段 2 | 合并访问、盒式滤波、可分离卷积 | `cpp/2_Concepts_and_Techniques/boxFilter/`、`convolutionSeparable/` |
| 阶段 3 | 转置与 bank conflict、tile 版转置 | `cpp/6_Performance/transpose/`、`cpp/9_CUDA_Tile/tileTranspose/` |
| 阶段 4 | reduction、warp shuffle、thread fence reduction | `cpp/2_Concepts_and_Techniques/reduction/`、`shfl_scan/`、`threadFenceReduction/` |
| 阶段 5 | scan、分段树 | `cpp/2_Concepts_and_Techniques/scan/`、`segmentationTreeThrust/` |
| 阶段 6 | histogram | `cpp/2_Concepts_and_Techniques/histogram/` |
| 阶段 7 | CUDA Graph 性能扩展、Unified Memory 性能 | `cpp/6_Performance/cudaGraphsPerfScaling/`、`UnifiedMemoryPerf/` |
| 阶段 9 | stream 序分配、stream 优先级、CUDA Graph、locality domain | `cpp/2_Concepts_and_Techniques/streamOrderedAllocation/`、`cpp/3_CUDA_Features/StreamPriorities/`、`simpleCudaGraphs/`、`graphMemoryNodes/`、`graphConditionalNodes/`、`localityDomains/` |
| 阶段 9 | Cooperative Groups 网格二分与 warp 聚合原子操作 | `cpp/3_CUDA_Features/binaryPartitionCG/`、`warpAggregatedAtomicsCG/` |
| 阶段 10 | Tensor Core GEMM、PTX JIT | `cpp/3_CUDA_Features/cudaTensorCoreGemm/`、`immaTensorCoreGemm/`、`ptxjit/`（`bf16TensorCoreGemm/` 与 `tf32TensorCoreGemm/` 需 sm_80+，T4 不可运行） |
| 阶段 11 | global→shared 异步拷贝、CuTe tile 抽象 | `cpp/3_CUDA_Features/globalToShmemAsyncCopy/`（`cp.async`，需 sm_80+）、`cpp/9_CUDA_Tile/helloTile/`、`tileMatmul/`、`tileLayerNorm/` |

`cpp/2_Concepts_and_Techniques/README.md`、`cpp/3_CUDA_Features/README.md`、`cpp/6_Performance/README.md` 与 `cpp/9_CUDA_Tile/README.md` 上游有说明，进入目录前先读。

## 使用边界

这里用于阶段 0~9 的 CUDA 编程模型、线程组织、global/shared memory 和基础并行模式。先运行上游样例，再把最小可读版本复制到自己的 task 中；不要直接修改 submodule。

注意 `cpp/6_Performance/` 目录当前只有少数几个样例（`transpose`、`cudaGraphsPerfScaling`、`alignedTypes`、`LargeKernelParameter`、`UnifiedMemoryPerf`），不要假设存在 `deviceQuery`、`transpose` 之外的其他经典样例目录。

## 测量替代方案

`ncu` 硬件计数器在本机报 `ERR_NVGPUCTRPERM`（容器内核参数 `RmProfilingAdminOnly: 1`，无法修改），这是稳定的环境限制。因此：

- 用“只改一个变量的重复计时”代替计数器。
- 用 `nsys` 拿 kernel 级时间线。
- 在 `result.md` 的待验证清单里写清需要哪个计数器、期望量级是多少，不要填猜测数值。

相关任务的约定见 `ROADMAP.md` 的“证据分级”。
