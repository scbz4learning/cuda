# 阶段 1：执行模型与线程组织

## 阶段目标

能预测任意索引表达式下 warp 的访问集合，并用实验解释 divergence 和 block 形状的代价。这是后面所有内存优化的前提：不知道 warp 怎么执行，就无法判断一次访存是否合并。

## Task 队列

| 进度 | Task | 只引入 |
| --- | --- | --- |
| [ ] | `task-001-warp-and-lane` | warp = 32、lane 与 `threadIdx` 的关系、每 SM 的 warp 调度 |
| [ ] | `task-002-simt-divergence` | SIMT 与 divergence；分支代价如何被观察到 |
| [ ] | `task-003-two-dimensional-indexing` | 2D/3D `blockDim`/`blockIdx`，与 1D 展开的等价性 |
| [ ] | `task-004-block-size-and-grid-shape` | block 大小作为性能参数；尾块与最后一个 block 的部分空闲 |

目录在本阶段开始时创建。

## 阶段验收

- [ ] 给定一个 3D 索引 kernel，能写出每个 warp 的 32 个 lane 各自算出的线性下标。
- [ ] 能用实验区分“divergence 慢”与“访存不合并慢”，并说明两者的区别。
- [ ] 能解释为什么 block 大小为 32 时同一 block 只有一个 warp。

## 资源

CUDA Programming Guide §2.3（Writing SIMT Kernels）、§5.8（Execution model）、[`resources/foundations/`](../../resources/foundations/)。硬件：T4 实测。
