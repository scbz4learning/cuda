# 阶段 02：合并访问与内存事务

## 阶段目标

能手算一次访存跨多少个 sector，并据此改写索引使它合并。这是全路线中复用价值最高的一项能力。

## Task 队列

| 进度 | Task | 只引入 |
| --- | --- | --- |
| [ ] | `task-011-stride-and-sector` | stride 与 sector/transaction 的关系 |
| [ ] | `task-012-coalesced-vs-strided` | 同一算法两种索引写法的因果对比 |
| [ ] | `task-013-vectorized-load-float4` | `float4` / `int4` 向量化访存与对齐要求 |
| [ ] | `task-014-transpose-access-patterns` | 转置中读与写不能同时合并 |
| [ ] | `task-015-l1-l2-and-caching` | L1/L2 容量与 read-only cache；解释“看起来不合并却很快” |

本阶段 task 编号 011–016。目录在本阶段开始时创建。

## 阶段验收

- [ ] 给定一个索引表达式和元素类型，能写出 32 个 lane 的地址、间隔字节数、覆盖的 32 字节 sector 数。
- [ ] 能用带宽曲线（stride 1/2/4/8/16）定位一个 kernel 的实际访存模式。
- [ ] 能解释为什么 `float4` 加载要求 16 字节对齐，以及不满足时会看到什么。

## 资源

CUDA Programming Guide §2.3、Best Practices Guide “Coalesced Access to Global Memory”、[`cuda-samples/cpp/6_Performance/transpose/`](../../resources/foundations/cuda-samples/cpp/6_Performance/transpose/)。硬件：T4 实测。

## 备注

`task-014` 承接原 `task-002-transpose-coalescing/` 的第 1、2 步内容（naive 转置的读合并/写不合并）。原 task 范围过大已删除，其 bank 模型与分块部分归 `task-019`~`021`。拆分说明见 [`ROADMAP.md`](../../ROADMAP.md) 的“现状映射”。
