# 从 0 到 Blackwell：能力路线

路线按“能运行 → 能测量 → 能解释 → 能修改”推进。方括号是学习进度：`[ ]` 未开始，`[-]` 进行中，`[x]` 已完成。每次只推进一个 task，下一项任务由助教放在 `experiments/stage-x/task-y-*-`（`x` 是阶段编号，`y` 是三位阶段内序号）。

## 怎么用这份路线

三条推进规则，用来防止 task 之间的跨度失控：

1. **一个 task 只引入一个新概念。** 任务书里必须写清“本 task 只引入 X”。需要两个新概念就拆成两个 task。
2. **前置知识必须由前一个 task 教过。** 不得把 shared memory、bank、warp、计时协议之类写成“已知的前置条件”；如果前一个 task 没教，就在中间补一个。
3. **每个 task 只有一个主要变量。** 只改 block 大小、只改 pitch、只改线程数，其余保持不变，并在 `result.md` 记录改了什么。

### 巩固任务

每个阶段末尾有一个编号 `task-9xx` 的**巩固任务**。它有几个固定特征：

- **不引入任何新概念。** 只允许复用本阶段已经教过的内容。
- **换一个题型。** 巩固任务的算子、形状或索引模式必须与阶段内的教学任务不同，避免靠记忆而不是靠推理过关。
- **需要独立完成。** 做完并记录之前不要看 [`resources/practice/`](resources/practice/) 里的参考实现；对答案时把差异写进 `result.md` 的“与参考实现的差异”一节。
- **允许跳过。** 巩固任务标 `[ ]`，做不做由学习者决定；但阶段 14 的综合项目要求阶段 2、3、4、5 的巩固任务至少完成一个。

阶段 14 是跨阶段的综合项目，检验的是组合能力而不是新知识。

## 硬件路线

本项目分两个阶段用不同硬件。这不是妥协，而是刻意的安排：**基础阶段在便宜且数量多的卡上做，深度阶段在有目标指令的卡上做。**

### 当前开发机：2 × Tesla T4（2026-09-25 实测）

| 项 | 值 | 获取方式 |
| --- | --- | --- |
| GPU | 2 × Tesla T4，15360 MiB | `nvidia-smi` |
| Compute Capability | 7.5（`sm_75`，Turing） | `cudaGetDeviceProperties` |
| 峰值显存带宽 | 320 GB/s（GDDR6，理论） | `2 * memoryClockRate * memoryBusWidth / 8` |
| Driver | 580.159.04 | `nvidia-smi` |
| CUDA Toolkit | 12.8（V12.8.93） | `nvcc --version` |
| Nsight Systems | 2024.6.2 | `nsys --version` |
| Nsight Compute | 2025.1.1.0，已安装但**硬件计数器不可用** | 见下 |

`ncu` 返回 `ERR_NVGPUCTRPERM`（容器内核参数 `RmProfilingAdminOnly: 1`，无法修改）。这是环境的稳定事实，不是偶发故障。处理方式：

- 不把任何 ncu 计数器写进验收标准。
- 阶段 7 安排一个任务建立“指标名 → 含义 → 期望量级 → 能支撑什么结论”的映射表，即使采不到数也要能写出期望量级。
- 换到有权限的环境时按那张表补测，不用重做结论。

### 目标机：NVIDIA B200（sm_100）

以下数值来自 [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/) v13.4，**不是实测**。换到 B200 后必须先用 `cudaGetDeviceProperties` 和 occupancy API 复核一遍再写入 `result.md`。

| 项 | 值 |
| --- | --- |
| Compute Capability | 10.0（`sm_100`；架构特定变体为 `sm_100a`） |
| 最大并发 warp / SM | 64（cc 12.0 为 48） |
| 寄存器文件 | 每 SM 64K 个 32 位寄存器；每线程最多 255 |
| 最大 thread block / SM | 32 |
| Shared memory / SM | 228 KB（cc 12.0 为 128 KB） |
| Shared memory / block | 最多 227 KB（静态分配仍限 48 KB，需 `cudaFuncSetAttribute` 显式放开） |
| L1 + Shared 统一容量 | 256 KB / SM；carveout 可选 0/8/16/32/64/100/132/164/196/228 KB |
| 显存 | HBM3 / HBM3e，最多 180 GB |
| L2（GB200） | 126 MB |
| Thread block cluster | 可移植最大 8；**B200 可放开到 16**，需设置 `cudaFuncAttributeNonPortableClusterSizeAllowed` |
| 互联 | 第五代 NVLink |

两个容易踩的坑，提前写进阶段 12、13 的验收：

- 用到 cluster 的 kernel，算 occupancy 要用 `cudaOccupancyMaxActiveClusters`，不能用普通的 per-SM block 公式。
- cluster size 开到 16 会降低全 GPU 的最大活跃 block 数，occupancy 会掉。

### 架构能力矩阵

下表是本机用 `nvcc 12.8` 实际编译验证的（T4 列为实测，B200 列依据官方文档），用于判断每个阶段在哪台机器上能跑。

| 指令 / 特性 | sm_75（T4） | sm_80 | sm_90 | sm_100（B200） | 依据 |
| --- | :---: | :---: | :---: | :---: | --- |
| `wmma` API（FP16 16x16x16） | 可用 | 可用 | 可用 | 可用 | 本机编译通过 |
| `mma.sync.m16n8k8.f16` | 可用 | 可用 | 可用 | 可用 | 本机编译通过 |
| `mma.sync.m16n8k16.f32.bf16.bf16.f32` | 不可用 | 可用 | 可用 | 可用 | ptxas：`requires .target sm_80 or higher` |
| `mma.sync.m16n8k32.s32.s8.s8.s32` | 不可用 | 可用 | 可用 | 可用 | ptxas：`requires .target sm_80 or higher` |
| `cp.async` | 不可用 | 可用 | 可用 | 可用 | ptxas：`requires .target sm_80 or higher` |
| `cp.async.bulk.tensor`（TMA） | 不可用 | 不可用 | 可用 | 可用 | sm_80 编译失败，sm_90 通过 |
| `barrier.cluster.arrive` / 分布式 shared memory | 不可用 | 不可用 | 可用 | 可用 | ptxas：`requires .target sm_90 or higher` |
| `wgmma.mma_async` | 不可用 | 不可用 | 可用 | 可用 | Hopper 指令，依据 PTX ISA 文档 |
| `tcgen05.*` / TMEM | 不可用 | 不可用 | 不可用 | 可用 | Blackwell 指令，依据 PTX ISA 文档 |

后两行没有在本机 ptxas 上取得门控报错（手写 inline asm 语法未通过校验），因此标注为文档依据，不与前面各行等同。

由此得到阶段与机器的对应关系：

- **T4 实测**：阶段 0~10。阶段 10 的 FP16 WMMA 是 T4 上能跑的最深一层。
- **T4 交叉编译 / 阅读**：阶段 11（`cp.async`、bf16、sm_80 tensorop）、阶段 12（Hopper）。用 `nvcc -arch=sm_90` 验证编译，**不写时间数字**。
- **B200 实测**：阶段 12、13 全部转为实测任务，同时保留 T4 上的交叉编译结论作为对照。换机后这两阶段的 `result.md` 需要重写而不是补写。
- 阶段 11 在 B200 上也能实测，但 T4 上已完成的部分不必重做，只补 B200 的数字。

## 证据分级

每条结论都要标明属于哪一级，不允许混写：

| 级别 | 含义 | 允许的表述 |
| --- | --- | --- |
| **实测** | 在目标机器上运行，有正确性输出和重复计时 | “测得 X us，达到峰值的 Y%” |
| **交叉编译** | 在非本机架构上编译通过，检查了 PTX/SASS | “在 sm_90 上可编译，PTX 中出现 X 指令” |
| **静态分析** | 读源码或文档得出，附依据位置 | “由 `文件:行号` 可知 X” |
| **待验证** | 需要对应硬件或 ncu 权限 | “若可获得 Z 计数器，期望值为 …” |

换机器时，**每条结论要重新标注属于哪一级**。T4 上写“实测”的结论搬到 B200 上只能算“交叉编译”或“待验证”，除非重测。

## 阶段总览

| 阶段 | 主题 | T4 | B200 | 累计能力 |
| --- | --- | :---: | :---: | --- |
| [0](#阶段-0工具链与可测量性) | 工具链与可测量性 | 实测 | — | 能编译、判错、计时、如实记录环境 |
| [1](#阶段-1执行模型与线程组织) | 执行模型与线程组织 | 实测 | — | 能预测 warp 行为并解释 divergence |
| [2](#阶段-2合并访问与内存事务) | 合并访问与内存事务 | 实测 | — | 能手算 sector 并据此改写索引 |
| [3](#阶段-3shared-memory-与-bank-conflict) | shared memory 与 bank conflict | 实测 | — | 能用分块和 padding 消除冲突 |
| [4](#阶段-4reduction-与-warp-原语) | Reduction 与 warp 原语 | 实测 | — | 能写出多层次 reduction |
| [5](#阶段-5prefix-sum--scan) | Prefix sum / scan | 实测 | — | 能写出 work-efficient scan |
| [6](#阶段-6histogram-与原子操作) | Histogram 与原子操作 | 实测 | — | 能用 privatization 降低 contention |
| [7](#阶段-7性能分析方法论) | 性能分析方法论 | 实测 | — | 能用 roofline/occupancy/nsys/sanitizer 定位瓶颈 |
| [8](#阶段-8sgemm-优化阶梯) | SGEMM 优化阶梯 | 实测 | — | 逐级优化并解释每级收益 |
| [9](#阶段-9cuda-库与并发执行) | CUDA 库与并发执行 | 实测 | — | 会用库、stream、cooperative groups、graphs |
| [10](#阶段-10tensor-core-与-wmma) | Tensor Core 与 WMMA | 实测 | 可选补测 | 能用 fragment 表达一次 MMA |
| [11](#阶段-11cute-与-sm_80-低层-mma) | CuTe 与 sm_80+ 低层 MMA | 交叉编译 | 实测 | 能读懂并改写 CUTLASS 3.x mainloop |
| [12](#阶段-12hopper) | Hopper | 交叉编译 | 实测 | 能解释并验证 WGMMA/TMA/cluster/warp specialization |
| [13](#阶段-13blackwell) | Blackwell | 阅读 | 实测 | 能实现 tcgen05/TMEM/NVFP4 的最小可验证程序 |
| [14](#阶段-14综合项目) | 综合项目 | 实测 | 实测 | 能独立把多个算子拼成完整流水线 |

---

## 阶段 0：工具链与可测量性

**能力目标**：能把一个 CUDA 程序编译、运行、判错、稳定计时，并如实记录环境。本阶段专门补上“会测量”这件事——它不会随第一个 kernel 自动获得。

**资源**：[`resources/foundations/`](resources/foundations/)、CUDA Programming Guide §2.7（NVCC）、§2.5（Asynchronous Execution）、[`resources/practice/pmpp-cuda-study/`](resources/practice/pmpp-cuda-study/)。

| 进度 | Task | 只引入 |
| --- | --- | --- |
| [x] | `task-001-vector-add` | 第一个 kernel、1D 索引、边界判断 |
| [ ] | `task-002-device-query-and-error-check` | 设备查询与 `cudaGetLastError` / 错误检查宏 |
| [ ] | `task-003-compile-flags-and-resource-usage` | `nvcc` 选项与 `cuobjdump -res-usage` 看到的寄存器/共享内存 |
| [ ] | `task-004-timing-protocol` | `cudaEvent` 计时、warm-up、重复、min/median/mean、空 kernel 的 launch 开销 |
| [ ] | `task-901-巩固-harness` | **巩固**：把计时 harness 固化成可复用文件 |

**阶段验收**

- [ ] 能用 `cudaGetDeviceProperties` 打印并解释 CC、SM 数、L2 大小、峰值带宽四个字段的来源。
- [ ] 能说出一个“编译通过但运行报错”的例子，并用 `cudaGetLastError` 与 `compute-sanitizer` 区分是启动失败还是运行中出错。
- [ ] 能用 `cuobjdump -res-usage` 读出某个 kernel 的寄存器数和静态共享内存数，并说明它受哪个编译选项影响。
- [ ] 给出空 kernel 的 launch 开销中位数，后续所有 kernel 计时都以此为噪声下限。
- [ ] 巩固任务产出的 harness 被阶段 1~7 复用。

**硬件**：T4 实测。

---

## 阶段 1：执行模型与线程组织

**能力目标**：能预测任意索引表达式下 warp 的访问集合，并用实验解释 divergence 和 block 形状的代价。

**资源**：CUDA Programming Guide §2.3（Writing SIMT Kernels）、§5.8（Execution model）、[`cuda-samples/cpp/2_Concepts_and_Techniques/threadMigration/`](resources/foundations/cuda-samples/cpp/2_Concepts_and_Techniques/threadMigration/)。

| 进度 | Task | 只引入 |
| --- | --- | --- |
| [ ] | `task-001-warp-and-lane` | warp = 32、lane 与 `threadIdx` 的关系、每 SM 的 warp 调度 |
| [ ] | `task-002-simt-divergence` | SIMT 与 divergence；分支代价如何被观察到 |
| [ ] | `task-003-two-dimensional-indexing` | 2D/3D `blockDim`/`blockIdx`，与 1D 展开的等价性 |
| [ ] | `task-004-block-size-and-grid-shape` | block 大小作为性能参数；尾块与最后一个 block 的部分空闲 |
| [ ] | `task-901-巩固-2d-stencil` | **巩固**：2D stencil，找出会退化的索引写法并解释 |

**阶段验收**

- [ ] 给定一个 3D 索引 kernel，能写出每个 warp 的 32 个 lane 各自算出的线性下标。
- [ ] 能用实验区分“divergence 慢”与“访存不合并慢”，并说明两者的区别。
- [ ] 能解释为什么 block 大小为 32 时同一 block 只有一个 warp。
- [ ] 巩固任务中至少找到一种写法使相邻线程读到同一 cache line 的不同部分，并说明代价。

**硬件**：T4 实测。

---

## 阶段 2：合并访问与内存事务

**能力目标**：能手算一次访存跨多少个 sector，并据此改写索引使它合并。这是全路线中复用价值最高的一项能力。

**资源**：CUDA Programming Guide §2.3、Best Practices Guide “Coalesced Access to Global Memory”、[`cuda-samples/cpp/6_Performance/transpose/`](resources/foundations/cuda-samples/cpp/6_Performance/transpose/)、[`cuda-samples/cpp/2_Concepts_and_Techniques/boxFilter/`](resources/foundations/cuda-samples/cpp/2_Concepts_and_Techniques/boxFilter/)。

| 进度 | Task | 只引入 |
| --- | --- | --- |
| [ ] | `task-001-stride-and-sector` | stride 与 sector/transaction 的关系 |
| [ ] | `task-002-coalesced-vs-strided` | 同一算法两种索引写法的因果对比 |
| [ ] | `task-003-vectorized-load-float4` | `float4` / `int4` 向量化访存与对齐要求 |
| [ ] | `task-004-transpose-access-patterns` | 转置中读与写不能同时合并 |
| [ ] | `task-005-l1-l2-and-caching` | L1/L2 容量与 read-only cache；解释“看起来不合并却很快” |
| [ ] | `task-901-巩固-sector-prediction` | **巩固**：先手算再实测，验证对陌生 kernel 的预测 |

**阶段验收**

- [ ] 给定一个索引表达式和元素类型，能写出 32 个 lane 的地址、间隔字节数、覆盖的 32 字节 sector 数。
- [ ] 能用带宽曲线（stride 1/2/4/8/16）定位一个 kernel 的实际访存模式。
- [ ] 能解释为什么 `float4` 加载要求 16 字节对齐，以及不满足时会看到什么。
- [ ] 巩固任务中，先写下手算预测再运行，预测与实测的偏差有解释。

**硬件**：T4 实测。

---

## 阶段 3：shared memory 与 bank conflict

**能力目标**：能用分块让读写都合并，并用 padding 或 swizzle 消除 bank conflict。

**资源**：CUDA Programming Guide §2.4（Writing Tile Kernels）、[`cuda-samples/cpp/9_CUDA_Tile/tileTranspose/`](resources/foundations/cuda-samples/cpp/9_CUDA_Tile/tileTranspose/)、GPU Gems 3 第 39 章、CUB 头文件 `/usr/local/cuda/include/cub/`。

| 进度 | Task | 只引入 |
| --- | --- | --- |
| [ ] | `task-001-shared-memory-basics` | `__shared__` 的生命周期、per-block 独占、`__syncthreads()` |
| [ ] | `task-002-dynamic-shared-memory` | `extern __shared__`、启动配置第三参数、`cudaFuncSetAttribute` 扩到 48KB 以上 |
| [ ] | `task-003-bank-model` | bank 编号模型；构造并观测 N 路冲突 |
| [ ] | `task-004-tiled-transpose` | 分块同时让读和写都合并 |
| [ ] | `task-005-padding-and-swizzle` | `pitch` padding、XOR swizzle、转置存储 |
| [ ] | `task-901-巩固-transpose-ladder` | **巩固**：把全部转置变体收进一个可比较的阶梯 |

**阶段验收**

- [ ] 能对任意 shared memory 索引式逐 lane 手算 bank 号并判断冲突路数。
- [ ] 能区分“同一地址被多个 lane 读”（广播，不算冲突）和“同 bank 不同地址”（冲突）。
- [ ] 能用计时证明 padding 确实改变了耗时，或如实记录“差距在噪声内”并解释原因。
- [ ] 巩固任务产出的阶梯至少含 5 个变体，每个变体标注它消除了哪种不合并或冲突。

**硬件**：T4 实测。

---

## 阶段 4：Reduction 与 warp 原语

**能力目标**：能写出 block 内、跨 block 的 reduction，并解释每层归约用什么原语实现。

**资源**：PMPP 第 10 章、CUDA Programming Guide §4.4（Cooperative Groups）、[`cuda-samples/cpp/2_Concepts_and_Techniques/reduction/`](resources/foundations/cuda-samples/cpp/2_Concepts_and_Techniques/reduction/)、[`cuda-samples/cpp/3_CUDA_Features/warpAggregatedAtomicsCG/`](resources/foundations/cuda-samples/cpp/3_CUDA_Features/warpAggregatedAtomicsCG/)、[`learn-cuda/03_sum/`](resources/modern-architectures/learn-cuda/03_sum/)。

| 进度 | Task | 只引入 |
| --- | --- | --- |
| [ ] | `task-001-naive-reduction` | 每线程部分和 + `atomicAdd`；先得到可用的 baseline |
| [ ] | `task-002-shared-memory-reduction` | block 内树形归约与同步 |
| [ ] | `task-003-warp-shuffle` | `__shfl_down_sync` / `__shfl_xor_sync` 的同步语义 |
| [ ] | `task-004-atomic-vs-shuffle` | 两种实现的 contention 与耗时对比 |
| [ ] | `task-005-multi-block-reduction` | 两阶段归约；grid sync 或单值原子累加 |
| [ ] | `task-006-max-then-sum` | 为 softmax 做准备的两趟归约 |
| [ ] | `task-901-巩固-segmented-reduction` | **巩固**：变长分段归约，只用本阶段原语 |

**阶段验收**

- [ ] 能说出 `__shfl_*_sync` 的 mask 参数为什么必须包含所有参与的 lane。
- [ ] 能画出归约树并说明每一步的并行度变化。
- [ ] 能解释 atomic 版本在数据倾斜时的行为。
- [ ] 巩固任务能处理段长极不均匀的输入，并说明为什么不能沿用定长假设。

**硬件**：T4 实测。

---

## 阶段 5：Prefix sum / scan

**能力目标**：能写出 work-efficient 的并行 scan，并说明 bank conflict 在其中出现的位置。

**资源**：PMPP 第 11 章、[`cuda-samples/cpp/2_Concepts_and_Techniques/scan/`](resources/foundations/cuda-samples/cpp/2_Concepts_and_Techniques/scan/)、CUB 的 `cub::DeviceScan` 头文件 `/usr/local/cuda/include/cub/device/device_scan.cuh`。

| 进度 | Task | 只引入 |
| --- | --- | --- |
| [ ] | `task-001-sequential-baseline` | 串行定义与 host 参考，明确 inclusive/exclusive |
| [ ] | `task-002-hillis-steele` | 最naive并行 scan；O(n log n) 与额外读写 |
| [ ] | `task-003-blelloch-work-efficient` | up-sweep/down-sweep；用 padding 处理非 2 的幂 |
| [ ] | `task-004-block-scan-with-carry` | block 内 scan + block 间 carry；串行或 decoupled look-back |
| [ ] | `task-901-巩固-stream-compaction` | **巩固**：用 scan 实现 stream compaction |

**阶段验收**

- [ ] 能手算 Blelloch scan 在 `n = 8` 和 `n = 5` 时的每一步中间数组。
- [ ] 能说明为什么 Blelloch 的某些阶段会出现 bank conflict，以及 padding 或转置存储如何消除。
- [ ] 能区分 inclusive 与 exclusive 的下标约定，并用测试覆盖边界。
- [ ] 巩固任务能处理过滤后元素数量不确定的情况，并说明为什么需要两趟扫描。

**硬件**：T4 实测。

---

## 阶段 6：Histogram 与原子操作

**能力目标**：能用 shared memory privatization 降低原子操作竞争。

**资源**：PMPP 第 9 章、[`cuda-samples/cpp/2_Concepts_and_Techniques/histogram/`](resources/foundations/cuda-samples/cpp/2_Concepts_and_Techniques/histogram/)、CUB 的 `cub::DeviceHistogram` 头文件。

| 进度 | Task | 只引入 |
| --- | --- | --- |
| [ ] | `task-001-global-atomic-histogram` | 朴素全局 atomic 版本，作为 baseline |
| [ ] | `task-002-shared-memory-private-histogram` | 私有化副本与块间合并 |
| [ ] | `task-003-contention-and-replication` | 数据倾斜下的竞争；复制多份 histogram |
| [ ] | `task-901-巩固-counting-sort` | **巩固**：用 histogram 实现 counting sort |

**阶段验收**

- [ ] 能构造倾斜输入（所有元素落到同一个 bucket）并解释为何此时 atomic 版本最慢。
- [ ] 能算出每个 block 私有 histogram 的 shared memory 开销，并判断是否会超出限制。
- [ ] 能说明复制份数与 bucket 分布之间的关系。
- [ ] 巩固任务的 counting sort 输出稳定，并说明稳定性来自哪里。

**硬件**：T4 实测。

---

## 阶段 7：性能分析方法论

**能力目标**：能在没有 ncu 计数器权限的情况下，仍然系统地定位瓶颈并给出可验证的结论。这是本路线的“方法论阶段”，不引入新的 kernel。

**资源**：[`resources/performance/`](resources/performance/)、[`cutlass/media/docs/cpp/gemm_performance_measurement_methodology_guidelines.md`](resources/modern-architectures/cutlass/media/docs/cpp/gemm_performance_measurement_methodology_guidelines.md)（NVIDIA 官方的 GEMM 测量方法学，可直接借用其测量协议）、CUDA Programming Guide §5.5、§5.1。

| 进度 | Task | 只引入 |
| --- | --- | --- |
| [ ] | `task-001-arithmetic-intensity-and-roofline` | 算术强度定义与 roofline 判据 |
| [ ] | `task-002-occupancy-theory-and-api` | `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 与理论占用率 |
| [ ] | `task-003-latency-hiding-ilp-vs-tlp` | 指令级并行与线程级并行；展开与预取 |
| [ ] | `task-004-nsys-workflow` | `nsys profile`、`nsys stats`、NVTX range 标记 |
| [ ] | `task-005-ncu-metric-mapping` | 建立指标名 → 含义 → 期望量级 → 结论的映射表（采不到数也要写完） |
| [ ] | `task-006-compute-sanitizer` | `--tool memcheck/racecheck/synccheck/initcheck` 定位真实 bug |
| [ ] | `task-901-巩固-performance-report` | **巩固**：给一个陌生 kernel 写完整性能报告 |

**阶段验收**

- [ ] 能对一个 kernel 算出算术强度，并用 T4 的 320 GB/s 与 FP32 峰值判断它是 memory-bound 还是 compute-bound。
- [ ] 能解释一个 occupancy cliff：哪个资源（寄存器、shared memory、block 数）先成为瓶颈。
- [ ] 产出一张 ncu 指标映射表，即使本机采不到数也要写清每个指标的期望量级和它能支撑的结论。
- [ ] 能用 `compute-sanitizer --tool racecheck` 找出一个真实的 shared memory 竞争。
- [ ] 巩固任务的性能报告包含：环境、编译命令、warm-up 策略、min/median/mean、有效带宽、roofline 位置、瓶颈判断、证据分级。

**硬件**：T4 实测；`task-005` 的计数器部分为“待验证”级证据。

---

## 阶段 8：SGEMM 优化阶梯

**能力目标**：逐级优化 SGEMM，每一级只加一个手段，并能用 roofline 解释收益从哪来。

**资源**：PMPP 第 6 章、[`cuda-kernel-academy/01-sgemm-tutorial/`](resources/performance/cuda-kernel-academy/01-sgemm-tutorial/)（内含 naive/tiled/bank_conflict_free/double_buffer/tensor_core 五个版本的 `.cuh`，以及 `tests/test_sgemm.cu` 的测试写法）、[`learn-cuda/02_matmul_simt/`](resources/modern-architectures/learn-cuda/02_matmul_simt/)。

| 进度 | Task | 只引入 |
| --- | --- | --- |
| [ ] | `task-001-naive-sgemm` | 三重循环 baseline 与正确性 |
| [ ] | `task-002-coalesced-sgemm` | 只改 B 的访问模式使其合并 |
| [ ] | `task-003-shared-memory-tiling` | 用 shared memory tile 复用数据 |
| [ ] | `task-004-thread-tiling-register-blocking` | 每线程算多个输出，减少读次数 |
| [ ] | `task-005-warp-tiling-and-vectorized` | warp 级分块与向量化加载 |
| [ ] | `task-006-occupancy-cliff` | 处理一个具体的占用率悬崖并解释 |
| [ ] | `task-007-compare-cublas` | 与 cuBLAS SGEMM 对比，给出 TFLOP/s 与 roofline 差距 |
| [ ] | `task-901-巩固-full-ladder` | **巩固**：不参考已有实现，自己重建完整阶梯并与 cuBLAS 对齐 |

**阶段验收**

- [ ] 保留全部中间版本，能画出每一级相对上一级的加速比。
- [ ] 能对每一级回答“时间减少是因为读少了、写少了、还是复用多了”。
- [ ] 最终版本达到 cuBLAS 约 50% 以上，或如实记录实际比例并说明差距来源。
- [ ] 巩固任务得到的阶梯与 `cuda-kernel-academy` 的版本逐级对照，差异（tile 尺寸、分块顺序、寄存器复用方式）有记录。

**硬件**：T4 实测（FP32，无 Tensor Core 参与）。

---

## 阶段 9：CUDA 库与并发执行

**能力目标**：知道什么时候该用库而不是自己写，并能用 stream 和 graph 把执行组织起来。

**资源**：CUDA Programming Guide §4.4、§4.2（CUDA Graphs）、§4.3（Stream-Ordered Memory Allocator）、§2.5、§4.20（Dynamic Parallelism）、Thrust 与 CUB 头文件 `/usr/local/cuda/include/{thrust,cub}/`、[`cuda-samples/cpp/6_Performance/cudaGraphsPerfScaling/`](resources/foundations/cuda-samples/cpp/6_Performance/cudaGraphsPerfScaling/)、[`cuda-samples/cpp/3_CUDA_Features/binaryPartitionCG/`](resources/foundations/cuda-samples/cpp/3_CUDA_Features/binaryPartitionCG/)。

| 进度 | Task | 只引入 |
| --- | --- | --- |
| [ ] | `task-001-cublas-baseline` | handle、stream、指针模式与 alpha/beta |
| [ ] | `task-002-thrust` | host/device vector、`reduce`、`sort` |
| [ ] | `task-003-cub-block-primitives` | block/warp/device 级原语与自定义算子 |
| [ ] | `task-004-streams-and-async-copy` | `cudaStream`、pinned memory、`cudaMemcpyAsync` |
| [ ] | `task-005-overlap-proof` | 用 event 证明拷贝与计算确实重叠 |
| [ ] | `task-006-cooperative-groups` | grid sync、tiled partition、网格二分 |
| [ ] | `task-007-cuda-graphs` | 捕获与重放，与 launch 开销对比 |
| [ ] | `task-901-巩固-pipelined-pipeline` | **巩固**：把阶段 8 的 GEMM 与阶段 6 的 softmax 串成 stream 流水并用 graph 加速 |

**阶段验收**

- [ ] 能用 event 时间线证明重叠发生，并说明为什么需要 pinned memory。
- [ ] 能说出什么时候 `cudaDeviceSynchronize` 会毁掉流水。
- [ ] 能解释 graph 降低的是什么开销，以及它不解决什么问题。
- [ ] 巩固任务的流水线在 graph 模式下相比逐 kernel 启动有可测量的差距，并给出差距数值。

**硬件**：T4 实测（本机有 2 张 T4，可做最小 P2P 与多流实验，但不做 NCCL 集群内容）。

---

## 阶段 10：Tensor Core 与 WMMA

**能力目标**：能用 fragment 表达一次 MMA，并说明数据在寄存器里的布局。

**T4 上的限制**：T4 是 sm_75，只有 FP16（`mma.sync.m16n8k8.f16`）和 `wmma` API 可用。**没有** bf16（需 sm_80+）、**没有** `cp.async`、**没有** INT8 `m16n8k32`。本阶段只走 FP16，不要按 `m16n8k16.bf16` 的教程写。切到 B200 后，本阶段可补做 bf16 与 FP8 的对照，但那属于阶段 11、13。

**资源**：[`LeetCUDA/kernels/hgemm/wmma/`](resources/tensor-cores/LeetCUDA/kernels/hgemm/wmma/)、[`LeetCUDA/kernels/hgemm/mma/`](resources/tensor-cores/LeetCUDA/kernels/hgemm/mma/)、[`cuda-kernel-academy/01-sgemm-tutorial/src/kernels/tensor_core_sgemm.cuh`](resources/performance/cuda-kernel-academy/01-sgemm-tutorial/src/kernels/tensor_core_sgemm.cuh)、[`cuda-samples/cpp/3_CUDA_Features/ptxjit/`](resources/foundations/cuda-samples/cpp/3_CUDA_Features/ptxjit/)、CUTLASS `include/cutlass/arch/wmma_sm75.h`、CUDA Programming Guide §5.5。

| 进度 | Task | 只引入 |
| --- | --- | --- |
| [ ] | `task-001-fp16-dtypes-and-error` | `__half` / `__half2`；T4 上 bf16 不可用的实测证据；累加精度与容差 |
| [ ] | `task-002-wmma-fragments` | `fragment` 的 load / `mma_sync` / store 四个动作 |
| [ ] | `task-003-fragment-layout-verification` | 自己写 host 侧参考，反推 fragment 的实际内存布局 |
| [ ] | `task-004-wmma-gemm-vs-cublas-hgemm` | 与 cuBLAS HGEMM 对比，量化 T4 上 Tensor Core 的实际收益 |
| [ ] | `task-005-mma-sync-ptx-m16n8k8` | 直接写 `mma.sync.m16n8k8.f16` inline PTX，与 WMMA 结果对拍 |
| [ ] | `task-901-巩固-hgemm-independent` | **巩固**：不看参考实现独立写 WMMA HGEMM，再逐项对照 |

**阶段验收**

- [ ] 能画出 `wmma::fragment<matrix_a, 16, 16, 16, half, row_major>` 中每个 lane 持有的元素位置。
- [ ] 能说明累加器为什么用 FP32，以及 T4 上 FP16 Tensor Core 与 FP32 CUDA Core 的峰值差距。
- [ ] 能用 inline PTX 版本和 WMMA 版本对拍，误差在容差内。
- [ ] 巩固任务的独立实现与 `LeetCUDA` 版本的差异（tile 尺寸、加载顺序、epilogue）逐条记录。

**硬件**：T4 实测（FP16 路径）；B200 上可选补做 bf16 对照。

---

## 阶段 11：CuTe 与 sm_80+ 低层 MMA

**能力目标**：能读懂 CUTLASS 3.x 的 mainloop，认出数据从 global 到 Tensor Core 的每一步，并能自己改写一段。

**T4 上的限制**：T4 不支持 `cp.async` 和 bf16。编译验证用 `nvcc -arch=sm_80`，**不写时间数字**。切到 B200 后本阶段全部任务转为实测，只需补测，不必重做分析。

**资源**：[`cutlass/media/docs/cpp/cute/`](resources/modern-architectures/cutlass/media/docs/cpp/cute/)（`00_quickstart.md`、`01_layout.md`、`02_layout_algebra.md`、`03_tensor.md`、`04_algorithms.md`、`0t_mma_atom.md`、`0z_tma_tensors.md`、`0x_gemm_tutorial.md`）、[`cutlass/media/docs/cpp/cutlass_3x_design.md`](resources/modern-architectures/cutlass/media/docs/cpp/cutlass_3x_design.md)、[`cutlass/media/docs/cpp/pipeline.md`](resources/modern-architectures/cutlass/media/docs/cpp/pipeline.md)、[`learn-cuda/02_matmul_sm80/`](resources/modern-architectures/learn-cuda/02_matmul_sm80/)、[`cuda-samples/cpp/3_CUDA_Features/globalToShmemAsyncCopy/`](resources/foundations/cuda-samples/cpp/3_CUDA_Features/globalToShmemAsyncCopy/)、[`cuda-samples/cpp/9_CUDA_Tile/helloTile/`](resources/foundations/cuda-samples/cpp/9_CUDA_Tile/helloTile/)、CUTLASS `include/cutlass/arch/mma_sm80.h`。

| 进度 | Task | 只引入 |
| --- | --- | --- |
| [ ] | `task-001-cute-layout-basics` | `Layout` / `TiledCopy` 的概念；用 `03_visualize_layout` 打印 layout |
| [ ] | `task-002-ldmatrix-fragment-load` | `ldmatrix` / `ldmatrix.trans` 的作用与限制 |
| [ ] | `task-003-cp-async-multistage-pipeline` | `cp.async` + 多级流水；commit/wait 的配对关系 |
| [ ] | `task-004-sm80-tensorop-mainloop` | 读 `14_ampere_tensorop_gemm`，标出 mainloop 的三个阶段 |
| [ ] | `task-901-巩固-cute-rewrite` | **巩固**：用 CuTe 重写一个阶段 3 已完成的 shared memory kernel |

**阶段验收**

- [ ] 能用 `nvcc -arch=sm_80` 编译通过一个 `cp.async` 多级流水的最小 kernel，并用 `cuobjdump -sass` 找到 `LDGSTS` 指令。
- [ ] 能画出 CUTLASS 3.x mainloop 中 global→shared→fragment→mma→epilogue 的完整数据流。
- [ ] 巩固任务能用 CuTe 表达一个已有的 shared memory kernel，并说明 CuTe 版本与手写 `__shared__` 版本的对应关系。
- [ ] 在 T4 上完成的部分不出现任何未实测的时间或 TFLOP/s 数字。

**硬件**：T4 交叉编译 + 静态分析；B200 实测。

---

## 阶段 12：Hopper

**能力目标**：能解释 WGMMA、TMA、thread block cluster 和 warp specialization 各自解决什么瓶颈。

**T4 上的限制**：T4 不支持 cluster、TMA、WGMMA。编译验证用 `nvcc -arch=sm_90`，**不写时间数字**。切到 B200（sm_100，向后兼容 Hopper 特性）后本阶段全部转为实测。

**资源**：CUTLASS `examples/48_hopper_warp_specialized_gemm`、`examples/54_hopper_fp8_warp_specialized_gemm`、`examples/88_hopper_fmha`、[`cutlass/include/cutlass/arch/mma_sm90.h`](resources/modern-architectures/cutlass/include/cutlass/arch/mma_sm90.h)、[`cuda-kernel-academy/03-hpc-advanced/src/07_cuda13_features/tma.cu`](resources/performance/cuda-kernel-academy/03-hpc-advanced/src/07_cuda13_features/tma.cu) 与 [`cluster.cu`](resources/performance/cuda-kernel-academy/03-hpc-advanced/src/07_cuda13_features/cluster.cu)、CUDA Programming Guide §4.10、§4.11、§5.1、PTX ISA。

| 进度 | Task | 只引入 |
| --- | --- | --- |
| [ ] | `task-001-tma-and-mbarrier` | `cp.async.bulk.tensor`、TMA descriptor、`mbarrier` 的 arrive/wait |
| [ ] | `task-002-cluster-and-distributed-shmem` | thread block cluster、分布式 shared memory、`cluster.map` |
| [ ] | `task-003-wgmma-vs-wmma` | A/B 驻留位置、scale 语义、异步完成与等待方式 |
| [ ] | `task-004-warp-specialization` | producer/consumer warp 分工与各自的 barrier |
| [ ] | `task-005-pipeline-stages` | 把 stages 改成 2/3/4，统计 shared memory 与寄存器用量变化 |
| [ ] | `task-006-hopper-fmha-dataflow` | 读 `88_hopper_fmha`，画出 attention 的数据路径 |
| [ ] | `task-901-巩固-cluster-occupancy` | **巩固**：用 `cudaOccupancyMaxActiveClusters` 解释 cluster 形状如何影响 occupancy |

**阶段验收**

- [ ] 能用 `nvcc -arch=sm_90` 编译通过含 `barrier.cluster.arrive` 的最小 kernel，并引用 ptxas 错误原文说明它在 sm_75 上被拒绝。
- [ ] 能解释为什么 WGMMA 不需要像 WMMA 那样显式把 A/B 载入寄存器。
- [ ] 能从 `48_hopper_warp_specialized_gemm` 中指出 TMA producer 和 WGMMA consumer 的分界线。
- [ ] 巩固任务能用 `cudaOccupancyMaxActiveClusters` 说明 cluster size 从 1 增到 8 再到 16（需放开非可移植属性）时活跃 cluster 数如何变化。
- [ ] 在 T4 上完成的部分不出现未实测的性能数字。

**硬件**：T4 交叉编译 + 静态分析；B200 实测。

---

## 阶段 13：Blackwell

**能力目标**：能**实现**并解释 `tcgen05`、TMEM、NVFP4/MXFP block scaling 改变了什么硬件瓶颈。本阶段在 T4 上只能阅读，在 B200 上是完整实测阶段。

**资源**：[`learn-cuda/02_matmul_sm100/`](resources/modern-architectures/learn-cuda/02_matmul_sm100/)（`matmul_v0.cu` 到 `matmul_v7.cu` 的演进链、`_matmul_tmem.cu`）、[`cutlass/media/docs/cpp/blackwell_functionality.md`](resources/modern-architectures/cutlass/media/docs/cpp/blackwell_functionality.md)、[`cutlass/media/docs/cpp/blackwell_cluster_launch_control.md`](resources/modern-architectures/cutlass/media/docs/cpp/blackwell_cluster_launch_control.md)、[`cutlass/include/cutlass/arch/mma_sm100.h`](resources/modern-architectures/cutlass/include/cutlass/arch/mma_sm100.h)、CUTLASS `examples/70_blackwell_gemm`、`examples/77_blackwell_fmha`、`examples/72_blackwell_narrow_precision_gemm`、`examples/73_blackwell_gemm_preferred_cluster`、`examples/75_blackwell_grouped_gemm`、Blackwell Tuning Guide、PTX ISA 的 tcgen05 章节。

| 进度 | Task | 只引入 |
| --- | --- | --- |
| [ ] | `task-001-tmem-alloc-and-fence` | TMEM 的分配/释放、`tcgen05.fence`、为什么需要独立于 shared memory 的存储 |
| [ ] | `task-002-tcgen05-mma` | `tcgen05.mma` 的 operand 布局与 A 在 TMEM 中的表示 |
| [ ] | `task-003-learn-cuda-sm100-ladder` | 逐版读 `matmul_v0`→`v7`，每版只回答“新增了什么、为什么需要” |
| [ ] | `task-004-tcgen05-matmul-minimal` | 写一个最小 tcgen05 matmul，**在 B200 上跑通并验证正确性** |
| [ ] | `task-005-nvfp4-and-mxfp` | block scaling 的 scale factor 存放位置；`72_*` 与 `75_*` 的差异 |
| [ ] | `task-006-cluster-shapes-and-cta-pair` | 可变 cluster 形状、`73_*` 的 preferred cluster、B200 的非可移植 cluster size 16 |
| [ ] | `task-007-capstone-dataflow` | 完整数据流：global → TMA → shared/TMEM → Tensor Core → epilogue |
| [ ] | `task-901-巩固-tmem-vs-shared` | **巩固**：同一 GEMM 分别用 shared memory pipeline 与 TMEM 实现，对比资源占用与数据流 |

**阶段验收**

- [ ] 能解释 TMEM 相对 shared memory 的容量与访问方式差异，以及它为什么改变 mainloop 的结构。
- [ ] 能在 B200 上跑通一个 `tcgen05` matmul，用 host 参考验证数值正确，并记录 kernel 时间与 `cudaOccupancyMaxActiveClusters` 的结果。
- [ ] 能用实测数据核对 Blackwell Tuning Guide 的资源上限：每 SM 228 KB shared memory、每 block 227 KB、每 SM 64 warp、每 SM 32 block。
- [ ] 能画出 `matmul_v0` 到 `matmul_v7` 每一步新增的能力，形成一条可讲述的演进链。
- [ ] 能说明 NVFP4 与 MXFP 的 scale 粒度差异。
- [ ] 巩固任务能说清 TMEM 版本相对 shared memory 版本省掉了什么数据搬运。
- [ ] 在 T4 上完成的部分（`task-001`、`002`、`003`、`006` 的分析部分）不出现未实测的性能数字，并注明“B200 上需补测”。

**硬件**：T4 阅读 + 交叉编译；B200 实测（本阶段的主要目标机器）。

---

## 阶段 14：综合项目

**能力目标**：把前面各阶段的算子拼成完整流水线，检验组合能力。这一阶段不引入新指令，用的全是已经学过的东西。

**前置**：阶段 2、3、4、5 的巩固任务至少完成一个；阶段 8、10、13 的验收标准已达成。

**资源**：[`cuda-kernel-academy/04-inference-engine/`](resources/performance/cuda-kernel-academy/04-inference-engine/)、[`learn-cuda/04_softmax/`](resources/modern-architectures/learn-cuda/04_softmax/)（naive softmax、online softmax、`atomicCAS`）、[`learn-cuda/07_attention/`](resources/modern-architectures/learn-cuda/07_attention/)、[`cuda-kernel-academy/03-hpc-advanced/src/05_attention/`](resources/performance/cuda-kernel-academy/03-hpc-advanced/src/05_attention/)。

| 进度 | Task | 只引入 |
| --- | --- | --- |
| [ ] | `task-001-softmax-from-scratch` | 组合：阶段 4 归约 + 阶段 5 scan + 阶段 2 访存。含 max 减法保证数值稳定 |
| [ ] | `task-002-layernorm-from-scratch` | 组合：两趟归约（mean/var）+ 向量化访存；与 `layernorm` 库对比 |
| [ ] | `task-003-attention-naive` | 组合：矩阵乘 + softmax + 矩阵乘，先求对再优化 |
| [ ] | `task-004-attention-tiled-and-fused` | 组合：阶段 3 分块 + 阶段 14 阶段的 online softmax，避免物化 N² 矩阵 |
| [ ] | `task-005-fused-kernel-and-graph` | 组合：kernel fusion + CUDA Graph 降低 launch 开销 |
| [ ] | `task-006-capstone-report` | **巩固**：完整技术报告，串起全路线的证据分级 |
| [ ] | `task-901-巩固-port-to-blackwell` | **巩固**：把阶段 14 的核心算子在 B200 上用 Tensor Core 重做并对比 |

**阶段验收**

- [ ] 每个算子都有 host 参考实现和覆盖边界的正确性测试。
- [ ] softmax 任务能说明为什么需要减 max，以及不减去会怎样（构造溢出输入证明）。
- [ ] attention 任务能给出朴素版与 fused 版的峰值显存占用对比。
- [ ] 融合前后有可测量的时间差，且能说明差值主要来自 launch 开销还是访存量减少。
- [ ] `task-006` 的报告包含：环境、全部编译命令、每个 kernel 的 min/median/mean、有效带宽或 TFLOP/s、roofline 位置、瓶颈判断、证据分级、以及仍未验证的清单。
- [ ] `task-901` 在 B200 上给出 FP32 CUDA Core 与 FP16/BF16 Tensor Core 的同输入对照。

**硬件**：阶段 14 全程 T4 实测；`task-901` 在 B200 上实测。

---

## 现状映射

本路线重构后，仓库中已有内容对应关系如下。重构前已存在的 task 不移动目录，只在此登记；后续 task 按新编号创建。

| 已有目录 | 现状 | 对应新阶段 |
| --- | --- | --- |
| `experiments/stage-0/task-001-vector-add` | 已完成 | 阶段 0 `task-001` |
| `experiments/stage-0/task-002-device-query-and-error-check` | 待开始 | 阶段 0 `task-002` |

原 `task-002-transpose-coalescing/` 一次引入二维索引、shared memory、动态共享内存、bank 模型、计时协议共五项，已删除。其内容按新路线拆到阶段 2 的 `task-004`（转置的读与写不能同时合并）与阶段 3 的 `task-003`~`005`（bank 模型、分块、padding），计时协议移到阶段 0 的 `task-004`。

阶段目录只在开始该阶段时创建，不预先建立空目录。

## 进度记录规则

阶段完成必须同时满足：对应 task 的 `result.md` 已填写、正确性有输出证据、测量条件完整、学习者能用自己的话解释结果。助教只在这些证据出现后更新本文件；单纯读完链接不算完成。

每个 task 的 `result.md` 至少记录：GPU / 驱动 / CUDA / compute capability、编译命令原文、输入规模与数据类型、正确性结果、warm-up 策略与重复次数、min/median/mean 计时、profiler 命令与关键输出、结论属于“实测 / 交叉编译 / 静态分析 / 待验证”中的哪一级，以及与 [`resources/practice/`](resources/practice/) 参考实现的差异（如适用）。
