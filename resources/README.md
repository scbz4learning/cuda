# 资源地图

资源按“先建立上下文、再运行代码、最后核对官方定义”组织。submodule 只作为上游参考源；助教布置 task 时，应在任务书中写出具体路径、文件或搜索关键词，而不是只给仓库首页。

## 硬件路线

本项目分两个阶段用不同硬件，各分类的 README 都写明了在不同 compute capability 上哪些内容能跑。

| | 当前开发机 | 目标机 |
| --- | --- | --- |
| GPU | 2 × Tesla T4 | NVIDIA B200 |
| Compute Capability | 7.5（`sm_75`） | 10.0（`sm_100`） |
| 显存 | 2 × 15360 MiB | HBM3/HBM3e，最多 180 GB |
| 可实测阶段 | 阶段 00~10 | 阶段 00~14 全部 |
| `ncu` 计数器 | 不可用（`ERR_NVGPUCTRPERM`） | 需重新确认 |

阶段 11~13 在 T4 上是交叉编译与静态分析，切到 B200 后转为实测。换机时每条结论要重新标注证据级别：T4 上的“实测”搬到 B200 上只能算“交叉编译”或“待验证”，除非重测。

B200 的官方资源上限（每 SM 228 KB shared memory、每 block 227 KB、每 SM 64 warp、每 SM 32 block、最大可移植 cluster 8 / B200 可放开到 16）来自 [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)，属于文档依据，使用前需用 `cudaGetDeviceProperties` 与 occupancy API 复核。

## 本地资源

六个 submodule 均已就位，`git submodule status` 可核对 commit。

| 分类 | 路径 | 适用阶段 | 适用内容 |
| --- | --- | --- | --- |
| 基础 CUDA | [`foundations/`](foundations/) | 0~9 | CUDA Samples：kernel 基础、内存访问、transpose、reduction、scan、histogram、graph |
| 性能工程 | [`performance/`](performance/) | 7~8、12~13 | SGEMM 优化阶梯、ctest 框架、TMA/cluster/FP8 示例 |
| Tensor Core | [`tensor-cores/`](tensor-cores/) | 10~11 | WMMA、MMA、CuTe 入门和练习 |
| 现代架构 | [`modern-architectures/`](modern-architectures/) | 11~13 | CUTLASS、Hopper、Blackwell、TMA、WGMMA、tcgen05、TMEM |
| 巩固练习 | [`practice/`](practice/) | 全阶段 | 150 个 PMPP 顺序的参考实现，用作对答案而非教材 |

Thrust 与 CUB 不需要 submodule，随 CUDA Toolkit 提供，版本 2.7.0，头文件在 `/usr/local/cuda/include/thrust/` 与 `/usr/local/cuda/include/cub/`。`cub::DeviceScan`、`cub::DeviceReduce`、`cub::DeviceHistogram` 的头文件路径写任务书时要给出。

## 在线资料

- [中文优先清单](zh-first.md)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)（本仓库引用的是 v13.4.2 的五部分结构）
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [PTX ISA Reference](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)
- [Nsight Compute Documentation](https://docs.nvidia.com/cuda/nsight-compute/)
- [Thrust Documentation](https://nvidia.github.io/thrust/) / [CUB Documentation](https://nvlabs.github.io/cub/)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)

任务书引用编程指南时写章节号（如 §2.4 Writing Tile Kernels、§4.12 Asynchronous Data Copies），不要只写“见文档”。

## 使用顺序

1. 先读 [中文优先清单](zh-first.md)，建立术语和问题列表。
2. 进入对应分类的本地 submodule，定位一个能编译或能阅读的最小示例。
3. 先复现，再只改一个变量、tile、stage 或数据类型。
4. 用重复计时（必要时加 `nsys`）验证变化，把命令、输出和解释写入 task 的 `result.md`。**T4 上 `ncu` 计数器不可用，不要把 ncu 指标写进验收标准**；有权限时按阶段 07 `task-043` 的映射表补测。
5. 遇到具体指令或架构语义时回到官方文档核对，不以博客单独作为最终依据。
6. 巩固任务先独立完成再查 [`practice/`](practice/)，顺序反了测的就不是设计能力。
