# 资源地图

资源按“先建立上下文、再运行代码、最后核对官方定义”组织。submodule 只作为上游参考源；助教布置 task 时，应在任务书中写出具体路径、文件或搜索关键词，而不是只给仓库首页。

## 本地资源

| 分类 | 路径 | 适用内容 |
| --- | --- | --- |
| 基础 CUDA | [`foundations/`](foundations/) | CUDA Samples、kernel 基础、内存访问、transpose、reduction、scan、histogram |
| 性能工程 | [`performance/`](performance/) | CUDA Core、GEMM、roofline、occupancy、Nsight Compute |
| Tensor Core | [`tensor-cores/`](tensor-cores/) | WMMA、MMA、CuTe 入门和练习 |
| 现代架构 | [`modern-architectures/`](modern-architectures/) | Hopper、Blackwell、TMA、WGMMA、TCGen05、TMEM |

每个分类目录中的 `README.md` 说明 submodule 的用途、推荐入口、适用阶段和已知硬件限制。不要直接修改 submodule；需要补充学习提示时修改分类说明或 `zh-first.md`。

## 在线资料

- [中文优先清单](zh-first.md)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)
- [PTX ISA Reference](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [Nsight Compute Documentation](https://docs.nvidia.com/cuda/nsight-compute/)

## 使用顺序

1. 先读 [中文优先清单](zh-first.md)，建立术语和问题列表。
2. 进入对应分类的本地 submodule，定位一个能编译或能阅读的最小示例。
3. 先复现，再只改一个变量、tile、stage 或数据类型。
4. 用 `ncu` 或重复计时验证变化，把命令、输出和解释写入 task 的 `result.md`。
5. 遇到具体指令或架构语义时回到官方文档核对，不以博客单独作为最终依据。
