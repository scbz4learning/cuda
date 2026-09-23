# 实验目录

实验按阶段放置。每次实验建议建立独立目录或复制 [记录模板](log-template.md)，不要只保存最终数字。

## 阶段目录

- `stage-0/`：基础 CUDA、transpose、reduction、scan、histogram
- `stage-1/`：CUDA Core GEMM、性能工程、roofline、occupancy
- `stage-2/`：WMMA / Ampere Tensor Core
- `stage-3/`：Hopper MMA + CuTe 基础
- `stage-4/`：Blackwell tcgen05 + TMEM
- `stage-5/`：Blackwell 新特性拓展（NVFP4/MXFP8、2-SM MMA、TMA）

## 每个实验至少保存

- 源码或上游 commit
- 编译命令和运行命令
- GPU、驱动、CUDA、compute capability
- 输入尺寸和数据类型
- 正确性检查结果
- 至少 3 次计时或明确的 warm-up 策略
- ncu 命令与关键指标
- 一句话结论和下一步改动
