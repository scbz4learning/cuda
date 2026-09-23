# 实验目录

实验按阶段放置。每次实验建议建立一个独立目录或复制 [记录模板](log-template.md)，不要只保存最终数字。

## 阶段目录

- `stage-0/`：环境检查、vector add、第一次 `ncu`
- `stage-1/`：transpose、reduction、scan、tiled GEMM
- `stage-2/`：roofline、occupancy、bank conflict、persistent kernel
- `stage-3/`：建议后续新增 WMMA / `mma.sync`
- `stage-4/`：建议在 Hopper 环境新增 TMA / WGMMA / CUTLASS 实验
- `stage-5/`：建议在 Blackwell 环境新增 `tcgen05` / TMEM / cluster 实验

## 每个实验至少保存

- 源码或上游 commit
- 编译命令和运行命令
- GPU、驱动、CUDA、compute capability
- 输入尺寸和数据类型
- 正确性检查结果
- 至少 3 次计时或明确的 warm-up 策略
- ncu 命令与关键指标
- 一句话结论和下一步改动
