# 从 0 到 Blackwell 的 CUDA 实验路线

这是一个以 **读代码、跑实验、看 profiler** 为主的 CUDA 学习资料仓库。主线从 CUDA 编程模型开始，经过 CUDA Core、性能工程、WMMA/Tensor Core、Hopper，再到 Blackwell 的 `tcgen05` / UMMA / TMEM。

本仓库保存：
- 可访问的资料链接与阅读顺序
- 阶段检查清单
- 实验目录与记录模板
- 本地环境、硬件和结果记录
- 通过 git submodule 管理的关键仓库代码

## 快速开始

```bash
git clone <your-repository-url>
cd cuda-from-zero-to-blackwell
git submodule update --init --recursive
```

先阅读：

1. [学习路线](ROADMAP.md)
2. [中文优先资料](resources/zh-first.md)
3. [实验总览](experiments/README.md)
4. [实验记录模板](experiments/log-template.md)

## 目录

```text
.
├── README.md
├── ROADMAP.md
├── resources/
│   ├── README.md
│   ├── zh-first.md
│   ├── cutlass/        # git submodule: NVIDIA/CUTLASS
│   ├── learn-cuda/     # git submodule: gau-nernst/learn-cuda
│   ├── LeetCUDA/       # git submodule: xlite-dev/LeetCUDA
│   └── cuda-samples/   # git submodule: NVIDIA/cuda-samples
├── experiments/
│   ├── README.md
│   ├── log-template.md
│   ├── stage-0/
│   ├── stage-1/
│   ├── stage-2/
│   ├── stage-3/
│   ├── stage-4/
│   └── stage-5/
├── notes/
│   └── .gitkeep
└── .gitignore
```

## 学习约束

- 以代码和可重复实验为主，视频只作为可选补充。
- 每个性能结论必须记录 GPU 型号、CUDA 版本、编译参数、输入规模和 profiler 结果。
- 阶段 4 需要 Hopper；阶段 5 的完整 `SM100` / TMEM 实验需要数据中心 Blackwell。RTX 50 系列是 `SM120`，不能替代全部 `SM100` 实验。
- 外部链接可能随 CUDA、CUTLASS 和文档版本变化；以仓库上游当前内容为准。

## 进度

- [ ] 阶段 0：基础 CUDA + 内存层级
- [ ] 阶段 1：CUDA Core GEMM 性能工程
- [ ] 阶段 2：WMMA / Ampere Tensor Core
- [ ] 阶段 3：Hopper MMA + CuTe 基础
- [ ] 阶段 4：Blackwell tcgen05 + TMEM
- [ ] 阶段 5：Blackwell 新特性拓展
