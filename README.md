# 从 0 到 Blackwell 的 CUDA 学习项目

这是一个由 AI 担任助教、以代码和可重复实验为主线的 CUDA 学习仓库。学习顺序从 CUDA 编程模型、内存层级和并行模式开始，逐步进入 CUDA Core 性能工程、Tensor Core、Hopper，再到 Blackwell 的 `tcgen05` 和 TMEM。完整路线共 14 个阶段，见 [学习路线与进度](ROADMAP.md)。

仓库的基本单位不是“看完一章”，而是一个可以运行、测量、解释和复盘的 task。助教每次只布置一个 task，学习者完成后提交代码、命令、输出和结论，再进入下一步。

## 本机环境

路线分两个阶段用不同硬件。

**当前开发机**（2026-09-25 实测）：2 × Tesla T4（compute capability 7.5，15 GB），峰值显存带宽 320 GB/s，Driver 580.159.04，CUDA Toolkit 12.8，`ncu` 2025.1.1 已安装但硬件计数器不可用（`ERR_NVGPUCTRPERM`）。

**目标机**：NVIDIA B200（compute capability 10.0，HBM3/HBM3e 最多 180 GB，第五代 NVLink）。

这决定了每个阶段的证据类型：

- 阶段 00~10 在 T4 上实测。阶段 10 的 FP16 WMMA 是 T4 能跑的最深一层——T4 没有 bf16、`cp.async`、INT8 `m16n8k32`。
- 阶段 11（`cp.async`、bf16）需 sm_80+，阶段 12（Hopper）需 sm_90+。在 T4 上用 `nvcc -arch=sm_90` 做交叉编译与静态分析，**不写时间数字**；切到 B200 后转为实测。
- 阶段 13（Blackwell `tcgen05`、TMEM、NVFP4）需 sm_100+，主要目标机器是 B200。
- 阶段 14 综合项目在 T4 上完成，末尾有一个 B200 上的 Tensor Core 对照。

完整架构能力矩阵、B200 资源上限和证据分级见 [学习路线与进度](ROADMAP.md#架构能力矩阵)。

## 开始学习

```bash
git clone <your-repository-url>
cd cuda-from-zero-to-blackwell
git submodule update --init --recursive
```

按下面顺序阅读：

1. [学习路线与进度](ROADMAP.md)：确定当前阶段和下一项能力目标。
2. [资源地图](resources/README.md)：知道每个资料库解决什么问题。
3. [实验与 task 规范](experiments/README.md)：了解任务目录、交付物和记录方式。
4. 当前 `experiments/stage-x/task-y-*` 下的任务书：只做助教指定的一个 task。

没有本地 NVIDIA GPU 时，可以先完成代码阅读、编译检查和静态推理；涉及真实性能、`ncu` 或特定架构指令的结论必须标注为未验证，必要时使用云 GPU。

## 项目结构

```text
.
├── README.md                 # 项目目标与学习入口
├── ROADMAP.md                # 阶段目标、前置条件和进度
├── resources/                # 分类后的上游代码与资料索引
├── experiments/
│   ├── README.md             # task 设计和交付规范
│   ├── log-template.md       # 实验结果模板
│   └── stage-x/task-y-topic/ # 每次教学任务的独立目录
├── docs/                     # 文档站内容，遵守 AGENTS.md 的权限规则
└── AGENTS.md                # AI 助教工作流与文件权限
```

## 学习原则

- 先保证正确性，再进行测量和优化；每次实验只改变一个主要变量。
- 一个 task 只引入一个新概念。需要两个新概念就拆成两个 task。
- 所有性能结论都要绑定 GPU、compute capability、驱动、CUDA、编译参数、输入规模和 profiler 结果，并标明属于“实测 / 交叉编译 / 静态分析 / 待验证”中的哪一级。
- 中文资料用于建立上下文；具体指令、硬件行为和 API 约束回到官方文档、PTX ISA、CUTLASS 源码和实验验证。
- 阶段 11 及以后依赖特定硬件。没有对应 GPU 时仍可阅读和完成可交叉编译的部分，但不能把编译结果或阅读结果当成目标硬件实测。
