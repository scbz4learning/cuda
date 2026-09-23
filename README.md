# 从 0 到 Blackwell 的 CUDA 学习项目

这是一个由 AI 担任助教、以代码和可重复实验为主线的 CUDA 学习仓库。学习顺序从 CUDA 编程模型、内存层级和并行模式开始，逐步进入 CUDA Core 性能工程、Tensor Core、Hopper，再到 Blackwell 的 `tcgen05`、UMMA 和 TMEM。

仓库的基本单位不是“看完一章”，而是一个可以运行、测量、解释和复盘的 task。助教每次只布置一个 task，学习者完成后提交代码、命令、输出和结论，再进入下一步。

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
- 所有性能结论都要绑定 GPU、compute capability、驱动、CUDA、编译参数、输入规模和 profiler 结果。
- 中文资料用于建立上下文；具体指令、硬件行为和 API 约束回到官方文档、PTX ISA、CUTLASS 源码和实验验证。
- 阶段 3 及以后依赖特定硬件。没有 Hopper 或 Blackwell 时仍可阅读和完成可移植部分，但不能把模拟或阅读结果当成目标硬件实测。
