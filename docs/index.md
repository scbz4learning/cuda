---
icon: lucide/graduation-cap
---

# 从 0 到 Blackwell 的 CUDA 学习项目

这是一个由 AI 担任助教、面向初学者的 CUDA 学习仓库。项目不以收集资料为目标，而是通过可运行、可测量、可解释的实验，逐步理解 CUDA 编程模型、GPU 性能和现代 NVIDIA 架构。

## 项目结构

```text
.
├── README.md                 # 项目说明与快速开始
├── ROADMAP.md                # 学习阶段与进度
├── resources/                # 按主题整理的资料和上游代码
├── experiments/              # 按阶段和 task 保存学习实验
│   └── stage-x/task-y-topic/
├── docs/                     # GitHub Pages 文档
└── AGENTS.md                # AI 助教工作规范
```

## 学习方式

- 学习以 task 为基本单位。每个 task 都应有任务书、代码、运行命令和结果记录。
- 助教每次只布置一个 task，并根据路线图、资源和上一个 task 的结果决定下一步。
- 实验遵循“先正确，再测量，后优化”；性能结论必须有可复现的环境、输入和 profiler 或计时结果。
- 当前路线从基础 CUDA 和内存层级开始，经过 CUDA Core、Tensor Core、Hopper，逐步进入 Blackwell 的 `tcgen05`、UMMA 和 TMEM。

## 重要约定

- `ROADMAP.md` 记录阶段目标和实际进度；没有完成证据时不标记为完成。
- `resources/` 中的上游仓库按学习主题组织，具体实验应记录所使用的路径和 commit。
- `docs/original/` 保存用户的关键词版笔记，AI 不得修改、移动或删除。
- 只有用户明确要求根据 `docs/original/`、task 进展和代码整理笔记时，AI 才生成或修改完整文档。
- `docs/other-ai/` 中的内容属于 AI 辅助材料，未经用户审查，不代表最终结论。

## 开始阅读

- [项目说明](../README.md)
- [学习路线与进度](../ROADMAP.md)
- [资源地图](../resources/README.md)
- [实验与 task 规范](../experiments/README.md)
- [AI 助教工作规范](../AGENTS.md)
