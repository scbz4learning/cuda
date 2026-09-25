# 巩固练习

## 本地入口

`pmpp-cuda-study/`（commit `370c238e9a2d4d744a00fe24b6050155e9ff367f`）。150 个按 PMPP 顺序排列的小例子，每个目录包含 `main.cu`、`README.md`、`benchmark.yaml`、`meta.yaml`，可单独编译运行，自带 CPU 参考实现和 `--check` 校验。

课程地图见 `docs/curriculum-map.md`，示例约定见 `docs/example-conventions.md`。

## 使用边界

**这是参考，不是答案。** 使用顺序必须是：

1. 先在自己的 task 里独立实现，做对、测到、记录完。
2. 再来这里读同一个例子的实现，比较索引写法、循环结构和同步位置。
3. 差异写进 `result.md` 的“与参考实现的差异”一节。

不允许先读实现再回来写自己的版本；那样测的是抄写能力，不是设计能力。

## 按阶段索引

| 阶段 | 对应例子 |
| --- | --- |
| 阶段 00 | `002_vector-addition`、`007_saxpy`、`008_copy-array-kernel` |
| 阶段 01 | `003_vector-subtraction`、`018_matrix-addition`、`019_matrix-transpose-naive`（多维索引） |
| 阶段 02 | `117_coalescing-study`、`130_irregular-gather-scatter-study` |
| 阶段 03 | `020_matrix-transpose-with-shared-memory`、`116_bank-conflict-study`、`118_shared-memory-staging-patterns`、`119_transpose-optimization-ladder` |
| 阶段 04 | `023_sum-reduction`、`101_segmented-reduction`、`103_two-phase-reduction`、`105_warp-aggregated-atomics`、`111_warp-shuffle-reduction` |
| 阶段 05 | `026_prefix-sum-naive-scan`、`027_prefix-sum-work-efficient-scan`、`102_segmented-scan`、`107_stream-compaction-prefix-sum-fused`、`112_warp-shuffle-scan` |
| 阶段 06 | `028_histogram-global-atomics`、`029_histogram-shared-memory`、`030_stream-compaction`、`110_counting-sort-parallel` |
| 阶段 07 | `137_heat-diffusion-tiled-2d`（带宽/算术强度判据的练习载体） |
| 阶段 08 | `042_naive-matrix-multiply`、`043_tiled-matrix-multiply`、`113_cpasync-style-tiled-matmul-demo`、`114_double-buffered-tiling`、`115_register-tiling-gemm` |
| 阶段 09 | `104_persistent-reduction-kernel`、`150_mini-inference-pipeline` |
| 阶段 10~13 | 上游多数例子以 CUDA Core 为主；Tensor Core 与 Hopper/Blackwell 内容以 `resources/modern-architectures/` 和 `resources/tensor-cores/` 为主 |
| 阶段 14 | `131_sobel-filter-optimized`、`132_box-blur-separable-optimized`、`141_layernorm-forward`、`142_softmax-stable`、`143_fused-softmax-scale-mask`、`144_attention-score-demo`、`145_im2col-convolution`、`147_winograd-conv-demo` |

## 工具

```bash
python scripts/build_examples.py --help   # 批量构建
python scripts/run_smoke_tests.py         # 冒烟测试
python scripts/validate_repo.py            # 仓库结构校验
```

`benchmark.yaml` 记录该例子的输入规模与测量方式，可以作为自己 `result.md` 测量表的格式参考。

## 注意

这些例子的计时是教学用的微基准，足够用来学习访存行为、同步开销和 baseline 与改进版的结构差异，不能当作受控硬件上的生产级性能结论。自己重新测一遍，以自己的数据为准。
