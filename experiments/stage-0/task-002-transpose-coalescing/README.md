# Task 002：转置、合并访问与 shared memory bank conflict

## 目标

用同一个转置任务，把 task-001 只观察过一次的“线程索引”推进成可以定量比较的结论：naive 转置慢在哪里、共享内存分块为什么能同时让读和写都合并、为什么 `TILE` 行距会引入 32-way bank conflict、以及 padding 如何把它消掉。本 task 交付一个可复现的计时协议和一份手算的访问模式分析。

硬件限制：本机 `ncu` 计数器不可用（`ERR_NVGPUCTRPERM`，内核参数 `RmProfilingAdminOnly: 1`，容器内无法修改）。所以本 task 不依赖 ncu 指标，改为用“只改一个变量的重复计时”把因果关系做实，并在 `result.md` 中写清哪些结论是计时推断、哪些是手算推断、哪些必须等 ncu 才能确认。

## 前置条件

- 已完成 `experiments/stage-0/task-001-vector-add/`，能解释 `index = blockIdx.x * blockDim.x + threadIdx.x` 和 host/device 数据流。
- 能读懂 32 个 lane 的线程编号（`lane = threadIdx.x`，warp size = 32，sm_75）。
- 知道 shared memory 的一个 bank 宽 4 字节、共有 32 个 bank、连续 4 字节地址映射到不同 bank。
- 上游参考实现已就位：`resources/foundations/cuda-samples/cpp/6_Performance/transpose/transpose.cu`（submodule commit `5443602d89ed99aede2e4b7bf329daddeadb320e`）。只看 `transposeNaive`、`transposeCoalesced`、`transposeNoBankConflicts` 三个函数，其余变体本 task 不要求。

## 目录内文件

- `transpose.cu`：host harness（设备信息、host 参考结果、warm-up + 重复计时、正确性、GB/s）已给出；`transpose_naive` 已给出；**需要你实现 `transpose_tiled`**。
- `bank_probe.cu`：host harness 已给出；**需要你实现 `probe_no_pad` 和 `probe_padded`**。
- `run.sh`：编译并按默认参数运行两个程序。
- `result.md`：实验记录，最后填写。

## 约定（三个 kernel 都必须遵守）

- 矩阵为方阵 `N x N`，`float`，row-major：输入 `in[y * N + x]`，输出 `out[y * N + x]`。
- 转置定义：`out[y * N + x] == in[x * N + y]`。
- 元素比较使用精确 `==`。harness 填入的数据全部是 0.5 的整数倍，**不要**往数据里放 NaN，否则 `==` 失效。
- `TILE = 32`，block 为 `dim3(32, 16)`（512 线程），grid 为 `dim3(N / 32, N / 32)`。三个变体使用完全相同的 block/grid，`N` 必须是 32 的倍数。

## 步骤

### 第 1 步：先算，不写代码

打开 `transpose.cu`，在纸上或 `result.md` 里先完成三张手算表，再看 `transpose_naive` 的代码验证你的理解：

1. 一个 warp（32 个 lane，`threadIdx.x = 0..31`，`threadIdx.y` 固定）读 `in` 时，32 个地址的间隔是多少字节？跨了几个 32 字节 sector？写 `out` 时的间隔又是多少字节？
2. `transpose_naive` 中每个 block 需要处理多少元素？`for (int i = 0; i < TILE; i += BLOCK_ROWS)` 循环几次？每个线程处理几个元素？
3. 如果把读和写都改成合并访问，可能吗？在 `result.md` 里给出你的判断和理由。

### 第 2 步：实现 `transpose_tiled`

要求（不要直接抄上游 `transposeCoalesced`，自己写，写完再和上游对照）：

- 用一块 `32 x 32` 的 shared memory tile 中转。shared memory 用**动态共享内存**（`extern __shared__`，行距由 `pitch` 参数决定），因为本 task 要用同一个 kernel 跑 `pitch = 32` 和 `pitch = 33` 两种布局。
- 阶段 1：block `(bx, by)` 把输入 tile（行 `[by*32, by*32+32)`，列 `[bx*32, bx*32+32)`）搬进 shared memory，要求**全局读合并**。
- 阶段 2：`__syncthreads()` 之后，把转置后的 tile 写回输出 tile（输出 tile 的行区间来自 `bx`、列区间来自 `by`），要求**全局写合并**。
- 两个阶段之间必须有同步；循环复用同一块 shared memory 时，阶段 2 结束到下一轮阶段 1 开始之间也要同步。
- 行 `r`、列 `c` 的 shared memory 地址必须是 `r * pitch + c`，其中 `pitch` 就是参数 `pitch`，不能写死 32。
- 边界：`N` 是 32 的倍数，所以 tile 恰好填满，`result.md` 里仍要说明为什么这里不需要 `if (x < N && y < N)`，以及 `N` 不是 32 的倍数时会怎样。
- 允许自己加 `__restrict__`、`const` 修饰，但只能一次改一项并记录。

harness 会用 `pitch = 32` 和 `pitch = 33` 各启动一次这个 kernel，分别打印为 `tiled_pitch32` 和 `tiled_pitch33`。

### 第 3 步：手算 bank conflict

对 `transpose_tiled` 的阶段 2（从 shared memory 读、往 global 写）逐 lane 列出：

- lane 的 `threadIdx.x`（记为 `tx`）和 `threadIdx.y`（记为 `ty`）各是多少；
- 它读的 shared memory 地址（4 字节单位）；
- 它的 bank 号 = `地址 % 32`；
- 在 `pitch = 32` 下，32 个 lane 的 bank 号是否相同？如果相同，是几路 conflict？同一地址的线程算不算 conflict（不算，是广播）？
- 在 `pitch = 33` 下重算一遍。

把这两张表完整写进 `result.md`。这是本 task 的核心交付物之一：必须能用自己的话说明“为什么列距取 32 会让整 warp 撞在同一个 bank”。

### 第 4 步：跑计时

```bash
./transpose 2048
./transpose 4096
./transpose 8192
```

至少记录三个尺寸 × 三个变体（`naive`、`tiled_pitch32`、`tiled_pitch33`）的 min / median / mean 时间和有效带宽。有效带宽 = `2 * N * N * sizeof(float) / 时间`，即读加写。三个变体里 `tiled_pitch32` 与 `tiled_pitch33` 的差值就是 bank conflict 的计时证据；如果差距很小（例如不到 5%），不要编造结论，把它如实记录，并解释可能的原因（DRAM 带宽压力、`N` 偏小、编译把 `pitch` 常量化、时钟波动等）。

补充事实：`T4` 的 L2 是 4 MiB，2048² 的矩阵是 16 MiB，已经超过 L2，三个尺寸都在 DRAM 带宽区间内。harness 会打印设备属性和按 `2 * memoryClockRate * memoryBusWidth / 8` 算出的理论峰值带宽，抄进 `result.md`，并用“有效带宽 / 峰值”给出三个变体离硬件上限还有多远。

### 第 5 步：隔离出 bank conflict 的代价

transpose 是带宽受限的，shared memory 冲突的额外开销有多少能反映在 kernel 时间上，取决于带宽压力和冲突程度。`bank_probe.cu` 用一个只碰 shared memory 的循环把冲突单独放大：

- 读入一个 `32 x 32` 的 tile 到 shared memory，同步一次；
- 之后循环 `iterations` 次，把一个 shared memory 元素累加到寄存器 `acc`；
- 把 `acc` 写到 `out[global_tid]`，防止被优化掉。`out` 有 `blocks * 512` 个元素，每个线程必须写到互不相同的一个下标上（`blockDim.x == 32`，用 `threadIdx.y * 32 + threadIdx.x` 之类的方式展平线程，写错会撞下标）；
- `probe_no_pad` 和 `probe_padded` 的**唯一差别**必须是 shared memory 的布局和索引方式，其他一切（循环次数、线程数、grid、global 访存）完全一致。

验收条件（两条都要在代码里成立）：

1. `probe_no_pad` 中，同一 warp 的 32 个 lane 读**同一个 bank 的 32 个不同地址**（32-way conflict）；`probe_padded` 中，同一 warp 的 32 个 lane 读**32 个不同 bank**。
2. 索引必须随循环变量变化，否则编译器会把 load 提到循环外，你测到的就不是 shared memory 访问。

`acc` 的期望值由 harness 校验：tile 中每个元素都是 1.0f，所以每个线程的 `acc` 应等于 `iterations`，一个 block 内 512 个线程的和应等于 `512 * iterations`，harness 汇总所有 block 后与 `blocks * 512 * iterations` 比较。如果你看到 `FAIL` 且时间短得离谱（例如 1 μs 级别），先怀疑 load 被优化掉了。

```bash
./bank_probe
```

记录两个 kernel 的中位时间和比值。比值小于 2 时不要下结论，如实记录并检查条件 1 是否真的成立。

### 第 6 步：交叉验证与记录

- 用 `nsys` 记录一次 `./transpose 4096`，把 kernel 列表和每行的时间填进 `result.md`，用于核对 harness 的 event 计时：

  ```bash
  nsys profile -o transpose_trace ./transpose 4096
  nsys stats transpose_trace.nsys-rep
  ```

- 尝试一次 `ncu` 并把失败信息原样记录：

  ```bash
  ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum ./transpose 4096
  ```

  这三个计数器分别对应 shared conflict、global load sector、global store sector。如果仍然报 `ERR_NVGPUCTRPERM`，把错误原文写进 `result.md`，并在“未能验证”一节列出：如果 ncu 可用，你要用哪三个指标确认第 3、4 步的推断、各指标的期望数量级是多少。**不要**填入猜测的数值。

- 把命令、输出、环境、改动、表格和结论写入 `result.md`。

## 交付物

1. `transpose.cu` 中可用的 `transpose_tiled` 实现（两个 pitch 都正确）。
2. `bank_probe.cu` 中可用的 `probe_no_pad` 和 `probe_padded`。
3. `result.md` 第 1、3 步的两组手算表（lane -> 地址 -> bank）。
4. 三个尺寸 × 三个变体的计时表和 `bank_probe` 的时间比值。
5. `nsys` 输出一次；`ncu` 成功则给出三个计数器，失败则给出错误原文和“需要哪三个指标、期望数量级”的说明。
6. 一段你自己的话解释：naive 转置慢在哪一步，共享内存分块把慢的那一步移到了哪里，padding 解决的是什么问题。

## 验收标准

- 三个变体在 2048 / 4096 / 8192 上都输出 `PASS`，且 `FAIL` 时打印了首个不匹配位置。
- 手算表与实际实现一致：`pitch = 32` 阶段 2 是 32-way conflict，`pitch = 33` 是 0 路。
- 能说清 naive 版本的哪一次 global 访存不合并，间隔多少字节。
- 计时表包含 warm-up 策略、重复次数、min/median/mean，且带宽用 `2 * N * N * 4` 字节计算。
- `bank_probe` 两版中位时间比值有具体数字，并解释这个比值为什么能代表 bank conflict 的代价。
- 明确区分“计时测到的”“手算推出来的”“需要 ncu 才能确认的”三类结论。

## 提示边界

- 先做第 1、3 步的手算，再写 kernel。手算错了，代码能跑通但结论是错的。
- 上游 `transpose.cu` 的 `transposeCoalesced` / `transposeNoBankConflicts` 只在你自己的版本通过正确性之后用来对照，不要直接复制替换；本 task 的索引约定（`out[y * N + x] == in[x * N + y]`）和它的写法不完全一样，直接抄会得到一个能跑但方向相反的结果。
- shared memory 的地址和 bank 号都用 4 字节为单位算，不要混用字节。
- `bank_probe` 的两条验收条件如果只满足一条，测出来的时间没有意义；此时应该回到索引表达式，而不是调 `iterations`。
- 遇到编译错误、运行时错误或 `nsys`/`ncu` 报错，提交完整原文，不要省略。
- 完成后只提一个你下一步想改的变量，不要一次列五个。
