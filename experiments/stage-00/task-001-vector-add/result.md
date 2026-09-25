# Task 001 实验记录：运行第一个 CUDA kernel

- 日期：2026-09-25
- 阶段：stage-00
- 代码位置：`experiments/stage-00/task-001-vector-add/vector_add.cu`
- 上游 commit / 参考链接：无；本 task 为仓库内实验

## 环境

- GPU：Tesla T4（`nvidia-smi` 列出 2 张可见 GPU；本次运行使用的 device index 未单独记录）
- Compute Capability：7.5
- Driver：580.159.04
- CUDA Toolkit：12.8
- 编译器：nvcc V12.8.93
- Nsight Systems：2024.6.2

## 输入与正确性

- 初始输入：`count = 1 << 20` 个 `float`；`left` 全为 1，`right` 全为 2，预期输出全为 3。
- 基线配置：`block_size = 256`，保留 `index < count` 判断。
- 编译命令：`nvcc -O3 -lineinfo vector_add.cu -o vector_add`
- 基线运行：`./vector_add`，输出 `vector_add: PASS`。
- 故意触发错误：将 `block_size` 改为 192，并注释掉 `if (index < count)`；运行报告 `an illegal memory access was encountered`。
- 小规模演示：改为 `count = 14`、`block_size = 4`，恢复边界判断；程序启动 4 个 block、共 16 个线程，输出 `vector_add: PASS`。

## Host / Device 数据流

1. `host_left`、`host_right` 和 `host_output` 是由 `std::vector` 管理的 host 内存。输入向量在 host 初始化为 1 和 2。
2. `cudaMalloc` 为输入、输出和线程映射记录分配 device 内存。`device_output` 无需从 host 预先复制或清零：kernel 对每个有效 `index` 都会写入一次 `output[index]`。
3. 两次 `cudaMemcpyHostToDevice` 将输入数据从 host 复制到 device。kernel 在 device 上读取 `device_left[index]`、`device_right[index]` 并写入 `device_output[index]`。
4. kernel 完成后，`cudaMemcpyDeviceToHost` 将结果复制到 `host_output`，再由 host 遍历检查每个值是否为 3。
5. 教学用的 `device_thread_indices` 和 `device_thread_is_valid` 也由 kernel 写入，之后分别复制回 host，用于打印每个线程的索引和有效性；它们是观察信息，不是 vector-add 正确性检查本身。
6. 最后通过 `cudaFree` 释放 device 分配。

## 线程映射与错误分析

线程用下面的公式计算元素索引：

```text
index = blockIdx.x * blockDim.x + threadIdx.x
```

小规模演示中的映射：

| `blockIdx.x` | 线程产生的 `index` | 结果 |
| --- | --- | --- |
| 0 | 0, 1, 2, 3 | 有效 |
| 1 | 4, 5, 6, 7 | 有效 |
| 2 | 8, 9, 10, 11 | 有效 |
| 3 | 12, 13, 14, 15 | 12、13 有效；14、15 越界并跳过 |

数组合法索引为 `0..13`。启动的最后一个 block 仍有 4 个线程，其中两个计算出超出数组范围的索引；边界判断使它们跳过内存访问。

在故意触发错误的配置中，`1,048,576 = 192 * 5,461 + 64`。因此需要 5,462 个 block，共启动 1,048,704 个线程；最后一个 block 有 64 个有效线程和 128 个越界线程。由于边界判断被注释，这些线程尝试越界读写。实验同时改了 block 大小和边界判断，故错误不能归因于 `block_size = 192` 单独一项；根因是越界线程没有被保护。

## 测量与 profiler

### Nsight Systems

基线 trace 命令：

```bash
nsys profile -o vector_add_trace ./vector_add
nsys stats vector_add_trace.nsys-rep
```

`vector_add` kernel 记录到 1 次，时间为 **46,272 ns（约 46.3 μs）**。这是 profiler 下的一次观测，不是重复计时或性能基准。CUDA API 汇总包含分配、拷贝等 host API 开销，不等同于 kernel 时间。

### Nsight Compute

尝试命令：`ncu --set basic ./vector_add`。程序输出 `vector_add: PASS`，但 profiler 报告 `ERR_NVGPUCTRPERM`，当前用户没有访问 GPU performance counters 的权限，因此没有采集到 NCU 指标。另有 `No module named 'wrapt'` 的 Python 环境警告；它不是计数器权限错误本身。没有填写或推测 DRAM throughput、occupancy 等指标。

## 结论

- 正确性：带边界判断的基线和 `count=14` 教学演示均通过；移除边界判断且 block 大小不能整除 count 时发生非法内存访问。
- 原因：grid 按完整 block 向上取整，额外线程仍会运行；必须用 `index < count` 阻止它们访问数组范围之外的地址。
- 性能：只得到一次 Nsight Systems kernel 时间，未做重复计时；Nsight Compute 硬件计数器受权限限制，不能据此判断 occupancy、带宽或优化效果。
- 下一步：如要比较 block 大小，保持边界判断不变，只改 block 大小，并采用 warm-up 与重复计时。
