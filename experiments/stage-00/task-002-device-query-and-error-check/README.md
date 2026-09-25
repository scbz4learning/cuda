# Task 002：设备查询与错误检查

## 本 task 只引入

- `cudaGetDeviceProperties` 的查询与解释
- `CUDA_CHECK` 错误检查宏，以及“错误在什么位置被报出”

**不引入**：计时（属于 `task-004-timing-protocol`）、2D/3D 索引、shared memory、bank、性能分析。遇到需要这些才能回答的问题，写下来，不要在本 task 里顺手做掉。

## 前置条件

- 已完成 [`../task-001-vector-add/`](../task-001-vector-add/)，能解释 `index = blockIdx.x * blockDim.x + threadIdx.x` 和 host/device 数据流。
- 能编译并运行 `task-001` 里的 `vector_add.cu`。
- 知道 `nvidia-smi` 与 `cudaGetDeviceProperties` 都能报告设备信息，但后者能拿到驱动不暴露的字段。

## 目录内文件

- `device_query.cu`：host 流程已给出；**需要你实现 `print_device_info`**。
- `error_lab.cu`：六个故意写错的 case 已给出，**不需要你修改任何 kernel**，你只负责运行、预测和解释。
- `run.sh`：编译并按进程隔离运行上面六个 case。
- `result.md`：实验记录，最后填写。

## 步骤

### 第 1 步：先预测，再运行

**在运行任何东西之前**，打开 `result.md` 的预测表，对六个 case 各写三件事：

1. 错误会在**哪个调用**上被报出（启动语句后的 `cudaGetLastError()`、`cudaDeviceSynchronize()`、还是那一个 host API 调用本身）；
2. 错误属于哪一类（启动配置、内存越界、地址不对齐、host API 参数、显存不足）；
3. 这个错误**会不会让后续调用继续失败**（下面把这种性质叫做“粘性”）。

预测必须写在纸上或 `result.md` 里，然后再运行。第 6 步要拿预测和实测对照，预测错了比预测对了更有价值，但**不要为了改答案去改已经写下的预测**——在旁边标注“预测错，实测是 X”。

### 第 2 步：实现 `print_device_info`

`device_query.cu` 里 `print_device_info` 的函数体是空的。需要查询的字段列表写在文件里的 TODO 注释中（也可以直接在 `/usr/local/cuda/include/driver_types.h` 里搜 `struct cudaDeviceProp` 自己找）。

要求：

- 逐行打印，一个字段一行，字段名和值都要能看清。
- 派生量一并打印：compute capability 字符串（如 `sm_75`）、各类容量的 KiB / MiB / GiB 换算、峰值显存带宽。
- 不要直接打印整个结构体。

### 第 3 步：推导峰值带宽公式

峰值显存带宽只能由 `cudaDeviceProp` 里的某两个字段算出来。你需要：

1. 写清用的是哪两个字段，以及每个字段的单位和含义（`memoryClockRate` 的单位容易看错）。
2. 写出完整的算式，注意 `memoryBusWidth` 的单位是**位**不是字节。
3. **判断需不需要乘一个系数**（提示：GDDR 内存每个时钟传输两次数据）。把你的判断和理由写下来。
4. 用一个**外部可查的规格数字**校验你的公式正确——查这台卡（Tesla T4）官方标称的显存带宽，把你算出来的值和它对比。

如果第 3 步判断错了，第 4 步的对比会直接暴露出来：你会算出规格数字的一半或两倍。

### 第 4 步：运行设备查询

```bash
./device_query 0
./device_query 1
```

两台设备都跑一次。如果某个字段在两台卡上不同，说出哪些字段会因卡型而异、哪些不会。

### 第 5 步：运行错误 lab

```bash
./run.sh
```

或者单独跑一个 case：

```bash
./error_lab illegal
```

六个 case 的含义（`error_lab.cu` 里有对应代码，先读代码再跑）：

| case | 做法 |
| --- | --- |
| `launchcfg` | 依次用非法的 block / grid 维度启动 |
| `dynsmem` | 申请 64 KB 动态共享内存，超过 48 KB 静态上限，然后尝试放开上限再启动一次 |
| `baddevice` | 对一个不存在的 device 编号调用 `cudaSetDevice` |
| `hugealloc` | 先小额分配成功，再申请 `1 << 62` 字节 |
| `illegal` | 只分配 1 KB，然后写到 512 MB 之外 |
| `misalign` | 把一个 4 字节对齐的地址当作 `float4` 读取 |

把实测输出原样抄进 `result.md`，然后与第 1 步的预测逐行对照。

### 第 6 步：回答四个问题

用自己的话回答，每条都要引用 `result.md` 里的具体某一行输出作为证据：

1. 同一个 `cudaGetLastError()` 紧跟在启动语句后面，为什么能抓到 `launchcfg` 和 `dynsmem` 的错误，却抓不到 `illegal` 和 `misalign` 的错误？把“启动配置检查”和“kernel 真正执行”分成两个时刻来解释。
2. `illegal` 发生之后，同一个进程里哪些调用继续返回错误、哪些仍然正常？你在输出里至少要指出两条继续失败的调用和一条仍然正常的调用，并解释差别。
3. `cudaDeviceReset()` 返回 `no error`，但这个进程之后再也拿不到这块卡。既然 reset 没用，工程上正确的做法是什么？这就是 `run.sh` 为什么按进程隔离调用每个 case 的原因。
4. `hugealloc` 的 `out of memory` 不具有粘性（后面的 `cudaMalloc(4096)` 成功），而 `illegal` 具有粘性。解释为什么这两种失败的性质不同。

再补一条：

5. `dynsmem` 的输出里，`cudaGetLastError()` 之后紧跟的 `cudaPeekAtLastError()` 返回了 `no error`，而不是重复报上一次的 `invalid argument`。解释这两个函数的差别，以及为什么 host API 的错误在 `baddevice` 那个 case 里表现得不一样。

## 交付物

1. `device_query.cu` 中可用的 `print_device_info`，两台设备都能跑。
2. `result.md` 第 1 步的预测表（六个 case × 三列）。
3. `run.sh` 的完整输出，含六个 case。
4. 峰值带宽的推导过程：两个字段、完整算式、是否乘系数的判断、以及与官方规格数字的对比。
5. 第 6 步五个问题的回答，每条引用具体输出行。
6. 一段你自己的话说明：`CUDA_CHECK` 应该放在哪些位置，为什么 kernel 启动之后光检查启动语句不够。

## 验收标准

- `print_device_info` 输出的字段覆盖 TODO 注释里列出的全部字段。
- 峰值带宽算式正确，且你判断的“是否乘系数”与第 3 步的判断一致（不要求与参考答案一致，但必须与第 4 步的规格对比自洽）。
- 六个 case 的实测输出完整，且与第 1 步的预测做了逐行对照，错处有标注。
- 第 6 步的五个问题都给出了引用具体输出行的答案。
- 能说清“启动失败”和“运行中出错”在报错时机上的区别。
- 明确区分“实测到的”和“推断的”：哪几条是你在输出里直接看到的，哪几条是你根据 CUDA 的执行模型推断的。

## 提示边界

- `error_lab.cu` 里的 kernel 是实验材料，**不要修**。它们坏了才有观察价值。
- 不要为了让输出好看而给 `error_lab` 加 `CUDA_CHECK` 提前退出——那样你就看不到后面的行，也就回答不了第 6 步的问题。
- 越界写的偏移量必须远大于 2 MB。如果你把偏移量改小到几 KB 之内，写入可能落在同一映射页内而**不报错**；这个现象本身值得记进 `result.md`。
- 遇到编译错误、运行时错误或 `nvcc`/`run.sh` 报错，提交完整原文，不要省略。
- 不要在本 task 里加计时逻辑。计时是 `task-004` 的内容。
- 完成后只提一个你下一步想改的变量，不要一次列五个。
