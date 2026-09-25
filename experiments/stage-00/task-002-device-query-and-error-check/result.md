# 实验记录：设备查询与错误检查

- 日期：
- 阶段：stage-00 / task-002
- 代码位置：`experiments/stage-00/task-002-device-query-and-error-check/device_query.cu`、`error_lab.cu`
- 上游 commit / 参考链接：无；本 task 为仓库内实验。设备字段定义见 `/usr/local/cuda/include/driver_types.h` 的 `struct cudaDeviceProp`

## 环境

- GPU（型号 + device index）：
- Compute Capability：
- Driver：
- CUDA Toolkit：
- 编译器：`nvcc --version`
- 编译命令：`./run.sh` 中的 `ARCH_FLAGS`，实际生效的架构：
- 可见设备数量：

## 第 1 步：预测表（在运行之前填写）

| case | 错误在哪个调用上报出 | 错误类别 | 是否粘性 |
| --- | --- | --- | --- |
| `launchcfg` |  |  |  |
| `dynsmem` |  |  |  |
| `baddevice` |  |  |  |
| `hugealloc` |  |  |  |
| `illegal` |  |  |  |
| `misalign` |  |  |  |

预测与实测不符的地方，在下表右侧单独标注，不要回头改预测。

## 第 2、3 步：设备查询与峰值带宽推导

- 实现的字段清单（对照 TODO 注释逐条确认是否都打了）：

| 字段 | device 0 的值 | device 1 的值 | 这个字段会不会因卡型而异 |
| --- | --- | --- | --- |
| `name` |  |  |  |
| `major` / `minor` |  |  |  |
| `multiProcessorCount` |  |  |  |
| `warpSize` |  |  |  |
| `l2CacheSize` |  |  |  |
| `sharedMemPerBlock` |  |  |  |
| `sharedMemPerMultiprocessor` |  |  |  |
| `regsPerBlock` |  |  |  |
| `regsPerMultiprocessor` |  |  |  |
| `maxThreadsPerBlock` |  |  |  |
| `maxThreadsPerMultiProcessor` |  |  |  |
| `memoryClockRate` |  |  |  |
| `memoryBusWidth` |  |  |  |
| `totalGlobalMem` |  |  |  |
| `clockRate` |  |  |  |
| `asyncEngineCount` |  |  |  |
| `deviceOverlap` |  |  |  |

- compute capability 字符串（`sm_XX`）是怎么拼出来的：
- 峰值带宽用了哪两个字段：
- 两个字段各自的单位（`memoryBusWidth` 是位还是字节？`memoryClockRate` 是 kHz 还是 MHz？）：
- 完整算式：

  ```text
  （自己写）
  ```

- 是否需要乘一个系数（GDDR 每个时钟传两次数据）：判断是 / 否
- 判断理由：
- 官方标称显存带宽（写明来源，例如 NVIDIA T4 数据表）：
- 算出来的值与标称值的差异（百分比）：
- 差异的解释：

### `./device_query` 输出

```text
（粘贴 ./device_query 0 和 ./device_query 1 的完整输出）
```

- 两台设备上不同的字段：
- 两台设备上相同的字段：

## 第 5 步：错误 lab 实测输出

```text
（粘贴 ./run.sh 的完整输出，含六个 case）
```

### 逐 case 对照

| case | 预测的报错调用 | 实测的报错调用 | 一致？ | 预测的类别 | 实测的错误字符串 | 预测粘性 | 实测粘性 |
| --- | --- | --- | :---: | --- | --- | --- | --- |
| `launchcfg` |  |  |  |  |  |  |  |
| `dynsmem` |  |  | :---: |  |  |  |  |
| `baddevice` |  |  | :---: |  |  |  |  |
| `hugealloc` |  |  | :---: |  |  |  |  |
| `illegal` |  |  | :---: |  |  |  |  |
| `misalign` |  |  | :---: |  |  |  |  |

- 预测错了的 case，以及你当时为什么这么想：
- 把越界偏移量改小到几 KB 之内时的现象（写入落在同一映射页内而不报错）：

## 第 6 步：解释

每条都要引用上面输出里的具体某一行。

1. 为什么 `cudaGetLastError()` 紧跟启动语句能抓到 `launchcfg` / `dynsmem`，却抓不到 `illegal` / `misalign`？把“启动配置检查”和“kernel 真正执行”分成两个时刻解释。证据行：
2. `illegal` 之后，哪些调用继续返回错误、哪些仍然正常？各举一条并解释差别。证据行：
3. `cudaDeviceReset()` 返回 `no error`，但进程之后拿不到这块卡。既然 reset 没用，工程上正确做法是什么？证据行：
4. 为什么 `hugealloc` 的 `out of memory` 不粘性，而 `illegal` 粘性？证据行：
5. `dynsmem` 里 `cudaGetLastError()` 之后的 `cudaPeekAtLastError()` 返回 `no error`，而 `baddevice` 里 host API 错误表现得不同。`cudaGetLastError()` 与 `cudaPeekAtLastError()` 的差别是什么？证据行：

## 结论

- 实测到的（直接来自输出）：
- 推断的（依据 CUDA 执行模型，输出里没有直接证据）：
- 需要 `compute-sanitizer` 或其他工具才能确认的：
- `CUDA_CHECK` 应该放在哪些位置，为什么启动语句之后光检查启动不够：
- 下一次只改：

## 附

- 你在推导峰值带宽时踩到的坑（单位、系数、字段含义）：
- 两个 case 之间你原本以为会一样、实测却不同的现象：
