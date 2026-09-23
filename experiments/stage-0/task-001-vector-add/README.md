# Task 001：运行第一个 CUDA kernel

## 目标

建立最小 CUDA 实验闭环：能编译 host/device 混合程序，能解释线程索引和边界判断，能用输出确认结果，并记录第一次 profiler 观察。

## 前置条件

- 已安装 CUDA Toolkit，或能使用带 `nvcc` 的 CUDA 环境。
- 已阅读根目录 README、ROADMAP 和 `resources/foundations/README.md`。

## 步骤

1. 进入本目录，阅读 `vector_add.cu`，先画出 `grid_size`、`block_size` 和 `index` 的关系。
2. 编译并运行：

   ```bash
   nvcc -O3 -lineinfo vector_add.cu -o vector_add
   ./vector_add
   ```

3. 确认输出包含 `vector_add: PASS`。
4. 只修改一个参数或实现细节，例如将 `block_size` 改为 128 或 512，重新编译并确认结果仍正确。
5. 有 `ncu` 时运行 `ncu --set basic ./vector_add`；没有时说明工具缺失，不要伪造指标。
6. 将命令、输出、环境、改动和解释填写到 `result.md`。

## 交付物

- `vector_add.cu` 的一次可解释修改，或说明为什么保持原实现。
- 两次运行的正确性输出。
- 填写完整的 `result.md`。

## 验收标准

- 能解释 host 到 device 的数据流、线程索引和 `index < count` 的必要性。
- 原始版本和修改版本都通过正确性检查。
- 能指出当前实验还没有回答的性能问题。

## 提示边界

先自己定位代码和运行命令；遇到编译错误时提交完整错误输出。不要直接复制一个更复杂的 vector add 实现来替换本任务。
