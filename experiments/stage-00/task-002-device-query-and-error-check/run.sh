#!/usr/bin/env bash
set -euo pipefail

# -arch=native 让 nvcc 按本机 GPU 生成代码；如报 unknown option，去掉它并加
# -Wno-deprecated-gpu-targets。
ARCH_FLAGS=(-O3 -lineinfo -arch=native)

nvcc "${ARCH_FLAGS[@]}" device_query.cu -o device_query
nvcc "${ARCH_FLAGS[@]}" error_lab.cu -o error_lab

echo "== device query (device 0) =="
./device_query 0
echo

# 每个 case 必须是独立的进程：illegal 和 misalign 会让本进程的 CUDA
# 上下文进入不可用状态，后续 case 全都会失败。
for case in launchcfg dynsmem baddevice hugealloc illegal misalign; do
    echo "== error_lab ${case} =="
    ./error_lab "${case}"
    echo
done
