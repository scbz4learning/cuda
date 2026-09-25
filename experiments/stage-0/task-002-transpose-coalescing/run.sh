#!/usr/bin/env bash
set -euo pipefail

# -arch=native 让 nvcc 按本机 GPU 生成代码；如报 unknown option，去掉它并加
# -Wno-deprecated-gpu-targets。
ARCH_FLAGS=(-O3 -lineinfo -arch=native)

nvcc "${ARCH_FLAGS[@]}" transpose.cu -o transpose
nvcc "${ARCH_FLAGS[@]}" bank_probe.cu -o bank_probe

echo "== transpose =="
./transpose 4096
echo
echo "== bank probe =="
./bank_probe
