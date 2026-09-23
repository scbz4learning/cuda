# 阶段 0：第一次 CUDA 实验

## vector add

```bash
nvcc -O3 -lineinfo vector_add.cu -o vector_add
./vector_add
ncu --set basic ./vector_add
```

预期输出包含 `PASS`。第一次 profile 只观察，不急着优化。把结果写入上级模板。
