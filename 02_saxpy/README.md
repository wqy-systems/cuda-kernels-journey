# 02 · SAXPY —— 自己写，不许抄

`y = a * x + y`，一个 float 标量 + 两个向量。
它比 vector add 只多一个标量乘法，但它是 BLAS Level-1 的原型，也是后面所有 kernel 的模板。

## 任务

打开 `saxpy.cu`，把里面 4 个 `TODO` 填掉：

1. `saxpyKernel` 的循环体
2. host 侧的内存分配
3. kernel 启动配置（block / grid 自己定，并在注释里写清为什么这么选）
4. 结果校验

## 验收标准

- 和 CPU 参考实现的最大绝对误差 < 1e-4（float 有精度损失，不是 0 才叫对）
- 实测带宽达到理论峰值的 **70% 以上**
- 用 ncu 跑一遍，能回答这两个问题：
  - memory throughput 是多少？离峰值差多少？
  - 如果只有 40%，最主要的 stall 原因是什么？

## 提示（卡住了再看）

- 每个元素读 8 bytes（x 和 y 各 4），写 4 bytes → 总算力需求 12 bytes/element
- 试试 BLOCK_SIZE = 256、GRID = SM数量 × 足够多的倍数，看带宽怎么变
- `__restrict__` 告诉编译器两个指针不重叠，它才敢做更激进的优化
- 别用 `cudaMalloc` 分配 host 内存，那是错误的用法（不会报错，但会慢）

## 完成后

在 `../README.md` 的 W07 那行记下你的实测带宽。
