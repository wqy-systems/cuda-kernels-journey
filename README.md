# cuda-kernels-journey

从零学 CUDA 算子的实验记录。每个目录一个 kernel，每个 kernel 都有实测数据和失败记录。

目标岗位：AI 系统 / 算子工程师（CUDA 算子、NCCL/RDMA 通信、MLIR/Triton 编译栈）。
计划：96 周，2028 年 9 月秋招。

## 当前进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| S0 地基 | Linux、C++、计算机组成、线性代数、工具链 | 完成 |
| S1 CUDA 入门 | 六个基础 kernel + ncu 瓶颈判定 | 完成 |
| S2 性能优化 | SGEMM 九版本，10311 GFLOP/s（峰值 52%） | 完成 |
| S3 算子库 | CUTLASS / FlashAttention / Triton | 进行中 |

## 成果

**SGEMM 优化：从算力峰值 6% 到 52%**

矩阵 2048×2048，单精度，RTX 4070 Laptop（36 SM，8GB，256 GB/s）：

| 版本 | GFLOP/s | 相对分块版 | 手法 |
|---|---|---|---|
| v2_tiled16 | 1593.7 | 1.00x | 共享内存分块，一输出一线程 |
| v3_tt4 | 7729.2 | 4.85x | 4×4 线程分块 |
| v4_tt4v | 8085.4 | 5.07x | + float4 访存 |
| v5_tt4v_lb4 | 8692.9 | 5.45x | + 占用率 67% |
| v6_tt8v | 7854.8 | 4.93x | 8×8（失败） |
| **v7_db** | **10311.7** | **6.47x** | + 双缓冲 |
| v8_db_lb4 | 9636.5 | 6.05x | 双缓冲 + 强制 4 块 |

完整数据和每条结论见 [`24_gemm_tt/GEMM_final_table.md`](24_gemm_tt/GEMM_final_table.md)。

**文章**

- 知乎：[把 SGEMM 从 6% 跑到 52%：九个版本的实验记录](https://zhuanlan.zhihu.com/p/2108300534721557012)
- 掘金：[同名文章](https://juejin.cn/post/7711248788611121161)

## 目录

### 基础 kernel

| 目录 | 内容 | 最好成绩 |
|---|---|---|
| `01_vector_add` | 向量加法，理解 grid-stride loop 与合并访问 | 240 GB/s（94% 峰值） |
| `02_saxpy` | 原地修改、const、可信的测量方法 | 240 GB/s |
| `03_transpose` | 共享内存分块 + `tile[32][33]` 消除 bank conflict | 243 GB/s，1.8–2.1x |
| `04_reduce` | 树形归约、warp shuffle、原子操作与浮点精度 | 262.7 GB/s |
| `22_conv1d` | 一维卷积，验证 L1 缓存兜住邻域复用 | 226 GB/s（理想流量 88%） |
| `23_softmax` | 三趟扫描的 softmax，online 最大值的起点 | 228 GB/s |

### GEMM 优化路径

| 目录 | 内容 |
|---|---|
| `05_gemm` | 朴素版，算术强度 0.25，撞上内存墙 |
| `06_gemm_tiled` | 共享内存分块 TILE=16，只涨 1.3 倍 |
| `07_gemm_unroll` | 四路累加，证明瓶颈不在加法依赖链 |
| `08_gemm_wmma` | Tensor Core，一条指令 4096 次乘加，5.63x |
| `24_gemm_tt` | 线程分块到双缓冲的完整擂台，九版本对比 |

### 自己独立写的 kernel

`09_my_first_kernel`、`10_my_saxpy`、`12_my_reduce` —— 这三个是遮住答案自己敲的，不是抄的。

### C++ 与工具链

| 目录 | 内容 |
|---|---|
| `00_cpp_warmup` | C++ 最小集：指针、数组、函数、const、循环 |
| `11_gdb_practice` | gdb 调试两个崩溃程序 |
| `13_pointer_gdb` | 指针与内存布局 |
| `14_const_ref` | const 与引用的组合 |
| `15_my_class` | 类的构造、析构、成员函数 |
| `16_my_template` | 函数模板与类模板 |
| `17_cache` | 亲手测出 cache line 大小与组相联 |
| `18_pipeline` | 亲手测出流水线与分支预测的 4 倍差 |
| `19_raii` | RAII：设备内存封装、异常安全、移动语义、显存池 |
| `20_stl` | vector 扩容、指针失效、reserve、benchmark 工具 |
| `21_compile` | 编译链接：`.i` / `.s` / `.o`、符号表、静态库与动态库 |

## 环境

| 项 | 值 |
|---|---|
| GPU | RTX 4070 Laptop，CC 8.9，36 SM，8GB |
| 主开发 | WSL2 · Ubuntu 22.04 · CUDA 12.6 |
| 真机 | Ubuntu 26.04.1 · CUDA 13.5.37 · ncu 2026.4.0.0 |
| 编译 | nvcc + CMake + Ninja |

ncu 只能在真机跑，WSL2 报 `ERR_NVGPUCTRPERM`。

## 怎么跑

```bash
# 全部编译
make

# 单个
make 01_vector_add/vector_add
./01_vector_add/vector_add

# CMake 构建
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j

# GEMM 擂台（九版本对比）
nvcc -O3 -Iinclude -Xptxas -v -o 24_gemm_tt/gemm_bench 24_gemm_tt/gemm_bench.cu
./24_gemm_tt/gemm_bench

# cp.async 版本必须指定架构
nvcc -O3 -arch=sm_89 -Iinclude -o 24_gemm_tt/gemm_cpasync 24_gemm_tt/gemm_cpasync.cu
```

## 三条带得走的结论

**一，性能账按「显存被碰了几次」记，不按「代码里读了几次」记。** 一维卷积纸面算出 1130 GB/s，超过峰值四倍，原因是那些重复读全被 L1 接住了。

**二，双缓冲省的是 `__syncthreads()` 的停顿，不是带宽。** DRAM 吞吐只占 2.76% 的时候，双缓冲仍然涨了 18.6%。

**三，编译参数的影响大于后期优化。** 同一个 kernel，加 `-arch=sm_89` 后慢 8%，而 v4 之后的每一步优化收益只有 3%–19%。

## 还没做的

- `04_reduce` 缺 unroll 与 warp shuffle 两版
- compute-sanitizer 查内存越界
- S3 的 CUTLASS / FlashAttention
