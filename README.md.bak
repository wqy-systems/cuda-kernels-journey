# CUDA Kernels Journey

从零手写 GPU kernel 的学习仓库。目标岗位：AI 系统 / 算子工程师。  
起点：2026-11-16（大二上）｜终点：2028-09 秋招。

## 设备

- RTX 4070 Laptop GPU，**8GB 显存**，sm_89，driver 621.24
- 编译环境：WSL2 + Ubuntu 22.04 + CUDA 12.x
- 注意：8G 显存 → GEMM 用 4096³ 以内，FlashAttention 用 seq_len ≤ 2048 / head_dim 64

## 目录约定

```
cuda-kernels-journey/
├── Makefile            编译规则（自动识别子目录下的 .cu）
├── check_env.sh        环境自检
├── include/common.h    计时 / 校验 / 错误检查工具
├── 01_vector_add/      第 1 个 kernel（参考实现，逐行有注释）
├── 02_saxpy/           第 2 个 kernel（自己写，看同目录 README）
└── notes/              每周笔记（每周至少一条）
```

## 常用命令

**CMake（推荐，日常用这个）**

```bash
cmake -B build              # 配置：只有第一次，或改了 CMakeLists.txt 才需要
cmake --build build -j      # 编译（增量，只重编改过的文件）
./build/bin/transpose       # 运行
rm -rf build                # 环境搞乱了就清空重来
```

**Makefile（备用，单文件编译够用）**

```bash
make                            # 编译全部
make 01_vector_add/vector_add   # 只编一个
./01_vector_add/vector_add      # 运行
bash check_env.sh               # 环境自检
```

**profiling**

```bash
ncu --set full -o report ./build/bin/transpose   # 出一份完整 profile
ncu -i report.ncu-rep                            # 在界面/终端里看
nsys profile -o timeline ./build/bin/transpose   # 看时间线
```

> CMake 的 `CMAKE_CUDA_ARCHITECTURES` 设成 89（4070 Laptop = sm_89）。换机器要改这里，否则编译巨慢或跑不起来。

## 欠账清单

按优先级排，别忘了：

- [ ] **gdb 调试**：现在程序崩了只能靠 printf。等第一次段错误之前补上，半小时够。
- [ ] **Linux 基础**：进程 / 权限 / 环境变量，按需查，不用专门学。
- [ ] **内存泄漏检查**：compute-sanitizer 还没用过，写 GEMM 之前过一遍。
- [x] ~~CMake 最小工程~~（2026-11-15 补完）

## 进度（W01-W08 · S0 地基）

- [x] W01 环境打通：`nvcc -V` + `nvidia-smi` 正常，跑通 `01_vector_add`
- [x] W01 建 GitHub 仓库，push 首次提交（commit `be4770c`）
- [ ] W02 Linux 基础：进程 / 权限 / 环境变量 / gdb 断点
- [x] W02 CMake 最小工程能编译 .cu（2026-11-15，五个目标全部构建通过）
- [ ] W03 C++ 指针与内存：手写 3 个小程序（数组反转 / 字符串拷贝 / 动态二维数组）
- [ ] W04 C++ 类与模板：手写简化版 vector（含拷贝构造、析构、operator[]）
- [ ] W05 CSAPP 第 3 章：汇编、内存层次、Cache 行大小实测
- [ ] W06 CSAPP 第 5-6 章：性能优化、SIMD
- [ ] W07 写 `02_saxpy`，用 ncu 看第一个 kernel 的指标
- [ ] W08 期末周：只保持每天 30 分钟，不掉线

## 性能基线

测量方法：先空跑 2 次热身，再连续跑 10 次取最快那一次（笔记本 GPU 空闲会降频，单次测量不可信）。

| kernel              | 带宽         | 占理论峰值 | 备注                   |
| ------------------- | ---------- | ----- | -------------------- |
| 公式算的理论峰值 | 256 GB/s | 100% | **低估了，别当基准用** |
| 实测最好成绩 | **262.7 GB/s** | — | reduce v3，目前最高 |
| 01 vector add       | ~240 GB/s  | 94%   | 纯流式，已到极限             |
| 02 saxpy            | ~240 GB/s  | 94%   | 同上，无优化空间             |
| 03 transpose 朴素 | ~118 GB/s | 46% | 散写，浪费一半带宽 |
| 03 transpose 分块 | ~243 GB/s | 95% | 共享内存中转，1.8–2.1 倍提速 |
| 04 reduce 原子加 | 219 GB/s | 85.6% | 52 万次原子排队，被带宽掩盖 |
| 04 reduce 树形 | 261 GB/s | 102% | 每个 block 只原子加一次 |
| 04 reduce shuffle | 262.7 GB/s | 102.6% | 寄存器交换，免共享内存 |
| H2D 拷贝              | ~12.3 GB/s | —     | PCIe 4.0 x8 + WSL 开销 |

**结论**
- 纯 streaming 和归约类 kernel 在这块卡上 ~95% 就是天花板，再优化也就 5% 的空间。
- **理论峰值公式（2×频率×位宽÷8）算出的 256 GB/s 偏低**，实测至少 263 GB/s。判断"离极限还有多远"要用实测最好成绩当分母。
- 真正值得优化的是 GEMM——它有数据复用、能吃满算力，跟这些带宽受限的 kernel 完全不是一类问题。

**浮点精度提醒**：reduce v1（原子加）相对误差 -4.3e-12，比树形归约的 -1e-15 大 4000 倍，而且每次运行结果都不一样——原子加的顺序不确定，不可复现。浮点加法不满足结合律，后面做 GEMM 精度对齐会反复遇到。

## GEMM 实验记录（2026-11-16）

算力峰值 **20.0 TFLOPS**（36 SM × 128 核 × 2 × 2.17 GHz），平衡点 78.3 FLOP/byte。

| 版本 | 1024³ | 占算力峰值 | 提速 |
|---|---|---|---|
| 05 朴素 | 1219 GFLOP/s | 6.1% | — |
| 06 分块16 | 1864 GFLOP/s | 9.3% | 1.3x |
| 07 四路累加 | 1882 GFLOP/s | 9.4% | 1.01x（相对分块） |
| TILE=32 | 1562 GFLOP/s | 7.8% | 反而更慢，已改回 16 |
| **08 Tensor Core (wmma)** | **10486 GFLOP/s** | 占 TC 峰值约 10% | **5.63x**，校验 9.7e-4 PASS |

**四次预测全错，这是收获**：
1. 以为主性能掉到 1% 以下 → 实际 6%
2. 以为分块快 5–20 倍 → 实际 1.3 倍
3. 以为四路累加快 2–4 倍 → 实际 1.01 倍
4. 以为 TILE 加大更好 → 反而更慢

说明瓶颈不是"加法排队"也不是"共享内存读取量"。**光靠改参数已经到顶**，要再往上必须换武器：Tensor Core（专用矩阵乘硬件，算力是 FP32 的十几倍）。

**Tensor Core 那一跳**：把 CUDA 核心换成 Tensor Core（wmma API），一条 `mma_sync` 指令 = 16×16×16 = 4096 次乘加，直接 5.63 倍。
注意 `matrix_b` 必须写 `wmma::row_major`（B 在内存里按行存），写成 `col_major` 会性能正常但结果全错——**性能正常、结果错，先查布局和索引**。
现在 10.5 TFLOPS 仍只有 Tensor Core 峰值（100+ TFLOPS）的 ~10%，因为每个 warp 各自从全局读块，没有复用。下一步：多 warp 共享块 + 双缓冲。

**WSL2 限制**：`ncu --set full` 报 `ERR_NVGPUCTRPERM`，sudo 也不行——WSL2 子系统里拿不到 GPU 硬件计数器。要做 profile 得在 Windows 侧用 Nsight Compute GUI 连 WSL target，或者上真 Linux / 云主机。目前用"改参数→看数字"的实验法代替。

## 踩坑记录

- **VSCode 显示旧内容**：从 Windows 侧改写 WSL 里的文件时，WSL 的文件监听可能不触发，VSCode 会一直显示旧缓存。判断方法看文件大小/修改时间，或在 VSCode 里 `Ctrl+Shift+P` → `File: Revert File`。**缓存是旧的时候千万别按 Ctrl+S**，会把磁盘上的新内容覆盖掉。
- **单次测性能不可信**：同一份代码三次能差 15%–60%，因为笔记本 GPU 空闲时频率会降到几百 MHz。必须热身 + 多次取最快。
- **nano 会留 `.save` 备份**：编辑过的文件旁边会多个 `xxx.save`，容易看错文件，用完删掉。

## 每周记录

| 周   | 主题                | 投入(h) | 卡住的地方                                                              | 一句话收获                                                |
| --- | ----------------- | ----- | ------------------------------------------------------------------ | ---------------------------------------------------- |
| W01 | 环境搭建 + 第一个 kernel | 4     | GitHub：github.com 被 DNS 污染成 127.0.0.1、HTTPS 被 SNI 重置、SSH 公钥三次指纹不匹配 | vector add 只有 47.6% 带宽——每个线程每轮只搬 4 字节，并行粒度不够粗，不是 bug |
| W02 |                   |       |                                                                    |                                                      |
| W03 |                   |       |                                                                    |                                                      |
| W04 |                   |       |                                                                    |                                                      |
| W05 |                   |       |                                                                    |                                                      |
| W06 |                   |       |                                                                    |                                                      |
| W07 |                   |       |                                                                    |                                                      |
| W08 |                   |       |                                                                    |                                                      |

## 一条硬规矩

每周至少一次 commit，哪怕只改了笔记。  
仓库的绿色格子连续不断，本身就是能力证明；面试官真的会点进来看提交记录。
