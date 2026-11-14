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

```bash
make                      # 编译全部
make 01_vector_add/vector_add   # 只编一个
./01_vector_add/vector_add      # 运行
bash check_env.sh               # 环境自检

ncu --set full -o report ./01_vector_add/vector_add   # 出一份完整 profile
ncu -i report.ncu-rep                                 # 在界面/终端里看
```

## 进度（W01-W08 · S0 地基）

- [x] W01 环境打通：`nvcc -V` + `nvidia-smi` 正常，跑通 `01_vector_add`
- [x] W01 建 GitHub 仓库，push 首次提交（commit `be4770c`）
- [ ] W02 Linux 基础：进程 / 权限 / 环境变量 / gdb 断点
- [ ] W02 CMake 最小工程能编译 .cu
- [ ] W03 C++ 指针与内存：手写 3 个小程序（数组反转 / 字符串拷贝 / 动态二维数组）
- [ ] W04 C++ 类与模板：手写简化版 vector（含拷贝构造、析构、operator[]）
- [ ] W05 CSAPP 第 3 章：汇编、内存层次、Cache 行大小实测
- [ ] W06 CSAPP 第 5-6 章：性能优化、SIMD
- [ ] W07 写 `02_saxpy`，用 ncu 看第一个 kernel 的指标
- [ ] W08 期末周：只保持每天 30 分钟，不掉线

## 每周记录

| 周 | 主题 | 投入(h) | 卡住的地方 | 一句话收获 |
|---|---|---|---|---|
| W01 | 环境搭建 + 第一个 kernel | 4 | GitHub：github.com 被 DNS 污染成 127.0.0.1、HTTPS 被 SNI 重置、SSH 公钥三次指纹不匹配 | vector add 只有 47.6% 带宽——每个线程每轮只搬 4 字节，并行粒度不够粗，不是 bug |
| W02 | | | | |
| W03 | | | | |
| W04 | | | | |
| W05 | | | | |
| W06 | | | | |
| W07 | | | | |
| W08 | | | | |

## 一条硬规矩

每周至少一次 commit，哪怕只改了笔记。
仓库的绿色格子连续不断，本身就是能力证明；面试官真的会点进来看提交记录。
