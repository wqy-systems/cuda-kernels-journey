#!/usr/bin/env bash
# 一键：构建 + 运行两个 GEMM 程序，结果存到 Windows 侧
#
# 用法（WSL 里）：
#   bash ~/dev/cuda-kernels-journey/run_gemm.sh

set -u

REPO="$HOME/dev/cuda-kernels-journey"
OUT="/mnt/c/Users/MECHREVO/WorkBuddy/2026-11-14-17-16-22/last_run.txt"

cd "$REPO" || { echo "找不到仓库目录 $REPO"; exit 1; }

{
  echo "===== 构建时间 $(date '+%Y-%m-%d %H:%M:%S') ====="
  echo "--- cmake 配置（新增了 06_gemm_tiled 目录，必须重新配置）---"
  cmake -B build 2>&1 | tail -12
  echo
  echo "--- 编译 ---"
  cmake --build build -j 2>&1 | tail -12

  echo
  echo "===== 05 朴素 GEMM：撞内存墙 ====="
  ./build/bin/gemm

  echo
  echo "===== 06 分块 GEMM：提高数据复用 ====="
  ./build/bin/gemm_tiled

  echo
  echo "===== 07 四路累加：治排队 ====="
  ./build/bin/gemm_unroll

} 2>&1 | tee "$OUT"

echo
echo "-------------------------------------------"
echo "全部结果已保存到："
echo "$OUT"
echo "跟大鲸鱼说一声，我自己去读。"
