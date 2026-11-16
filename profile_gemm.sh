#!/usr/bin/env bash
# 用 Nsight Compute 实测 GEMM 的瓶颈，结果写到 Windows 侧
#
# 用法：bash ~/dev/cuda-kernels-journey/profile_gemm.sh

set -u

OUT="/mnt/c/Users/MECHREVO/WorkBuddy/2026-11-14-17-16-22/profile_gemm.txt"
BIN="$HOME/dev/cuda-kernels-journey/build/bin/gemm_unroll"

cd "$HOME/dev/cuda-kernels-journey" || exit 1

{
  echo "===== ncu profile $(date '+%Y-%m-%d %H:%M:%S') ====="
  echo "目标 kernel: gemm1acc（单累加器分块版）"
  echo
  sudo /opt/nvidia/nsight-compute/2023.2.2/ncu --set full --launch-count 1 --kernel-name regex:gemm1acc "$BIN" 2>&1
  echo
  echo "===== 结束 ====="
} > "$OUT" 2>&1

echo "profile 已保存到："
echo "$OUT"
echo "如果上面报权限错误，把报错发我，我换 sudo 方案。"
