#!/usr/bin/env bash
# 环境自检：装完 WSL + CUDA 后跑这一遍，全绿再往下走
set -u

green() { printf "\033[32m  OK  \033[0m %s\n" "$1"; }
red()   { printf "\033[31m FAIL \033[0m %s\n" "$1"; }
warn()  { printf "\033[33m WARN \033[0m %s\n" "$1"; }

echo "==============================================="
echo " CUDA 环境自检"
echo "==============================================="

# 1. 系统
if grep -qi ubuntu /etc/os-release 2>/dev/null; then
    green "系统: $(. /etc/os-release && echo "$PRETTY_NAME")"
else
    warn "不是 Ubuntu，其他发行版路径可能不同"
fi

# 2. 显卡与驱动
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "---- nvidia-smi ----"
    nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap \
               --format=csv,noheader 2>/dev/null | sed 's/^/     /'
    green "驱动可用"
else
    red "nvidia-smi 找不到 —— WSL 里没透传 GPU，检查 Windows 端驱动是否 5xx 以上"
fi

# 3. nvcc
if command -v nvcc >/dev/null 2>&1; then
    green "nvcc: $(nvcc --version | grep -oP 'release \K[0-9.]+')"
    CUDA_HOME=$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")
    green "CUDA_HOME=$CUDA_HOME"
else
    red "nvcc 未安装 —— 装 CUDA Toolkit 12.x（只装 toolkit，不要装驱动）"
fi

# 4. 编译链
for t in gcc g++ make git; do
    if command -v "$t" >/dev/null 2>&1; then
        green "$t: $("$t" --version 2>/dev/null | head -1 | cut -c1-50)"
    else
        red "$t 缺失 —— sudo apt install build-essential git"
    fi
done

# 5. profiling 工具
for t in ncu nsys; do
    if command -v "$t" >/dev/null 2>&1; then
        green "$t: 已安装"
    else
        warn "$t 未安装 —— 在 Linux 里装 Nsight Compute / Systems（WSL 里装的是 Linux 版）"
    fi
done

# 6. 磁盘与内存
echo "---- 资源 ----"
df -h "$HOME" | tail -1 | awk '{printf "     磁盘可用: %s / %s\n", $4, $2}'
free -h | awk '/^Mem:/{printf "     内存: 已用 %s / 共 %s\n", $3, $2}'

# 7. 实测一次编译
echo "---- 编译测试 ----"
TMPD=$(mktemp -d)
cat > "$TMPD/t.cu" <<'EOF'
#include <cstdio>
__global__ void k(){ printf("hello from block %d thread %d\n", blockIdx.x, threadIdx.x); }
int main(){ k<<<1,4>>>(); cudaDeviceSynchronize(); return 0; }
EOF
if nvcc -arch=sm_89 -o "$TMPD/t" "$TMPD/t.cu" 2>"$TMPD/err"; then
    if "$TMPD/t" >/dev/null 2>&1; then
        green "编译 + 运行 kernel 成功"
    else
        red "编译过了但运行失败，看上面的报错"
    fi
else
    red "编译失败："
    sed 's/^/     /' "$TMPD/err"
fi
rm -rf "$TMPD"

echo "==============================================="
echo " 全绿就可以开始 W01 了：make && ./01_vector_add/vector_add"
echo "==============================================="
