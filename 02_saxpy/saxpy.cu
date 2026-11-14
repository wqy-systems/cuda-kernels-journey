// 02 · SAXPY: y = a * x + y
//
// 自己写。规则：不许看 01_vector_add，卡住了再看本目录 README 的提示。
// 写不出来就说明 01 还没消化，回去重读一遍。

#include <cstdio>
#include <cstdlib>
#include "common.h"

constexpr int N = 1 << 24;

// TODO 1: 写出 kernel 函数体
// 提示：参数是 (const float* x, float* y, float a, int n)
//       每个线程处理哪些元素？循环怎么写才不越界？
__global__ void saxpyKernel(const float* __restrict__ x,
                           float* __restrict__ y,
                           float a,
                           int n) {
    // 在这里写
}

void saxpyCPU(const float* x, float* y, float a, int n) {
    for (int i = 0; i < n; ++i) y[i] = a * x[i] + y[i];
}

int main() {
    printDeviceInfo();

    const size_t bytes = sizeof(float) * N;
    const float a = 2.5f;

    // TODO 2: 分配 host 内存（h_x / h_y / h_ref）并初始化数据
    //         记住 01 里为什么用 cudaMallocHost
    float *h_x = nullptr, *h_y = nullptr, *h_ref = nullptr;

    // TODO 3: 分配 device 内存、拷贝、启动 kernel、拷回
    //         kernel 启动配置自己定，并在注释里写清理由
    float *d_x = nullptr, *d_y = nullptr;

    // TODO 4: 用 maxAbsError 校验，并 reportBandwidth 打印实测带宽
    //         参考：(读 x + 读 y + 写 y) = 3 * N * 4 bytes

    fprintf(stderr, "[TODO] 还没写完，去把 saxpy.cu 里的 4 个 TODO 填掉\n");
    return 1;
}
