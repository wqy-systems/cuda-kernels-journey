#include <cstdio>
#include <stdexcept>
#include <cuda_runtime.h>

constexpr size_t MB = 1024 * 1024;

// 场景 1：提前 return，cudaFree 被跳过
void work_return() {
    float* p = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&p), 256 * MB);
    printf("  work_return: 拿到 256MB, p = %p\n", (void*)p);

    bool bad = true;
    if (bad) {
        printf("  work_return: 出错，直接 return —— cudaFree 那行不会执行\n");
        return;
    }
    cudaFree(p);
}

// 场景 2：抛异常，异常之后的代码全部跳过
void work_throw() {
    float* p = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&p), 256 * MB);
    printf("  work_throw : 拿到 256MB, p = %p\n", (void*)p);
    throw std::runtime_error("数据非法");
    cudaFree(p);   // 永远到不了
}

int main() {
    size_t free0 = 0, total = 0;
    cudaMemGetInfo(&free0, &total);
    printf("启动 : 空闲 %6.0f MB / 总 %.0f MB\n", free0 / (double)MB, total / (double)MB);

    work_return();
    try { work_throw(); }
    catch (const std::exception& e) { printf("  捕获异常: %s\n", e.what()); }

    size_t free1 = 0;
    cudaMemGetInfo(&free1, &total);
    printf("结束 : 空闲 %6.0f MB\n", free1 / (double)MB);
    printf("泄漏 : %.0f MB\n", (free0 - free1) / (double)MB);
    return 0;
}
