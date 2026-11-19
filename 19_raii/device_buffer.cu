#include <cstdio>
#include <stdexcept>
#include <cuda_runtime.h>

constexpr size_t MB = 1024 * 1024;

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t n) : ptr_(nullptr), n_(n) {
        cudaMalloc(reinterpret_cast<void**>(&ptr_), n_ * sizeof(T));
        printf("  构造：拿到 %zu 字节, p = %p\n", n_ * sizeof(T), (void*)ptr_);
    }

    ~DeviceBuffer() {
        if (ptr_) {
            cudaFree(ptr_);
            printf("  析构：还回去了\n");
        }
    }

    // 禁止拷贝：否则两个对象拥有同一块显存，析构两次 = double free
    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return ptr_; }

private:
    T*     ptr_;
    size_t n_;
};

void f_return() {
    DeviceBuffer<float> buf(64 * MB / sizeof(float));
    printf("  f_return: 在里面用 buf\n");
    return;
}

void f_throw() {
    DeviceBuffer<float> buf(64 * MB / sizeof(float));
    size_t f = 0, t = 0;
    cudaMemGetInfo(&f, &t);
    printf("  此刻空闲 %.0f MB\n", f / (double)MB);
    throw std::runtime_error("出错");
}

int main() {
    size_t f0 = 0, t = 0;
    cudaMemGetInfo(&f0, &t);
    printf("启动空闲 %.0f MB\n", f0 / (double)MB);

    f_return();
    try { f_throw(); } catch (const std::exception& e) { printf("  捕获: %s\n", e.what()); }

    size_t f1 = 0;
    cudaMemGetInfo(&f1, &t);
    printf("结束空闲 %.0f MB, 泄漏 %.0f MB\n", f1 / (double)MB, (f0 - f1) / (double)MB);

    // 取消下面两行的注释，看编译器怎么拦你
    DeviceBuffer<float> c1(1024);
    DeviceBuffer<float> c2 = c1;
    return 0;
}
