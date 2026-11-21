#include <cstdio>
#include <vector>
#include <utility>
#include <chrono>
#include <cuda_runtime.h>

constexpr size_t MB = 1024 * 1024;

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t n) : ptr_(nullptr), n_(n) {
        cudaMalloc(reinterpret_cast<void**>(&ptr_), n_ * sizeof(T));
        printf("     构造 [%p] %zu 字节\n", (void*)ptr_, n_ * sizeof(T));
    }
    ~DeviceBuffer() {
        if (ptr_) { cudaFree(ptr_); printf("     析构 [%p]\n", (void*)ptr_); }
    }

    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(o.ptr_), n_(o.n_) {
        o.ptr_ = nullptr; o.n_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this == &o) return *this;
        if (ptr_) cudaFree(ptr_);
        ptr_ = o.ptr_; n_ = o.n_;
        o.ptr_ = nullptr; o.n_ = 0;
        return *this;
    }

    T*     get()  const { return ptr_; }
    size_t size() const { return n_; }

private:
    T*     ptr_;
    size_t n_;
};

class WorkspacePool {
public:
    WorkspacePool(size_t count, size_t bytes_each) {
        free_list_.reserve(count);
        for (size_t i = 0; i < count; ++i)
            free_list_.push_back(DeviceBuffer<char>(bytes_each));
    }

    DeviceBuffer<char> acquire() {
        DeviceBuffer<char> b = std::move(free_list_.back());   // 从池里拿走
        free_list_.pop_back();
        return b;                                              // 交给调用者
    }

    void release(DeviceBuffer<char>&& b) {
        free_list_.push_back(std::move(b));                    // 还回池子
    }

    size_t available() const { return free_list_.size(); }

private:
    std::vector<DeviceBuffer<char>> free_list_;
};

int main() {
    const size_t N = 8, ONE = 16 * MB;
    size_t f0 = 0, t = 0;
    cudaMemGetInfo(&f0, &t);

    printf("--- A. 池子：只在初始化时分配 ---\n");
    WorkspacePool pool(N, ONE);
    size_t f1 = 0;
    cudaMemGetInfo(&f1, &t);
    printf("  池子就绪，占 %.0f MB，可用 %zu 块\n\n", (f0 - f1) / (double)MB, pool.available());

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        auto buf = pool.acquire();
        cudaMemset(buf.get(), 0, 1024);
        pool.release(std::move(buf));
    }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();

    size_t f2 = 0;
    cudaMemGetInfo(&f2, &t);
    printf("  借还 1000 次：%.2f ms，显存变化 %.0f MB（必须是 0）\n\n",
           std::chrono::duration<double, std::milli>(t1 - t0).count(),
           (f1 - f2) / (double)MB);

    printf("--- B. 不用池子：每次现分配现释放 ---\n");
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        void* p = nullptr;
        cudaMalloc(&p, ONE);
        cudaMemset(p, 0, 1024);
        cudaFree(p);
    }
    cudaDeviceSynchronize();
    auto t3 = std::chrono::high_resolution_clock::now();
    printf("  分配释放 1000 次：%.2f ms\n", std::chrono::duration<double, std::milli>(t3 - t2).count());

    return 0;
}
