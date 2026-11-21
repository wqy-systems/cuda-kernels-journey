#include <cstdio>
#include <utility>
#include <vector>
#include <cuda_runtime.h>

constexpr size_t MB = 1024 * 1024;

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t n) : ptr_(nullptr), n_(n) {
        cudaMalloc(reinterpret_cast<void**>(&ptr_), n_ * sizeof(T));
        printf("  构造     [%p]\n", (void*)ptr_);
    }

    ~DeviceBuffer() {
        if (ptr_) { cudaFree(ptr_); printf("  析构     [%p]\n", (void*)ptr_); }
    }

    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // 移动构造：偷过来，然后把对方清空
    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(o.ptr_), n_(o.n_) {
        o.ptr_ = nullptr;
        o.n_   = 0;
        printf("  移动构造 [%p] 换主人\n", (void*)ptr_);
    }

    // 移动赋值：先还掉自己原有的，再偷
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this == &o) return *this;
        if (ptr_) { printf("  释放旧的 [%p]\n", (void*)ptr_); cudaFree(ptr_); }
        ptr_   = o.ptr_;
        n_     = o.n_;
        o.ptr_ = nullptr;
        o.n_   = 0;
        printf("  移动赋值 [%p] 换主人\n", (void*)ptr_);
        return *this;
    }

    T*       get()        { return ptr_; }
    const T* get()  const { return ptr_; }
    size_t   size() const { return n_; }

private:
    T*     ptr_;
    size_t n_;
};

int main() {
    printf("--- 1. 建一个 64MB ---\n");
    DeviceBuffer<float> a(64 * MB / sizeof(float));

    printf("\n--- 2. 移动构造：b 抢走 a 的 ---\n");
    size_t f1 = 0, t = 0;
    cudaMemGetInfo(&f1, &t);
    DeviceBuffer<float> b(std::move(a));
    size_t f2 = 0;
    cudaMemGetInfo(&f2, &t);
    printf("  a = %p   <- 必须变成 0x0\n", (void*)a.get());
    printf("  b = %p   <- 必须和 a 刚才的一模一样\n", (void*)b.get());
    printf("  移动前后空闲显存: %.0f -> %.0f MB  <- 必须完全相等\n", f1 / (double)MB, f2 / (double)MB);

    printf("\n--- 3. 移动赋值：c 换掉自己原有的 ---\n");
    DeviceBuffer<float> c(32 * MB / sizeof(float));
    printf("  c 原来 = %p\n", (void*)c.get());
    c = std::move(b);
    printf("  c 现在 = %p   <- 必须等于 b 的\n", (void*)c.get());
    printf("  b = %p   <- 必须变成 0x0\n", (void*)b.get());

    printf("\n--- 4. 塞进 vector ---\n");
    std::vector<DeviceBuffer<float>> v;
    v.push_back(DeviceBuffer<float>(16 * MB / sizeof(float)));
    printf("  vector 里 %zu 个, v[0] = %p\n", v.size(), (void*)v[0].get());

    printf("\n还活着的: a=%p b=%p c=%p\n", (void*)a.get(), (void*)b.get(), (void*)c.get());
    return 0;
}
