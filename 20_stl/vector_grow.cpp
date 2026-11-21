#include <cstdio>
#include <vector>

int main() {
    printf("=== 1. push_back 时容量怎么变 ===\n");
    printf("   i  size  cap        data\n");
    std::vector<int> v;
    for (int i = 0; i < 20; ++i) {
        v.push_back(i);
        printf("  %2d  %4zu  %3zu   %p\n", i, v.size(), v.capacity(), (void*)v.data());
    }

    printf("\n=== 2. 指针什么时候失效 ===\n");
    std::vector<int> w = {1, 2, 3};
    int* p = &w[1];
    printf("  push 前: data=%p, p 指向 %d\n", (void*)w.data(), *p);
    w.push_back(4);
    printf("  push 后: data=%p, p 指向 %d\n", (void*)w.data(), *p);

    printf("\n=== 3. reserve 之后 ===\n");
    std::vector<int> u;
    u.reserve(20);
    const int* base = u.data();
    int moved = 0;
    for (int i = 0; i < 20; ++i) {
        u.push_back(i);
        if (u.data() != base) { ++moved; base = u.data(); }
    }
    printf("  reserve(20) 后 push 20 次，搬家次数 = %d\n", moved);
    printf("  size=%zu  cap=%zu\n", u.size(), u.capacity());

    printf("\n=== 4. 连续内存意味着什么 ===\n");
    std::vector<float> h(1024, 1.0f);
    printf("  &h[0]   = %p\n", (void*)&h[0]);
    printf("  &h[1023]= %p\n", (void*)&h[1023]);
    printf("  差 = %zu 字节（正好 1023 * 4）\n",
           (size_t)((char*)&h[1023] - (char*)&h[0]));
    printf("  所以 h.data() 可以直接丢给 cudaMemcpy\n");
    return 0;
}
