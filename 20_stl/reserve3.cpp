#include <cstdio>
#include <vector>

int main() {
    std::vector<int> v;
    v.reserve(3);
    for (int i = 0; i < 5; ++i) {
        v.push_back(i);
        printf("放第 %d 个: size=%zu cap=%zu data=%p\n", i, v.size(), v.capacity(), (void*)v.data());
    }
    return 0;
}
