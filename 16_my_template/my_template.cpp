// 16 · 我写的第一个 C++ 模板：arrayMax
//
// 由王庆尧于 2026-11-18 独立写出。
// 一份代码，int 数组和 double 数组都能用 —— 编译器自动造出两个版本。
//
// 编译运行：
//   g++ -o 16_my_template/my_template 16_my_template/my_template.cpp
//   ./16_my_template/my_template

#include <cstdio>

template <typename T>
T arrayMax(const T* arr, int n) {
    T best = arr[0];
    for (int i = 1; i < n; ++i) {
        if (arr[i] > best) {
            best = arr[i];
        }
    }
    return best;
}

int main() {
    int a[] = {3, 7, 2, 9, 4};
    double b[] = {1.5, 8.2, 3.1};

    printf("%d\n", arrayMax(a, 5));      // 9   —— 编译器造了 T=int 版
    printf("%.1f\n", arrayMax(b, 3));    // 8.2 —— 编译器造了 T=double 版

    return 0;
}
