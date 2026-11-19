// 亲手测出 CPU 流水线和分支预测
//
// 两份数据完全一样，只是有序 vs 无序。
// 跑同一段代码，看速度差多少。
//
// 编译运行：
//   g++ -O2 -o 18_pipeline/branch 18_pipeline/branch.cpp
//   ./18_pipeline/branch

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <algorithm>

constexpr int N = 1 << 23;      // 8,388,608 个 int

long long process(const int* data, int n) {
    long long sum = 0;
    for (int i = 0; i < n; ++i) {
        if (data[i] < 128) {    // ← 就这一个 if，一切玄机都在这
            sum += data[i];
        }
    }
    return sum;
}

double bench(const int* data, const char* name) {
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    long long r = process(data, N);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1000.0
              + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("  %-12s %8.3f ms   (sum = %lld)\n", name, ms, r);
    return ms;
}

int main() {
    int* data = (int*)malloc(sizeof(int) * N);
    srand(12345);
    for (int i = 0; i < N; ++i) data[i] = rand() % 256;   // 0~255 随机数

    int* sorted = (int*)malloc(sizeof(int) * N);
    memcpy(sorted, data, sizeof(int) * N);
    std::sort(sorted, sorted + N);                        // 排好序

    printf("数据 : %d 个 int，全是 0~255 的数字\n", N);
    printf("代码 : if (data[i] < 128) sum += data[i];\n");
    printf("两份数组内容完全一样，只是有序/无序的区别\n\n");

    process(data, N);   // 预热

    double t_rand = bench(data, "随机数组");
    double t_sort = bench(sorted, "有序数组");

    printf("\n  有序比随机快  %.2f 倍\n", t_rand / t_sort);
    printf("\n数据一模一样，代码一模一样，只是顺序不同！\n\n");

    free(data);
    free(sorted);
    return 0;
}
