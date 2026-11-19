// 分支预测实验 · 严谨版
//
// 上一版两个问题：
//   1. 数组 32MB，正好卡在 L2 容量边界，缓存抖动会淹没分支预测的效果
//   2. 各跑一次，顺序固定 —— 频率/缓存状态会污染结果
//
// 这一版：数组 8MB（稳稳装进 32MB L2），两种数据交错跑 20 轮，各自取最快。
//
// 编译运行：
//   g++ -O2 -o 18_pipeline/branch2 18_pipeline/branch2.cpp
//   ./18_pipeline/branch2

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <algorithm>

constexpr int N = 1 << 21;       // 2,097,152 个 int = 8MB，装得进 L2
constexpr int ROUNDS = 20;

long long process(const int* data, int n) {
    long long sum = 0;
    for (int i = 0; i < n; ++i) {
        if (data[i] < 128) sum += data[i];
    }
    return sum;
}

static inline double now_ms() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

int main() {
    int* rnd = (int*)malloc(sizeof(int) * N);
    srand(12345);
    for (int i = 0; i < N; ++i) rnd[i] = rand() % 256;

    int* srt = (int*)malloc(sizeof(int) * N);
    memcpy(srt, rnd, sizeof(int) * N);
    std::sort(srt, srt + N);

    printf("数据 : %d 个 int = 8MB（装得进 32MB 的 L2）\n", N);
    printf("方法 : 交错跑 %d 轮，各自取最快\n\n", ROUNDS);
    printf(" 轮次    随机(ms)   有序(ms)   比值\n");
    printf(" ---------------------------------------\n");

    // 热身：两种都跑
    process(rnd, N);
    process(srt, N);

    double bestR = 1e9, bestS = 1e9;
    volatile long long sink = 0;     // volatile：不许编译器优化掉
    for (int r = 0; r < ROUNDS; ++r) {
        double t0 = now_ms();
        sink += process(rnd, N);     // 用掉返回值
        double tr = now_ms() - t0;

        double t1 = now_ms();
        sink += process(srt, N);
        double ts = now_ms() - t1;

        if (tr < bestR) bestR = tr;
        if (ts < bestS) bestS = ts;

        if (r < 5 || r == ROUNDS - 1) {
            printf(" %4d   %9.3f  %9.3f   %5.2f\n", r, tr, ts, tr / ts);
        }
    }

    printf(" ---------------------------------------\n");
    printf(" 最快   %9.3f  %9.3f   %5.2f  ← 有序快 %.2f 倍\n",
           bestR, bestS, bestR / bestS, bestR / bestS);
    printf("\n如果有序明显更快，就是分支预测在起作用：\n");
    printf("  有序：if 的结果连续一样，CPU 猜得全对，流水线不停\n");
    printf("  随机：if 的结果乱跳，CPU 一半猜错，每次猜错要清空流水线重来\n\n");

    printf("校验 sum = %lld（非零说明真的算了）\n\n", (long long)sink);

    free(rnd);
    free(srt);
    return 0;
}
