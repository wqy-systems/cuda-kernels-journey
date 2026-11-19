// 亲手测出 Cache Line 的存在
//
// 每次都访问 100 万个元素，访问次数完全一样。
// 唯一的区别：步长（相邻两次访问隔多远）。
//
// 如果内存是"一个一个数着拿"，步长不应该影响速度。
// 但如果 CPU 每次搬一整条（cache line），那 stride=1 会超快，
// stride 大到跨过一条，就会慢下来。
//
// 编译运行：
//   g++ -O2 -o 17_cache/cache_line 17_cache/cache_line.cpp
//   ./17_cache/cache_line

#include <cstdio>
#include <cstdlib>
#include <ctime>

constexpr int SIZE = 32 * 1024 * 1024 / sizeof(int);   // 32M 个 int = 128MB
constexpr int VISITS = 1 << 20;                        // 每次访问 100 万个

int main() {
    int* arr = (int*)malloc(sizeof(int) * SIZE);
    for (int i = 0; i < SIZE; ++i) arr[i] = i & 1;      // 填点数据

    printf("数组大小 128MB，每次访问 %d 个元素\n", VISITS);
    printf("（L2 缓存只有 32MB，所以数据装不下，必须真的去内存拿）\n\n");
    printf(" 步长   间隔(字节)   耗时(ms)   相对 stride=1\n");
    printf(" ------------------------------------------------\n");

    double base = 0;
    const int strides[] = {1, 2, 4, 8, 16, 32, 64, 128};

    for (int s = 0; s < 8; ++s) {
        int stride = strides[s];
        volatile int sink = 0;      // volatile 防止编译器把循环优化掉

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        for (int i = 0; i < VISITS; ++i) {
            sink += arr[(i * stride) % SIZE];
        }

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ms = (t1.tv_sec - t0.tv_sec) * 1000.0
                  + (t1.tv_nsec - t0.tv_nsec) / 1e6;

        if (s == 0) base = ms;
        printf(" %5d  %9d  %9.3f   %6.2f 倍\n",
               stride, stride * 4, ms, ms / base);
    }

    printf(" ------------------------------------------------\n");
    printf("看步长从几开始明显变慢 —— 那个临界点就是 cache line 的大小。\n");
    printf("（x86 通常是 64 字节 = 16 个 int）\n\n");

    free(arr);
    return 0;
}
