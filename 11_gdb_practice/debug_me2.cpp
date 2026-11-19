// 第二个 gdb 练习：崩在深层函数里
// 不用你写，你只要用 gdb 找出：崩在哪、是谁调用了谁

#include <cstdio>

int computeSum(int* values, int count) {
    int total = 0;
    for (int i = 0; i < count; ++i) {
        total += values[i];        // ← 这里会出事
    }
    return total;
}

int analyze(int* values, int count) {
    int s = computeSum(values, count);
    return s * 2;
}

int main() {
    int* data = nullptr;           // 空的
    int n = 10;

    int result = analyze(data, n);  // 把空的传下去

    printf("result = %d\n", result);
    return 0;
}
