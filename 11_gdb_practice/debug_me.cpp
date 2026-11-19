// 一个会崩溃的小程序 —— 专门拿来练 gdb
// 不用你写，你只要用 gdb 找出它崩在哪一行、为什么崩。

#include <cstdio>

int add(int a, int b) {
    return a + b;
}

int main() {
    int scores[3] = {90, 80, 70};

    printf("总分 = %d\n", add(scores[0], scores[1]));

    int* p = nullptr;      // 空指针
    *p = 42;               // ← 这一行会崩溃

    printf("这行不会打印出来\n");
    return 0;
}
