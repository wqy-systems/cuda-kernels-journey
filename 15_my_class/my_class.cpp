// 15 · 我写的第一个 C++ 类：Counter
//
// 两个问号由王庆尧于 2026-11-18 填出：
//   构造函数里 value = 0
//   add 里 value += x
//
// 编译运行：
//   g++ -o 15_my_class/my_class 15_my_class/my_class.cpp
//   ./15_my_class/my_class

#include <cstdio>

struct Counter {
    int value;                    // 数据：一个整数（对象的"私人物品"）

    Counter() {                   // 构造函数：造对象时自动跑
        value = 0;                // ← 你填的
    }

    void add(int x) {             // 成员函数：这个盒子能干的事
        value += x;               // ← 你填的
    }

    void show() {
        printf("value = %d\n", value);
    }
};

int main() {
    Counter c;        // 造一个盒子，构造函数自动跑，value 变成 0
    c.show();         // 打印 0

    c.add(5);
    c.add(3);
    c.show();         // 打印 8

    // 再造一个，互不干扰——每个对象有自己的一份 value
    Counter d;
    d.add(100);
    c.show();         // c 还是 8
    d.show();         // d 是 100

    return 0;
}
