#include <cstdio>
#include <vector>
#include <string>
#include <algorithm>

struct Entry {
    std::string name;
    double      ms;
    double      gbps;
};

int main() {
    std::vector<Entry> log;

    auto add = [&](const char* name, double ms, double bytes) {
        log.push_back(Entry{name, ms, bytes / (ms * 1e6)});
    };

    const size_t N = 64ull * 1024 * 1024;
    add("vector_add", 3.36, 3.0 * N * 4);
    add("saxpy",      3.36, 3.0 * N * 4);
    add("transpose",  2.21, 2.0 * N * 4);
    add("reduce",     1.02, 1.0 * N * 4);

    std::sort(log.begin(), log.end(),
              [](const Entry& a, const Entry& b) { return a.ms < b.ms; });

    printf("%-14s %9s %9s\n", "kernel", "ms", "GB/s");
    for (const auto& e : log)
        printf("%-14s %9.3f %9.1f\n", e.name.c_str(), e.ms, e.gbps);

    return 0;
}
