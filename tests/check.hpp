// Minimal header-only test harness — no external deps. Each test_*.cpp has a
// main() that runs CHECK macros; nonzero exit on failure (ctest reads that).
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace c4d::test {
inline int g_fails = 0;
inline int g_checks = 0;

inline void report(bool ok, std::string_view expr, const char* file, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fails;
        std::fprintf(stderr, "FAIL %s:%d: %.*s\n", file, line,
                     static_cast<int>(expr.size()), expr.data());
    }
}
inline int finish() {
    std::fprintf(stderr, "%d/%d checks passed%s\n", g_checks - g_fails, g_checks,
                 g_fails ? "  *** FAILURES ***" : "");
    return g_fails ? 1 : 0;
}
} // namespace c4d::test

#define CHECK(cond) ::c4d::test::report((cond), #cond, __FILE__, __LINE__)
#define CHECK_NEAR(a, b, tol) \
    ::c4d::test::report(std::fabs((a) - (b)) <= (tol), #a " ~= " #b, __FILE__, __LINE__)
#define RUN_TESTS_RETURN() return ::c4d::test::finish()
