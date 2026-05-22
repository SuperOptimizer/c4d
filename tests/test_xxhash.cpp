// xxhash64 known-answer tests. xxhash is the integrity backbone of the .c4d
// archive (every payload/index/directory checksum), so it's verified against
// canonical XXH64 reference values (cross-checked against the reference C
// implementation and the Python `xxhash` package). The reference sanity buffer
// is the standard xxHash one: byte[i] = (PRIME32 >> 24); PRIME32 *= PRIME32.
#include "c4d/xxhash.hpp"
#include "check.hpp"
#include <span>
#include <vector>

using namespace c4d;

int main() {
    std::vector<u8> buf(101);
    u32 bs = 2654435761u;                      // PRIME32
    for (int i = 0; i < 101; ++i) { buf[i] = static_cast<u8>(bs >> 24); bs *= bs; }
    auto h = [&](int n, u64 seed) { return xxhash64(std::span<const u8>(buf.data(), size_t(n)), seed); };
    constexpr u64 P32 = 2654435761ull;

    // Canonical XXH64 known-answer vectors.
    CHECK(xxhash64(std::span<const u8>{}, 0)            == 0xef46db3751d8e999ull);  // empty
    CHECK(h(1, 0)                                       == 0x4fce394cc88952d8ull);
    CHECK(h(1, P32)                                     == 0x739840cb819fa723ull);
    CHECK(h(14, 0)                                      == 0xcffa8db881bc3a3dull);
    CHECK(h(14, P32)                                    == 0x5b9611585efcc9cbull);
    CHECK(h(101, 0)                                     == 0x0eab543384f878adull);
    CHECK(h(101, P32)                                   == 0xcaa65939306f1e21ull);

    // Determinism + seed/length sensitivity (catches a stuck or truncated hash).
    CHECK(h(101, 0) == h(101, 0));
    CHECK(h(101, 0) != h(101, 1));             // seed matters
    CHECK(h(100, 0) != h(101, 0));             // length matters
    CHECK(h(31, 0)  != h(32, 0));              // crosses the 32-byte block boundary
    CHECK(h(32, 0)  != h(33, 0));

    // A single flipped byte changes the digest (the property the archive relies on).
    {
        std::vector<u8> a(64, 0xAB), b = a;
        b[37] ^= 0x01;
        CHECK(xxhash64(a) != xxhash64(b));
    }

    RUN_TESTS_RETURN();
}
