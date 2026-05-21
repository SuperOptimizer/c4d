// HybridUint framing: exhaustive encode/decode roundtrip over the magnitude
// range coefficients actually reach, and a bounded token alphabet. Spec §4.7.
#include "c4d/hybrid_uint.hpp"
#include "check.hpp"

using namespace c4d;

int main() {
    u32 max_token = 0;
    // Coefficients after DWT of u8 (×~5 levels of synthesis gain) stay well under
    // 2^20; test the full 0..(1<<20)-1 range exhaustively-ish (step to keep fast).
    bool ok = true;
    for (u32 u = 0; u < (1u << 16); ++u) {
        auto s = hybrid::encode(u);
        max_token = std::max(max_token, s.token);
        CHECK(hybrid::raw_bits_of(s.token) == s.nbits);
        u32 back = hybrid::decode(s.token, s.raw);
        if (back != u) { ok = false; std::fprintf(stderr, "  u=%u token=%u raw=%u -> %u\n", u, s.token, s.raw, back); break; }
    }
    CHECK(ok);

    // sparse sweep over the high range
    ok = true;
    for (u32 u = (1u << 16); u < (1u << 22); u += 13) {
        auto s = hybrid::encode(u);
        max_token = std::max(max_token, s.token);
        if (hybrid::decode(s.token, s.raw) != u) { ok = false; break; }
    }
    CHECK(ok);

    std::fprintf(stderr, "  max token over [0, 2^22) = %u (alphabet ceiling %u)\n",
                 max_token, hybrid::MAX_TOKEN);
    CHECK(max_token < hybrid::MAX_TOKEN);

    RUN_TESTS_RETURN();
}
