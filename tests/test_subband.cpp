// Subband map: every coordinate in the transformed cube maps to a valid
// subband; coefficient counts per subband match the dyadic pyramid geometry.
#include "c4d/subband.hpp"
#include "c4d/dwt.hpp"
#include "check.hpp"
#include <map>
#include <vector>

using namespace c4d;

int main() {
    const u32 L = DWT_LEVELS;

    // Count coefficients per subband; totals must equal CHUNK^3 and match the
    // closed-form sizes of a dyadic 3D pyramid.
    std::map<std::pair<u32,u32>, u64> counts;
    for (u32 z = 0; z < CHUNK; ++z)
        for (u32 y = 0; y < CHUNK; ++y)
            for (u32 x = 0; x < CHUNK; ++x) {
                Subband sb = subband_of(z, y, x, L);
                CHECK(sb.level < L && sb.orient < 8);
                counts[{sb.level, sb.orient}] += 1;
            }

    u64 total = 0;
    for (auto& [k, v] : counts) total += v;
    CHECK(total == CHUNK_VOX);

    // LLL approximation band: (CHUNK>>L)^3 voxels.
    u64 lll = u64(CHUNK >> L) * (CHUNK >> L) * (CHUNK >> L);
    u64 lll_count = counts[std::pair<u32,u32>{L - 1, 0}];
    CHECK(lll_count == lll);

    // Each level l has 7 detail orientations; a detail subband at level l holds
    // (CHUNK>>(l+1))^3 voxels. Sum over levels of 7*that + lll == CHUNK^3.
    u64 check_total = lll;
    for (u32 l = 0; l < L; ++l) {
        u64 band = u64(CHUNK >> (l + 1)) * (CHUNK >> (l + 1)) * (CHUNK >> (l + 1));
        for (u32 o = 1; o < 8; ++o) {
            u64 c = counts[std::pair<u32,u32>{l, o}];
            CHECK(c == band);
        }
        check_total += 7 * band;
    }
    CHECK(check_total == CHUNK_VOX);

    RUN_TESTS_RETURN();
}
