// DWT correctness: forward∘inverse ≈ identity (perfect reconstruction within
// float32 tolerance), mirror boundary, multi-level. Spec §4.1, §4.4, §4.5.
#include "c4d/dwt.hpp"
#include "check.hpp"
#include <random>
#include <vector>

using namespace c4d;

static f32 max_abs_err(const std::vector<f32>& a, const std::vector<f32>& b) {
    f32 m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

int main() {
    std::mt19937 rng(12345);

    // --- 1D roundtrip on a single mirror-extended line ---
    {
        std::uniform_real_distribution<f32> d(-300.f, 300.f);
        std::vector<f32> line(CHUNK), orig;
        for (auto& v : line) v = d(rng);
        orig = line;
        dwt::fwd_1d(line.data(), CHUNK, 1);
        dwt::inv_1d(line.data(), CHUNK, 1);
        f32 e = max_abs_err(orig, line);
        f32 maxabs = 0; for (auto v : orig) maxabs = std::max(maxabs, std::fabs(v));
        CHECK(e < 1e-3f * maxabs);  // float32 relative tolerance
    }

    // --- full 3D multi-level roundtrip on random data ---
    {
        std::vector<f32> vol(CHUNK_VOX), orig;
        std::uniform_real_distribution<f32> d(0.f, 255.f);
        for (auto& v : vol) v = d(rng);
        orig = vol;
        dwt::forward(vol.data());
        dwt::inverse(vol.data());
        f32 e = max_abs_err(orig, vol);
        CHECK(e < 0.05f);  // 255-range float32, 5 levels: sub-0.05 abs error
    }

    // --- roundtrip on a smooth ramp (worst case for boundary handling) ---
    {
        std::vector<f32> vol(CHUNK_VOX), orig;
        for (u32 z = 0; z < CHUNK; ++z)
            for (u32 y = 0; y < CHUNK; ++y)
                for (u32 x = 0; x < CHUNK; ++x)
                    vol[vox_index(z, y, x)] = static_cast<f32>(z + y + x);
        orig = vol;
        dwt::forward(vol.data());
        dwt::inverse(vol.data());
        CHECK(max_abs_err(orig, vol) < 0.01f);
    }

    // --- a constant block: all DETAIL subbands ~0, energy in the LLL band ---
    // (energy compaction sanity for a flat block). After `levels` decompositions
    // the LLL approximation occupies the leading (CHUNK>>levels)^3 octant; every
    // coefficient outside that octant is a detail coefficient and must be ~0.
    {
        std::vector<f32> vol(CHUNK_VOX, 100.f);
        dwt::forward(vol.data());
        const u32 lll = CHUNK >> DWT_LEVELS;  // LLL band edge (=4 for 5 levels)
        f64 detail_energy = 0, lll_energy = 0;
        for (u32 z = 0; z < CHUNK; ++z)
            for (u32 y = 0; y < CHUNK; ++y)
                for (u32 x = 0; x < CHUNK; ++x) {
                    f32 c = vol[vox_index(z, y, x)];
                    if (z < lll && y < lll && x < lll) lll_energy += f64(c) * c;
                    else detail_energy += f64(c) * c;
                }
        CHECK(detail_energy < 1e-2);
        CHECK(lll_energy > 1.0);  // approximation band carries the energy
    }

    RUN_TESTS_RETURN();
}
