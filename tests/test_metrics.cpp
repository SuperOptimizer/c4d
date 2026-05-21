// Metric basket sanity: identical inputs score ~perfect; blurring/noise lowers
// each metric monotonically; geomean and floors behave. Spec §7.1.
#include "c4d/metrics.hpp"
#include "check.hpp"
#include <algorithm>
#include <random>
#include <vector>

using namespace c4d;

int main() {
    std::mt19937 rng(3);
    std::vector<u8> ref(CHUNK_VOX);
    std::normal_distribution<f32> n(0.f, 5.f);
    for (u32 z = 0; z < CHUNK; ++z)
        for (u32 y = 0; y < CHUNK; ++y)
            for (u32 x = 0; x < CHUNK; ++x) {
                f32 v = 100 + 40 * std::sin(x * 0.05f) + 30 * std::cos(y * 0.06f);
                if ((y / 8) % 3 == 0) v += 60;
                ref[vox_index(z, y, x)] = static_cast<u8>(std::clamp(v + n(rng), 0.f, 255.f));
            }

    // identical -> near-perfect on all metrics
    {
        auto b = metrics::evaluate(ref, ref);
        CHECK(b.psnr_db > 90);
        CHECK(b.gmsd > 0.99);
        CHECK(b.ms_ssim > 0.99);
        CHECK(b.haarpsi > 0.99);
        CHECK(b.sharpness > 0.99);
        CHECK(b.epi > 0.99);
        CHECK(b.passes_floors());
    }

    // a 3x3 box-blurred copy (over-smoothing) -> lower edge/sharpness metrics
    auto blur = [&](const std::vector<u8>& src) {
        std::vector<u8> o = src;
        for (u32 z = 0; z < CHUNK; ++z)
            for (u32 y = 1; y + 1 < CHUNK; ++y)
                for (u32 x = 1; x + 1 < CHUNK; ++x) {
                    f32 s = 0;
                    for (i32 dy = -1; dy <= 1; ++dy) for (i32 dx = -1; dx <= 1; ++dx)
                        s += metrics::at(src, z, y + dy, x + dx);
                    o[vox_index(z, y, x)] = static_cast<u8>(s / 9.f + 0.5f);
                }
        return o;
    };
    {
        auto blurred = blur(ref);
        auto bb = metrics::evaluate(ref, blurred);
        std::fprintf(stderr, "  blurred: PSNR=%.1f GMSD=%.3f MS-SSIM=%.3f HaarPSI=%.3f sharp=%.3f EPI=%.3f geo=%.3f\n",
                     bb.psnr_db, bb.gmsd, bb.ms_ssim, bb.haarpsi, bb.sharpness, bb.epi, bb.geomean());
        // sharpness retention must drop clearly under blur
        CHECK(bb.sharpness < 0.9);
        CHECK(bb.gmsd < 0.999);
        CHECK(bb.geomean() < 1.0);
    }

    // additive noise -> GMSD/HaarPSI drop; geomean below identity
    {
        std::vector<u8> noisy = ref;
        std::normal_distribution<f32> nn(0.f, 15.f);
        for (auto& v : noisy) v = static_cast<u8>(std::clamp(f32(v) + nn(rng), 0.f, 255.f));
        auto bn = metrics::evaluate(ref, noisy);
        std::fprintf(stderr, "  noisy:   PSNR=%.1f GMSD=%.3f MS-SSIM=%.3f HaarPSI=%.3f sharp=%.3f EPI=%.3f geo=%.3f\n",
                     bn.psnr_db, bn.gmsd, bn.ms_ssim, bn.haarpsi, bn.sharpness, bn.epi, bn.geomean());
        CHECK(bn.gmsd < 0.999);
        CHECK(bn.geomean() < metrics::evaluate(ref, ref).geomean());
    }

    RUN_TESTS_RETURN();
}
