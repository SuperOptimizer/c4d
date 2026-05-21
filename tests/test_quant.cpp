// Quantizer: dead-zone roundtrip error bounded by step; full
// DWT→quant→dequant→IDWT pipeline yields sane PSNR and obeys monotone
// rate/quality vs q. Spec §4.1, §4.5.
#include "c4d/quant.hpp"
#include "c4d/dwt.hpp"
#include "check.hpp"
#include <algorithm>
#include <random>
#include <vector>
#include <cmath>

using namespace c4d;

static f64 psnr(const std::vector<f32>& a, const std::vector<f32>& b) {
    f64 se = 0;
    for (size_t i = 0; i < a.size(); ++i) { f64 d = a[i] - b[i]; se += d * d; }
    f64 mse = se / a.size();
    if (mse <= 0) return 999.0;
    return 10.0 * std::log10(255.0 * 255.0 / mse);
}

int main() {
    std::mt19937 rng(7);

    // --- scalar dead-zone error bound: |c - dequant(quant(c))| < step ---
    {
        std::uniform_real_distribution<f32> d(-500.f, 500.f);
        f32 step = 12.f;
        f32 worst = 0;
        for (int i = 0; i < 100000; ++i) {
            f32 c = d(rng);
            f32 r = dead_zone_dequantize(dead_zone_quantize(c, step), step);
            worst = std::max(worst, std::fabs(c - r));
        }
        CHECK(worst < step);  // mid-tread dead-zone: max error < one step
    }

    // --- full pipeline PSNR on a structured synthetic volume ---
    // (smooth gradient + a few sharp edges + mild noise — scroll-like)
    auto make_vol = [&]() {
        std::vector<f32> v(CHUNK_VOX);
        std::normal_distribution<f32> n(0.f, 3.f);
        for (u32 z = 0; z < CHUNK; ++z)
            for (u32 y = 0; y < CHUNK; ++y)
                for (u32 x = 0; x < CHUNK; ++x) {
                    f32 base = 80.f + 40.f * std::sin(x * 0.05f) + 30.f * std::cos(y * 0.03f);
                    if (((x / 8) + (y / 8) + (z / 8)) % 5 == 0) base += 90.f;  // edges
                    f32 val = base + n(rng);
                    v[vox_index(z, y, x)] = std::clamp(val, 0.f, 255.f);
                }
        return v;
    };

    std::vector<f32> orig = make_vol();
    std::vector<i32> ql(CHUNK_VOX);
    std::vector<f32> coef(CHUNK_VOX), recon(CHUNK_VOX);

    f64 prev_psnr = 1e9; u64 prev_nnz = 0; bool first = true;
    for (f32 q : {4.f, 8.f, 16.f, 32.f}) {
        coef = orig;
        // subtract DC (mean) like the real pipeline
        f64 mean = 0; for (f32 v : coef) mean += v; mean /= CHUNK_VOX;
        for (f32& v : coef) v -= static_cast<f32>(mean);

        dwt::forward(coef.data());
        StepTable t = StepTable::from_q(q);
        quantize(coef, t, ql);
        u64 nnz = 0; for (i32 v : ql) if (v) ++nnz;
        dequantize(ql, t, coef);
        dwt::inverse(coef.data());
        for (u32 i = 0; i < CHUNK_VOX; ++i) recon[i] = coef[i] + static_cast<f32>(mean);

        f64 p = psnr(orig, recon);
        CHECK(p > 25.0);                       // even at q=32, sane PSNR
        if (!first) {
            CHECK(p <= prev_psnr + 0.5);        // coarser q => lower/equal PSNR
            CHECK(nnz <= prev_nnz);             // coarser q => fewer nonzeros
        }
        prev_psnr = p; prev_nnz = nnz; first = false;
        std::fprintf(stderr, "  q=%5.1f  PSNR=%.2f dB  nnz=%llu (%.1f%%)\n",
                     q, p, (unsigned long long)nnz, 100.0 * nnz / CHUNK_VOX);
    }

    RUN_TESTS_RETURN();
}
