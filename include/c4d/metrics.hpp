// Perceptual / structural quality metrics for the encoder objective (§7.1).
// All operate on u8 cubes (reference vs reconstruction). The 2D-defined metrics
// (GMSD, MS-SSIM, HaarPSI) are computed per Z-slice and averaged; the volumetric
// ones (Laplacian sharpness-retention, EPI, PSNR) use the whole cube.
//
// These are ENCODER-SIDE instrumentation — they are NOT part of the frozen
// format. The deliverable is a per-metric table across a parameter sweep, sliced
// post-hoc by whichever aggregator answers the question (avg / geomean / floors).
#pragma once
#include "core.hpp"
#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

namespace c4d::metrics {

// --- helpers ---------------------------------------------------------------
inline f32 at(std::span<const u8> v, u32 z, u32 y, u32 x) { return f32(v[vox_index(z, y, x)]); }

// Sobel-ish gradient magnitude on a 2D slice (central differences).
inline void grad_mag_slice(std::span<const u8> v, u32 z, std::vector<f32>& g) {
    g.assign(CHUNK * CHUNK, 0.f);
    for (u32 y = 1; y + 1 < CHUNK; ++y)
        for (u32 x = 1; x + 1 < CHUNK; ++x) {
            f32 gx = at(v, z, y, x + 1) - at(v, z, y, x - 1);
            f32 gy = at(v, z, y + 1, x) - at(v, z, y - 1, x);
            g[y * CHUNK + x] = std::sqrt(gx * gx + gy * gy);
        }
}

// --- PSNR (whole cube) -----------------------------------------------------
[[nodiscard]] inline f64 psnr(std::span<const u8> a, std::span<const u8> b) {
    f64 se = 0; for (u32 i = 0; i < CHUNK_VOX; ++i) { f64 d = f64(a[i]) - b[i]; se += d * d; }
    f64 mse = se / CHUNK_VOX;
    return mse <= 0 ? 99.0 : 10.0 * std::log10(255.0 * 255.0 / mse);
}

// --- GMSD: gradient-magnitude-similarity DEVIATION (std-pooled) ------------
// Primary edge term (§7.1). Lower = better; its variance pooling specifically
// punishes "edges smoothed in some regions but not others". We return 1 - GMSD
// so higher = better and it composes in a geomean with the others.
[[nodiscard]] inline f64 gmsd_score(std::span<const u8> ref, std::span<const u8> rec) {
    constexpr f32 C = 170.f;  // stability constant (GMSD paper scale for 0..255)
    std::vector<f32> gr, gc;
    f64 sum = 0, sum2 = 0; u64 n = 0;
    for (u32 z = 0; z < CHUNK; ++z) {
        grad_mag_slice(ref, z, gr);
        grad_mag_slice(rec, z, gc);
        for (u32 i = 0; i < CHUNK * CHUNK; ++i) {
            f32 gms = (2.f * gr[i] * gc[i] + C) / (gr[i] * gr[i] + gc[i] * gc[i] + C);
            sum += gms; sum2 += f64(gms) * gms; ++n;
        }
    }
    f64 mean = sum / n;
    f64 var = std::max(0.0, sum2 / n - mean * mean);
    f64 gmsd = std::sqrt(var);          // deviation of the GMS map
    return 1.0 - gmsd;                  // higher = better
}

// --- SSIM (single scale) per slice, averaged. MS-SSIM approximated by a 3-scale
// average (downsample by 2 between scales). Structural floor (§7.1). -----------
[[nodiscard]] inline f64 ssim_slice(std::span<const f32> a, std::span<const f32> b,
                                    u32 w, u32 h) {
    // 8x8 windowed SSIM, non-overlapping, mean over windows.
    constexpr f64 C1 = 6.5025, C2 = 58.5225;  // (0.01*255)^2, (0.03*255)^2
    f64 acc = 0; u64 nw = 0;
    for (u32 by = 0; by + 8 <= h; by += 8)
        for (u32 bx = 0; bx + 8 <= w; bx += 8) {
            f64 ma = 0, mb = 0, va = 0, vb = 0, cov = 0;
            for (u32 y = 0; y < 8; ++y) for (u32 x = 0; x < 8; ++x) {
                f64 pa = a[(by + y) * w + bx + x], pb = b[(by + y) * w + bx + x];
                ma += pa; mb += pb;
            }
            ma /= 64; mb /= 64;
            for (u32 y = 0; y < 8; ++y) for (u32 x = 0; x < 8; ++x) {
                f64 pa = a[(by + y) * w + bx + x] - ma, pb = b[(by + y) * w + bx + x] - mb;
                va += pa * pa; vb += pb * pb; cov += pa * pb;
            }
            va /= 63; vb /= 63; cov /= 63;
            f64 s = ((2 * ma * mb + C1) * (2 * cov + C2)) /
                    ((ma * ma + mb * mb + C1) * (va + vb + C2));
            acc += s; ++nw;
        }
    return nw ? acc / nw : 1.0;
}

inline void downsample2(std::span<const f32> in, u32 w, u32 h, std::vector<f32>& out,
                        u32& ow, u32& oh) {
    ow = w / 2; oh = h / 2; out.assign(ow * oh, 0.f);
    for (u32 y = 0; y < oh; ++y) for (u32 x = 0; x < ow; ++x)
        out[y * ow + x] = 0.25f * (in[(2*y)*w + 2*x] + in[(2*y)*w + 2*x+1]
                                 + in[(2*y+1)*w + 2*x] + in[(2*y+1)*w + 2*x+1]);
}

[[nodiscard]] inline f64 ms_ssim(std::span<const u8> ref, std::span<const u8> rec) {
    f64 acc = 0; u64 ns = 0;
    std::vector<f32> a(CHUNK * CHUNK), b(CHUNK * CHUNK);
    for (u32 z = 0; z < CHUNK; z += 4) {   // every 4th slice (speed)
        for (u32 i = 0; i < CHUNK * CHUNK; ++i) { a[i] = ref[z*CHUNK*CHUNK + i]; b[i] = rec[z*CHUNK*CHUNK + i]; }
        u32 w = CHUNK, h = CHUNK;
        std::vector<f32> ca = a, cb = b, da, db;
        f64 prod = 1.0; int scales = 3;
        for (int s = 0; s < scales; ++s) {
            prod *= std::max(1e-6, ssim_slice(ca, cb, w, h));
            if (s + 1 < scales) {
                u32 nw, nh; downsample2(ca, w, h, da, nw, nh); downsample2(cb, w, h, db, nw, nh);
                ca = da; cb = db; w = nw; h = nh;
            }
        }
        acc += std::pow(prod, 1.0 / scales); ++ns;
    }
    return ns ? acc / ns : 1.0;
}

// --- HaarPSI (Haar perceptual similarity index), simplified single-level -----
// Wavelet-native (§7.1): one-level Haar high-pass response similarity, weighted
// by local activity. We use a 2x2 Haar HL/LH response per slice.
[[nodiscard]] inline f64 haarpsi(std::span<const u8> ref, std::span<const u8> rec) {
    constexpr f64 C = 30.0;
    f64 num = 0, den = 0;
    for (u32 z = 0; z < CHUNK; z += 4)
        for (u32 y = 0; y + 1 < CHUNK; y += 2)
            for (u32 x = 0; x + 1 < CHUNK; x += 2) {
                auto haar = [&](std::span<const u8> v) {
                    f64 a = at(v,z,y,x), b = at(v,z,y,x+1), c = at(v,z,y+1,x), d = at(v,z,y+1,x+1);
                    f64 hl = (a + c) - (b + d);     // horizontal high
                    f64 lh = (a + b) - (c + d);     // vertical high
                    return std::sqrt(hl*hl + lh*lh);
                };
                f64 fr = haar(ref), fc = haar(rec);
                f64 sim = (2 * fr * fc + C) / (fr * fr + fc * fc + C);
                f64 w = std::max(fr, fc);            // perceptual weight = local activity
                num += sim * w; den += w;
            }
    return den > 0 ? num / den : 1.0;
}

// --- Laplacian sharpness-retention (one-sided floor, §7.1) ------------------
// ratio of reconstruction Laplacian-variance to reference; ~1 = sharpness kept,
// <1 = over-smoothed. Clamp to [0,1] (over-sharpening counts as full retention).
[[nodiscard]] inline f64 lap_var(std::span<const u8> v) {
    f64 sum = 0, sum2 = 0; u64 n = 0;
    for (u32 z = 0; z < CHUNK; ++z)
        for (u32 y = 1; y + 1 < CHUNK; ++y)
            for (u32 x = 1; x + 1 < CHUNK; ++x) {
                f64 l = -4 * at(v,z,y,x) + at(v,z,y,x+1) + at(v,z,y,x-1)
                                         + at(v,z,y+1,x) + at(v,z,y-1,x);
                sum += l; sum2 += l * l; ++n;
            }
    f64 mean = sum / n; return sum2 / n - mean * mean;
}
[[nodiscard]] inline f64 sharpness_retention(std::span<const u8> ref, std::span<const u8> rec) {
    f64 lr = lap_var(ref), lc = lap_var(rec);
    if (lr <= 0) return 1.0;
    return std::min(1.0, lc / lr);
}

// --- EPI (edge preservation index, CT validation column) --------------------
// Pratt-style correlation of high-pass responses (Laplacian) between ref/rec.
[[nodiscard]] inline f64 epi(std::span<const u8> ref, std::span<const u8> rec) {
    f64 sr = 0, sc = 0; u64 n = 0;
    std::vector<f64> hr, hc; hr.reserve(CHUNK_VOX); hc.reserve(CHUNK_VOX);
    auto hp = [&](std::span<const u8> v, u32 z, u32 y, u32 x) {
        return -4.0*at(v,z,y,x) + at(v,z,y,x+1)+at(v,z,y,x-1)+at(v,z,y+1,x)+at(v,z,y-1,x);
    };
    for (u32 z = 0; z < CHUNK; z += 2)
        for (u32 y = 1; y + 1 < CHUNK; ++y)
            for (u32 x = 1; x + 1 < CHUNK; ++x) {
                f64 a = hp(ref,z,y,x), b = hp(rec,z,y,x);
                hr.push_back(a); hc.push_back(b); sr += a; sc += b; ++n;
            }
    f64 mr = sr / n, mc = sc / n, cov = 0, vr = 0, vc = 0;
    for (u64 i = 0; i < n; ++i) { f64 da = hr[i]-mr, db = hc[i]-mc; cov += da*db; vr += da*da; vc += db*db; }
    f64 denom = std::sqrt(vr * vc);
    return denom > 0 ? cov / denom : 1.0;
}

// --- the basket ------------------------------------------------------------
struct Basket {
    f64 psnr_db, gmsd, ms_ssim, haarpsi, sharpness, epi;
    // geomean of the perceptual terms (normalized-PSNR via a soft map), §7.1.
    [[nodiscard]] f64 geomean() const {
        f64 npsnr = std::min(1.0, psnr_db / 50.0);   // normalize PSNR to ~[0,1]
        f64 g = npsnr * gmsd * ms_ssim * haarpsi;
        return std::pow(std::max(1e-9, g), 1.0 / 4.0);
    }
    // hard floors (§7.1): MS-SSIM >= 0.90, sharpness-retention >= ~0.85.
    [[nodiscard]] bool passes_floors() const { return ms_ssim >= 0.90 && sharpness >= 0.85; }
};

[[nodiscard]] inline Basket evaluate(std::span<const u8> ref, std::span<const u8> rec) {
    return { psnr(ref, rec), gmsd_score(ref, rec), ms_ssim(ref, rec),
             haarpsi(ref, rec), sharpness_retention(ref, rec), epi(ref, rec) };
}

} // namespace c4d::metrics
