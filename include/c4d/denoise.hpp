// Noise-aware dead-zone (denoise-in-transform), §4.10. Encoder-only: it widens
// the per-subband dead-zone for noise-dominated bands so c4d declines to spend
// bits coding sub-noise-floor coefficients. This tunes the *step values* the
// decoder reads (StepTable) — the decoder and bitstream are unchanged, so this
// is freely improvable encoder policy, not part of the frozen format.
//
// Dead-zone quantization and wavelet shrinkage are nearly the same operation; we
// estimate the noise level from the data and set each subband's dead-zone from a
// BayesShrink-style threshold. This helps RATIO and true-signal QUALITY together
// on noisy scans (the corpus-confirmed hard case, e.g. the 3.24um scrolls): it
// suppresses noise that would otherwise eat bits, while preserving edge-bearing
// coefficients.
#pragma once
#include "core.hpp"
#include "subband.hpp"
#include "quant.hpp"
#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

namespace c4d {

// Robust noise estimate from the finest diagonal (HHH at level 0) subband, which
// is ~pure noise in CT: sigma = median(|coeff|) / 0.6745 (MAD estimator).
[[nodiscard]] inline f64 estimate_noise_sigma(std::span<const f32> coeffs) {
    const SubbandMap& m = subband_map();
    std::vector<f32> hh;
    hh.reserve(CHUNK_VOX / 8);
    const u8 finest_hhh = static_cast<u8>((0u << 3) | 7u);  // level 0, orient HHH
    for (u32 i = 0; i < CHUNK_VOX; ++i)
        if (m.id[i] == finest_hhh) hh.push_back(std::fabs(coeffs[i]));
    if (hh.empty()) return 0.0;
    size_t mid = hh.size() / 2;
    std::nth_element(hh.begin(), hh.begin() + mid, hh.end());
    f64 median = hh[mid];
    return median / 0.6745;
}

// Per-subband signal-std estimate and BayesShrink threshold. For subband with
// observed variance var_obs and noise variance sigma^2:
//   sigma_signal = sqrt(max(0, var_obs - sigma^2));  T = sigma^2 / sigma_signal.
// If sigma_signal -> 0 (band is all noise) the threshold -> large => band fully
// suppressed. We compute per-subband variance in one pass.
struct NoiseShrink {
    f64 sigma = 0;                                  // chunk noise level
    std::array<std::array<f64, 8>, DWT_LEVELS> thr{};  // BayesShrink T per band

    static NoiseShrink analyze(std::span<const f32> coeffs) {
        NoiseShrink ns;
        ns.sigma = estimate_noise_sigma(coeffs);
        const SubbandMap& m = subband_map();
        // accumulate per-band sum, sumsq, count
        std::array<std::array<f64, 8>, DWT_LEVELS> s{}, sq{};
        std::array<std::array<u64, 8>, DWT_LEVELS> n{};
        for (u32 i = 0; i < CHUNK_VOX; ++i) {
            u32 lvl = m.level(i), o = m.orient(i);
            f64 c = coeffs[i];
            s[lvl][o] += c; sq[lvl][o] += c * c; ++n[lvl][o];
        }
        f64 s2 = ns.sigma * ns.sigma;
        for (u32 l = 0; l < DWT_LEVELS; ++l)
            for (u32 o = 0; o < 8; ++o) {
                if (n[l][o] == 0) { ns.thr[l][o] = 0; continue; }
                f64 mean = s[l][o] / n[l][o];
                f64 var = std::max(0.0, sq[l][o] / n[l][o] - mean * mean);
                f64 sig_sig = std::sqrt(std::max(0.0, var - s2));
                // never shrink the LLL approximation (carries the DC/structure)
                if (o == 0) { ns.thr[l][o] = 0; continue; }
                ns.thr[l][o] = (sig_sig > 1e-6) ? (s2 / sig_sig) : (3.0 * ns.sigma);
            }
        return ns;
    }

    // Fold the BayesShrink thresholds into a base StepTable: the effective
    // dead-zone half-width is max(base step/2, threshold). We realize that by
    // raising the per-subband step so its dead zone (step/2) covers the
    // threshold. strength in [0,1] scales how aggressively we apply shrinkage.
    StepTable apply(const StepTable& base, f64 strength = 1.0) const {
        StepTable t = base;
        for (u32 l = 0; l < DWT_LEVELS; ++l)
            for (u32 o = 0; o < 8; ++o) {
                f64 shrink_step = 2.0 * thr[l][o] * strength;   // step whose dead-zone = T
                t.step[l][o] = static_cast<f32>(std::max<f64>(t.step[l][o], shrink_step));
            }
        return t;
    }
};

// --- Perceptual / energy-preserving RDO (§4.10, encoder-only) ---------------
// Bias the HF-subband steps by the chunk's gradient COHERENCE, gated by the MAD
// noise floor. Real edges/texture are oriented (high structure-tensor
// anisotropy); noise is high-variance but incoherent (low anisotropy). So:
//   - high coherence above the noise floor  => REDUCE HF steps (preserve detail)
//   - low coherence / at the noise floor     => leave coarse (don't spend bits on noise)
// This directly counters "the wavelet throws away HF" without amplifying noise.
// Coherence is one cheap scalar per chunk from the spatial-domain structure
// tensor; no per-candidate metric evaluation.
struct PerceptualRDO {
    // Global gradient coherence in [0,1]: anisotropy (λ1-λ2)/(λ1+λ2) of the
    // accumulated 3D structure tensor (here the dominant-plane 2D tensor over
    // central slices, which is cheap and tracks oriented sheet structure well).
    [[nodiscard]] static f64 coherence(std::span<const u8> vox) {
        f64 Jxx = 0, Jyy = 0, Jxy = 0;
        for (u32 z = 0; z < CHUNK; z += 4)
            for (u32 y = 1; y + 1 < CHUNK; ++y)
                for (u32 x = 1; x + 1 < CHUNK; ++x) {
                    f64 gx = f64(vox[vox_index(z,y,x+1)]) - vox[vox_index(z,y,x-1)];
                    f64 gy = f64(vox[vox_index(z,y+1,x)]) - vox[vox_index(z,y-1,x)];
                    Jxx += gx*gx; Jyy += gy*gy; Jxy += gx*gy;
                }
        f64 tr = Jxx + Jyy;
        if (tr < 1e-9) return 0.0;
        f64 det = Jxx*Jyy - Jxy*Jxy;
        f64 disc = std::sqrt(std::max(0.0, tr*tr/4 - det));
        f64 l1 = tr/2 + disc, l2 = tr/2 - disc;
        return (l1 + l2 > 1e-9) ? (l1 - l2) / (l1 + l2) : 0.0;
    }

    // Apply HF-preservation: scale HF-subband steps by (1 - strength*coherence_gain)
    // where coherence_gain rises with coherence ABOVE a noise-floor gate. The LLL
    // and coarsest-level bands are left alone (they carry structure, not noise).
    // Returns a modified StepTable (steps the decoder reads — §4.1).
    static StepTable apply(const StepTable& base, std::span<const u8> vox,
                           f64 sigma, f64 strength = 0.5) {
        f64 coh = coherence(vox);
        // gate: only act when there's coherent structure AND it's above noise.
        // noise gate: more noise (large sigma) => demand higher coherence to act.
        f64 gate = coh / (coh + 0.15 + 0.01 * sigma);   // 0..~1, smooth
        f64 reduce = strength * gate;                    // fraction to tighten HF
        StepTable t = base;
        for (u32 l = 0; l < DWT_LEVELS; ++l)
            for (u32 o = 1; o < 8; ++o) {                // skip LLL (o==0)
                // tighten the finest levels most (where edge detail lives).
                f64 lvl_w = (l == 0) ? 1.0 : (l == 1 ? 0.6 : 0.3);
                f64 factor = 1.0 - reduce * lvl_w;       // <1 => finer step => more detail
                t.step[l][o] = static_cast<f32>(t.step[l][o] * std::max(0.5, factor));
            }
        return t;
    }
};

} // namespace c4d
