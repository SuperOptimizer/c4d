// c4d 9/7 lifting constants + per-subband synthesis-L2 weight table, all
// computed at compile time. Spec §4.1, §4.5, §4.9.
//
// The CDF 9/7 lifting scheme (Daubechies-Sweldens) with the standard four
// lifting steps and the K scaling. We use the *unit-L2-normalized* convention
// (§4.5): the analysis low/high outputs are scaled so a single global quantizer
// step yields near-MSE-optimal allocation. Because the 9/7 lifting is only
// *approximately* orthonormal, the synthesis basis L2 gain differs per subband;
// we bake the exact per-subband multiplier so coefficient-domain L2 ≈
// reconstructed-domain L2.
#pragma once
#include "core.hpp"
#include <array>
#include <cmath>

namespace c4d::dwt {

// Standard CDF 9/7 lifting coefficients (Cohen-Daubechies-Feauveau, the JPEG2000
// irreversible filter). Predict/update alternate; K is the scaling normalizer.
inline constexpr f64 ALPHA = -1.586134342059924;
inline constexpr f64 BETA  = -0.052980118572961;
inline constexpr f64 GAMMA =  0.882911075530934;
inline constexpr f64 DELTA =  0.443506852043971;
inline constexpr f64 KAPPA =  1.230174104914001;   // scaling for the high band

// In the lifting normalization, after the four lifts the even (low) samples are
// multiplied by 1/KAPPA and the odd (high) samples by KAPPA. That is *not*
// unit-L2; the per-orientation analysis L2 gains below correct toward unit-L2.
//
// For a single separable 1D step, the low-pass and high-pass analysis L2 gains
// (energy multipliers) under this lifting are, to machine precision:
inline constexpr f64 L1_LOW_GAIN  = 1.0 / KAPPA;   // applied to even samples
inline constexpr f64 L1_HIGH_GAIN = KAPPA;         // applied to odd samples

// --- Per-subband synthesis L2 weight (§4.5) -------------------------------
// A 3D separable transform tags each subband by its (z,y,x) orientation triple
// of L (low) / H (high) per axis, at a given decomposition level. The synthesis
// L2 gain of a subband is the product of the per-axis 1D synthesis gains, and
// each additional coarse level multiplies the LLL path again.
//
// We index subbands by (level, orient) where orient in [0,8): bit0=x, bit1=y,
// bit2=z, set => high on that axis. orient 0 at the coarsest level is the final
// LLL approximation (DC-band of the pyramid).

// 1D synthesis L2 gain per orientation bit (inverse of analysis gain — the
// reconstruction multiplier). Low synthesis gain = KAPPA, high = 1/KAPPA, since
// synthesis inverts the analysis scaling.
[[nodiscard]] constexpr f64 axis_synth_gain(bool high) noexcept {
    return high ? (1.0 / L1_HIGH_GAIN) : (1.0 / L1_LOW_GAIN);
}

// Per-subband synthesis L2 weight: product over the 3 axes of this level's
// orientation, times the accumulated low-pass synthesis gain of all coarser
// levels above it (each coarser level applies an extra LLL synthesis on all 3
// axes to this subband's spatial support).
[[nodiscard]] constexpr f64 subband_synth_l2(u32 level, u32 orient) noexcept {
    // level 0 = finest (one DWT step from full res); larger = coarser.
    f64 g = 1.0;
    g *= axis_synth_gain(orient & 0b001);  // x
    g *= axis_synth_gain(orient & 0b010);  // y
    g *= axis_synth_gain(orient & 0b100);  // z
    // Each coarser level this subband sits under contributes an LLL synthesis
    // (all three axes low). There are `level` coarser steps below it.
    const f64 lll = axis_synth_gain(false) * axis_synth_gain(false)
                  * axis_synth_gain(false);
    for (u32 l = 0; l < level; ++l) g *= lll;
    return g;
}

// Per-subband quantizer weight: to make coefficient-domain L2 ≈ recon-domain L2,
// the *step* for a subband scales inversely with its synthesis L2 gain, so that
// a single global q is MSE-optimal across subbands. weight = 1 / synth_l2.
[[nodiscard]] constexpr f64 subband_quant_weight(u32 level, u32 orient) noexcept {
    return 1.0 / subband_synth_l2(level, orient);
}

// Baked table of quant weights for every (level, orient) up to DWT_LEVELS.
// Index: [level][orient]. orient 0 at level==DWT_LEVELS-1 is the LLL approx.
struct QuantWeights {
    std::array<std::array<f32, 8>, DWT_LEVELS> w{};
};

[[nodiscard]] consteval QuantWeights make_quant_weights() noexcept {
    QuantWeights q{};
    for (u32 lvl = 0; lvl < DWT_LEVELS; ++lvl)
        for (u32 o = 0; o < 8; ++o)
            q.w[lvl][o] = static_cast<f32>(subband_quant_weight(lvl, o));
    return q;
}

inline constexpr QuantWeights QUANT_WEIGHTS = make_quant_weights();

} // namespace c4d::dwt
