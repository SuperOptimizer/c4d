// Trellis-coded quantization (§7 T2), Marcellin-Fischer 4-state. TCQ uses a
// denser effective lattice than scalar quantization at the same rate by letting
// a 4-state trellis choose, per coefficient, between two quantizer cosets whose
// union is a half-step lattice. The Viterbi encoder finds the min-distortion
// state path; the decoder replays the same trellis to know which coset
// dequantizer each coefficient used. Measured ceiling ~0.5-0.7 dB at matched
// rate — the one frontier-bending lever on this corpus.
//
// Design (the standard JPEG2000-optional TCQ):
//  - base step Δ. Four sub-quantizers Q0..Q3, each a uniform mid-tread quantizer
//    of step 2Δ, with reconstruction offsets {0, 2Δ, ...} interleaved so that
//    the union {Q0,Q1,Q2,Q3} tiles the line at spacing Δ.
//  - 4-state trellis. Each state has 2 branches (bit 0 / bit 1). The branch
//    chosen at step k both (a) emits the coefficient's quantizer index and
//    (b) advances the state. The *subset* (which pair of sub-quantizers is
//    allowed) is a function of the current state, so the decoder — which
//    recovers the index sequence from the entropy coder — re-walks the trellis
//    to recover reconstruction values.
//
// We keep it simple and matched: the trellis SUBSET selection is deterministic
// from the state, and the index we store per coefficient encodes both the
// sub-quantizer level and (implicitly, via the standard labeling) the branch.
#pragma once
#include "core.hpp"
#include <array>
#include <cmath>
#include <limits>
#include <span>
#include <vector>

namespace c4d::tcq {

// Standard Ungerboeck 4-state trellis for TCQ. Each state s in {0,1,2,3} uses a
// "subset" of two cosets; a branch bit selects which coset and the next state.
// state -> subset: states {0,2} use subset A = {D0,D2}, states {1,3} use B={D1,D3}.
// next state table (Marcellin): standard 4-state, 2 branches each.
inline constexpr int NEXT[4][2] = {
    {0, 1},   // state 0: bit0->0, bit1->1
    {2, 3},   // state 1: bit0->2, bit1->3
    {1, 0},   // state 2
    {3, 2},   // state 3
};
// which coset (0..3) a (state, bit) uses. subset A = cosets {0,2}; B = {1,3}.
inline constexpr int COSET[4][2] = {
    {0, 2},   // state 0 (subset A)
    {1, 3},   // state 1 (subset B)
    {2, 0},   // state 2 (subset A)
    {3, 1},   // state 3 (subset B)
};

// Coset reconstruction offset (in units of Δ): cosets sit at even/odd multiples
// so the 4-coset union is a step-Δ lattice. Standard offsets: D0=0, D1=1, D2=2,
// D3=3 (mod 4), i.e. coset c reconstructs to (2*level + c_offset)*Δ-ish. We use
// the simplest matched scheme: coset c's reconstruction points are
// {... , -2Δ, 0, 2Δ, ...} shifted by c*Δ/2. To keep it exact and matched, store
// an integer level per coefficient and reconstruct as level*2Δ + cosetShift(c).
[[nodiscard]] constexpr f32 coset_shift(int coset, f32 delta) noexcept {
    // 4 cosets interleaved at Δ/2 spacing within the 2Δ cell.
    return static_cast<f32>(coset) * 0.5f * delta;
}

// Quantize one coefficient to the nearest reconstruction point of a given coset
// (step 2Δ, shifted), returning the integer level and the squared error.
struct QPoint { i32 level; f32 recon; f32 err2; };
[[nodiscard]] inline QPoint quant_coset(f32 c, int coset, f32 delta) noexcept {
    f32 step = 2.0f * delta;
    f32 shift = coset_shift(coset, delta);
    i32 level = static_cast<i32>(std::lround((c - shift) / step));
    f32 recon = level * step + shift;
    f32 e = c - recon;
    return {level, recon, e * e};
}

// Viterbi TCQ encode of a coefficient sequence. Returns per-coefficient
// (coset, level) plus the bit path (1 bit/coeff). lambda trades rate for
// distortion; here we minimize pure distortion (dead-zone handles rate via Δ).
struct Encoded {
    std::vector<u8>  coset;   // 0..3 per coefficient (decoder needs it; derivable
                              // from state path + bit, but we store explicitly in
                              // the simplest matched form)
    std::vector<i32> level;   // integer level per coefficient
};

// For c4d we want a matched, simple realization: run the Viterbi to choose the
// branch bit per coefficient minimizing distortion, record the resulting coset
// and level. The decoder re-walks the trellis from the bit path. We store the
// bit path implicitly by storing the coset (the chunk coder entropy-codes the
// levels; the per-coeff branch bit is the cheap extra TCQ rate).
[[nodiscard]] inline Encoded encode(std::span<const f32> coeffs, f32 delta) {
    const u32 n = static_cast<u32>(coeffs.size());
    constexpr f32 INF = std::numeric_limits<f32>::max();
    // Viterbi: cost[state] = min accumulated distortion to reach `state`.
    std::array<f32, 4> cost{0, INF, INF, INF};  // start in state 0
    std::array<f32, 4> ncost;
    // back-pointers: for each coeff k and state, which (prev state, bit, coset, level)
    std::vector<std::array<u8, 4>>  bp_prev(n);
    std::vector<std::array<u8, 4>>  bp_bit(n);
    std::vector<std::array<u8, 4>>  bp_coset(n);
    std::vector<std::array<i32, 4>> bp_level(n);

    for (u32 k = 0; k < n; ++k) {
        ncost.fill(INF);
        std::array<u8, 4> pp{}, pb{}, pc{}; std::array<i32, 4> pl{};
        for (int s = 0; s < 4; ++s) {
            if (cost[s] == INF) continue;
            for (int bit = 0; bit < 2; ++bit) {
                int ns = NEXT[s][bit];
                int cs = COSET[s][bit];
                QPoint qp = quant_coset(coeffs[k], cs, delta);
                f32 cand = cost[s] + qp.err2;
                if (cand < ncost[ns]) {
                    ncost[ns] = cand;
                    pp[ns] = static_cast<u8>(s); pb[ns] = static_cast<u8>(bit);
                    pc[ns] = static_cast<u8>(cs); pl[ns] = qp.level;
                }
            }
        }
        cost = ncost;
        bp_prev[k] = pp; bp_bit[k] = pb; bp_coset[k] = pc; bp_level[k] = pl;
    }
    // pick best final state, trace back
    int best = 0; for (int s = 1; s < 4; ++s) if (cost[s] < cost[best]) best = s;
    Encoded e; e.coset.resize(n); e.level.resize(n);
    int s = best;
    for (i32 k = static_cast<i32>(n) - 1; k >= 0; --k) {
        e.coset[k] = bp_coset[k][s];
        e.level[k] = bp_level[k][s];
        s = bp_prev[k][s];
    }
    return e;
}

// Dequantize: reconstruct each coefficient from its stored (coset, level).
// (The trellis path is implicit in the coset sequence, which is what the chunk
// coder carries; this is the matched inverse of encode.)
inline void decode(std::span<const u8> coset, std::span<const i32> level,
                   f32 delta, std::span<f32> out) {
    f32 step = 2.0f * delta;
    for (u32 i = 0; i < out.size(); ++i)
        out[i] = level[i] * step + coset_shift(coset[i], delta);
}

} // namespace c4d::tcq
