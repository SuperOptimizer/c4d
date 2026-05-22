// Dead-zone scalar quantization with per-subband, L2-weighted steps.
// Spec §4.1, §4.5. A single global quality knob `q` sets the base step; each
// subband's effective step is q * (1/synth_L2_gain) so a single q is
// near-MSE-optimal across subbands (the unit-L2 property, §4.5).
//
// The decoder reads the per-subband steps from the chunk (§4.1) — it never
// recomputes them — so the step *policy* below is encoder-side and retunable;
// only the dead-zone *mechanism* (carried steps) is frozen.
#pragma once
#include "core.hpp"
#include "subband.hpp"
#include "dwt_tables.hpp"
#include <array>
#include <cmath>
#include <span>
#include <vector>
#if __has_include(<experimental/simd>)
#  include <experimental/simd>
#  define C4D_QUANT_SIMD 1
#endif

namespace c4d {

// The 8 subband orientations across DWT_LEVELS levels, but only the meaningful
// (level, orient) pairs exist: for each level all 8 orients except that only the
// coarsest level uses orient 0 (LLL). We store a flat step table indexed by
// [level][orient]; the encoder fills it, the chunk carries it, decode reads it.
struct StepTable {
    std::array<std::array<f32, 8>, DWT_LEVELS> step{};

    // Build from a global quality knob using the L2 weights. Larger q => coarser.
    static StepTable from_q(f32 q) noexcept {
        StepTable t;
        for (u32 l = 0; l < DWT_LEVELS; ++l)
            for (u32 o = 0; o < 8; ++o)
                t.step[l][o] = q * dwt::QUANT_WEIGHTS.w[l][o];
        return t;
    }
};

// Dead-zone quantize one coefficient: the central [-step/2, step/2) bin maps to
// 0 (the dead zone), outer bins are uniform width `step`. Standard mid-tread
// dead-zone rule. Returns a signed integer level.
[[nodiscard]] inline i32 dead_zone_quantize(f32 c, f32 step) noexcept {
    if (step <= 0.f) return 0;
    f32 a = std::fabs(c);
    if (a < 0.5f * step) return 0;                  // dead zone
    i32 q = static_cast<i32>(std::floor(a / step - 0.5f)) + 1;
    return c < 0.f ? -q : q;
}

// Reconstruct a coefficient from its quantized level (mid-point of the bin).
[[nodiscard]] inline f32 dead_zone_dequantize(i32 q, f32 step) noexcept {
    if (q == 0) return 0.f;
    f32 a = (static_cast<f32>(std::abs(q))) * step;  // bin lower edge + half ~ q*step
    return q < 0 ? -a : a;
}

// Precomputed per-voxel subband id for the transformed cube (built once; reused
// across all chunks since the geometry is fixed). Stored as a packed
// (level<<3 | orient) byte per voxel.
struct SubbandMap {
    std::array<u8, CHUNK_VOX> id{};
    SubbandMap() noexcept {
        for (u32 z = 0; z < CHUNK; ++z)
            for (u32 y = 0; y < CHUNK; ++y)
                for (u32 x = 0; x < CHUNK; ++x) {
                    Subband sb = subband_of(z, y, x, DWT_LEVELS);
                    id[vox_index(z, y, x)] = static_cast<u8>((sb.level << 3) | sb.orient);
                }
    }
    [[nodiscard]] u32 level(u32 i)  const noexcept { return id[i] >> 3; }
    [[nodiscard]] u32 orient(u32 i) const noexcept { return id[i] & 7; }
};

// Process-wide singleton (the map is identical for every chunk).
[[nodiscard]] inline const SubbandMap& subband_map() noexcept {
    static const SubbandMap m;
    return m;
}

// Reusable per-thread scratch for the per-voxel step field — avoids a 256 KB
// heap alloc + zero-fill on every quantize/dequantize call (measured in the asm
// as an operator new/delete + memset per call). Thread-local keeps the codec
// reentrant/thread-safe (each calling thread gets its own buffer). The field is
// memoized against the StepTable that built it: a whole region encodes/decodes
// at one q, so the 256K fill happens once and is reused across all 64 chunks in
// both directions (the 40-float compare is far cheaper than the fill).
struct StepFieldCache {
    std::vector<f32> field{std::vector<f32>(CHUNK_VOX)};
    StepTable cached{};        // the table that produced `field`
    bool valid = false;
};
inline StepFieldCache& step_cache() noexcept {
    thread_local StepFieldCache c;
    return c;
}

[[nodiscard]] inline bool step_tables_equal(const StepTable& a, const StepTable& b) noexcept {
    for (u32 l = 0; l < DWT_LEVELS; ++l)
        for (u32 o = 0; o < 8; ++o) if (a.step[l][o] != b.step[l][o]) return false;
    return true;
}

// Per-voxel quantizer step for `t`, materialized into a flat array so the
// dead-zone loop is a contiguous SIMD pass (no per-voxel table indirection).
// Returns the cached field if `t` is unchanged from the last call; otherwise
// rebuilds it. The subband id per voxel is fixed geometry; only the 40 step
// values change per chunk.
[[nodiscard]] inline const f32* step_field(const StepTable& t) noexcept {
    StepFieldCache& c = step_cache();
    if (c.valid && step_tables_equal(c.cached, t)) return c.field.data();
    const SubbandMap& m = subband_map();
    std::array<f32, 256> lut{};   // flatten (level,orient) -> id-byte lookup
    for (u32 l = 0; l < DWT_LEVELS; ++l)
        for (u32 o = 0; o < 8; ++o) lut[(l << 3) | o] = t.step[l][o];
    f32* field = c.field.data();
    for (u32 i = 0; i < CHUNK_VOX; ++i) field[i] = lut[m.id[i]];
    c.cached = t; c.valid = true;
    return field;
}

// Quantize a whole transformed cube into integer levels using the step table.
// Returns the number of NONZERO levels — free here (this pass already visits
// every coefficient), and it lets the token generator pick a sparse vs dense
// code path (the high- vs low-compression regimes).
inline u32 quantize(std::span<const f32> coeffs, const StepTable& t,
                    std::span<i32> out) noexcept {
    std::span<const f32> step(step_field(t), CHUNK_VOX);
    u32 nnz = 0;
#ifdef C4D_QUANT_SIMD
    // Work entirely in the float domain (masks are native to vf), produce a
    // signed float level, then one cast to int. Matches dead_zone_quantize.
    namespace stdx = std::experimental;
    using vf = stdx::native_simd<f32>;
    using vi = stdx::rebind_simd_t<i32, vf>;
    constexpr u32 W = vf::size();
    u32 i = 0;
    for (; i + W <= CHUNK_VOX; i += W) {
        vf c(&coeffs[i], stdx::element_aligned);
        vf s(&step[i],   stdx::element_aligned);
        vf a = stdx::abs(c);
        vf mag = stdx::floor(a / s - 0.5f) + 1.0f;          // |level|
        vf sgn = vf(1.0f);
        where(c < 0.f, sgn) = -1.0f;
        vf qf = mag * sgn;
        where((a < 0.5f * s) || (s <= 0.f), qf) = 0.0f;     // dead zone / no step
        vi q = stdx::static_simd_cast<vi>(qf);
        q.copy_to(&out[i], stdx::element_aligned);
        nnz += stdx::popcount(q != 0);                      // nonzero lanes
    }
    for (; i < CHUNK_VOX; ++i) { i32 q = dead_zone_quantize(coeffs[i], step[i]); out[i] = q; nnz += (q != 0); }
#else
    for (u32 i = 0; i < CHUNK_VOX; ++i) { i32 q = dead_zone_quantize(coeffs[i], step[i]); out[i] = q; nnz += (q != 0); }
#endif
    return nnz;
}

// Quantize directly from the DWT's slice-padded buffer (dwt::forward_to_padded)
// into dense `out`, avoiding the 256KB unpack-to-dense pass. Each slice is
// CHUNK*CHUNK contiguous floats at `pad + z*slice_pitch`, mapping to the dense
// run out[z*CHUNK*CHUNK ..]; the inner SIMD loop is identical to quantize(),
// just run per-slice. Byte-identical result to forward()+quantize().
inline u32 quantize_padded(const f32* pad, u32 slice_pitch, const StepTable& t,
                           std::span<i32> out) noexcept {
    const f32* step = step_field(t);
    constexpr u32 SVOX = CHUNK * CHUNK;       // floats per dense slice
    u32 nnz = 0;
    for (u32 z = 0; z < CHUNK; ++z) {
        const f32* c = pad + z * slice_pitch;
        const f32* s = step + z * SVOX;
        i32* o = out.data() + z * SVOX;
#ifdef C4D_QUANT_SIMD
        namespace stdx = std::experimental;
        using vf = stdx::native_simd<f32>;
        using vi = stdx::rebind_simd_t<i32, vf>;
        constexpr u32 W = vf::size();
        u32 i = 0;
        for (; i + W <= SVOX; i += W) {
            vf cc(&c[i], stdx::element_aligned);
            vf ss(&s[i], stdx::element_aligned);
            vf a = stdx::abs(cc);
            vf mag = stdx::floor(a / ss - 0.5f) + 1.0f;
            vf sgn = vf(1.0f); where(cc < 0.f, sgn) = -1.0f;
            vf qf = mag * sgn;
            where((a < 0.5f * ss) || (ss <= 0.f), qf) = 0.0f;
            vi q = stdx::static_simd_cast<vi>(qf);
            q.copy_to(&o[i], stdx::element_aligned);
            nnz += stdx::popcount(q != 0);
        }
        for (; i < SVOX; ++i) { i32 q = dead_zone_quantize(c[i], s[i]); o[i] = q; nnz += (q != 0); }
#else
        for (u32 i = 0; i < SVOX; ++i) { i32 q = dead_zone_quantize(c[i], s[i]); o[i] = q; nnz += (q != 0); }
#endif
    }
    return nnz;
}

// Dequantize integer levels back to coefficients (out = q * step, mid-tread).
inline void dequantize(std::span<const i32> q, const StepTable& t,
                       std::span<f32> out) noexcept {
    std::span<const f32> step(step_field(t), CHUNK_VOX);
#ifdef C4D_QUANT_SIMD
    namespace stdx = std::experimental;
    using vf = stdx::native_simd<f32>;
    using vi = stdx::rebind_simd_t<i32, vf>;
    constexpr u32 W = vf::size();
    u32 i = 0;
    for (; i + W <= CHUNK_VOX; i += W) {
        vi qi(&q[i], stdx::element_aligned);
        vf qf = stdx::static_simd_cast<vf>(qi);
        vf s(&step[i], stdx::element_aligned);
        (qf * s).copy_to(&out[i], stdx::element_aligned);    // 0*s = 0, sign preserved
    }
    for (; i < CHUNK_VOX; ++i) out[i] = dead_zone_dequantize(q[i], step[i]);
#else
    for (u32 i = 0; i < CHUNK_VOX; ++i) out[i] = dead_zone_dequantize(q[i], step[i]);
#endif
}

// Dequantize dense `q` directly INTO the DWT's slice-padded buffer, avoiding the
// 256KB pack-to-padded that dwt::inverse would otherwise do. Each dense slice
// q[z*CHUNK*CHUNK ..] writes the padded run pad[z*slice_pitch ..]; inner SIMD is
// identical to dequantize(), per-slice. Byte-identical to dequantize()+pack.
inline void dequantize_padded(std::span<const i32> q, u32 slice_pitch,
                              const StepTable& t, f32* pad) noexcept {
    const f32* step = step_field(t);
    constexpr u32 SVOX = CHUNK * CHUNK;
    for (u32 z = 0; z < CHUNK; ++z) {
        const i32* qi = q.data() + z * SVOX;
        const f32* s = step + z * SVOX;
        f32* o = pad + z * slice_pitch;
#ifdef C4D_QUANT_SIMD
        namespace stdx = std::experimental;
        using vf = stdx::native_simd<f32>;
        using vi = stdx::rebind_simd_t<i32, vf>;
        constexpr u32 W = vf::size();
        u32 i = 0;
        for (; i + W <= SVOX; i += W) {
            vi qq(&qi[i], stdx::element_aligned);
            vf qf = stdx::static_simd_cast<vf>(qq);
            vf ss(&s[i], stdx::element_aligned);
            (qf * ss).copy_to(&o[i], stdx::element_aligned);
        }
        for (; i < SVOX; ++i) o[i] = dead_zone_dequantize(qi[i], s[i]);
#else
        for (u32 i = 0; i < SVOX; ++i) o[i] = dead_zone_dequantize(qi[i], s[i]);
#endif
    }
}

} // namespace c4d
