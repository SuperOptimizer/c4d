// c4d separable 3D CDF 9/7 lifting DWT, float32, unit-L2 normalized, with
// whole-sample symmetric (mirror) boundary extension at chunk faces.
// Spec §4.1, §4.4 (mirror REQUIRED), §4.5 (unit-L2).
//
// Layout: the transform operates in place on a dense CHUNK^3 float buffer in
// Z,Y,X order (index = (z*CHUNK + y)*CHUNK + x). After a forward transform the
// subbands occupy the standard Mallat pyramid layout: the level-0 high bands
// fill the outer half-cubes, with each coarser level nested in the LLL octant.
//
// This is the correctness reference: a straightforward separable lifting along
// each axis. The fused single-loop "cube core" + SIMD variant (§4.9) is a drop-in
// replacement to be added once roundtrip parity is locked.
#pragma once
#include "core.hpp"
#include "dwt_tables.hpp"
#include <array>
#include <cstring>
#include <vector>
#if __has_include(<experimental/simd>)
#  include <experimental/simd>
#  define C4D_HAVE_SIMD 1
namespace c4d::dwt { namespace stdx = std::experimental; }
#endif

namespace c4d::dwt {

// One forward lifting step over a CONTIGUOUS line `v` of length n (n even):
// `c * (left + right)` accumulated into the target parity. The interior runs
// branch-free; only the two boundary samples consult the mirror (x[-1]=x[1],
// x[n]=x[n-2], whole-sample symmetric). `odd` selects which parity is updated.
template <bool ODD>
inline void lift_pass(f32* __restrict v, u32 n, f64 c) noexcept {
    const u32 start = ODD ? 1u : 0u;
    // boundary-aware first/last targets; interior is branch-free.
    if constexpr (ODD) {
        // odd targets: i=1..n-1 step 2; neighbors i-1,i+1 always in range for
        // i<=n-3; the last odd index n-1 needs the mirror (i+1 == n -> n-2).
        for (u32 i = 1; i + 1 < n; i += 2)
            v[i] += static_cast<f32>(c * (f64(v[i - 1]) + v[i + 1]));
        u32 last = n - 1;  // odd (n even)
        v[last] += static_cast<f32>(c * (f64(v[last - 1]) + v[last - 1]));  // mirror
    } else {
        // even targets: i=0..n-2 step 2; i=0 needs mirror (i-1 == -1 -> 1).
        v[0] += static_cast<f32>(c * (f64(v[1]) + v[1]));                   // mirror
        for (u32 i = 2; i < n; i += 2)
            v[i] += static_cast<f32>(c * (f64(v[i - 1]) + v[i + 1]));
    }
    (void)start;
}

// Forward 1D 9/7 lifting on a strided line, packing the result directly into
// the Mallat halves: gather strided -> lift branch-free in contiguous scratch
// -> scatter the even (low) samples to the first half and odd (high) to the
// second half of the strided output. One strided read + one strided write per
// axis (the separate deinterleave pass is folded in here).
inline void fwd_1d(f32* p, u32 n, u32 stride) noexcept {
    f32 v[CHUNK];
    for (u32 i = 0; i < n; ++i) v[i] = p[i * stride];
    lift_pass<true >(v, n, ALPHA);
    lift_pass<false>(v, n, BETA);
    lift_pass<true >(v, n, GAMMA);
    lift_pass<false>(v, n, DELTA);
    const f32 lo = static_cast<f32>(1.0 / KAPPA), hi = static_cast<f32>(KAPPA);
    const u32 h = n / 2;
    for (u32 i = 0; i < h; ++i) {
        p[i * stride]       = v[2 * i]     * lo;   // low band -> first half
        p[(h + i) * stride] = v[2 * i + 1] * hi;   // high band -> second half
    }
}

#ifdef C4D_HAVE_SIMD
// --- SIMD multi-line lifting (§4.9) ----------------------------------------
// Process W parallel lines at once, one SIMD lane per line. The W lines are W
// consecutive innermost-x positions (lane_stride 1 => contiguous loads), each
// line stepping by `stride` along the transform axis. Used for the Y and Z
// passes where the scalar strided gather/scatter dominated. Bit-for-bit
// equivalent to running fwd_1d/inv_1d on each of the W lines (modulo f32
// associativity, which -ffast-math already permits).
using vf = stdx::native_simd<f32>;
inline constexpr u32 VW = vf::size();

template <bool ODD>
inline void lift_pass_v(vf* __restrict v, u32 n, f32 c) noexcept {
    if constexpr (ODD) {
        for (u32 i = 1; i + 1 < n; i += 2) v[i] += c * (v[i - 1] + v[i + 1]);
        u32 last = n - 1;
        v[last] += c * (v[last - 1] + v[last - 1]);          // mirror
    } else {
        v[0] += c * (v[1] + v[1]);                           // mirror
        for (u32 i = 2; i < n; i += 2) v[i] += c * (v[i - 1] + v[i + 1]);
    }
}

// Forward lifting of VW lines starting at `base`, axis stride `stride`.
inline void fwd_1d_v(f32* base, u32 n, u32 stride) noexcept {
    vf v[CHUNK];
    for (u32 i = 0; i < n; ++i) v[i] = vf(&base[i * stride], stdx::element_aligned);
    lift_pass_v<true >(v, n, f32(ALPHA));
    lift_pass_v<false>(v, n, f32(BETA));
    lift_pass_v<true >(v, n, f32(GAMMA));
    lift_pass_v<false>(v, n, f32(DELTA));
    const f32 lo = f32(1.0 / KAPPA), hi = f32(KAPPA);
    const u32 h = n / 2;
    for (u32 i = 0; i < h; ++i) {
        (v[2 * i]     * lo).copy_to(&base[i * stride],       stdx::element_aligned);
        (v[2 * i + 1] * hi).copy_to(&base[(h + i) * stride], stdx::element_aligned);
    }
}

inline void inv_1d_v(f32* base, u32 n, u32 stride) noexcept {
    vf v[CHUNK];
    const f32 lo = f32(KAPPA), hi = f32(1.0 / KAPPA);
    const u32 h = n / 2;
    for (u32 i = 0; i < h; ++i) {
        v[2 * i]     = vf(&base[i * stride],       stdx::element_aligned) * lo;
        v[2 * i + 1] = vf(&base[(h + i) * stride], stdx::element_aligned) * hi;
    }
    lift_pass_v<false>(v, n, f32(-DELTA));
    lift_pass_v<true >(v, n, f32(-GAMMA));
    lift_pass_v<false>(v, n, f32(-BETA));
    lift_pass_v<true >(v, n, f32(-ALPHA));
    for (u32 i = 0; i < n; ++i) v[i].copy_to(&base[i * stride], stdx::element_aligned);
}
#endif

// Inverse 1D 9/7 lifting — exact reverse of fwd_1d, including the interleave:
// gather the two Mallat halves back into interleaved order, undo scale, undo
// the four lifts, scatter back.
inline void inv_1d(f32* p, u32 n, u32 stride) noexcept {
    f32 v[CHUNK];
    const f32 lo = static_cast<f32>(KAPPA), hi = static_cast<f32>(1.0 / KAPPA);
    const u32 h = n / 2;
    for (u32 i = 0; i < h; ++i) {
        v[2 * i]     = p[i * stride]       * lo;   // low half -> even
        v[2 * i + 1] = p[(h + i) * stride] * hi;   // high half -> odd
    }
    lift_pass<false>(v, n, -DELTA);
    lift_pass<true >(v, n, -GAMMA);
    lift_pass<false>(v, n, -BETA);
    lift_pass<true >(v, n, -ALPHA);
    for (u32 i = 0; i < n; ++i) p[i * stride] = v[i];
}

// One separable forward DWT step over the leading s^3 sub-cube of a CHUNK^3
// buffer, transforming all three axes and packing into Mallat octant layout.
// fwd_1d folds the deinterleave into its scatter, so each axis is one pass.
inline void fwd_step(f32* vol, u32 s) noexcept {
    // X axis (stride 1): contiguous lines, scalar gather is already cache-good.
    for (u32 z = 0; z < s; ++z)
        for (u32 y = 0; y < s; ++y)
            fwd_1d(vol + (z * CHUNK + y) * CHUNK, s, 1);
    // Y axis (stride CHUNK): VW consecutive x per SIMD pass (contiguous loads),
    // scalar remainder. Z axis (stride CHUNK^2) likewise.
#ifdef C4D_HAVE_SIMD
    for (u32 z = 0; z < s; ++z) {
        u32 x = 0;
        for (; x + VW <= s; x += VW) fwd_1d_v(vol + z * CHUNK * CHUNK + x, s, CHUNK);
        for (; x < s; ++x)           fwd_1d  (vol + z * CHUNK * CHUNK + x, s, CHUNK);
    }
    for (u32 y = 0; y < s; ++y) {
        u32 x = 0;
        for (; x + VW <= s; x += VW) fwd_1d_v(vol + y * CHUNK + x, s, CHUNK * CHUNK);
        for (; x < s; ++x)           fwd_1d  (vol + y * CHUNK + x, s, CHUNK * CHUNK);
    }
#else
    for (u32 z = 0; z < s; ++z) for (u32 x = 0; x < s; ++x) fwd_1d(vol + z*CHUNK*CHUNK + x, s, CHUNK);
    for (u32 y = 0; y < s; ++y) for (u32 x = 0; x < s; ++x) fwd_1d(vol + y*CHUNK + x, s, CHUNK*CHUNK);
#endif
}

// One separable inverse DWT step (exact reverse of fwd_step).
inline void inv_step(f32* vol, u32 s) noexcept {
#ifdef C4D_HAVE_SIMD
    for (u32 y = 0; y < s; ++y) {
        u32 x = 0;
        for (; x + VW <= s; x += VW) inv_1d_v(vol + y * CHUNK + x, s, CHUNK * CHUNK);
        for (; x < s; ++x)           inv_1d  (vol + y * CHUNK + x, s, CHUNK * CHUNK);
    }
    for (u32 z = 0; z < s; ++z) {
        u32 x = 0;
        for (; x + VW <= s; x += VW) inv_1d_v(vol + z * CHUNK * CHUNK + x, s, CHUNK);
        for (; x < s; ++x)           inv_1d  (vol + z * CHUNK * CHUNK + x, s, CHUNK);
    }
#else
    for (u32 y = 0; y < s; ++y) for (u32 x = 0; x < s; ++x) inv_1d(vol + y*CHUNK + x, s, CHUNK*CHUNK);
    for (u32 z = 0; z < s; ++z) for (u32 x = 0; x < s; ++x) inv_1d(vol + z*CHUNK*CHUNK + x, s, CHUNK);
#endif
    // X axis (stride 1)
    for (u32 z = 0; z < s; ++z)
        for (u32 y = 0; y < s; ++y)
            inv_1d(vol + (z * CHUNK + y) * CHUNK, s, 1);
}

// Full multi-level forward transform over a CHUNK^3 buffer. Each level halves
// the active sub-cube (recursing into the LLL octant), `levels` deep.
inline void forward(f32* vol, u32 levels = DWT_LEVELS) noexcept {
    u32 s = CHUNK;
    for (u32 l = 0; l < levels; ++l) { fwd_step(vol, s); s /= 2; }
}

// Full multi-level inverse transform (coarsest level first).
inline void inverse(f32* vol, u32 levels = DWT_LEVELS) noexcept {
    // Recompute the sub-cube size at the coarsest level, then grow back.
    std::array<u32, 16> sizes{};
    u32 s = CHUNK;
    for (u32 l = 0; l < levels; ++l) { sizes[l] = s; s /= 2; }
    for (i32 l = static_cast<i32>(levels) - 1; l >= 0; --l) inv_step(vol, sizes[static_cast<u32>(l)]);
}

} // namespace c4d::dwt
