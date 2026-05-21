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

namespace c4d::dwt {

// In-place 1D forward 9/7 lifting on a strided line of length n (n even).
// Whole-sample symmetric extension: index reflection x[-1]=x[1], x[n]=x[n-2]
// (mirror about the boundary sample). After lifting, even indices hold the
// low band, odd indices the high band (deinterleaving is done by the caller's
// pyramid packing — here we keep the in-place interleaved form).
inline void fwd_1d(f32* p, u32 n, u32 stride) noexcept {
    auto at = [&](i64 i) -> f32& {
        // mirror about endpoints (whole-sample symmetric): reflect into [0,n-1].
        if (i < 0) i = -i;
        if (i >= static_cast<i64>(n)) i = 2 * (static_cast<i64>(n) - 1) - i;
        return p[static_cast<u32>(i) * stride];
    };
    const f64 a = ALPHA, b = BETA, g = GAMMA, d = DELTA, k = KAPPA;
    // predict 1 (odd += a*(even neighbors))
    for (i64 i = 1; i < n; i += 2) at(i) += static_cast<f32>(a * (at(i - 1) + at(i + 1)));
    // update 1 (even += b*(odd neighbors))
    for (i64 i = 0; i < n; i += 2) at(i) += static_cast<f32>(b * (at(i - 1) + at(i + 1)));
    // predict 2
    for (i64 i = 1; i < n; i += 2) at(i) += static_cast<f32>(g * (at(i - 1) + at(i + 1)));
    // update 2
    for (i64 i = 0; i < n; i += 2) at(i) += static_cast<f32>(d * (at(i - 1) + at(i + 1)));
    // scale: even (low) *= 1/k, odd (high) *= k
    for (u32 i = 0; i < n; i += 2) p[i * stride] = static_cast<f32>(p[i * stride] * (1.0 / k));
    for (u32 i = 1; i < n; i += 2) p[i * stride] = static_cast<f32>(p[i * stride] * k);
}

// In-place 1D inverse 9/7 lifting — exact reverse of fwd_1d.
inline void inv_1d(f32* p, u32 n, u32 stride) noexcept {
    auto at = [&](i64 i) -> f32& {
        if (i < 0) i = -i;
        if (i >= static_cast<i64>(n)) i = 2 * (static_cast<i64>(n) - 1) - i;
        return p[static_cast<u32>(i) * stride];
    };
    const f64 a = ALPHA, b = BETA, g = GAMMA, d = DELTA, k = KAPPA;
    // undo scale
    for (u32 i = 0; i < n; i += 2) p[i * stride] = static_cast<f32>(p[i * stride] * k);
    for (u32 i = 1; i < n; i += 2) p[i * stride] = static_cast<f32>(p[i * stride] * (1.0 / k));
    // undo update 2
    for (i64 i = 0; i < n; i += 2) at(i) -= static_cast<f32>(d * (at(i - 1) + at(i + 1)));
    // undo predict 2
    for (i64 i = 1; i < n; i += 2) at(i) -= static_cast<f32>(g * (at(i - 1) + at(i + 1)));
    // undo update 1
    for (i64 i = 0; i < n; i += 2) at(i) -= static_cast<f32>(b * (at(i - 1) + at(i + 1)));
    // undo predict 1
    for (i64 i = 1; i < n; i += 2) at(i) -= static_cast<f32>(a * (at(i - 1) + at(i + 1)));
}

// After an in-place interleaved lifting pass over a line of length n, separate
// the low (even) and high (odd) samples into the contiguous Mallat halves:
// [low(0..n/2) | high(0..n/2)]. Inverse interleaves them back.
inline void deinterleave(f32* p, u32 n, u32 stride, f32* tmp) noexcept {
    const u32 h = n / 2;
    for (u32 i = 0; i < h; ++i) { tmp[i] = p[(2 * i) * stride]; tmp[h + i] = p[(2 * i + 1) * stride]; }
    for (u32 i = 0; i < n; ++i) p[i * stride] = tmp[i];
}
inline void interleave(f32* p, u32 n, u32 stride, f32* tmp) noexcept {
    const u32 h = n / 2;
    for (u32 i = 0; i < h; ++i) { tmp[2 * i] = p[i * stride]; tmp[2 * i + 1] = p[(h + i) * stride]; }
    for (u32 i = 0; i < n; ++i) p[i * stride] = tmp[i];
}

// One separable forward DWT step over the leading s^3 sub-cube of a CHUNK^3
// buffer, transforming all three axes and packing into Mallat octant layout.
inline void fwd_step(f32* vol, u32 s) noexcept {
    std::array<f32, CHUNK> tmp{};
    // X axis (stride 1)
    for (u32 z = 0; z < s; ++z)
        for (u32 y = 0; y < s; ++y) {
            f32* line = vol + (z * CHUNK + y) * CHUNK;
            fwd_1d(line, s, 1);
            deinterleave(line, s, 1, tmp.data());
        }
    // Y axis (stride CHUNK)
    for (u32 z = 0; z < s; ++z)
        for (u32 x = 0; x < s; ++x) {
            f32* line = vol + z * CHUNK * CHUNK + x;
            fwd_1d(line, s, CHUNK);
            deinterleave(line, s, CHUNK, tmp.data());
        }
    // Z axis (stride CHUNK*CHUNK)
    for (u32 y = 0; y < s; ++y)
        for (u32 x = 0; x < s; ++x) {
            f32* line = vol + y * CHUNK + x;
            fwd_1d(line, s, CHUNK * CHUNK);
            deinterleave(line, s, CHUNK * CHUNK, tmp.data());
        }
}

// One separable inverse DWT step (exact reverse of fwd_step).
inline void inv_step(f32* vol, u32 s) noexcept {
    std::array<f32, CHUNK> tmp{};
    // Z axis
    for (u32 y = 0; y < s; ++y)
        for (u32 x = 0; x < s; ++x) {
            f32* line = vol + y * CHUNK + x;
            interleave(line, s, CHUNK * CHUNK, tmp.data());
            inv_1d(line, s, CHUNK * CHUNK);
        }
    // Y axis
    for (u32 z = 0; z < s; ++z)
        for (u32 x = 0; x < s; ++x) {
            f32* line = vol + z * CHUNK * CHUNK + x;
            interleave(line, s, CHUNK, tmp.data());
            inv_1d(line, s, CHUNK);
        }
    // X axis
    for (u32 z = 0; z < s; ++z)
        for (u32 y = 0; y < s; ++y) {
            f32* line = vol + (z * CHUNK + y) * CHUNK;
            interleave(line, s, 1, tmp.data());
            inv_1d(line, s, 1);
        }
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
