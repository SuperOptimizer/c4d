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
#if defined(__AVX512F__)
#  include <immintrin.h>
#  define C4D_AVX512_TRANSPOSE 1
#endif

namespace c4d::dwt {

// Slice (z) pitch padding (deep cache fix). The Z/Y separable passes stride by
// CHUNK^2 floats; for CHUNK=64 that's 4096 floats = 16 KiB, an exact power of
// two, so every voxel in a z-column maps to the same handful of L1 cache sets
// (set-aliasing) — cachegrind measured the Z pass at 27% L1 miss vs 1.2% with a
// padded stride (23x fewer misses). Pad the per-slice pitch by a non-power-of-2
// so z-columns scatter across all sets. Rows stay dense (CHUNK), so x/y passes
// and the Mallat octant packing within a slice are unchanged. The codec's
// external buffer stays dense CHUNK_VOX; forward()/inverse() repack in and out.
inline constexpr u32 SLICE_PAD = 8;                        // floats, non-pow2 sum
inline constexpr u32 SLICE     = CHUNK * CHUNK + SLICE_PAD; // padded z-stride
inline constexpr u32 PADDED_VOX = CHUNK * SLICE;           // padded buffer size

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

// --- X-pass transpose tile (VW rows x n positions) -------------------------
// The X axis is stride-1, so to lift VW rows in parallel (one row per SIMD lane)
// we must transpose: tile[i][l] = rows[l*rowstride + i]. The scalar nested loop
// was ~8% of all codec instructions (cachegrind). On AVX-512 a register 16x16
// transpose replaces it; n is a multiple of 16 here (CHUNK and its halves), so
// the n x VW gather is ceil(n/16) independent 16x16 transposes. Fallback to the
// scalar loop on non-AVX512 / VW!=16 (keeps the portable build correct).
#if defined(C4D_AVX512_TRANSPOSE)
// Transpose one 16x16 f32 block: src[r*src_stride + c] -> dst[c*16 + r].
inline void transpose16x16(const f32* src, u32 src_stride, f32* dst16) noexcept {
    __m512 r[16];
    for (int i = 0; i < 16; ++i) r[i] = _mm512_loadu_ps(src + i * src_stride);
    __m512 t[16];
    for (int i = 0; i < 8; ++i) {
        t[2*i]   = _mm512_unpacklo_ps(r[2*i], r[2*i+1]);
        t[2*i+1] = _mm512_unpackhi_ps(r[2*i], r[2*i+1]);
    }
    __m512 u[16];
    for (int i = 0; i < 4; ++i) {
        u[4*i+0]=_mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t[4*i+0]),_mm512_castps_pd(t[4*i+2])));
        u[4*i+1]=_mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t[4*i+0]),_mm512_castps_pd(t[4*i+2])));
        u[4*i+2]=_mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t[4*i+1]),_mm512_castps_pd(t[4*i+3])));
        u[4*i+3]=_mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t[4*i+1]),_mm512_castps_pd(t[4*i+3])));
    }
    __m512 v[16];
    for (int i = 0; i < 4; ++i) {
        v[i+0] =_mm512_shuffle_f32x4(u[i+0],u[i+4],0x88);
        v[i+4] =_mm512_shuffle_f32x4(u[i+0],u[i+4],0xdd);
        v[i+8] =_mm512_shuffle_f32x4(u[i+8],u[i+12],0x88);
        v[i+12]=_mm512_shuffle_f32x4(u[i+8],u[i+12],0xdd);
    }
    for (int i = 0; i < 8; ++i) {
        _mm512_storeu_ps(dst16 + (i)   * 16, _mm512_shuffle_f32x4(v[i], v[i+8], 0x88));
        _mm512_storeu_ps(dst16 + (i+8) * 16, _mm512_shuffle_f32x4(v[i], v[i+8], 0xdd));
    }
}
#endif

// Gather VW rows x n positions into tile[i][0..VW): tile[i][l] = rows[l*rs + i].
inline void transpose_load(const f32* rows, u32 n, u32 rs, f32 (*tile)[VW]) noexcept {
#if defined(C4D_AVX512_TRANSPOSE)
    if constexpr (VW == 16) {
        for (u32 i = 0; i + 16 <= n; i += 16)         // each 16-wide column block
            transpose16x16(rows + i, rs, &tile[i][0]);
        for (u32 i = (n / 16) * 16; i < n; ++i)       // ragged tail (n not /16)
            for (u32 l = 0; l < VW; ++l) tile[i][l] = rows[l * rs + i];
        return;
    }
#endif
    for (u32 i = 0; i < n; ++i)
        for (u32 l = 0; l < VW; ++l) tile[i][l] = rows[l * rs + i];
}

// Scatter back: rows[l*rs + i] = tile[i][l]  (inverse of transpose_load).
inline void transpose_store(f32* rows, u32 n, u32 rs, const f32 (*tile)[VW]) noexcept {
#if defined(C4D_AVX512_TRANSPOSE)
    if constexpr (VW == 16) {
        // The store is the same 16x16 transpose with src/dst roles swapped: write
        // 16 rows from a transposed 16x16 of the tile block. Build the transposed
        // block, then store each of its 16 rows to rows[l*rs + i..].
        alignas(64) f32 tb[256];
        for (u32 i = 0; i + 16 <= n; i += 16) {
            transpose16x16(&tile[i][0], VW, tb);      // tile block (16 pos x 16 lane) -> (16 lane x 16 pos)
            for (u32 l = 0; l < 16; ++l) _mm512_storeu_ps(rows + l * rs + i, _mm512_loadu_ps(tb + l * 16));
        }
        for (u32 i = (n / 16) * 16; i < n; ++i)
            for (u32 l = 0; l < VW; ++l) rows[l * rs + i] = tile[i][l];
        return;
    }
#endif
    for (u32 i = 0; i < n; ++i)
        for (u32 l = 0; l < VW; ++l) rows[l * rs + i] = tile[i][l];
}

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

// X-axis (stride-1) multi-line lifting: process VW adjacent ROWS (each row a
// contiguous line of length n at row stride `rowstride`=CHUNK) as SIMD lanes.
// Gather transposes VW rows into lane-major scratch; lift; scatter transposes
// back. Folds the deinterleave into the scatter like fwd_1d. `rows` points at
// the first of VW consecutive rows.
inline void fwd_1d_vx(f32* rows, u32 n, u32 rowstride) noexcept {
    alignas(64) f32 tile[CHUNK][VW];
    transpose_load(rows, n, rowstride, tile);         // VW-vectorized transpose
    vf v[CHUNK];
    for (u32 i = 0; i < n; ++i) v[i] = vf(&tile[i][0], stdx::vector_aligned);
    lift_pass_v<true >(v, n, f32(ALPHA));
    lift_pass_v<false>(v, n, f32(BETA));
    lift_pass_v<true >(v, n, f32(GAMMA));
    lift_pass_v<false>(v, n, f32(DELTA));
    const f32 lo = f32(1.0 / KAPPA), hi = f32(KAPPA);
    const u32 h = n / 2;
    for (u32 i = 0; i < h; ++i) {
        (v[2*i]   * lo).copy_to(&tile[i][0],     stdx::vector_aligned);
        (v[2*i+1] * hi).copy_to(&tile[h+i][0],   stdx::vector_aligned);
    }
    transpose_store(rows, n, rowstride, tile);
}

inline void inv_1d_vx(f32* rows, u32 n, u32 rowstride) noexcept {
    alignas(64) f32 tile[CHUNK][VW];
    transpose_load(rows, n, rowstride, tile);
    vf v[CHUNK];
    const f32 lo = f32(KAPPA), hi = f32(1.0 / KAPPA);
    const u32 h = n / 2;
    for (u32 i = 0; i < h; ++i) {
        v[2*i]   = vf(&tile[i][0],   stdx::vector_aligned) * lo;
        v[2*i+1] = vf(&tile[h+i][0], stdx::vector_aligned) * hi;
    }
    lift_pass_v<false>(v, n, f32(-DELTA));
    lift_pass_v<true >(v, n, f32(-GAMMA));
    lift_pass_v<false>(v, n, f32(-BETA));
    lift_pass_v<true >(v, n, f32(-ALPHA));
    for (u32 i = 0; i < n; ++i) v[i].copy_to(&tile[i][0], stdx::vector_aligned);
    transpose_store(rows, n, rowstride, tile);
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
    // X axis (stride 1): VW adjacent y-rows per SIMD pass (transpose-tile), scalar
    // remainder. (Was the slowest axis — 7x the SIMD Y pass — when done scalar.)
    // Slice base is z*SLICE (padded z-pitch); rows within a slice stay dense.
#ifdef C4D_HAVE_SIMD
    for (u32 z = 0; z < s; ++z) {
        u32 y = 0;
        for (; y + VW <= s; y += VW) fwd_1d_vx(vol + z * SLICE + y * CHUNK, s, CHUNK);
        for (; y < s; ++y)           fwd_1d  (vol + z * SLICE + y * CHUNK, s, 1);
    }
#else
    for (u32 z = 0; z < s; ++z)
        for (u32 y = 0; y < s; ++y)
            fwd_1d(vol + z * SLICE + y * CHUNK, s, 1);
#endif
    // Y axis (stride CHUNK): VW consecutive x per SIMD pass (contiguous loads),
    // scalar remainder. Z axis (stride SLICE = padded CHUNK^2) likewise.
#ifdef C4D_HAVE_SIMD
    for (u32 z = 0; z < s; ++z) {
        u32 x = 0;
        for (; x + VW <= s; x += VW) fwd_1d_v(vol + z * SLICE + x, s, CHUNK);
        for (; x < s; ++x)           fwd_1d  (vol + z * SLICE + x, s, CHUNK);
    }
    for (u32 y = 0; y < s; ++y) {
        u32 x = 0;
        for (; x + VW <= s; x += VW) fwd_1d_v(vol + y * CHUNK + x, s, SLICE);
        for (; x < s; ++x)           fwd_1d  (vol + y * CHUNK + x, s, SLICE);
    }
#else
    for (u32 z = 0; z < s; ++z) for (u32 x = 0; x < s; ++x) fwd_1d(vol + z*SLICE + x, s, CHUNK);
    for (u32 y = 0; y < s; ++y) for (u32 x = 0; x < s; ++x) fwd_1d(vol + y*CHUNK + x, s, SLICE);
#endif
}

// One separable inverse DWT step (exact reverse of fwd_step).
inline void inv_step(f32* vol, u32 s) noexcept {
#ifdef C4D_HAVE_SIMD
    for (u32 y = 0; y < s; ++y) {
        u32 x = 0;
        for (; x + VW <= s; x += VW) inv_1d_v(vol + y * CHUNK + x, s, SLICE);
        for (; x < s; ++x)           inv_1d  (vol + y * CHUNK + x, s, SLICE);
    }
    for (u32 z = 0; z < s; ++z) {
        u32 x = 0;
        for (; x + VW <= s; x += VW) inv_1d_v(vol + z * SLICE + x, s, CHUNK);
        for (; x < s; ++x)           inv_1d  (vol + z * SLICE + x, s, CHUNK);
    }
#else
    for (u32 y = 0; y < s; ++y) for (u32 x = 0; x < s; ++x) inv_1d(vol + y*CHUNK + x, s, SLICE);
    for (u32 z = 0; z < s; ++z) for (u32 x = 0; x < s; ++x) inv_1d(vol + z*SLICE + x, s, CHUNK);
#endif
    // X axis (stride 1): VW adjacent y-rows per SIMD pass, scalar remainder.
#ifdef C4D_HAVE_SIMD
    for (u32 z = 0; z < s; ++z) {
        u32 y = 0;
        for (; y + VW <= s; y += VW) inv_1d_vx(vol + z * SLICE + y * CHUNK, s, CHUNK);
        for (; y < s; ++y)           inv_1d  (vol + z * SLICE + y * CHUNK, s, 1);
    }
#else
    for (u32 z = 0; z < s; ++z)
        for (u32 y = 0; y < s; ++y)
            inv_1d(vol + z * SLICE + y * CHUNK, s, 1);
#endif
}

// Per-thread padded scratch (slice-pitched, see SLICE). Reused across calls so
// the repack adds no allocation. The two repack copies are sequential CHUNK^2
// runs (cache-friendly) and are paid back many times over by the strided
// Z/Y-pass miss reduction.
[[nodiscard]] inline f32* padded_scratch() noexcept {
    thread_local std::vector<f32> buf(PADDED_VOX);
    return buf.data();
}

// Repack dense CHUNK^3 (z*CHUNK^2 + y*CHUNK + x) <-> slice-padded (z*SLICE + ...).
// Rows are dense in both, so each slice is one contiguous CHUNK^2-float copy.
inline void pack_padded(const f32* dense, f32* pad) noexcept {
    for (u32 z = 0; z < CHUNK; ++z)
        std::memcpy(pad + z * SLICE, dense + z * CHUNK * CHUNK, CHUNK * CHUNK * sizeof(f32));
}
// Pack with a DC bias subtracted during the copy — folds the encoder's
// `coef[i] -= mean` pass (a full 256K read+write that missed L1 because coef
// was already evicted) into the pack copy that has to read coef anyway.
inline void pack_padded_bias(const f32* dense, f32* pad, f32 bias) noexcept {
    for (u32 z = 0; z < CHUNK; ++z) {
        const f32* src = dense + z * CHUNK * CHUNK;
        f32* dst = pad + z * SLICE;
        for (u32 i = 0; i < CHUNK * CHUNK; ++i) dst[i] = src[i] - bias;
    }
}
inline void unpack_padded(const f32* pad, f32* dense) noexcept {
    for (u32 z = 0; z < CHUNK; ++z)
        std::memcpy(dense + z * CHUNK * CHUNK, pad + z * SLICE, CHUNK * CHUNK * sizeof(f32));
}

// Full multi-level forward transform over a CHUNK^3 buffer. Each level halves
// the active sub-cube (recursing into the LLL octant), `levels` deep. Transforms
// in the slice-padded scratch to avoid Z/Y-stride cache-set aliasing. `bias` is
// subtracted from every sample during the pack (the encoder's DC removal) —
// pass 0 for a plain transform.
inline void forward(f32* vol, f32 bias = 0.f, u32 levels = DWT_LEVELS) noexcept {
    f32* pad = padded_scratch();
    if (bias != 0.f) pack_padded_bias(vol, pad, bias);
    else             pack_padded(vol, pad);
    u32 s = CHUNK;
    for (u32 l = 0; l < levels; ++l) { fwd_step(pad, s); s /= 2; }
    unpack_padded(pad, vol);
}

// Full multi-level inverse transform (coarsest level first).
inline void inverse(f32* vol, u32 levels = DWT_LEVELS) noexcept {
    f32* pad = padded_scratch();
    pack_padded(vol, pad);
    std::array<u32, 16> sizes{};
    u32 s = CHUNK;
    for (u32 l = 0; l < levels; ++l) { sizes[l] = s; s /= 2; }
    for (i32 l = static_cast<i32>(levels) - 1; l >= 0; --l) inv_step(pad, sizes[static_cast<u32>(l)]);
    unpack_padded(pad, vol);
}

} // namespace c4d::dwt
