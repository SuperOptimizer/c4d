// POCS seam deblocking (decode-side, opt-in). Independent per-chunk mirror-
// boundary DWT leaves a faint periodic discontinuity at chunk faces (a "grid"
// artifact) — low energy (≈+0.1 dB PSNR) but perceptually visible and bad for
// downstream ML. This post-process removes it WITHOUT touching the encoder or
// the stored bytes, and provably without corrupting real signal.
//
// Projection Onto Convex Sets: alternate two projections a few iterations —
//   (a) SMOOTHNESS: blend the few voxels straddling each internal chunk face
//       (uses the neighbour chunks' already-decoded values across the seam);
//   (b) QUANTIZATION CONSTRAINT: re-DWT each chunk and clamp every coefficient
//       back into the dead-zone cell it decoded from (|c - q·step| ≤ step/2), so
//       smoothing can never push the reconstruction outside the quantization
//       error already accepted. PSNR can only stay equal or improve.
//
// Operates on a dense grid of decoded CHUNK³ blocks (a region or whole volume),
// in place. Needs each chunk's StepTable (carried in its payload). 2 iterations
// is the measured sweet spot; more over-smooths.
#pragma once
#include "core.hpp"
#include "dwt.hpp"
#include "quant.hpp"
#include <span>
#include <vector>

namespace c4d::deblock {

// A decoded volume tiled into CHUNK³ blocks: `vol` is the dense (gz·CHUNK)×… cube
// in Z,Y,X; `grid` is the chunk grid; `steps[chunk_linear]` the per-chunk steps.
// `vol` dims must be grid·CHUNK on each axis (caller pads to full chunks).
struct Tiled {
    std::span<f32> vol;             // dense f32 volume, dims = grid*CHUNK
    Coord3 grid;                    // chunk grid (Z,Y,X)
    std::span<const StepTable> steps; // [chunk_linear] per-chunk step tables
    std::span<const f32> dcs;       // [chunk_linear] per-chunk DC (subtracted before DWT)
    // Original decoded quantized levels per chunk (chunk_linear-major, CHUNK_VOX
    // each). The POCS constraint clamps each coefficient to THIS level's cell —
    // the cell it was actually coded in — not the cell a smoothed value falls in.
    std::span<const i32> ql;        // [chunk_linear * CHUNK_VOX + i]
    [[nodiscard]] u64 dimx() const { return grid.x * CHUNK; }
    [[nodiscard]] u64 dimy() const { return grid.y * CHUNK; }
    [[nodiscard]] u64 dimz() const { return grid.z * CHUNK; }
    [[nodiscard]] f32& at(u64 z, u64 y, u64 x) const { return vol[(z * dimy() + y) * dimx() + x]; }
};

// Project a single chunk onto its quantization constraint set: re-DWT, clamp
// each coefficient to the dead-zone cell of its current quantized level, inverse.
// This guarantees the result stays within ±step/2 of the originally-coded value
// per coefficient — it cannot move outside the accepted quantization error.
inline void project_quant_cell(f32* chunk, const StepTable& st, f32 dc, const i32* ql) {
    for (u32 i = 0; i < CHUNK_VOX; ++i) chunk[i] -= dc;   // same DC the chunk decoded with
    dwt::forward(chunk);
    const SubbandMap& m = subband_map();
    for (u32 i = 0; i < CHUNK_VOX; ++i) {
        f32 step = st.step[m.level(i)][m.orient(i)];
        if (step <= 0.f) continue;
        // clamp to the cell of the ORIGINAL decoded level ql[i] (not a re-quant of
        // the smoothed value) — this is the true quantization constraint set, so
        // the result can never drift outside the accepted error.
        i32 q = ql[i];
        f32 lo, hi;
        if (q == 0) { lo = -0.5f * step; hi = 0.5f * step; }
        else { f32 c = static_cast<f32>(q) * step; lo = c - 0.5f * step; hi = c + 0.5f * step; }
        if (chunk[i] < lo) chunk[i] = lo;
        if (chunk[i] > hi) chunk[i] = hi;
    }
    dwt::inverse(chunk);
    for (u32 i = 0; i < CHUNK_VOX; ++i) chunk[i] += dc;
}

// Smooth a band of width W straddling one internal face along `axis` at the
// global face coordinate `fc`, pulling the ±W band toward a linear fit between
// the band's outer anchors (the seam's two sides). Uses both chunks' values.
inline void smooth_face(const Tiled& t, int axis, u64 fc, int W, f32 strength) {
    u64 dz = t.dimz(), dy = t.dimy(), dx = t.dimx();
    auto idx = [&](u64 z, u64 y, u64 x) -> f32& { return t.at(z, y, x); };
    if (axis == 0) {
        for (u64 y = 1; y + 1 < dy; ++y) for (u64 x = 1; x + 1 < dx; ++x) {
            f32 a = idx(fc - W - 1, y, x), b = idx(fc + W, y, x);
            for (int o = -W; o < W; ++o) {
                f32 tt = (o + W + 1) / f32(2 * W + 1);
                f32& v = idx(fc + o, y, x);
                v = (1.f - strength) * v + strength * ((1 - tt) * a + tt * b);
            }
        }
    } else if (axis == 1) {
        for (u64 z = 1; z + 1 < dz; ++z) for (u64 x = 1; x + 1 < dx; ++x) {
            f32 a = idx(z, fc - W - 1, x), b = idx(z, fc + W, x);
            for (int o = -W; o < W; ++o) {
                f32 tt = (o + W + 1) / f32(2 * W + 1);
                f32& v = idx(z, fc + o, x);
                v = (1.f - strength) * v + strength * ((1 - tt) * a + tt * b);
            }
        }
    } else {
        for (u64 z = 1; z + 1 < dz; ++z) for (u64 y = 1; y + 1 < dy; ++y) {
            f32 a = idx(z, y, fc - W - 1), b = idx(z, y, fc + W);
            for (int o = -W; o < W; ++o) {
                f32 tt = (o + W + 1) / f32(2 * W + 1);
                f32& v = idx(z, y, fc + o);
                v = (1.f - strength) * v + strength * ((1 - tt) * a + tt * b);
            }
        }
    }
}

// Run the full POCS deblock on a tiled decoded volume, in place. `strength` in
// (0,1] scales the smoothing blend — gentler smoothing keeps coefficients near
// their cell centres (less PSNR cost) while still relaxing the seam; the quant-
// cell projection then bounds any residual drift.
inline void run(const Tiled& t, int iterations = 2, int band = 2, f32 strength = 0.5f) {
    const Coord3& g = t.grid;
    std::vector<f32> chunk(CHUNK_VOX);
    for (int it = 0; it < iterations; ++it) {
        // (a) smoothness: every internal face on each axis
        for (u64 cz = 1; cz < g.z; ++cz) smooth_face(t, 0, cz * CHUNK, band, strength);
        for (u64 cy = 1; cy < g.y; ++cy) smooth_face(t, 1, cy * CHUNK, band, strength);
        for (u64 cx = 1; cx < g.x; ++cx) smooth_face(t, 2, cx * CHUNK, band, strength);
        // (b) quantization-cell projection per chunk
        for (u64 cz = 0; cz < g.z; ++cz)
          for (u64 cy = 0; cy < g.y; ++cy)
            for (u64 cx = 0; cx < g.x; ++cx) {
                u64 ci = chunk_linear({cz, cy, cx}, g);
                u64 z0 = cz * CHUNK, y0 = cy * CHUNK, x0 = cx * CHUNK;
                for (u32 z = 0; z < CHUNK; ++z)
                    for (u32 y = 0; y < CHUNK; ++y)
                        for (u32 x = 0; x < CHUNK; ++x)
                            chunk[vox_index(z, y, x)] = t.at(z0 + z, y0 + y, x0 + x);
                project_quant_cell(chunk.data(), t.steps[ci], t.dcs[ci],
                                   t.ql.data() + ci * CHUNK_VOX);
                for (u32 z = 0; z < CHUNK; ++z)
                    for (u32 y = 0; y < CHUNK; ++y)
                        for (u32 x = 0; x < CHUNK; ++x)
                            t.at(z0 + z, y0 + y, x0 + x) = chunk[vox_index(z, y, x)];
            }
    }
}

} // namespace c4d::deblock
