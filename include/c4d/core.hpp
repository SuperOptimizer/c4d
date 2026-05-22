// c4d core: fixed geometry, constexpr-baked constants, fundamental types.
// Spec §0 (impl stance), §1.1 (hard constraints), §2 (geometry), §4.1 (fixed props).
#pragma once
#include <array>
#include <cstdint>
#include <cstddef>
#include <span>

namespace c4d {

using u8  = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i8  = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;
using f32 = float;
using f64 = double;

// --- Geometry (spec §2, frozen) -------------------------------------------
// The chunk is the only structural unit: codec atom AND random-access unit.
// Shards (a 2048^3 object-grouping unit) were dropped — the single-file archive
// + footer index (§5/§6) is the I/O unit, so the shard's only purpose (avoiding
// S3 LIST storms over a million small chunk-files) no longer exists. Prefetch
// locality, if wanted, is a write-time payload ordering hint (Morton/Z-order),
// not a format structure.
// Chunk edge, pinned by measurement. 64³ (not 128³): with cross-chunk SHARED
// entropy tables (now allowed — chunk independence is relaxed, §3), 64³ matches
// or slightly beats 128³ on the RD frontier (small chunks adapt local statistics
// better; the per-chunk table overhead that used to penalize them is amortized
// across a table-group). 64³ also gives 8× finer random access and an 8× cheaper
// cold neighbor-fetch for the decode-time seam heal. 32³ adds a few % more ratio
// but 64× the chunk count (index/seam/metadata all blow up) — 64³ is the sweet
// spot. Overridable for re-sweeps.
#ifndef C4D_CHUNK
#define C4D_CHUNK 64
#endif
inline constexpr u32 CHUNK   = C4D_CHUNK;      // codec atom edge, voxels
inline constexpr u32 CHUNK_VOX = CHUNK * CHUNK * CHUNK;

static_assert((CHUNK & (CHUNK - 1)) == 0, "chunk edge must be a power of two");

// --- Transform -------------------------------------------------------------
// 9/7 lifting, float32, separable Z then Y then X. Level count pinned by
// measurement: at 64³ compression saturates by 2–3 levels (the LLL approximation
// at 3 levels = 8³ holds the LF energy; deeper adds only coding overhead). 3 is
// the sweet spot — best PSNR, clean 8³ LLL, matching 128³-at-4-levels. Overridable.
#ifndef C4D_DWT_LEVELS
#define C4D_DWT_LEVELS 3
#endif
inline constexpr u32 DWT_LEVELS = C4D_DWT_LEVELS;
static_assert(DWT_LEVELS >= 1 && (CHUNK >> DWT_LEVELS) >= 1,
              "cannot take more DWT levels than the chunk supports");

// Axis order is Z, Y, X everywhere (spec §1.1). Index helpers over a dense
// CHUNK^3 voxel cube (logical, no stride padding — that's a layout concern).
[[nodiscard]] constexpr u32 vox_index(u32 z, u32 y, u32 x) noexcept {
    return (z * CHUNK + y) * CHUNK + x;
}

// 3D coordinate triple (Z,Y,X). Used for chunk-grid and shard-grid math.
struct Coord3 {
    u64 z = 0, y = 0, x = 0;
    friend constexpr bool operator==(const Coord3&, const Coord3&) = default;
};

// Number of chunks along each axis to cover `extent`, padded up to full chunks.
[[nodiscard]] constexpr Coord3 chunk_grid(Coord3 extent) noexcept {
    auto up = [](u64 n) constexpr { return (n + CHUNK - 1) / CHUNK; };
    return {up(extent.z), up(extent.y), up(extent.x)};
}

// Total padded chunk count for a volume of the given logical extent.
[[nodiscard]] constexpr u64 chunk_count(Coord3 extent) noexcept {
    Coord3 g = chunk_grid(extent);
    return g.z * g.y * g.x;
}

// Linearize / delinearize a chunk coordinate over the padded grid (Z-major,
// row-major Z,Y,X — matches the implicit-by-position chunk index, spec §5.2).
[[nodiscard]] constexpr u64 chunk_linear(Coord3 c, Coord3 grid) noexcept {
    return (c.z * grid.y + c.y) * grid.x + c.x;
}
[[nodiscard]] constexpr Coord3 chunk_unravel(u64 i, Coord3 grid) noexcept {
    u64 x = i % grid.x;  i /= grid.x;
    u64 y = i % grid.y;  i /= grid.y;
    return {i, y, x};
}

} // namespace c4d
