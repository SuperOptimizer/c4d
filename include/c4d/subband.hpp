// Subband geometry for the Mallat pyramid layout produced by dwt::forward.
// Maps a voxel coordinate in the transformed CHUNK^3 cube to its (level, orient)
// subband id, and enumerates subbands. Spec §4.1, §4.5.
//
// Layout recap: dwt::forward runs `levels` steps; step l (l=0 finest) transforms
// the leading s^3 octant (s = CHUNK>>l) and packs low into [0,s/2), high into
// [s/2,s) per axis. So after all levels:
//   - the LLL approximation band occupies [0, CHUNK>>levels) on every axis;
//   - level l's detail bands occupy the shell between (CHUNK>>(l+1)) and
//     (CHUNK>>l) on at least one axis, with orient bits set where the coord is
//     in the high half of that level's octant.
#pragma once
#include "core.hpp"
#include <array>

namespace c4d {

// Subband identity: which decomposition level (0=finest .. levels-1=coarsest)
// and orientation (bit0=x high, bit1=y high, bit2=z high). The LLL approximation
// is reported as {level = levels-1, orient = 0}.
struct Subband {
    u32 level;
    u32 orient;  // 0..7
    friend constexpr bool operator==(const Subband&, const Subband&) = default;
};

// The finest level at which a coordinate sits in a high ring [s/2, s) for
// s = CHUNK>>l, else `levels` (the coordinate is inside the approximation
// range [0, CHUNK>>levels)). This is the per-axis "outermost high level".
[[nodiscard]] constexpr u32 axis_high_level(u32 c, u32 levels) noexcept {
    for (u32 l = 0; l < levels; ++l) {
        u32 s = CHUNK >> l;
        if (c >= s / 2 && c < s) return l;   // first (finest) high ring hit
    }
    return levels;  // in the approximation low range
}

// Full subband id for a transformed-cube coordinate. The detail level is the
// FINEST level at which *any* axis is in a high ring; orientation is then read
// relative to that level (an axis is "high" iff it lies in that level's high
// ring [CHUNK>>(level+1), CHUNK>>level)). If no axis is ever high, the
// coordinate is the LLL approximation, reported at the coarsest level.
[[nodiscard]] constexpr Subband subband_of(u32 z, u32 y, u32 x, u32 levels) noexcept {
    u32 lz = axis_high_level(z, levels);
    u32 ly = axis_high_level(y, levels);
    u32 lx = axis_high_level(x, levels);
    u32 level = levels;
    if (lz < level) level = lz;
    if (ly < level) level = ly;
    if (lx < level) level = lx;
    if (level == levels) return {levels - 1, 0};  // LLL approximation
    u32 lo = CHUNK >> (level + 1), hi = CHUNK >> level;  // this level's high ring
    auto is_high = [&](u32 c) constexpr { return c >= lo && c < hi; };
    u32 orient = (is_high(x) ? 1u : 0u) | (is_high(y) ? 2u : 0u) | (is_high(z) ? 4u : 0u);
    return {level, orient};
}

} // namespace c4d
