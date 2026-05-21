// Coefficient scan order for the chunk coder (§4.7). Coefficients are visited
// subband-by-subband (so the entropy model's context changes only at block-
// aligned boundaries) and WITHIN each subband in natural raster (linear-index)
// order. This keeps each band's coefficients contiguous so the dead-zone zeros
// form long runs the zero-run/EOB tokens collapse cheaply.
//
// MEASURED (2026-05-21): §4.7 also recommends a *frequency-diagonal* within-band
// order (radial sort, JPEG-zigzag analog) claiming ~2x longer runs. On the real
// dense scroll corpus it was WORSE — it FRAGMENTS runs (avg run 11.2 -> 10.1,
// runs 149518 -> 164931 on a PHerc0172 chunk at q16; corpus ratio 7.5x -> 7.3x).
// The dense interior's spatial locality already clusters zeros, and the radial
// re-sort scatters that locality. The spec marks the diagonal scan "recommended,
// measure before committing" — the measurement says no for this data, so we keep
// raster order. (Revisit if a future corpus shows the opposite.)
#pragma once
#include "core.hpp"
#include "subband.hpp"
#include "quant.hpp"   // SubbandMap
#include <array>
#include <vector>

namespace c4d {

// The canonical visiting order over all CHUNK_VOX coefficients: subband id
// ascending (level<<3|orient), raster index within each band. Built once.
struct ScanOrder {
    std::vector<u32> order;   // [CHUNK_VOX] coefficient indices in visit order
    ScanOrder() {
        order.resize(CHUNK_VOX);
        const SubbandMap& m = subband_map();
        std::array<std::vector<u32>, 256> buckets;
        for (u32 i = 0; i < CHUNK_VOX; ++i) buckets[m.id[i]].push_back(i);
        u32 k = 0;
        for (auto& b : buckets) for (u32 i : b) order[k++] = i;
    }
};

// Process-wide singleton (identical for every chunk, fixed geometry).
[[nodiscard]] inline const ScanOrder& scan_order() noexcept {
    static const ScanOrder s;
    return s;
}

} // namespace c4d
