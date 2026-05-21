// Outlier-correction pass (§4.6): a hard point-wise-error / constant-quality
// guarantee, separable from the coefficient coder. The encoder quantizes
// coefficients normally, internally decodes, finds every voxel where
// |original - reconstructed| > t, and codes that sparse set as a SECOND stream.
// On decode, after the normal reconstruction the corrections are applied so
// |z - x| <= t is guaranteed (the math only needs corrections coded to +/- t).
//
// Sparse coding: voxels sorted by linear index; positions delta-coded (gaps are
// small in dense regions), magnitudes quantized to multiples of t (almost always
// tiny integers), signs raw. We reuse c4d's own rANS + HybridUint + bypass.
#pragma once
#include "core.hpp"
#include "rans.hpp"
#include "hybrid_uint.hpp"
#include <span>
#include <vector>

namespace c4d::outlier {

// One correction: position (linear index) and a signed multiple of t.
struct Correction { u32 pos; i32 steps; };  // delta = steps * t (steps != 0)

// Find corrections: for each voxel where |orig - recon| > t, the correction
// rounds (orig - recon)/t to the nearest nonzero integer so |residual| <= t/2...
// actually we need |z - x| <= t after applying steps*t. Round to nearest integer
// k = round((orig-recon)/t); applying k*t leaves |orig - (recon + k*t)| <= t/2,
// comfortably within t. k!=0 exactly when |orig-recon| > t/2; we gate on > t so
// the set stays sparse while still guaranteeing the bound.
inline std::vector<Correction> find(std::span<const u8> orig, std::span<const u8> recon,
                                    f32 t) {
    std::vector<Correction> out;
    if (t <= 0) return out;
    const f32 inv_t = 1.0f / t;
    for (u32 i = 0; i < CHUNK_VOX; ++i) {
        f32 diff = f32(orig[i]) - f32(recon[i]);
        if (std::fabs(diff) <= t) continue;            // already within tolerance
        i32 k = static_cast<i32>(std::lround(diff * inv_t));
        if (k == 0) k = diff > 0 ? 1 : -1;             // ensure progress
        out.push_back({i, k});
    }
    return out;
}

// Encode the correction set into a flat blob: [u32 count][rANS pos-gap+mag
// tokens][bypass raw bits + signs]. Empty when there are no corrections.
inline std::vector<u8> encode(const std::vector<Correction>& cs) {
    std::vector<u8> blob;
    auto push32 = [&](u32 v) { for (int i = 0; i < 4; ++i) blob.push_back(u8((v >> (8 * i)) & 0xff)); };
    push32(static_cast<u32>(cs.size()));
    if (cs.empty()) return blob;

    // token alphabet: position-gap tokens and magnitude tokens both via
    // HybridUint, in separate logical passes but one shared histogram is fine.
    std::vector<u32> toks;
    rans::BitWriter bypass;
    std::vector<u32> counts(hybrid::MAX_TOKEN, 0);
    auto emit = [&](u32 u) {
        auto s = hybrid::encode(u);
        toks.push_back(s.token); ++counts[s.token];
        if (s.nbits) bypass.put(s.raw, s.nbits);
    };

    u32 prev = 0;
    for (const auto& c : cs) {
        emit(c.pos - prev);                            // position gap (>=0)
        prev = c.pos;
        u32 mag = static_cast<u32>(c.steps < 0 ? -c.steps : c.steps);
        emit(mag - 1);                                 // magnitude-1 (steps != 0)
        bypass.put(c.steps < 0 ? 1u : 0u, 1);          // sign
    }

    auto tbl = rans::FreqTable::build(counts);
    rans::Encoder enc;
    for (i64 i = static_cast<i64>(toks.size()) - 1; i >= 0; --i) enc.put(tbl, toks[i]);
    auto rans_bytes = enc.finish();
    auto bp = bypass.finish();

    tbl.serialize(blob);
    push32(static_cast<u32>(toks.size()));
    push32(static_cast<u32>(rans_bytes.size()));
    blob.insert(blob.end(), rans_bytes.begin(), rans_bytes.end());
    blob.insert(blob.end(), bp.begin(), bp.end());
    return blob;
}

// Apply a correction blob to a reconstructed cube in place: out[pos] += steps*t,
// clamped to u8. `t` must match the encode tolerance.
inline void apply(std::span<const u8> blob, f32 t, std::span<u8> out) {
    size_t pos = 0;
    auto rd32 = [&] { u32 v = 0; for (int i = 0; i < 4; ++i) v |= u32(blob[pos++]) << (8 * i); return v; };
    u32 count = rd32();
    if (count == 0) return;

    auto tbl = rans::FreqTable::deserialize(blob, pos);
    u32 ntok = rd32();
    u32 rlen = rd32();
    std::span<const u8> rans_bytes(blob.data() + pos, rlen); pos += rlen;
    std::span<const u8> bp(blob.data() + pos, blob.size() - pos);
    rans::Decoder dec(rans_bytes);
    rans::BitReader br(bp);
    (void)ntok;

    auto get = [&] {
        u32 token = dec.get(tbl);
        u32 rb = hybrid::raw_bits_of(token);
        u32 raw = rb ? br.get(rb) : 0;
        return hybrid::decode(token, raw);
    };

    u32 cur = 0;
    for (u32 c = 0; c < count; ++c) {
        cur += get();                                  // position via gap
        u32 mag = get() + 1;                           // magnitude
        u32 sign = br.get(1);
        i32 steps = sign ? -static_cast<i32>(mag) : static_cast<i32>(mag);
        f32 v = f32(out[cur]) + steps * t;
        out[cur] = static_cast<u8>(v < 0.f ? 0.f : (v > 255.f ? 255.f : v + 0.5f));
    }
}

} // namespace c4d::outlier
