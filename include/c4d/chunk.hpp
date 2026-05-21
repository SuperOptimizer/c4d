// Per-chunk encode/decode: the full §4 pipeline tying together DC removal, the
// 9/7 DWT, dead-zone quantization, "model the zeros" coefficient coding, and
// rANS. One encode_chunk call consumes a 128^3 u8 cube and returns a
// self-contained payload; decode_chunk inverts it. Spec §4.
//
// Coefficient coding (§4.7, "model the zeros"): coefficients are visited in
// subband order; the dead-zone makes HF subbands 70-95% zeros, so instead of
// coding each zero we code zero *runs*. The token alphabet is:
//   [0 .. RUN_BASE)            : HybridUint magnitude token of a NONZERO coeff
//   [RUN_BASE .. RUN_BASE+RUNS): a zero-run-length bucket (run of zeros)
// Each nonzero token is followed by raw mantissa bits (bypass) and a sign bit.
// Zero runs use a HybridUint-style length split too. This is the standout T0/T1
// ratio lever and also speeds decode (fewer rANS pulls).
#pragma once
#include "core.hpp"
#include "dwt.hpp"
#include "quant.hpp"
#include "rans.hpp"
#include "hybrid_uint.hpp"
#include <span>
#include <vector>

namespace c4d::chunk {

// Token alphabet layout. Magnitude tokens occupy [0, RUN_BASE); run tokens
// occupy [RUN_BASE, RUN_BASE + RUN_TOKENS).
inline constexpr u32 RUN_BASE   = hybrid::MAX_TOKEN;     // 64
inline constexpr u32 RUN_TOKENS = hybrid::MAX_TOKEN;     // run-length buckets
                                                         // (same hybrid scheme;
                                                         // a run can span a full
                                                         // 128^3 = 2^21 chunk)
inline constexpr u32 ALPHABET   = RUN_BASE + RUN_TOKENS; // total symbols

// Zero-run length split: same hybrid scheme, offset into the run token band.
struct RunSplit { u32 token; u32 raw; u32 nbits; };
[[nodiscard]] inline RunSplit run_encode(u32 len) noexcept {
    auto s = hybrid::encode(len);
    return {RUN_BASE + s.token, s.raw, s.nbits};
}
[[nodiscard]] inline u32 run_decode(u32 token, u32 raw) noexcept {
    return hybrid::decode(token - RUN_BASE, raw);
}
[[nodiscard]] inline u32 run_raw_bits(u32 token) noexcept {
    return hybrid::raw_bits_of(token - RUN_BASE);
}

// A decoded/encodable chunk payload: the entropy bytes plus the small header
// the decoder needs (q knob -> step table is rebuilt; mean DC; stream sizes).
struct Payload {
    f32 dc = 0;                 // subtracted mean
    f32 q = 0;                  // quality knob used (-> StepTable::from_q)
    std::vector<u8> tokens;     // rANS symbol stream
    std::vector<u8> bypass;     // raw mantissa + sign + run bits
    // Serialized form is produced by serialize(); the archive stores that blob.
    std::vector<u8> serialize() const;
    static Payload deserialize(std::span<const u8> bytes);
    size_t size_bytes() const { return tokens.size() + bypass.size() + 16; }
};

// --- token-stream generation (encode) --------------------------------------
// First pass builds the token list + bypass bits and gathers symbol counts;
// second pass rANS-encodes the tokens against the static histogram.
struct TokenStream {
    std::vector<u32> toks;          // in forward order
    rans::BitWriter  bypass;
    std::vector<u32> counts;        // [ALPHABET]
    TokenStream() : counts(ALPHABET, 0) {}

    void emit_nonzero(i32 v) {
        u32 mag = static_cast<u32>(v < 0 ? -v : v);
        auto s = hybrid::encode(mag);
        toks.push_back(s.token); ++counts[s.token];
        if (s.nbits) bypass.put(s.raw, s.nbits);
        bypass.put(v < 0 ? 1u : 0u, 1);     // sign
    }
    void emit_run(u32 len) {
        auto r = run_encode(len);
        toks.push_back(r.token); ++counts[r.token];
        if (r.nbits) bypass.put(r.raw, r.nbits);
    }
};

// Encode one 128^3 u8 chunk at quality knob q. `vox` is row-major Z,Y,X.
inline Payload encode_chunk(std::span<const u8> vox, f32 q) {
    // 1. to float, subtract DC (mean)
    std::vector<f32> coef(CHUNK_VOX);
    f64 mean = 0; for (u8 v : vox) mean += v; mean /= CHUNK_VOX;
    for (u32 i = 0; i < CHUNK_VOX; ++i) coef[i] = static_cast<f32>(vox[i]) - static_cast<f32>(mean);

    // 2. DWT, 3. quantize
    dwt::forward(coef.data());
    StepTable steps = StepTable::from_q(q);
    std::vector<i32> ql(CHUNK_VOX);
    quantize(coef, steps, ql);

    // 4. model-the-zeros token generation, in subband order (block-aligned).
    // For T0 we scan in the natural index order grouped by subband id so that
    // runs of zeros within dead HF bands coalesce. (Frequency-diagonal scan is a
    // T1 refinement that lengthens runs further.)
    TokenStream ts;
    const SubbandMap& m = subband_map();
    // Visit coefficients ordered by subband id, then linear index — keeps each
    // band's coefficients contiguous so zero runs don't break at band edges.
    // Precompute a stable order: bucket indices by subband id.
    static const std::vector<u32> order = [] {
        std::vector<u32> ord(CHUNK_VOX);
        const SubbandMap& mm = subband_map();
        std::vector<std::vector<u32>> buckets(256);
        for (u32 i = 0; i < CHUNK_VOX; ++i) buckets[mm.id[i]].push_back(i);
        u32 k = 0;
        for (auto& b : buckets) for (u32 i : b) ord[k++] = i;
        return ord;
    }();

    u32 zeros = 0;
    for (u32 idx : order) {
        i32 v = ql[idx];
        if (v == 0) { ++zeros; continue; }
        if (zeros) { ts.emit_run(zeros); zeros = 0; }
        ts.emit_nonzero(v);
    }
    if (zeros) ts.emit_run(zeros);   // trailing zero run

    // 5. rANS the tokens against a static histogram.
    auto tbl = rans::FreqTable::build(ts.counts);
    rans::Encoder enc;
    for (i64 i = static_cast<i64>(ts.toks.size()) - 1; i >= 0; --i) enc.put(tbl, ts.toks[i]);
    auto rans_bytes = enc.finish();

    Payload p;
    p.dc = static_cast<f32>(mean);
    p.q  = q;
    // tokens blob = [freq table | token count | rans bytes]
    tbl.serialize(p.tokens);
    for (int i = 0; i < 4; ++i) p.tokens.push_back(u8((u32(ts.toks.size()) >> (8 * i)) & 0xff));
    p.tokens.insert(p.tokens.end(), rans_bytes.begin(), rans_bytes.end());
    p.bypass = ts.bypass.finish();
    return p;
}

// Decode one chunk payload back to a 128^3 u8 cube.
inline void decode_chunk(const Payload& p, std::span<u8> out) {
    // rebuild freq table + token count
    size_t pos = 0;
    auto tbl = rans::FreqTable::deserialize(p.tokens, pos);
    u32 ntok = 0; for (int i = 0; i < 4; ++i) ntok |= u32(p.tokens[pos++]) << (8 * i);
    std::span<const u8> rans_bytes(p.tokens.data() + pos, p.tokens.size() - pos);
    rans::Decoder dec(rans_bytes);
    rans::BitReader br(p.bypass);

    // Reconstruct quantized levels in the same subband order, then scatter.
    static const std::vector<u32> order = [] {
        std::vector<u32> ord(CHUNK_VOX);
        const SubbandMap& mm = subband_map();
        std::vector<std::vector<u32>> buckets(256);
        for (u32 i = 0; i < CHUNK_VOX; ++i) buckets[mm.id[i]].push_back(i);
        u32 k = 0; for (auto& b : buckets) for (u32 i : b) ord[k++] = i;
        return ord;
    }();

    std::vector<i32> ql(CHUNK_VOX, 0);
    u32 oi = 0;  // position in `order`
    for (u32 t = 0; t < ntok; ++t) {
        u32 token = dec.get(tbl);
        if (token >= RUN_BASE) {                       // zero run
            u32 rb = run_raw_bits(token);
            u32 raw = rb ? br.get(rb) : 0;
            u32 len = run_decode(token, raw);
            oi += len;                                 // those positions stay 0
        } else {                                       // nonzero magnitude
            u32 rb = hybrid::raw_bits_of(token);
            u32 raw = rb ? br.get(rb) : 0;
            u32 mag = hybrid::decode(token, raw);
            u32 sign = br.get(1);
            ql[order[oi++]] = sign ? -static_cast<i32>(mag) : static_cast<i32>(mag);
        }
    }

    // dequantize, inverse DWT, add DC, clamp to u8.
    StepTable steps = StepTable::from_q(p.q);
    std::vector<f32> coef(CHUNK_VOX);
    dequantize(ql, steps, coef);
    dwt::inverse(coef.data());
    for (u32 i = 0; i < CHUNK_VOX; ++i) {
        f32 v = coef[i] + p.dc;
        out[i] = static_cast<u8>(v < 0.f ? 0.f : (v > 255.f ? 255.f : v + 0.5f));
    }
}

// --- payload serialization (flat blob for the archive) ---------------------
inline std::vector<u8> Payload::serialize() const {
    std::vector<u8> b;
    auto push32 = [&](u32 v) { for (int i = 0; i < 4; ++i) b.push_back(u8((v >> (8 * i)) & 0xff)); };
    auto pushf  = [&](f32 f) { u32 v; std::memcpy(&v, &f, 4); push32(v); };
    pushf(dc); pushf(q);
    push32(static_cast<u32>(tokens.size()));
    push32(static_cast<u32>(bypass.size()));
    b.insert(b.end(), tokens.begin(), tokens.end());
    b.insert(b.end(), bypass.begin(), bypass.end());
    return b;
}

inline Payload Payload::deserialize(std::span<const u8> bytes) {
    Payload p; size_t pos = 0;
    auto rd32 = [&] { u32 v = 0; for (int i = 0; i < 4; ++i) v |= u32(bytes[pos++]) << (8 * i); return v; };
    auto rdf  = [&] { u32 v = rd32(); f32 f; std::memcpy(&f, &v, 4); return f; };
    p.dc = rdf(); p.q = rdf();
    u32 tlen = rd32(), blen = rd32();
    p.tokens.assign(bytes.begin() + pos, bytes.begin() + pos + tlen); pos += tlen;
    p.bypass.assign(bytes.begin() + pos, bytes.begin() + pos + blen); pos += blen;
    return p;
}

} // namespace c4d::chunk
