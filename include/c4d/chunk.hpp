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
#include "denoise.hpp"
#include "rans.hpp"
#include "hybrid_uint.hpp"
#include "scan.hpp"
#include "outlier.hpp"
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

// Context modeling (§4.8). Each token is coded against one of NUM_CTX static
// histograms selected by a cheap context id: (level bucket {0,1,2+}) × (previous
// token was a zero-run vs a magnitude). MEASURED on the corpus: ~11% smaller
// token stream net of the per-context table cost (the rich per-subband context
// is clustered to these 6 — the §4.8 context-map idea). Per-symbol distribution
// switching is cheap in rANS (just point at another table).
inline constexpr u32 NUM_CTX = 6;
[[nodiscard]] inline u32 context_id(u32 subband_id, u32 prev_class) noexcept {
    u32 lvl = subband_id >> 3;
    u32 lb  = lvl >= 2 ? 2u : lvl;          // level bucket {0,1,2+}
    return (lb << 1) | (prev_class & 1u);   // 0..5
}

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
// the decoder needs. Per §4.1 the decoder READS the per-subband quantizer steps
// from the chunk (never recomputes them), so the encoder's step policy — global
// q, noise-aware shrinkage, future RDO — is not baked into the format. The full
// StepTable (DWT_LEVELS*8 floats) rides in the payload header.
struct Payload {
    f32 dc = 0;                 // subtracted mean
    f32 q = 0;                  // the global q knob (record-keeping / per-chunk field)
    StepTable steps{};          // per-subband steps the decoder dequantizes with
    f32 tolerance = 0;          // outlier-pass L-inf tolerance t (0 => no pass)
    std::vector<u8> tokens;     // rANS symbol stream
    std::vector<u8> bypass;     // raw mantissa + sign + run bits
    std::vector<u8> outliers;   // optional sparse correction stream (§4.6)
    // Serialized form is produced by serialize(); the archive stores that blob.
    std::vector<u8> serialize() const;
    static Payload deserialize(std::span<const u8> bytes);
    size_t size_bytes() const { return tokens.size() + bypass.size() + 16 + sizeof(StepTable); }
};

// --- token-stream generation (encode) --------------------------------------
// First pass builds the token list (each tagged with its context) + bypass bits
// and gathers per-context symbol counts; second pass rANS-encodes each token
// against its context's static histogram (§4.8).
struct TokenStream {
    std::vector<u32> toks;                          // forward order
    std::vector<u8>  ctx;                           // per-token context id (0..NUM_CTX)
    rans::BitWriter  bypass;
    std::array<std::vector<u32>, NUM_CTX> counts;   // per-context histograms
    TokenStream() { for (auto& c : counts) c.assign(ALPHABET, 0); }

    void emit_nonzero(i32 v, u32 c) {
        u32 mag = static_cast<u32>(v < 0 ? -v : v);
        auto s = hybrid::encode(mag);
        toks.push_back(s.token); ctx.push_back(static_cast<u8>(c)); ++counts[c][s.token];
        if (s.nbits) bypass.put(s.raw, s.nbits);
        bypass.put(v < 0 ? 1u : 0u, 1);     // sign
    }
    void emit_run(u32 len, u32 c) {
        auto r = run_encode(len);
        toks.push_back(r.token); ctx.push_back(static_cast<u8>(c)); ++counts[c][r.token];
        if (r.nbits) bypass.put(r.raw, r.nbits);
    }
};

inline void decode_chunk(const Payload& p, std::span<u8> out);  // fwd decl

// Encoder options. The decoder reads steps from the stream regardless, so these
// only affect the encoder's step-table choice (§4.10) — never the bitstream
// shape or the decoder.
struct EncodeOpts {
    f32 q = 16.f;                 // global quality knob (base step scale)
    bool noise_aware = false;     // MAD + BayesShrink dead-zone (§4.10)
    f64 shrink_strength = 1.0;    // 0..1, how aggressively to apply shrinkage
    f32 tolerance = 0;            // outlier pass: hard L-inf bound t (0 => off, §4.6)
};

// Encode one 128^3 u8 chunk. `vox` is row-major Z,Y,X.
inline Payload encode_chunk(std::span<const u8> vox, const EncodeOpts& opt) {
    // 1. to float, subtract DC (mean)
    std::vector<f32> coef(CHUNK_VOX);
    f64 mean = 0; for (u8 v : vox) mean += v; mean /= CHUNK_VOX;
    for (u32 i = 0; i < CHUNK_VOX; ++i) coef[i] = static_cast<f32>(vox[i]) - static_cast<f32>(mean);

    // 2. DWT
    dwt::forward(coef.data());

    // 3. choose per-subband steps (encoder policy; decoder reads them back).
    StepTable steps = StepTable::from_q(opt.q);
    if (opt.noise_aware)
        steps = NoiseShrink::analyze(coef).apply(steps, opt.shrink_strength);

    std::vector<i32> ql(CHUNK_VOX);
    quantize(coef, steps, ql);

    // 4. model-the-zeros token generation (§4.7). Coefficients are visited in
    // the canonical scan order (subband-contiguous, raster within band) so the
    // dead-zone zeros form long runs the zero-run/EOB tokens collapse cheaply.
    // (Frequency-diagonal within-band order measured worse here — see scan.hpp.)
    TokenStream ts;
    const std::vector<u32>& order = scan_order().order;
    const SubbandMap& sm = subband_map();

    // Each token's context = (level bucket of the position where it applies) ×
    // (previous token class: run=0, magnitude=1). A run token uses the subband at
    // its START position; a magnitude token uses its own position's subband.
    u32 zeros = 0, prev_class = 1, run_start_sb = 0;
    for (u32 idx : order) {
        i32 v = ql[idx];
        if (v == 0) { if (zeros == 0) run_start_sb = sm.id[idx]; ++zeros; continue; }
        if (zeros) { ts.emit_run(zeros, context_id(run_start_sb, prev_class)); zeros = 0; prev_class = 0; }
        ts.emit_nonzero(v, context_id(sm.id[idx], prev_class)); prev_class = 1;
    }
    if (zeros) ts.emit_run(zeros, context_id(run_start_sb, prev_class));

    // 5. rANS each token against its context's static histogram (§4.8). Encoding
    // is reverse (LIFO); the context per token is replayed from ts.ctx.
    std::array<rans::FreqTable, NUM_CTX> tbls;
    for (u32 c = 0; c < NUM_CTX; ++c) tbls[c] = rans::FreqTable::build(ts.counts[c]);
    rans::Encoder enc;
    for (i64 i = static_cast<i64>(ts.toks.size()) - 1; i >= 0; --i)
        enc.put(tbls[ts.ctx[i]], ts.toks[i]);
    auto rans_bytes = enc.finish();

    Payload p;
    p.dc = static_cast<f32>(mean);
    p.q  = opt.q;
    p.steps = steps;
    p.tolerance = opt.tolerance;
    // tokens blob = [NUM_CTX freq tables | token count | rans bytes]
    for (u32 c = 0; c < NUM_CTX; ++c) tbls[c].serialize(p.tokens);
    for (int i = 0; i < 4; ++i) p.tokens.push_back(u8((u32(ts.toks.size()) >> (8 * i)) & 0xff));
    p.tokens.insert(p.tokens.end(), rans_bytes.begin(), rans_bytes.end());
    p.bypass = ts.bypass.finish();

    // Outlier pass (§4.6): internally decode the coefficient stream we just
    // built, find voxels outside the L-inf tolerance, and code the sparse
    // corrections as a second stream. Decode is unaffected beyond applying them.
    if (opt.tolerance > 0) {
        std::vector<u8> recon(CHUNK_VOX);
        decode_chunk(p, recon);                        // recon WITHOUT corrections yet
        auto cs = outlier::find(vox, recon, opt.tolerance);
        p.outliers = outlier::encode(cs);
    }
    return p;
}

// Convenience overload: encode at a plain global quality knob.
inline Payload encode_chunk(std::span<const u8> vox, f32 q) {
    return encode_chunk(vox, EncodeOpts{.q = q});
}

// Decode one chunk payload back to a 128^3 u8 cube.
inline void decode_chunk(const Payload& p, std::span<u8> out) {
    // rebuild NUM_CTX freq tables + token count
    size_t pos = 0;
    std::array<rans::FreqTable, NUM_CTX> tbls;
    for (u32 c = 0; c < NUM_CTX; ++c) tbls[c] = rans::FreqTable::deserialize(p.tokens, pos);
    u32 ntok = 0; for (int i = 0; i < 4; ++i) ntok |= u32(p.tokens[pos++]) << (8 * i);
    std::span<const u8> rans_bytes(p.tokens.data() + pos, p.tokens.size() - pos);
    rans::Decoder dec(rans_bytes);
    rans::BitReader br(p.bypass);

    // Reconstruct quantized levels in the same canonical scan order, then scatter.
    const std::vector<u32>& order = scan_order().order;
    const SubbandMap& sm = subband_map();

    std::vector<i32> ql(CHUNK_VOX, 0);
    u32 oi = 0;            // position in `order`
    u32 prev_class = 1;    // matches the encoder's initial prev_class
    for (u32 t = 0; t < ntok; ++t) {
        // context = (level bucket at the current scan position) × prev_class.
        // The encoder used the subband at the run/magnitude START position; the
        // decoder is at exactly that position (oi) before consuming the token.
        u32 sb = (oi < CHUNK_VOX) ? sm.id[order[oi]] : 0;
        u32 token = dec.get(tbls[context_id(sb, prev_class)]);
        if (token >= RUN_BASE) {                       // zero run
            u32 rb = run_raw_bits(token);
            u32 raw = rb ? br.get(rb) : 0;
            u32 len = run_decode(token, raw);
            oi += len;                                 // those positions stay 0
            prev_class = 0;
        } else {                                       // nonzero magnitude
            u32 rb = hybrid::raw_bits_of(token);
            u32 raw = rb ? br.get(rb) : 0;
            u32 mag = hybrid::decode(token, raw);
            u32 sign = br.get(1);
            ql[order[oi++]] = sign ? -static_cast<i32>(mag) : static_cast<i32>(mag);
            prev_class = 1;
        }
    }

    // dequantize with the steps carried in the chunk (§4.1), inverse DWT, add
    // DC, clamp to u8.
    std::vector<f32> coef(CHUNK_VOX);
    dequantize(ql, p.steps, coef);
    dwt::inverse(coef.data());
    for (u32 i = 0; i < CHUNK_VOX; ++i) {
        f32 v = coef[i] + p.dc;
        out[i] = static_cast<u8>(v < 0.f ? 0.f : (v > 255.f ? 255.f : v + 0.5f));
    }

    // Apply the optional outlier corrections (§4.6) -> hard |z - x| <= t.
    if (p.tolerance > 0 && !p.outliers.empty())
        outlier::apply(p.outliers, p.tolerance, out);
}

// --- payload serialization (flat blob for the archive) ---------------------
inline std::vector<u8> Payload::serialize() const {
    std::vector<u8> b;
    auto push32 = [&](u32 v) { for (int i = 0; i < 4; ++i) b.push_back(u8((v >> (8 * i)) & 0xff)); };
    auto pushf  = [&](f32 f) { u32 v; std::memcpy(&v, &f, 4); push32(v); };
    pushf(dc); pushf(q); pushf(tolerance);
    // per-subband step table (decoder reads these, §4.1)
    for (u32 l = 0; l < DWT_LEVELS; ++l)
        for (u32 o = 0; o < 8; ++o) pushf(steps.step[l][o]);
    push32(static_cast<u32>(tokens.size()));
    push32(static_cast<u32>(bypass.size()));
    push32(static_cast<u32>(outliers.size()));
    b.insert(b.end(), tokens.begin(), tokens.end());
    b.insert(b.end(), bypass.begin(), bypass.end());
    b.insert(b.end(), outliers.begin(), outliers.end());
    return b;
}

inline Payload Payload::deserialize(std::span<const u8> bytes) {
    Payload p; size_t pos = 0;
    auto rd32 = [&] { u32 v = 0; for (int i = 0; i < 4; ++i) v |= u32(bytes[pos++]) << (8 * i); return v; };
    auto rdf  = [&] { u32 v = rd32(); f32 f; std::memcpy(&f, &v, 4); return f; };
    p.dc = rdf(); p.q = rdf(); p.tolerance = rdf();
    for (u32 l = 0; l < DWT_LEVELS; ++l)
        for (u32 o = 0; o < 8; ++o) p.steps.step[l][o] = rdf();
    u32 tlen = rd32(), blen = rd32(), olen = rd32();
    p.tokens.assign(bytes.begin() + pos, bytes.begin() + pos + tlen); pos += tlen;
    p.bypass.assign(bytes.begin() + pos, bytes.begin() + pos + blen); pos += blen;
    p.outliers.assign(bytes.begin() + pos, bytes.begin() + pos + olen); pos += olen;
    return p;
}

} // namespace c4d::chunk
