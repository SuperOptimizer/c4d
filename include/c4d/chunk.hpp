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
#include <cstdlib>
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

// Context modeling (§4.8). Each token is coded against a static histogram chosen
// by a cheap context id. v2 context (DATA-DRIVEN, measured on the corpus):
//   level-bucket {0,1,2+}  ×  prev-token-class {run,mag}  ×  neighbor-magnitude
//   bucket {0,1,2,3,4}  (sum of |causal spatial neighbors -z,-y,-x|, EBCOT/SPIHT)
// The spatial-neighbor axis is the canonical wavelet context and measured
// -4.3..-4.9% smaller vs the old (level×prev-class) model — it BEAT a real zstd
// dictionary on our data. Neighbors are causal (already decoded in subband-raster
// order), so encode and decode derive the context identically. Static-per-region
// tables keep decode SIMD (just point at another table per symbol).
inline constexpr u32 NLEV = 3;    // level buckets {0,1,2+}
inline constexpr u32 NPC  = 2;    // prev-token class {run, mag}
inline constexpr u32 NNB  = 5;    // neighbor-magnitude buckets
inline constexpr u32 NUM_CTX = NLEV * NPC * NNB;   // 30

[[nodiscard]] inline u32 neigh_bucket(u32 nsum) noexcept {
    return nsum == 0 ? 0u : nsum < 2 ? 1u : nsum < 4 ? 2u : nsum < 8 ? 3u : 4u;
}
[[nodiscard]] inline u32 context_id(u32 subband_id, u32 prev_class, u32 nbucket) noexcept {
    u32 lvl = subband_id >> 3;
    u32 lb  = lvl >= 2 ? 2u : lvl;                  // level bucket {0,1,2+}
    return ((lb * NPC + (prev_class & 1u)) * NNB) + nbucket;   // 0..29
}

// Sum of |causal spatial neighbors| (−z, −y, −x in the transformed cube) at a
// linear index, counting ONLY same-subband neighbors. Restricting to the same
// subband guarantees those neighbors are already decoded in subband-raster scan
// (a different-subband neighbor may be scanned later → encode/decode desync).
[[nodiscard]] inline u32 causal_neigh_sum(const std::vector<i32>& ql,
                                          const SubbandMap& sm, u32 idx) noexcept {
    // CHUNK is a power of two; derive the on-face tests with masks, no div/mod.
    constexpr u32 PLANE = CHUNK * CHUNK;
    const i32* q = ql.data(); const u8* id = sm.id.data();
    u8 c = id[idx];
    u32 s = 0;
    if ((idx & (CHUNK - 1))   && id[idx - 1]     == c) s += static_cast<u32>(std::abs(q[idx - 1]));
    if ((idx & (PLANE - 1)) >= CHUNK && id[idx - CHUNK] == c) s += static_cast<u32>(std::abs(q[idx - CHUNK]));
    if (idx >= PLANE && id[idx - PLANE] == c)          s += static_cast<u32>(std::abs(q[idx - PLANE]));
    return s;
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
    // Uniform fast-path (§7 T2): a perfectly-constant chunk codes as just its
    // value (no DWT/quant/entropy/tables). uniform=true => decode fills `uval`.
    bool uniform = false;
    u8   uval = 0;
    // Shared-tables mode: when true, the freq tables are NOT in `tokens` (they're
    // carried once at group/member level); decode must supply them externally.
    bool shared_tables = false;
    // Serialized form is produced by serialize(); the archive stores that blob.
    std::vector<u8> serialize() const;
    static Payload deserialize(std::span<const u8> bytes);
    size_t size_bytes() const {
        return uniform ? 2 : (tokens.size() + bypass.size() + 16 + sizeof(StepTable));
    }
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

struct SharedTables;
inline void decode_chunk(const Payload& p, std::span<u8> out);  // fwd decl
inline void decode_chunk_shared(const Payload& p, const SharedTables* shared,
                                std::span<u8> out);              // fwd decl

// Encoder options. The decoder reads steps from the stream regardless, so these
// only affect the encoder's step-table choice (§4.10) — never the bitstream
// shape or the decoder.
struct EncodeOpts {
    f32 q = 16.f;                 // global quality knob (base step scale)
    bool noise_aware = false;     // MAD + BayesShrink dead-zone (§4.10)
    f64 shrink_strength = 1.0;    // 0..1, how aggressively to apply shrinkage
    f32 tolerance = 0;            // outlier pass: hard L-inf bound t (0 => off, §4.6)
    bool perceptual_rdo = false;  // coherence-gated HF preservation (§4.10)
    f64 rdo_strength = 0.5;       // 0..1, how much to tighten coherent HF
};

// Per-context shared entropy tables (§ shared-tables / no-independence). When a
// group of chunks shares one table set, the per-chunk table overhead (NUM_CTX
// serialized freq tables, ~1.5 KB) is paid ONCE for the group instead of per
// chunk. Measured 4-8% smaller total at 64³. The decoder must be given the same
// tables (carried once at group/member level).
struct SharedTables {
    std::array<rans::FreqTable, NUM_CTX> tbl;
    std::array<std::vector<u32>, NUM_CTX> counts;   // accumulator during analyze
    SharedTables() { for (auto& c : counts) c.assign(ALPHABET, 0); }
    void accumulate(const std::array<std::vector<u32>, NUM_CTX>& src) {
        for (u32 c = 0; c < NUM_CTX; ++c)
            for (u32 s = 0; s < ALPHABET; ++s) counts[c][s] += src[c][s];
    }
    void build() { for (u32 c = 0; c < NUM_CTX; ++c) tbl[c] = rans::FreqTable::build(counts[c]); }
    void serialize(std::vector<u8>& out) const { for (u32 c = 0; c < NUM_CTX; ++c) tbl[c].serialize(out); }
    static SharedTables deserialize(std::span<const u8> in, size_t& pos) {
        SharedTables s;
        for (u32 c = 0; c < NUM_CTX; ++c) s.tbl[c] = rans::FreqTable::deserialize(in, pos);
        return s;
    }
};

// Phase 1 of encoding: everything up to (not including) the rANS pass — DC, DWT,
// quantization, token generation. Reused by both per-chunk and group encode.
struct ChunkAnalysis {
    bool uniform = false; u8 uval = 0;
    f32 dc = 0, q = 0, tolerance = 0;
    StepTable steps{};
    std::vector<u32> toks;                       // forward-order tokens
    std::vector<u8>  ctx;                        // per-token context id
    std::array<std::vector<u32>, NUM_CTX> counts;// per-context histograms (for tables)
    std::vector<u8>  bypass_bytes;               // finished bypass stream
    std::vector<u8>  orig;                        // kept for the outlier pass
};

inline ChunkAnalysis analyze_chunk(std::span<const u8> vox, const EncodeOpts& opt) {
    ChunkAnalysis a; a.q = opt.q; a.tolerance = opt.tolerance;
    { u8 v0 = vox.empty() ? 0 : vox[0]; bool uni = true;
      for (u8 v : vox) if (v != v0) { uni = false; break; }
      if (uni) { a.uniform = true; a.uval = v0; return a; } }

    std::vector<f32> coef(CHUNK_VOX);
    f64 mean = 0; for (u8 v : vox) mean += v; mean /= CHUNK_VOX;
    for (u32 i = 0; i < CHUNK_VOX; ++i) coef[i] = static_cast<f32>(vox[i]) - static_cast<f32>(mean);
    dwt::forward(coef.data());
    a.steps = StepTable::from_q(opt.q);
    f64 sigma = (opt.noise_aware || opt.perceptual_rdo) ? estimate_noise_sigma(coef) : 0.0;
    if (opt.noise_aware)   a.steps = NoiseShrink::analyze(coef).apply(a.steps, opt.shrink_strength);
    if (opt.perceptual_rdo) a.steps = PerceptualRDO::apply(a.steps, vox, sigma, opt.rdo_strength);
    std::vector<i32> ql(CHUNK_VOX);
    quantize(coef, a.steps, ql);

    TokenStream ts;
    const std::vector<u32>& order = scan_order().order;
    const SubbandMap& sm = subband_map();
    // Context per token: subband level × prev-class × causal-neighbor-magnitude
    // bucket. A run token uses the subband+neighbors at its START position; a
    // magnitude token uses its own position. Neighbors are causal (decoded).
    u32 zeros = 0, prev_class = 1, run_start_sb = 0, run_start_nb = 0;
    for (u32 idx : order) {
        i32 v = ql[idx];
        if (v == 0) {
            if (zeros == 0) { run_start_sb = sm.id[idx]; run_start_nb = neigh_bucket(causal_neigh_sum(ql, sm, idx)); }
            ++zeros; continue;
        }
        if (zeros) { ts.emit_run(zeros, context_id(run_start_sb, prev_class, run_start_nb)); zeros = 0; prev_class = 0; }
        u32 nb = neigh_bucket(causal_neigh_sum(ql, sm, idx));
        ts.emit_nonzero(v, context_id(sm.id[idx], prev_class, nb)); prev_class = 1;
    }
    if (zeros) ts.emit_run(zeros, context_id(run_start_sb, prev_class, run_start_nb));
    a.dc = static_cast<f32>(mean);
    a.toks = std::move(ts.toks);
    a.ctx = std::move(ts.ctx);
    a.counts = std::move(ts.counts);
    a.bypass_bytes = ts.bypass.finish();
    if (opt.tolerance > 0) a.orig.assign(vox.begin(), vox.end());
    return a;
}

// Phase 2: rANS the analyzed tokens. If `shared` is non-null, code against the
// shared tables and OMIT per-chunk table serialization (the group carries them);
// otherwise build+serialize per-chunk tables (the independent path).
inline Payload finalize_chunk(const ChunkAnalysis& a, const SharedTables* shared) {
    Payload p;
    if (a.uniform) { p.uniform = true; p.uval = a.uval; p.q = a.q; return p; }
    p.dc = a.dc; p.q = a.q; p.steps = a.steps; p.tolerance = a.tolerance;
    p.shared_tables = (shared != nullptr);

    const std::array<rans::FreqTable, NUM_CTX>* tbls;
    std::array<rans::FreqTable, NUM_CTX> own;
    if (shared) tbls = &shared->tbl;
    else { for (u32 c = 0; c < NUM_CTX; ++c) own[c] = rans::FreqTable::build(a.counts[c]); tbls = &own; }

    rans::Encoder enc;
    for (i64 i = static_cast<i64>(a.toks.size()) - 1; i >= 0; --i)
        enc.put((*tbls)[a.ctx[i]], a.toks[i]);
    auto rans_bytes = enc.finish();

    if (!shared) for (u32 c = 0; c < NUM_CTX; ++c) own[c].serialize(p.tokens);  // per-chunk tables
    for (int i = 0; i < 4; ++i) p.tokens.push_back(u8((u32(a.toks.size()) >> (8 * i)) & 0xff));
    p.tokens.insert(p.tokens.end(), rans_bytes.begin(), rans_bytes.end());
    p.bypass = a.bypass_bytes;
    return p;
}

// Encode one chunk (independent, per-chunk tables). `vox` is row-major Z,Y,X.
inline Payload encode_chunk(std::span<const u8> vox, const EncodeOpts& opt) {
    ChunkAnalysis a = analyze_chunk(vox, opt);
    Payload p = finalize_chunk(a, nullptr);            // per-chunk tables

    // Outlier pass (§4.6): internally decode the stream we just built, find voxels
    // outside the L-inf tolerance, code the sparse corrections as a second stream.
    if (opt.tolerance > 0 && !p.uniform) {
        std::vector<u8> recon(CHUNK_VOX);
        decode_chunk(p, recon);                        // recon WITHOUT corrections yet
        auto cs = outlier::find(vox, recon, opt.tolerance);
        p.outliers = outlier::encode(cs);
    }
    return p;
}

// --- Group encode (§ shared tables) ----------------------------------------
// Encode a set of chunks sharing ONE entropy-table set. Two passes: analyze all
// chunks (gather histograms), build the shared tables, then finalize each chunk
// against them. Returns the shared tables (caller stores them once) + payloads.
// Outlier pass is applied per chunk (it needs the chunk's own decode, which uses
// the shared tables — passed through here).
struct GroupEncoded {
    SharedTables tables;
    std::vector<Payload> payloads;
};
inline GroupEncoded encode_group(const std::vector<std::span<const u8>>& voxs,
                                 const EncodeOpts& opt) {
    GroupEncoded g;
    std::vector<ChunkAnalysis> ana;
    ana.reserve(voxs.size());
    for (auto vox : voxs) {
        ana.push_back(analyze_chunk(vox, opt));
        if (!ana.back().uniform) g.tables.accumulate(ana.back().counts);
    }
    g.tables.build();
    g.payloads.reserve(voxs.size());
    for (size_t i = 0; i < ana.size(); ++i) {
        Payload p = finalize_chunk(ana[i], ana[i].uniform ? nullptr : &g.tables);
        if (opt.tolerance > 0 && !p.uniform) {
            std::vector<u8> recon(CHUNK_VOX);
            decode_chunk_shared(p, &g.tables, recon);
            auto cs = outlier::find(voxs[i], recon, opt.tolerance);
            p.outliers = outlier::encode(cs);
        }
        g.payloads.push_back(std::move(p));
    }
    return g;
}

// Convenience overload: encode at a plain global quality knob.
inline Payload encode_chunk(std::span<const u8> vox, f32 q) {
    return encode_chunk(vox, EncodeOpts{.q = q});
}

// Decode a chunk payload to a CHUNK^3 u8 cube. If the payload is in shared-tables
// mode, `shared` must supply the group's tables (they're not in the payload).
inline void decode_chunk_shared(const Payload& p, const SharedTables* shared,
                                std::span<u8> out) {
    if (p.uniform) { std::fill(out.begin(), out.end(), p.uval); return; }  // §7 T2

    size_t pos = 0;
    std::array<rans::FreqTable, NUM_CTX> own;
    const std::array<rans::FreqTable, NUM_CTX>* tbls;
    if (p.shared_tables) { tbls = &shared->tbl; }     // tables external
    else { for (u32 c = 0; c < NUM_CTX; ++c) own[c] = rans::FreqTable::deserialize(p.tokens, pos); tbls = &own; }
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
        // context = level-bucket × prev_class × causal-neighbor-magnitude bucket,
        // all derived from the START position (oi) and already-decoded same-subband
        // neighbors — identical to what the encoder computed there.
        u32 idx0 = (oi < CHUNK_VOX) ? order[oi] : 0;
        u32 sb = (oi < CHUNK_VOX) ? sm.id[idx0] : 0;
        u32 nb = (oi < CHUNK_VOX) ? neigh_bucket(causal_neigh_sum(ql, sm, idx0)) : 0;
        u32 token = dec.get((*tbls)[context_id(sb, prev_class, nb)]);
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

// Decode an independent (per-chunk-tables) payload.
inline void decode_chunk(const Payload& p, std::span<u8> out) {
    decode_chunk_shared(p, nullptr, out);
}

// --- payload serialization (flat blob for the archive) ---------------------
inline std::vector<u8> Payload::serialize() const {
    std::vector<u8> b;
    // tag byte: 1 = uniform fast-path (value follows), 0 = normal payload.
    // tag: 1=uniform, 2=normal+shared-tables (tables external), 0=normal per-chunk
    if (uniform) { b.push_back(1); b.push_back(uval); return b; }
    b.push_back(shared_tables ? u8(2) : u8(0));
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
    u8 tag = bytes[pos++];
    if (tag == 1) { p.uniform = true; p.uval = bytes[pos]; return p; }  // §7 T2
    p.shared_tables = (tag == 2);
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
