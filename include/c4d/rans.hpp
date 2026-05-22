// Static rANS entropy coder — 32-bit state, 16-bit renormalization, static
// per-chunk histograms (12-bit total frequency precision). Spec §4.9.
//
// Correctness-first scalar implementation. The interleaved-SIMD variant
// (8/16/32-way, §4.9) is a later drop-in: same byte framing, lanes interleaved.
//
// Two cooperating streams (the JPEG XL split, §4.7 HybridUint):
//   - the rANS symbol stream: small token alphabet, static histogram;
//   - a raw "bypass" bitstream: uniform mantissa bits + signs, plain-packed
//     (rANS gives uniform data nothing, and bit-packing is faster).
// They are serialized independently and concatenated by the chunk coder.
#pragma once
#include "core.hpp"
#include <cstring>
#include <span>
#include <vector>

namespace c4d::rans {

inline constexpr u32 PROB_BITS  = 12;
inline constexpr u32 PROB_SCALE = 1u << PROB_BITS;   // freqs sum to 4096
inline constexpr u32 RANS_L     = 1u << 16;          // renorm lower bound

// --- Frequency table -------------------------------------------------------
struct FreqTable {
    u32 nsym = 0;
    std::vector<u16> freq;      // normalized, sum == PROB_SCALE
    std::vector<u16> cum;       // [nsym+1] cumulative starts
    std::vector<u16> slot2sym;  // [PROB_SCALE] slot -> symbol (decode)
    // Per-symbol reciprocals so Encoder::put replaces the per-token integer
    // `state/freq` (a ~20-cycle div) with a 128-bit multiply + shift (~5 cycles).
    std::vector<u64> rcp_mul64;

    static FreqTable build(std::span<const u32> counts) {
        FreqTable t;
        t.nsym = static_cast<u32>(counts.size());
        t.freq.assign(t.nsym, 0);
        u64 total = 0; for (u32 c : counts) total += c;
        if (total == 0) {                       // degenerate: all symbols freq 1
            for (u32 s = 0; s < t.nsym; ++s) t.freq[s] = 1;
            fixup_sum(t);
            t.finish();
            return t;
        }
        for (u32 s = 0; s < t.nsym; ++s) {
            if (counts[s] == 0) continue;
            u32 f = static_cast<u32>((u64(counts[s]) * PROB_SCALE) / total);
            t.freq[s] = static_cast<u16>(f == 0 ? 1u : f);
        }
        fixup_sum(t);
        t.finish();
        return t;
    }

    void finish() {
        cum.assign(nsym + 1, 0);
        for (u32 s = 0; s < nsym; ++s) cum[s + 1] = static_cast<u16>(cum[s] + freq[s]);
        slot2sym.assign(PROB_SCALE, 0);
        for (u32 s = 0; s < nsym; ++s)
            for (u32 i = cum[s]; i < cum[s + 1]; ++i) slot2sym[i] = static_cast<u16>(s);
        // Round-up reciprocal: m = floor(2^SH / f) + 1, so floor(x/f) =
        // (x*m) >> SH for all x < 2^32. SH=48 makes it exact across our range
        // (verified exhaustively); the product x*m needs 128 bits. f==0 unused.
        rcp_mul64.assign(nsym, 0);
        for (u32 s = 0; s < nsym; ++s)
            if (freq[s]) rcp_mul64[s] = (u64(1) << RCP_SH) / freq[s] + 1;
    }
    static constexpr u32 RCP_SH = 48;

    // floor(x / freq[s]) via the precomputed reciprocal (no integer div).
    [[nodiscard]] u32 divf(u32 x, u32 s) const noexcept {
        return static_cast<u32>((static_cast<unsigned __int128>(x) * rcp_mul64[s]) >> RCP_SH);
    }

    // Serialize the normalized frequencies (the decoder rebuilds cum/slot2sym).
    void serialize(std::vector<u8>& out) const {
        push_u32(out, nsym);
        for (u16 f : freq) { out.push_back(u8(f & 0xff)); out.push_back(u8(f >> 8)); }
    }
    static FreqTable deserialize(std::span<const u8> in, size_t& pos) {
        FreqTable t;
        t.nsym = read_u32(in, pos);
        t.freq.resize(t.nsym);
        for (u32 s = 0; s < t.nsym; ++s) {
            t.freq[s] = static_cast<u16>(in[pos] | (u32(in[pos + 1]) << 8));
            pos += 2;
        }
        t.finish();
        return t;
    }

private:
    static void fixup_sum(FreqTable& t) {
        u32 sum = 0; for (u16 f : t.freq) sum += f;
        if (sum == PROB_SCALE) return;
        u32 best = 0; for (u32 s = 1; s < t.nsym; ++s) if (t.freq[s] > t.freq[best]) best = s;
        t.freq[best] = static_cast<u16>(static_cast<i32>(t.freq[best])
                                        + (static_cast<i32>(PROB_SCALE) - static_cast<i32>(sum)));
    }
    static void push_u32(std::vector<u8>& o, u32 v) {
        for (int i = 0; i < 4; ++i) o.push_back(u8((v >> (8 * i)) & 0xff));
    }
    static u32 read_u32(std::span<const u8> in, size_t& pos) {
        u32 v = 0; for (int i = 0; i < 4; ++i) v |= u32(in[pos++]) << (8 * i); return v;
    }
};

// --- rANS symbol encoder (reverse) -----------------------------------------
class Encoder {
public:
    void put(const FreqTable& t, u32 sym) {
        u32 f = t.freq[sym], c = t.cum[sym];
        // Renorm threshold; u64 because freq up to PROB_SCALE makes this exceed
        // 2^32 (a single-symbol alphabet hits exactly that).
        u64 x_max = (u64((RANS_L >> PROB_BITS) << 16)) * f;
        while (state_ >= x_max) { out_.push_back(u16(state_ & 0xffff)); state_ >>= 16; }
        // state/f and state%f via the precomputed reciprocal (no integer div).
        u32 q = t.divf(state_, sym);
        state_ = (q << PROB_BITS) + (state_ - q * f) + c;
    }
    std::vector<u8> finish() {
        out_.push_back(u16(state_ & 0xffff));
        out_.push_back(u16((state_ >> 16) & 0xffff));
        std::vector<u8> bytes(out_.size() * 2);
        for (size_t i = 0; i < out_.size(); ++i) {  // reverse word order
            u16 w = out_[out_.size() - 1 - i];
            bytes[2 * i] = u8(w & 0xff); bytes[2 * i + 1] = u8(w >> 8);
        }
        return bytes;
    }
private:
    u32 state_ = RANS_L;
    std::vector<u16> out_;
};

// --- rANS symbol decoder ---------------------------------------------------
class Decoder {
public:
    explicit Decoder(std::span<const u8> bytes) : data_(bytes) {
        u32 hi = read_word(), lo = read_word();
        state_ = (hi << 16) | lo;
    }
    u32 get(const FreqTable& t) {
        u32 slot = state_ & (PROB_SCALE - 1);
        u32 sym = t.slot2sym[slot];
        state_ = t.freq[sym] * (state_ >> PROB_BITS) + slot - t.cum[sym];
        while (state_ < RANS_L) state_ = (state_ << 16) | read_word();
        return sym;
    }
private:
    u32 read_word() {
        u32 w = 0;
        if (pos_ < data_.size())     w  = data_[pos_];
        if (pos_ + 1 < data_.size()) w |= u32(data_[pos_ + 1]) << 8;
        pos_ += 2;
        return w;
    }
    std::span<const u8> data_;
    size_t pos_ = 0;
    u32 state_ = 0;
};

// --- Raw bypass bitstream (uniform bits: HybridUint mantissa + sign) -------
// Plain LSB-first bit packing. Optimal for uniform data and trivially fast.
class BitWriter {
public:
    BitWriter() { out_.reserve(1 << 16); }     // avoid early reallocs
    // Append nbits (<=32) of `bits`, LSB-first. Flushes whole 32-bit words when
    // the 64-bit accumulator can hold no more, in one memcpy-friendly push —
    // avoids the per-byte push_back loop that dominated tokengen.
    void put(u32 bits, u32 nbits) {
        if (nfill_ + nbits > 64) flush32();    // keep room (nbits<=32 => fits after)
        acc_ |= u64(bits & ((nbits < 32) ? ((1u << nbits) - 1) : 0xffffffffu)) << nfill_;
        nfill_ += nbits;
    }
    std::vector<u8> finish() {
        while (nfill_ >= 8) { out_.push_back(u8(acc_ & 0xff)); acc_ >>= 8; nfill_ -= 8; }
        if (nfill_ > 0) { out_.push_back(u8(acc_ & 0xff)); acc_ = 0; nfill_ = 0; }
        return std::move(out_);
    }
private:
    void flush32() {                            // emit the low 32 bits as 4 bytes
        u32 w = static_cast<u32>(acc_);
        out_.push_back(u8(w)); out_.push_back(u8(w >> 8));
        out_.push_back(u8(w >> 16)); out_.push_back(u8(w >> 24));
        acc_ >>= 32; nfill_ -= 32;
    }
    std::vector<u8> out_;
    u64 acc_ = 0;
    u32 nfill_ = 0;
};

class BitReader {
public:
    explicit BitReader(std::span<const u8> bytes) : data_(bytes) {}
    u32 get(u32 nbits) {
        if (nfill_ < nbits) refill();          // pull in a whole word at once
        u32 v = static_cast<u32>(acc_ & ((nbits < 32) ? ((1u << nbits) - 1) : 0xffffffffu));
        acc_ >>= nbits; nfill_ -= nbits;
        return v;
    }
private:
    void refill() {                            // top up the accumulator (nbits<=32)
        if (pos_ + 4 <= data_.size()) {        // fast path: one 32-bit load
            u32 w = data_[pos_] | (u32(data_[pos_+1]) << 8)
                  | (u32(data_[pos_+2]) << 16) | (u32(data_[pos_+3]) << 24);
            acc_ |= u64(w) << nfill_; nfill_ += 32; pos_ += 4;
        } else {
            while (nfill_ <= 56 && pos_ < data_.size()) {
                acc_ |= u64(data_[pos_++]) << nfill_; nfill_ += 8;
            }
            if (pos_ >= data_.size()) nfill_ = 64;   // pad with zeros past EOF
        }
    }
    std::span<const u8> data_;
    size_t pos_ = 0;
    u64 acc_ = 0;
    u32 nfill_ = 0;
};

} // namespace c4d::rans
