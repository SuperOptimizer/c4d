// HybridUint coefficient framing (§4.7): split a nonnegative integer into a
// small rANS *token* (the magnitude bucket) plus raw mantissa bits sent to the
// bypass stream. This keeps the rANS alphabet tiny and its histogram sharp.
//
// Encoding (the classic "exp-Golomb-like" token): for value u,
//   u < SPLIT             -> token = u                    (no raw bits)
//   u >= SPLIT            -> let n = floor(log2(u)); token encodes n and the
//                            top fractional bit(s); the remaining low bits are raw.
// We use the simple JPEG-XL-style hybrid with `split_token` direct values and
// then power-of-two buckets: token t>=SPLIT means u in [2^(t-SPLIT+k), ...) with
// (t-SPLIT+k) raw mantissa bits. We pick a compact scheme below and unit-test the
// roundtrip exhaustively over the u8-coefficient range we actually hit.
#pragma once
#include "core.hpp"
#include <bit>
#include <utility>

namespace c4d::hybrid {

// Values 0..DIRECT-1 are coded literally as their own token (no raw bits).
// Larger values: token = DIRECT + (nbits-1)*2 + topbit-ish bucket, with the
// remaining low bits raw. We use the standard "msb + 1 extra bit" bucketing,
// which is a good speed/ratio balance and keeps the alphabet small.
inline constexpr u32 DIRECT = 4;          // 0,1,2,3 literal
inline constexpr u32 MAX_TOKEN = 64;      // alphabet ceiling (u8 coeffs are small)

// Token + (raw value, nbits) for a nonnegative magnitude `u`.
struct Split { u32 token; u32 raw; u32 nbits; };

[[nodiscard]] inline Split encode(u32 u) noexcept {
    if (u < DIRECT) return {u, 0, 0};
    // n = position of MSB (>=2 here since u>=DIRECT>=4 => MSB>=2)
    u32 n = 31u - static_cast<u32>(std::countl_zero(u));   // floor(log2 u)
    // Use the top bit below the MSB to split each power-of-two band into two,
    // matching JPEG XL's hybrid: msb 'n', one selector bit 'b', (n-1) raw bits.
    u32 b = (u >> (n - 1)) & 1u;                 // bit just below the MSB
    u32 raw_bits = n - 1;
    u32 raw = u & ((1u << raw_bits) - 1);
    u32 token = DIRECT + (n - 2) * 2 + b;        // bands from n=2 upward
    return {token, raw, raw_bits};
}

[[nodiscard]] inline u32 decode(u32 token, u32 raw) noexcept {
    if (token < DIRECT) return token;
    u32 k = token - DIRECT;
    u32 n = (k / 2) + 2;          // recover MSB position
    u32 b = k & 1u;               // selector bit
    u32 raw_bits = n - 1;
    return (1u << n) | (b << (n - 1)) | (raw & ((1u << raw_bits) - 1));
}

// Number of raw mantissa bits a given token implies (decoder needs this before
// reading the bypass stream).
[[nodiscard]] inline u32 raw_bits_of(u32 token) noexcept {
    if (token < DIRECT) return 0;
    u32 n = ((token - DIRECT) / 2) + 2;
    return n - 1;
}

} // namespace c4d::hybrid
