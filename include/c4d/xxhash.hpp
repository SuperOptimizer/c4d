// Compact XXH64 — for per-chunk and index checksums (§5.2). Not cryptographic;
// just fast corruption detection. Standard xxHash64 (Yann Collet), single-shot.
#pragma once
#include "core.hpp"
#include <bit>
#include <cstring>
#include <span>

namespace c4d {

[[nodiscard]] inline u64 xxhash64(std::span<const u8> data, u64 seed = 0) noexcept {
    constexpr u64 P1 = 0x9E3779B185EBCA87ull, P2 = 0xC2B2AE3D27D4EB4Full,
                  P3 = 0x165667B19E3779F9ull, P4 = 0x85EBCA77C2B2AE63ull,
                  P5 = 0x27D4EB2F165667C5ull;
    const u8* p = data.data();
    const u8* end = p + data.size();
    auto rd64 = [](const u8* q) { u64 v; std::memcpy(&v, q, 8); return v; };
    auto rd32 = [](const u8* q) { u32 v; std::memcpy(&v, q, 4); return v; };
    auto round = [](u64 acc, u64 in) { acc += in * P2; acc = std::rotl(acc, 31); return acc * P1; };

    u64 h;
    if (data.size() >= 32) {
        u64 v1 = seed + P1 + P2, v2 = seed + P2, v3 = seed, v4 = seed - P1;
        const u8* limit = end - 32;
        do {
            v1 = round(v1, rd64(p)); p += 8;
            v2 = round(v2, rd64(p)); p += 8;
            v3 = round(v3, rd64(p)); p += 8;
            v4 = round(v4, rd64(p)); p += 8;
        } while (p <= limit);
        h = std::rotl(v1, 1) + std::rotl(v2, 7) + std::rotl(v3, 12) + std::rotl(v4, 18);
        auto merge = [&](u64 acc, u64 v) { v = round(0, v); acc ^= v; return acc * P1 + P4; };
        h = merge(h, v1); h = merge(h, v2); h = merge(h, v3); h = merge(h, v4);
    } else {
        h = seed + P5;
    }
    h += static_cast<u64>(data.size());
    while (p + 8 <= end) { u64 k = round(0, rd64(p)); h ^= k; h = std::rotl(h, 27) * P1 + P4; p += 8; }
    if (p + 4 <= end) { h ^= u64(rd32(p)) * P1; h = std::rotl(h, 23) * P2 + P3; p += 4; }
    while (p < end) { h ^= u64(*p) * P5; h = std::rotl(h, 11) * P1; ++p; }
    h ^= h >> 33; h *= P2; h ^= h >> 29; h *= P3; h ^= h >> 32;
    return h;
}

} // namespace c4d
