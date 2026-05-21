// Single-file .c4d archive: payloads, per-member fixed-width chunk index,
// member directory, JSON metadata, fixed trailer at EOF. Spec §5.
//
// Layout (low->high offset):
//   [ chunk payloads (concatenated) ]
//   [ per-member chunk indexes      ]
//   [ member directory              ]
//   [ metadata (JSON)               ]
//   [ fixed trailer (at EOF)        ]
// Read path: tail-read the trailer -> directory -> member -> its index ->
// range-GET payload(s). Append: write new payloads + fresh index/dir/meta/
// trailer at EOF, never rewriting existing bytes (§5.5).
//
// All integers little-endian (§1.1). This is a self-contained writer/reader over
// an in-memory or mmap'd byte span; the I/O (file / S3 range-GET) is the
// caller's, but the byte layout here is the wire format.
#pragma once
#include "core.hpp"
#include "chunk.hpp"
#include "xxhash.hpp"
#include <cstring>
#include <span>
#include <string>
#include <vector>

namespace c4d::archive {

inline constexpr u8  MAGIC[4]   = {'C', '4', 'D', 0};
inline constexpr u32 FORMAT_VER = 1;
inline constexpr u32 TRAILER_SIZE = 48;   // fixed tail read

enum class MemberType : u32 { Intensity = 0, ValidityMask = 1, Metadata = 2 };

// Per-chunk index entry (fixed width => O(1) random lookup, §5.2).
struct IndexEntry {
    u64 offset = 0;       // payload offset from file start
    u64 length = 0;       // payload byte length (0 + ABSENT flag => all-zero)
    u64 checksum = 0;     // xxhash64 of the payload
    f32 q = 0;            // per-chunk quality knob
    u8  flags = 0;        // bit0: ABSENT (all-zero / padded; no payload)
    u8  _pad[3] = {0,0,0};
    static constexpr u8 ABSENT = 1;
};
// 3*u64 + f32 + u8 + 3*u8 = 32 bytes (8-aligned). Fixed width => O(1) lookup.
static_assert(sizeof(IndexEntry) == 32, "fixed-width index entry");

// Member descriptor in the directory (§5.2).
struct Member {
    std::string name;
    MemberType type = MemberType::Intensity;
    u32 codec = 0;            // codec tag (0 = c4d wavelet)
    Coord3 shape{};           // logical Z,Y,X extent (bbox lives in metadata)
    u32 chunk_size = CHUNK;   // per-member chunk edge
    std::vector<IndexEntry> index;  // chunk-grid order, coord implicit
};

// --- Writer ----------------------------------------------------------------
// Builds an archive in memory. add_member appends a member's chunk payloads;
// finish() serializes payloads + indexes + directory + metadata + trailer.
class Writer {
public:
    // Register a member; `chunks` are payload blobs in chunk-grid order. An
    // empty blob marks an ABSENT (all-zero) chunk (no bytes stored).
    void add_member(std::string name, MemberType type, Coord3 shape,
                    const std::vector<std::vector<u8>>& chunks,
                    const std::vector<f32>& qs) {
        Member m;
        m.name = std::move(name); m.type = type; m.shape = shape;
        m.index.reserve(chunks.size());
        for (size_t i = 0; i < chunks.size(); ++i) {
            IndexEntry e;
            e.q = (i < qs.size()) ? qs[i] : 0.f;
            if (chunks[i].empty()) {
                e.flags = IndexEntry::ABSENT;
            } else {
                e.offset = blob_.size();
                e.length = chunks[i].size();
                e.checksum = xxhash64(chunks[i]);
                blob_.insert(blob_.end(), chunks[i].begin(), chunks[i].end());
            }
            m.index.push_back(e);
        }
        members_.push_back(std::move(m));
    }

    void set_metadata(std::string json) { metadata_ = std::move(json); }

    std::vector<u8> finish() {
        std::vector<u8> out = std::move(blob_);

        // per-member indexes (record their offsets/lengths for the directory)
        std::vector<u64> idx_off(members_.size()), idx_len(members_.size());
        std::vector<u64> idx_sum(members_.size());
        for (size_t mi = 0; mi < members_.size(); ++mi) {
            idx_off[mi] = out.size();
            const auto& idx = members_[mi].index;
            const u8* raw = reinterpret_cast<const u8*>(idx.data());
            size_t bytes = idx.size() * sizeof(IndexEntry);
            out.insert(out.end(), raw, raw + bytes);
            idx_len[mi] = bytes;
            idx_sum[mi] = xxhash64({raw, bytes});
        }

        // member directory
        u64 dir_off = out.size();
        put_u32(out, static_cast<u32>(members_.size()));
        for (size_t mi = 0; mi < members_.size(); ++mi) {
            const Member& m = members_[mi];
            put_u32(out, static_cast<u32>(m.name.size()));
            out.insert(out.end(), m.name.begin(), m.name.end());
            put_u32(out, static_cast<u32>(m.type));
            put_u32(out, m.codec);
            put_u64(out, m.shape.z); put_u64(out, m.shape.y); put_u64(out, m.shape.x);
            put_u32(out, m.chunk_size);
            put_u64(out, static_cast<u64>(m.index.size()));
            put_u64(out, idx_off[mi]); put_u64(out, idx_len[mi]); put_u64(out, idx_sum[mi]);
        }
        u64 dir_len = out.size() - dir_off;
        u64 dir_sum = xxhash64({out.data() + dir_off, dir_len});

        // metadata (JSON)
        u64 meta_off = out.size();
        out.insert(out.end(), metadata_.begin(), metadata_.end());
        u64 meta_len = metadata_.size();

        // fixed trailer at EOF (TRAILER_SIZE bytes)
        size_t tstart = out.size();
        out.insert(out.end(), MAGIC, MAGIC + 4);     // 0: magic
        put_u32(out, FORMAT_VER);                     // 4: version
        put_u64(out, dir_off);                        // 8: directory offset
        put_u64(out, dir_len);                        // 16: directory length
        put_u64(out, dir_sum);                        // 24: directory checksum
        put_u64(out, meta_off);                       // 32: metadata offset
        put_u32(out, static_cast<u32>(meta_len));     // 40: metadata length
        out.insert(out.end(), MAGIC, MAGIC + 4);      // 44: trailing magic
        (void)tstart;
        return out;
    }

private:
    static void put_u32(std::vector<u8>& o, u32 v) { for (int i=0;i<4;++i) o.push_back(u8((v>>(8*i))&0xff)); }
    static void put_u64(std::vector<u8>& o, u64 v) { for (int i=0;i<8;++i) o.push_back(u8((v>>(8*i))&0xff)); }

    std::vector<u8> blob_;          // payloads accumulate here first
    std::vector<Member> members_;
    std::string metadata_;
};

// --- Reader ----------------------------------------------------------------
// Opens an archive byte span; parses trailer -> directory. Payloads fetched on
// demand by member+chunk index (range-GET friendly).
class Reader {
public:
    explicit Reader(std::span<const u8> bytes) : data_(bytes) { parse(); }

    size_t member_count() const { return members_.size(); }
    const Member& member(size_t i) const { return members_[i]; }
    std::string_view metadata() const { return meta_; }

    // Find a member by name; returns index or SIZE_MAX.
    size_t find(std::string_view name) const {
        for (size_t i = 0; i < members_.size(); ++i) if (members_[i].name == name) return i;
        return SIZE_MAX;
    }

    // Fetch chunk `ci` of member `mi` as a payload span (empty if ABSENT).
    // Verifies the per-chunk checksum.
    std::span<const u8> chunk_payload(size_t mi, u64 ci) const {
        const IndexEntry& e = members_[mi].index[ci];
        if (e.flags & IndexEntry::ABSENT) return {};
        std::span<const u8> p = data_.subspan(e.offset, e.length);
        return p;
    }
    bool chunk_ok(size_t mi, u64 ci) const {
        const IndexEntry& e = members_[mi].index[ci];
        if (e.flags & IndexEntry::ABSENT) return true;
        return xxhash64(data_.subspan(e.offset, e.length)) == e.checksum;
    }

private:
    void parse() {
        const u8* t = data_.data() + data_.size() - TRAILER_SIZE;
        if (std::memcmp(t, MAGIC, 4) != 0) std::abort();             // §0 fail-fast
        u32 ver = rd_u32(t + 4); (void)ver;
        u64 dir_off = rd_u64(t + 8), dir_len = rd_u64(t + 16), dir_sum = rd_u64(t + 24);
        u64 meta_off = rd_u64(t + 32); u32 meta_len = rd_u32(t + 40);
        if (std::memcmp(t + 44, MAGIC, 4) != 0) std::abort();
        if (xxhash64(data_.subspan(dir_off, dir_len)) != dir_sum) std::abort();

        const u8* d = data_.data() + dir_off;
        u32 nm = rd_u32(d); d += 4;
        members_.resize(nm);
        for (u32 mi = 0; mi < nm; ++mi) {
            Member& m = members_[mi];
            u32 nlen = rd_u32(d); d += 4;
            m.name.assign(reinterpret_cast<const char*>(d), nlen); d += nlen;
            m.type = static_cast<MemberType>(rd_u32(d)); d += 4;
            m.codec = rd_u32(d); d += 4;
            m.shape.z = rd_u64(d); d += 8; m.shape.y = rd_u64(d); d += 8; m.shape.x = rd_u64(d); d += 8;
            m.chunk_size = rd_u32(d); d += 4;
            u64 nidx = rd_u64(d); d += 8;
            u64 ioff = rd_u64(d); d += 8; u64 ilen = rd_u64(d); d += 8; u64 isum = rd_u64(d); d += 8;
            if (xxhash64(data_.subspan(ioff, ilen)) != isum) std::abort();
            m.index.resize(nidx);
            std::memcpy(m.index.data(), data_.data() + ioff, nidx * sizeof(IndexEntry));
        }
        meta_ = std::string_view(reinterpret_cast<const char*>(data_.data() + meta_off), meta_len);
    }
    static u32 rd_u32(const u8* p) { u32 v=0; for(int i=0;i<4;++i) v|=u32(p[i])<<(8*i); return v; }
    static u64 rd_u64(const u8* p) { u64 v=0; for(int i=0;i<8;++i) v|=u64(p[i])<<(8*i); return v; }

    std::span<const u8> data_;
    std::vector<Member> members_;
    std::string_view meta_;
};

} // namespace c4d::archive
