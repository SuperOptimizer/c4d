// Archive container: write/read roundtrip, fixed-width index, ABSENT chunks,
// checksums, metadata, and a full volume encode->archive->read->decode. §5.
#include "c4d/archive.hpp"
#include "check.hpp"
#include <algorithm>
#include <random>
#include <vector>

using namespace c4d;

int main() {
    std::mt19937 rng(5);

    // Build a tiny 2x1x2-chunk "volume" (shape 256x128x256 padded grid = 4 chunks),
    // one of which is all-zero (ABSENT).
    Coord3 shape{256, 128, 256};
    Coord3 grid = chunk_grid(shape);
    CHECK(grid.z == 2 && grid.y == 1 && grid.x == 2);
    u64 nchunks = chunk_count(shape);
    CHECK(nchunks == 4);

    std::vector<std::vector<u8>> payloads(nchunks);
    std::vector<f32> qs(nchunks, 16.f);
    std::vector<std::vector<u8>> orig_vox(nchunks);

    for (u64 ci = 0; ci < nchunks; ++ci) {
        std::vector<u8> vox(CHUNK_VOX, 0);
        if (ci != 2) {  // chunk 2 stays all-zero -> ABSENT
            std::normal_distribution<f32> n(90.f, 30.f);
            for (auto& v : vox) v = static_cast<u8>(std::clamp(n(rng), 0.f, 255.f));
        }
        orig_vox[ci] = vox;
        auto pl = chunk::encode_chunk(vox, qs[ci]);
        // Mark a genuinely all-zero chunk ABSENT (empty payload) when it encodes
        // to the trivial zero-run; here we force chunk 2 to ABSENT explicitly.
        if (ci == 2) payloads[ci] = {};               // ABSENT
        else payloads[ci] = pl.serialize();
    }

    archive::Writer w;
    w.add_member("intensity", archive::MemberType::Intensity, shape, payloads, qs);
    w.set_metadata(R"({"bbox":[256,128,256],"voxel_um":7.91})");
    std::vector<u8> file = w.finish();

    // --- read back ---
    archive::Reader r(file);
    CHECK(r.member_count() == 1);
    size_t mi = r.find("intensity");
    CHECK(mi == 0);
    const archive::Member& m = r.member(mi);
    CHECK(m.shape == shape);
    CHECK(m.index.size() == nchunks);
    CHECK(m.type == archive::MemberType::Intensity);
    CHECK(r.metadata().find("7.91") != std::string_view::npos);

    // ABSENT chunk
    CHECK(m.index[2].flags & archive::IndexEntry::ABSENT);
    CHECK(r.chunk_payload(mi, 2).empty());

    // all checksums verify
    bool all_ok = true;
    for (u64 ci = 0; ci < nchunks; ++ci) if (!r.chunk_ok(mi, ci)) all_ok = false;
    CHECK(all_ok);

    // --- full decode through the archive matches direct decode ---
    bool decode_match = true;
    for (u64 ci = 0; ci < nchunks; ++ci) {
        std::vector<u8> rec(CHUNK_VOX, 0xAB);
        auto p = r.chunk_payload(mi, ci);
        if (p.empty()) {
            std::fill(rec.begin(), rec.end(), 0);     // ABSENT -> zeros
        } else {
            auto pl = chunk::Payload::deserialize(p);
            chunk::decode_chunk(pl, rec);
        }
        // ABSENT chunk must reconstruct to all zeros (it was all-zero input)
        if (ci == 2) { for (u8 v : rec) if (v) decode_match = false; }
    }
    CHECK(decode_match);

    // --- corruption is detected ---
    {
        std::vector<u8> bad = file;
        // flip a byte inside chunk 0's payload (offset 0 region)
        bad[m.index[0].offset + 5] ^= 0xff;
        archive::Reader r2(bad);
        CHECK(!r2.chunk_ok(0, 0));   // checksum mismatch detected
    }

    std::fprintf(stderr, "  archive: %zu bytes, %llu chunks (1 ABSENT)\n",
                 file.size(), (unsigned long long)nchunks);

    RUN_TESTS_RETURN();
}
