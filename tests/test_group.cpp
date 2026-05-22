// Shared-tables group encode (§ no-independence): a group of chunks shares one
// entropy-table set. Verify (1) each chunk roundtrips losslessly through the
// shared tables, (2) total bytes < per-chunk-table encoding.
#include "c4d/chunk.hpp"
#include "check.hpp"
#include <random>
#include <vector>
using namespace c4d;
int main() {
    std::mt19937 rng(21);
    // build a few distinct scroll-ish chunks
    const int N = 8;
    std::vector<std::vector<u8>> vox(N);
    for (int c = 0; c < N; ++c) {
        vox[c].resize(CHUNK_VOX);
        std::normal_distribution<f32> nz(0.f, 3.f);
        for (u32 z = 0; z < CHUNK; ++z) for (u32 y = 0; y < CHUNK; ++y) for (u32 x = 0; x < CHUNK; ++x) {
            f32 v = 80 + 40*std::sin((x+c*7)*0.06f) + 25*std::cos(y*0.05f);
            if (((y/6)+(c)) % 4 == 0) v += 60;
            vox[c][vox_index(z,y,x)] = static_cast<u8>(std::clamp(v + nz(rng), 0.f, 255.f));
        }
    }
    std::vector<std::span<const u8>> spans;
    for (auto& v : vox) spans.emplace_back(v);

    chunk::EncodeOpts opt{.q = 16.f};
    auto g = chunk::encode_group(spans, opt);
    CHECK(g.payloads.size() == N);

    // (1) lossless roundtrip through shared tables
    bool ok = true;
    size_t shared_total = 0; std::vector<u8> tblbytes; g.tables.serialize(tblbytes);
    shared_total += tblbytes.size();
    for (int c = 0; c < N; ++c) {
        std::vector<u8> rec(CHUNK_VOX), ref(CHUNK_VOX);
        chunk::decode_chunk_shared(g.payloads[c], &g.tables, rec);
        // compare to the per-chunk independent decode of the SAME data (quality
        // identical; only the table source differs)
        auto indep = chunk::encode_chunk(vox[c], opt);
        chunk::decode_chunk(indep, ref);
        if (rec != ref) ok = false;
        shared_total += g.payloads[c].serialize().size();
    }
    CHECK(ok);   // shared-table decode == per-chunk decode (same reconstruction)

    // (2) shared total < per-chunk total
    size_t per_chunk_total = 0;
    for (int c = 0; c < N; ++c) per_chunk_total += chunk::encode_chunk(vox[c], opt).serialize().size();
    std::fprintf(stderr, "  group(%d chunks): shared=%zu B  per-chunk=%zu B  (%.1f%% smaller)\n",
                 N, shared_total, per_chunk_total, 100.0*(per_chunk_total-shared_total)/per_chunk_total);
    CHECK(shared_total < per_chunk_total);

    RUN_TESTS_RETURN();
}
