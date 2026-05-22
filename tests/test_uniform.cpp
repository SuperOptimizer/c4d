// Uniform-block fast-path (§7 T2): constant chunks code to ~2 bytes, lossless,
// and roundtrip through serialize/deserialize. Spec §7.
#include "c4d/chunk.hpp"
#include "check.hpp"
#include <vector>
using namespace c4d;
int main() {
    for (u8 val : {u8(0), u8(1), u8(73), u8(255)}) {
        std::vector<u8> c(CHUNK_VOX, val);
        auto pl = chunk::encode_chunk(c, 16.f);
        CHECK(pl.uniform);
        CHECK(pl.uval == val);
        auto blob = pl.serialize();
        CHECK(blob.size() == 2);                 // tag + value
        auto p2 = chunk::Payload::deserialize(blob);
        std::vector<u8> rec(CHUNK_VOX, 9);
        chunk::decode_chunk(p2, rec);
        bool ok = true; for (u8 v : rec) if (v != val) ok = false;
        CHECK(ok);                                // lossless
    }
    // a non-uniform chunk must NOT take the fast-path
    {
        std::vector<u8> c(CHUNK_VOX, 100); c[vox_index(10,20,30)] = 101;
        auto pl = chunk::encode_chunk(c, 16.f);
        CHECK(!pl.uniform);
        auto blob = pl.serialize();
        CHECK(blob.size() > 2);
        CHECK(blob[0] == 0);                      // normal tag
    }
    std::fprintf(stderr, "  uniform chunk: 2 B (was ~1723 B)\n");
    RUN_TESTS_RETURN();
}
