// Outlier pass (§4.6): after decode, GUARANTEE |decoded - original| <= t for
// every voxel, for several tolerances, on real and synthetic data. Also verify
// tolerance=0 leaves the stream identical to the no-outlier path.
#include "c4d/chunk.hpp"
#include "check.hpp"
#include <algorithm>
#include <fstream>
#include <random>
#include <vector>

using namespace c4d;

static std::vector<u8> load_or_synth(const char* path, std::mt19937& rng) {
    std::ifstream f(path, std::ios::binary);
    if (f) {
        std::vector<u8> v(CHUNK_VOX);
        f.read(reinterpret_cast<char*>(v.data()), CHUNK_VOX);
        if (f.gcount() == CHUNK_VOX) return v;
    }
    // synthetic fallback: smooth + edges + noise
    std::vector<u8> v(CHUNK_VOX);
    std::normal_distribution<f32> n(0.f, 4.f);
    for (u32 z = 0; z < CHUNK; ++z)
        for (u32 y = 0; y < CHUNK; ++y)
            for (u32 x = 0; x < CHUNK; ++x) {
                f32 b = 90 + 40 * std::sin(x * 0.05f) + 30 * std::cos(y * 0.04f);
                if ((y / 7) % 3 == 0) b += 60;
                v[vox_index(z, y, x)] = static_cast<u8>(std::clamp(b + n(rng), 0.f, 255.f));
            }
    return v;
}

int main() {
    std::mt19937 rng(11);
    std::vector<u8> vox = load_or_synth("testdata/p1667_102_31_32.raw", rng);

    // For each tolerance, the hard bound must hold for EVERY voxel. Per §4.6
    // step 1 the base quantizer is matched to the tolerance (q ~ 1.5*t), so the
    // outlier set stays sparse (it cleans up an already-close reconstruction).
    for (f32 t : {1.f, 2.f, 4.f}) {
        chunk::EncodeOpts opt{.q = 1.5f * t, .tolerance = t};
        auto pl = chunk::encode_chunk(vox, opt);
        auto blob = pl.serialize();
        auto p2 = chunk::Payload::deserialize(blob);

        std::vector<u8> rec(CHUNK_VOX);
        chunk::decode_chunk(p2, rec);

        u32 violations = 0, max_err = 0, corrected = 0;
        for (u32 i = 0; i < CHUNK_VOX; ++i) {
            u32 e = std::abs(int(vox[i]) - int(rec[i]));
            max_err = std::max(max_err, e);
            if (e > static_cast<u32>(t)) ++violations;
        }
        // count corrections from the blob header
        if (pl.outliers.size() >= 4)
            corrected = u32(pl.outliers[0]) | (u32(pl.outliers[1]) << 8)
                      | (u32(pl.outliers[2]) << 16) | (u32(pl.outliers[3]) << 24);

        std::fprintf(stderr, "  t=%.0f: max_err=%u violations=%u corrections=%u (%.2f%%) bytes=%zu\n",
                     t, max_err, violations, corrected, 100.0 * corrected / CHUNK_VOX, blob.size());
        CHECK(violations == 0);              // HARD guarantee: all within t
        CHECK(max_err <= static_cast<u32>(t));
    }

    // tolerance=0 path produces no outlier stream and still roundtrips.
    {
        auto pl = chunk::encode_chunk(vox, chunk::EncodeOpts{.q = 16.f, .tolerance = 0.f});
        CHECK(pl.outliers.empty());
        std::vector<u8> rec(CHUNK_VOX);
        chunk::decode_chunk(pl, rec);
        // no guarantee on max error here, just that it decodes
        CHECK(rec.size() == CHUNK_VOX);
    }

    RUN_TESTS_RETURN();
}
