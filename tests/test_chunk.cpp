// Full per-chunk pipeline: u8 cube -> encode -> payload -> decode -> u8 cube.
// Verifies lossless framing (decoded == decode-of-serialized), PSNR vs q, and
// real compression ratio. Spec §4 pipeline + §4.7 model-the-zeros.
#include "c4d/chunk.hpp"
#include "check.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

using namespace c4d;

static f64 psnr_u8(std::span<const u8> a, std::span<const u8> b) {
    f64 se = 0;
    for (size_t i = 0; i < a.size(); ++i) { f64 d = f64(a[i]) - b[i]; se += d * d; }
    f64 mse = se / a.size();
    if (mse <= 0) return 999.0;
    return 10.0 * std::log10(255.0 * 255.0 / mse);
}

int main() {
    std::mt19937 rng(2024);

    // scroll-like synthetic: smooth background + sheets/edges + noise + a padded
    // (all-zero) air margin to exercise zero-run coding.
    std::vector<u8> vox(CHUNK_VOX);
    std::normal_distribution<f32> n(0.f, 2.5f);
    for (u32 z = 0; z < CHUNK; ++z)
        for (u32 y = 0; y < CHUNK; ++y)
            for (u32 x = 0; x < CHUNK; ++x) {
                f32 v;
                if (x < 24 || z > 110) v = 0.f;   // air / padding margins
                else {
                    v = 70.f + 35.f * std::sin(x * 0.06f) + 25.f * std::cos(y * 0.04f);
                    if ((y / 6) % 4 == 0) v += 70.f;   // periodic sheets
                    v += n(rng);
                }
                vox[vox_index(z, y, x)] = static_cast<u8>(std::clamp(v, 0.f, 255.f));
            }

    f64 prev_psnr = 1e9; size_t prev_size = ~0ull; bool first = true;
    for (f32 q : {4.f, 8.f, 16.f, 32.f}) {
        auto payload = chunk::encode_chunk(vox, q);

        // serialize/deserialize roundtrip must be byte-exact
        auto blob = payload.serialize();
        auto p2 = chunk::Payload::deserialize(blob);

        std::vector<u8> rec(CHUNK_VOX), rec2(CHUNK_VOX);
        chunk::decode_chunk(payload, rec);
        chunk::decode_chunk(p2, rec2);
        CHECK(rec == rec2);   // serialize roundtrip is lossless

        f64 p = psnr_u8(vox, rec);
        size_t sz = blob.size();
        f64 ratio = double(CHUNK_VOX) / sz;
        std::fprintf(stderr, "  q=%5.1f  PSNR=%.2f dB  size=%zu B  ratio=%.1fx\n",
                     q, p, sz, ratio);

        CHECK(p > 28.0);          // sane quality across the q range
        CHECK(ratio > 2.0);       // some compression even at the finest q
        if (!first) {
            CHECK(p <= prev_psnr + 0.5);    // coarser q => lower PSNR
            CHECK(sz <= prev_size);         // coarser q => smaller
        }
        prev_psnr = p; prev_size = sz; first = false;
    }

    // All-zero (padded) chunk: must be tiny and decode back to all zeros.
    {
        std::vector<u8> zero(CHUNK_VOX, 0);
        auto pl = chunk::encode_chunk(zero, 8.f);
        auto blob = pl.serialize();
        std::vector<u8> rec(CHUNK_VOX, 7);
        chunk::decode_chunk(pl, rec);
        bool all_zero = true; for (u8 v : rec) if (v) all_zero = false;
        CHECK(all_zero);
        std::fprintf(stderr, "  all-zero chunk: %zu B\n", blob.size());
        // Near-empty: a single zero-run token; the bytes are dominated by the
        // serialized frequency table (88-symbol alphabet), not coefficient data.
        CHECK(blob.size() < 400);
    }

    RUN_TESTS_RETURN();
}
