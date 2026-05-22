// Encoder-side noise-aware shrinkage (§4.10) and perceptual RDO: the opt-in
// paths in EncodeOpts. Verifies they (a) round-trip through the full chunk
// codec, (b) are decode-transparent (the decoder reads steps from the stream,
// so a noise_aware payload decodes with the plain decoder), (c) actually change
// the output / step table (the path is exercised, not a silent no-op), and
// (d) the MAD sigma estimate tracks injected noise. Default path stays exact.
#include "c4d/chunk.hpp"
#include "c4d/denoise.hpp"
#include "check.hpp"
#include <cmath>
#include <random>
#include <vector>

using namespace c4d;

// Smooth signal + Gaussian noise of std `nsig`, as a CHUNK^3 u8 cube.
static std::vector<u8> make_cube(std::mt19937& g, f32 nsig) {
    std::normal_distribution<f32> n(0.f, nsig);
    std::vector<u8> v(CHUNK_VOX);
    for (u32 z = 0; z < CHUNK; ++z)
        for (u32 y = 0; y < CHUNK; ++y)
            for (u32 x = 0; x < CHUNK; ++x) {
                f32 b = 110 + 50 * std::sin(0.05f * x) + 35 * std::cos(0.04f * y)
                            + 20 * std::sin(0.07f * z);
                f32 val = b + n(g);
                v[vox_index(z, y, x)] = static_cast<u8>(val < 0 ? 0 : val > 255 ? 255 : val);
            }
    return v;
}

static f64 psnr_u8(std::span<const u8> a, std::span<const u8> b) {
    f64 se = 0; for (size_t i = 0; i < a.size(); ++i) { f64 d = f64(a[i]) - b[i]; se += d * d; }
    f64 mse = se / a.size(); return mse <= 0 ? 999.0 : 10.0 * std::log10(255.0 * 255.0 / mse);
}

int main() {
    std::mt19937 g(7);

    // --- MAD sigma estimate tracks injected noise (monotone) ----------------
    {
        auto coef_of = [&](f32 nsig) {
            auto v = make_cube(g, nsig);
            std::vector<f32> c(CHUNK_VOX);
            for (u32 i = 0; i < CHUNK_VOX; ++i) c[i] = f32(v[i]);
            dwt::forward(c.data());
            return estimate_noise_sigma(c);
        };
        f64 s_lo = coef_of(2.f), s_hi = coef_of(16.f);
        CHECK(s_lo >= 0.0);
        CHECK(s_hi > s_lo);                     // more injected noise => larger MAD sigma
    }

    auto vox = make_cube(g, 10.f);

    // --- noise_aware path: round-trips, decode-transparent, changes output ---
    {
        chunk::EncodeOpts plain{.q = 16.f};
        chunk::EncodeOpts naw{.q = 16.f, .noise_aware = true, .shrink_strength = 1.0};

        auto p_plain = chunk::encode_chunk(vox, plain);
        auto p_naw   = chunk::encode_chunk(vox, naw);

        // (a) decode-transparent: the plain decoder handles the noise_aware payload
        std::vector<u8> rec_plain(CHUNK_VOX), rec_naw(CHUNK_VOX);
        chunk::decode_chunk(p_plain, rec_plain);
        chunk::decode_chunk(p_naw, rec_naw);

        // (b) both reconstruct sanely. noise_aware shrinks the injected (incom-
        //     pressible) HF noise, so vs the NOISY input its PSNR is lower by
        //     design — but it stays reasonable.
        CHECK(psnr_u8(vox, rec_plain) > 30.0);
        CHECK(psnr_u8(vox, rec_naw)   > 27.0);

        // (c) the path is actually exercised: noise_aware raises HF steps, which
        //     both changes the payload and collapses the noisy stream (the win).
        auto bp = p_plain.serialize(), bn = p_naw.serialize();
        CHECK(bp != bn);
        CHECK(bn.size() < bp.size() / 2);                           // big size cut on noisy data
        CHECK(p_naw.steps.step[0][7] >= p_plain.steps.step[0][7]);  // finest HHH step not smaller
    }

    // --- perceptual RDO path: round-trips and is decode-transparent ----------
    {
        chunk::EncodeOpts rdo{.q = 16.f, .perceptual_rdo = true, .rdo_strength = 0.5};
        auto p = chunk::encode_chunk(vox, rdo);
        std::vector<u8> rec(CHUNK_VOX);
        chunk::decode_chunk(p, rec);             // plain decoder, steps from stream
        CHECK(psnr_u8(vox, rec) > 28.0);
    }

    // --- strength=0 noise_aware is (near) a no-op vs plain -------------------
    {
        auto a = chunk::encode_chunk(vox, chunk::EncodeOpts{.q = 16.f});
        auto b = chunk::encode_chunk(vox, chunk::EncodeOpts{.q = 16.f, .noise_aware = true,
                                                            .shrink_strength = 0.0});
        // zero strength => steps unchanged from the base q table
        bool same_steps = true;
        for (u32 l = 0; l < DWT_LEVELS; ++l) for (u32 o = 0; o < 8; ++o)
            if (a.steps.step[l][o] != b.steps.step[l][o]) same_steps = false;
        CHECK(same_steps);
    }

    RUN_TESTS_RETURN();
}
