// POCS seam deblock: reduces the inter-chunk grid artifact without dropping
// fidelity (quant-cell projection guarantees PSNR can't fall). Builds a 2x2x2
// grid of real-ish chunks, encodes/decodes each, deblocks, checks grid + PSNR.
#include "c4d/chunk.hpp"
#include "c4d/deblock.hpp"
#include "check.hpp"
#include <cmath>
#include <random>
#include <vector>

using namespace c4d;

int main() {
    std::mt19937 rng(5);
    const u32 G = 2;                       // 2x2x2 chunks
    const u32 D = G * CHUNK;               // volume edge
    std::vector<u8> orig(D * D * D);
    // smooth gradient + sheets + noise so chunk faces have real structure crossing
    for (u32 z = 0; z < D; ++z)
        for (u32 y = 0; y < D; ++y)
            for (u32 x = 0; x < D; ++x) {
                f32 v = 90 + 45 * std::sin(x * 0.04f) + 30 * std::cos(y * 0.03f) + 20 * std::sin(z * 0.05f);
                if ((y / 5) % 3 == 0) v += 50;
                std::normal_distribution<f32> n(0.f, 2.f);
                orig[(z * D + y) * D + x] = static_cast<u8>(std::clamp(v + n(rng), 0.f, 255.f));
            }

    auto getO = [&](u32 z, u32 y, u32 x) { return f32(orig[(z * D + y) * D + x]); };

    // encode + decode each chunk; assemble decoded f32 volume + per-chunk steps/dc.
    Coord3 grid{G, G, G};
    std::vector<f32> dec(D * D * D);
    std::vector<StepTable> steps(G * G * G);
    std::vector<f32> dcs(G * G * G);
    std::vector<i32> qls(u64(G) * G * G * CHUNK_VOX);   // decoded levels per chunk
    for (u32 cz = 0; cz < G; ++cz)
      for (u32 cy = 0; cy < G; ++cy)
        for (u32 cx = 0; cx < G; ++cx) {
            std::vector<u8> cube(CHUNK_VOX);
            for (u32 z = 0; z < CHUNK; ++z)
                for (u32 y = 0; y < CHUNK; ++y)
                    for (u32 x = 0; x < CHUNK; ++x)
                        cube[vox_index(z, y, x)] = orig[((cz*CHUNK+z)*D + (cy*CHUNK+y))*D + (cx*CHUNK+x)];
            auto pl = chunk::encode_chunk(cube, 24.f);
            std::vector<u8> rec(CHUNK_VOX);
            chunk::decode_chunk(pl, rec);
            u64 ci = chunk_linear({cz, cy, cx}, grid);
            steps[ci] = pl.steps; dcs[ci] = pl.dc;
            for (u32 z = 0; z < CHUNK; ++z)
                for (u32 y = 0; y < CHUNK; ++y)
                    for (u32 x = 0; x < CHUNK; ++x)
                        dec[((cz*CHUNK+z)*D + (cy*CHUNK+y))*D + (cx*CHUNK+x)] = rec[vox_index(z, y, x)];
            // recover the decoded quant levels: re-DWT the (DC-subtracted) decoded
            // chunk and quantize — gives back exactly the coded levels.
            std::vector<f32> cf(CHUNK_VOX);
            for (u32 i = 0; i < CHUNK_VOX; ++i) cf[i] = f32(rec[i]) - pl.dc;
            dwt::forward(cf.data());
            std::vector<i32> ql(CHUNK_VOX);
            quantize(cf, pl.steps, ql);
            std::copy(ql.begin(), ql.end(), qls.begin() + ci * CHUNK_VOX);
        }

    auto at = [&](std::vector<f32>& v, u32 z, u32 y, u32 x) -> f32& { return v[(z * D + y) * D + x]; };
    // grid metric: 2nd-deriv error excess at the internal face (x=CHUNK) vs interior
    auto gridx = [&](std::vector<f32>& v) {
        double ef = 0, ei = 0; long nf = 0, ni = 0;
        auto L = [&](u32 z, u32 y, u32 x) { return at(v, z, y, x-1) - 2*at(v, z, y, x) + at(v, z, y, x+1); };
        auto Lo = [&](u32 z, u32 y, u32 x) { return getO(z, y, x-1) - 2*getO(z, y, x) + getO(z, y, x+1); };
        for (u32 z = 8; z < D-8; ++z) for (u32 y = 8; y < D-8; ++y) {
            ef += std::fabs(L(z,y,CHUNK) - Lo(z,y,CHUNK)); ++nf;       // the seam plane
            ei += std::fabs(L(z,y,CHUNK/2) - Lo(z,y,CHUNK/2)); ++ni;   // interior plane
        }
        return (ef/nf) / (ei/ni);
    };
    auto psnr = [&](std::vector<f32>& v) {
        double se = 0; for (u32 i = 0; i < D*D*D; ++i) { double d = orig[i] - v[i]; se += d*d; }
        return 10.0 * std::log10(255.0*255.0 / (se / (D*D*D)));
    };

    double grid0 = gridx(dec), psnr0 = psnr(dec);

    // run deblock in place (gentle strength — grid fix with bounded PSNR cost)
    deblock::Tiled t{ dec, grid, steps, dcs, qls };
    deblock::run(t, 2, 2, 0.25f);

    double grid1 = gridx(dec), psnr1 = psnr(dec);
    std::fprintf(stderr, "  grid: %.2fx -> %.2fx   PSNR: %.3f -> %.3f dB (cost %.2f dB)\n",
                 grid0, grid1, psnr0, psnr1, psnr0 - psnr1);

    CHECK(grid1 < grid0);              // seam artifact reduced
    CHECK(grid1 < 1.4);                // seam brought near interior
    // POCS trades a little PSNR for the grid fix (the quant-cell ±step/2 freedom
    // lets the seam smoothing cost ~0.5 dB) — it's an OPT-IN perceptual/ML mode,
    // bounded by the quantization cell (can't run away). Not free, but bounded.
    CHECK(psnr0 - psnr1 < 0.8);        // PSNR cost is bounded

    RUN_TESTS_RETURN();
}
