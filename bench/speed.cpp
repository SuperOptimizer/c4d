// Focused single-thread encode/decode throughput microbench. Generates a
// region of CT-like chunks (smooth + textured + noise), then times encode_group
// and decode_chunk_shared in isolation. Reports MB/s of u8 voxel throughput.
#include "c4d/chunk.hpp"
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

using namespace c4d;
using clk = std::chrono::steady_clock;

int main(int argc, char** argv) {
    const u32 NCH = (argc > 1) ? std::atoi(argv[1]) : 64;   // chunks (one region)
    const int ITERS = (argc > 2) ? std::atoi(argv[2]) : 5;
    const f32 Q = (argc > 3) ? (f32)std::atof(argv[3]) : 16.f;   // quality knob
    std::mt19937 rng(12345);
    std::normal_distribution<f32> noise(0.f, 6.f);

    std::vector<std::vector<u8>> cubes(NCH, std::vector<u8>(CHUNK_VOX));
    for (u32 c = 0; c < NCH; ++c)
        for (u32 z = 0; z < CHUNK; ++z)
            for (u32 y = 0; y < CHUNK; ++y)
                for (u32 x = 0; x < CHUNK; ++x) {
                    f32 base = 110.f + 70.f * std::sin(0.06f*x + 0.03f*c)
                                     + 40.f * std::cos(0.05f*y) + 25.f*std::sin(0.09f*z);
                    f32 tex  = 18.f * std::sin(0.5f*x + 0.4f*y) * std::cos(0.45f*z);
                    f32 v = base + tex + noise(rng);
                    cubes[c][vox_index(z,y,x)] = (u8)(v < 0 ? 0 : v > 255 ? 255 : v);
                }
    std::vector<std::span<const u8>> spans;
    for (auto& c : cubes) spans.emplace_back(c);

    const double mb = double(NCH) * CHUNK_VOX / (1024.0*1024.0);
    chunk::EncodeOpts opt{.q = Q};

    // warm up scratch arenas
    auto ge0 = chunk::encode_group(spans, opt);

    double best_enc = 1e30;
    for (int it = 0; it < ITERS; ++it) {
        auto t0 = clk::now();
        auto ge = chunk::encode_group(spans, opt);
        auto t1 = clk::now();
        double s = std::chrono::duration<double>(t1-t0).count();
        if (s < best_enc) best_enc = s;
    }

    auto ge = chunk::encode_group(spans, opt);
    std::vector<u8> out(CHUNK_VOX);
    double best_dec = 1e30;
    for (int it = 0; it < ITERS; ++it) {
        auto t0 = clk::now();
        for (auto& pl : ge.payloads) chunk::decode_chunk_shared(pl, &ge.tables, out);
        auto t1 = clk::now();
        double s = std::chrono::duration<double>(t1-t0).count();
        if (s < best_dec) best_dec = s;
    }

    size_t bytes = 0; for (auto& pl : ge.payloads) bytes += pl.size_bytes();
    std::printf("encode %.1f MB/s   decode %.1f MB/s   ratio %.2fx\n",
                mb/best_enc, mb/best_dec, double(NCH)*CHUNK_VOX/double(bytes));
    return 0;
}
