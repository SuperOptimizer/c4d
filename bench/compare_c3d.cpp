// Head-to-head: c4d (a 256^3 region = 4x4x4 = 64 chunks of 64^3, encoded as one
// shared-table group — the production path) vs c3d v1 (1x 256^3 chunk) on the
// SAME 256^3 volume. Reports, per quality setting, total bytes / ratio / PSNR /
// §7.1 basket, plus encode+decode wall time. Both compress identical voxels,
// each in its native chunk size, so the bytes/quality/speed comparison is fair.
#include "c4d/chunk.hpp"
#include "c4d/core.hpp"
#include "c4d/metrics.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <vector>

// c3d requires 64-byte aligned in/out buffers (C3D_ALIGN). Aligned scratch.
struct Aligned {
    c4d::u8* p = nullptr; size_t n = 0;
    explicit Aligned(size_t bytes) : n(bytes) {
        p = static_cast<c4d::u8*>(std::aligned_alloc(64, (bytes + 63) & ~size_t(63)));
    }
    ~Aligned() { std::free(p); }
};
extern "C" {
#include "c3d.h"
}
using namespace c4d;
using clk = std::chrono::high_resolution_clock;
static double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static std::vector<u8> read_file(const char* p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    auto n = f.tellg(); f.seekg(0);
    std::vector<u8> v(n); f.read((char*)v.data(), n); return v;
}

static constexpr u32 VEDGE = 256;                 // benchmark volume edge
static constexpr u32 GRID  = VEDGE / CHUNK;        // chunks per axis (4 for CHUNK=64)

// Extract one CHUNK^3 chunk (cz,cy,cx) from a VEDGE^3 volume (Z,Y,X).
static std::vector<u8> subchunk(const std::vector<u8>& vol, u32 cz, u32 cy, u32 cx) {
    std::vector<u8> c(CHUNK_VOX);
    for (u32 z = 0; z < CHUNK; ++z)
        for (u32 y = 0; y < CHUNK; ++y) {
            const u8* src = &vol[((cz*CHUNK+z)*VEDGE + (cy*CHUNK+y))*VEDGE + cx*CHUNK];
            std::memcpy(&c[(z*CHUNK+y)*CHUNK], src, CHUNK);
        }
    return c;
}
static void placechunk(std::vector<u8>& vol, const std::vector<u8>& c, u32 cz, u32 cy, u32 cx) {
    for (u32 z = 0; z < CHUNK; ++z)
        for (u32 y = 0; y < CHUNK; ++y) {
            u8* dst = &vol[((cz*CHUNK+z)*VEDGE + (cy*CHUNK+y))*VEDGE + cx*CHUNK];
            std::memcpy(dst, &c[(z*CHUNK+y)*CHUNK], CHUNK);
        }
}

// Average the §7.1 basket over the chunk grid (metrics are defined on CHUNK^3).
static metrics::Basket basket_vol(const std::vector<u8>& ref, const std::vector<u8>& rec) {
    metrics::Basket acc{0,0,0,0,0,0}; int n = 0;
    for (u32 cz = 0; cz < GRID; ++cz) for (u32 cy = 0; cy < GRID; ++cy) for (u32 cx = 0; cx < GRID; ++cx) {
        auto r = subchunk(ref, cz, cy, cx), c = subchunk(rec, cz, cy, cx);
        auto b = metrics::evaluate(r, c);
        acc.psnr_db += b.psnr_db; acc.gmsd += b.gmsd; acc.ms_ssim += b.ms_ssim;
        acc.haarpsi += b.haarpsi; acc.sharpness += b.sharpness; acc.epi += b.epi; ++n;
    }
    acc.psnr_db/=n; acc.gmsd/=n; acc.ms_ssim/=n; acc.haarpsi/=n; acc.sharpness/=n; acc.epi/=n;
    return acc;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: compare_c3d <vol256.raw> [c4d_q...]\n"); return 2; }
    auto vol = read_file(argv[1]);
    if (vol.size() != C3D_VOXELS_PER_CHUNK || C3D_VOXELS_PER_CHUNK != size_t(VEDGE)*VEDGE*VEDGE) {
        std::fprintf(stderr, "need a 256^3 raw (got %zu bytes; c3d chunk=%zu)\n",
                     vol.size(), size_t(C3D_VOXELS_PER_CHUNK));
        return 1;
    }
    std::vector<f32> c4dq;
    for (int i = 2; i < argc; ++i) c4dq.push_back(std::strtof(argv[i], nullptr));
    if (c4dq.empty()) c4dq = {8.f, 16.f, 24.f, 32.f, 48.f, 64.f};

    std::fprintf(stderr, "volume: %s (256^3); c4d region = %ux%ux%u chunks of %u^3\n\n",
                 argv[1], GRID, GRID, GRID, CHUNK);
    std::printf("%-5s %-6s %8s %8s %8s %8s %8s %9s %9s %9s\n",
                "codec", "q", "ratio", "PSNR", "GMSD", "MSSSIM", "HaarPSI", "geomean",
                "enc(ms)", "dec(ms)");

    // --- c3d v1: one 256^3 chunk, encode_at_q ---
    Aligned cap(c3d_chunk_encode_max_size());
    Aligned ain(C3D_VOXELS_PER_CHUNK);
    Aligned aout(C3D_VOXELS_PER_CHUNK);
    std::memcpy(ain.p, vol.data(), C3D_VOXELS_PER_CHUNK);
    for (f32 q : {0.05f, 0.1f, 0.2f, 0.4f}) {       // c3d's own q scale (target distortion)
        auto t0 = clk::now();
        size_t n = c3d_chunk_encode_at_q(ain.p, q, cap.p, cap.n);
        auto t1 = clk::now();
        c3d_chunk_decode(cap.p, n, aout.p);
        auto t2 = clk::now();
        std::vector<u8> rec(aout.p, aout.p + C3D_VOXELS_PER_CHUNK);
        auto b = basket_vol(vol, rec);
        f64 ratio = double(C3D_VOXELS_PER_CHUNK) / n;
        std::printf("%-5s %-6.3f %7.1fx %8.2f %8.3f %8.3f %8.3f %9.3f %9.1f %9.1f\n",
                    "c3d", q, ratio, b.psnr_db, b.gmsd, b.ms_ssim, b.haarpsi, b.geomean(),
                    ms(t0,t1), ms(t1,t2));
    }

    // --- c4d: the 256^3 region as one shared-table group (production path) ---
    std::vector<std::vector<u8>> cubes;
    for (u32 cz = 0; cz < GRID; ++cz) for (u32 cy = 0; cy < GRID; ++cy) for (u32 cx = 0; cx < GRID; ++cx)
        cubes.push_back(subchunk(vol, cz, cy, cx));
    for (f32 q : c4dq) {
        std::vector<std::span<const u8>> spans;
        for (auto& c : cubes) spans.emplace_back(c);
        auto t0 = clk::now();
        auto ge = chunk::encode_group(spans, chunk::EncodeOpts{.q = q});
        auto t1 = clk::now();
        std::vector<u8> rec(C3D_VOXELS_PER_CHUNK);
        u32 idx = 0;
        std::vector<u8> c(CHUNK_VOX);
        for (u32 cz = 0; cz < GRID; ++cz) for (u32 cy = 0; cy < GRID; ++cy) for (u32 cx = 0; cx < GRID; ++cx) {
            chunk::decode_chunk_shared(ge.payloads[idx++], &ge.tables, c);
            placechunk(rec, c, cz, cy, cx);
        }
        auto t2 = clk::now();
        size_t total = 0; for (auto& p : ge.payloads) total += p.size_bytes();
        { std::vector<u8> tb; ge.tables.serialize(tb); total += tb.size(); }   // shared-table overhead
        auto b = basket_vol(vol, rec);
        f64 ratio = double(C3D_VOXELS_PER_CHUNK) / total;
        std::printf("%-5s %-6.0f %7.1fx %8.2f %8.3f %8.3f %8.3f %9.3f %9.1f %9.1f\n",
                    "c4d", q, ratio, b.psnr_db, b.gmsd, b.ms_ssim, b.haarpsi, b.geomean(),
                    ms(t0,t1), ms(t1,t2));
    }
    return 0;
}
