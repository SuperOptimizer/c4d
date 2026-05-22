// Head-to-head: c4d (8x 128^3 chunks) vs c3d v1 (1x 256^3 chunk) on the SAME
// 256^3 volume. Reports, per quality setting, total bytes / ratio / PSNR /
// §7.1 basket, plus encode+decode wall time. Both compress identical voxels,
// each in its native chunk size, so the bytes/quality/speed comparison is fair.
#include "c4d/chunk.hpp"
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

// extract a 128^3 chunk (cz,cy,cx in {0,1}) from a 256^3 volume (Z,Y,X)
static std::vector<u8> sub128(const std::vector<u8>& vol, u32 cz, u32 cy, u32 cx) {
    std::vector<u8> c(CHUNK_VOX);
    for (u32 z = 0; z < 128; ++z)
        for (u32 y = 0; y < 128; ++y) {
            const u8* src = &vol[((cz*128+z)*256 + (cy*128+y))*256 + cx*128];
            std::memcpy(&c[(z*128+y)*128], src, 128);
        }
    return c;
}
static void place128(std::vector<u8>& vol, const std::vector<u8>& c, u32 cz, u32 cy, u32 cx) {
    for (u32 z = 0; z < 128; ++z)
        for (u32 y = 0; y < 128; ++y) {
            u8* dst = &vol[((cz*128+z)*256 + (cy*128+y))*256 + cx*128];
            std::memcpy(dst, &c[(z*128+y)*128], 128);
        }
}

static metrics::Basket basket256(const std::vector<u8>& ref, const std::vector<u8>& rec) {
    // average the per-128-chunk basket over the 8 octants (metrics are defined on
    // CHUNK^3); good enough for a comparison column.
    metrics::Basket acc{0,0,0,0,0,0}; int n = 0;
    for (u32 cz = 0; cz < 2; ++cz) for (u32 cy = 0; cy < 2; ++cy) for (u32 cx = 0; cx < 2; ++cx) {
        auto r = sub128(ref, cz, cy, cx), c = sub128(rec, cz, cy, cx);
        auto b = metrics::evaluate(r, c);
        acc.psnr_db += b.psnr_db; acc.gmsd += b.gmsd; acc.ms_ssim += b.ms_ssim;
        acc.haarpsi += b.haarpsi; acc.sharpness += b.sharpness; acc.epi += b.epi; ++n;
    }
    acc.psnr_db/=n; acc.gmsd/=n; acc.ms_ssim/=n; acc.haarpsi/=n; acc.sharpness/=n; acc.epi/=n;
    return acc;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: compare_c3d <vol256.raw> [q...]\n"); return 2; }
    auto vol = read_file(argv[1]);
    if (vol.size() != C3D_VOXELS_PER_CHUNK) { std::fprintf(stderr, "need a 256^3 raw\n"); return 1; }
    std::vector<f32> qs;
    for (int i = 2; i < argc; ++i) qs.push_back(std::strtof(argv[i], nullptr));
    if (qs.empty()) qs = {0.05f, 0.1f, 0.2f, 0.4f};   // c3d q range; c4d uses its own scale

    std::fprintf(stderr, "volume: %s (256^3)\n\n", argv[1]);
    std::printf("%-5s %-6s %8s %8s %8s %8s %8s %8s %9s %9s\n",
                "codec", "q", "ratio", "PSNR", "GMSD", "MSSSIM", "HaarPSI", "geomean",
                "enc(ms)", "dec(ms)");

    Aligned cap(c3d_chunk_encode_max_size());
    Aligned ain(C3D_VOXELS_PER_CHUNK);
    Aligned aout(C3D_VOXELS_PER_CHUNK);
    std::memcpy(ain.p, vol.data(), C3D_VOXELS_PER_CHUNK);

    for (f32 q : qs) {
        // --- c3d v1: one 256^3 chunk, encode_at_q ---
        {
            auto t0 = clk::now();
            size_t n = c3d_chunk_encode_at_q(ain.p, q, cap.p, cap.n);
            auto t1 = clk::now();
            c3d_chunk_decode(cap.p, n, aout.p);
            auto t2 = clk::now();
            std::vector<u8> rec(aout.p, aout.p + C3D_VOXELS_PER_CHUNK);
            auto b = basket256(vol, rec);
            f64 ratio = double(C3D_VOXELS_PER_CHUNK) / n;
            std::printf("%-5s %-6.3f %7.1fx %8.2f %8.3f %8.3f %8.3f %9.3f %9.1f %9.1f\n",
                        "c3d", q, ratio, b.psnr_db, b.gmsd, b.ms_ssim, b.haarpsi, b.geomean(),
                        ms(t0,t1), ms(t1,t2));
        }
    }

    // c4d uses a different q scale; sweep its own knobs so ratios overlap c3d's.
    for (f32 q : {8.f, 16.f, 24.f, 32.f, 48.f, 64.f}) {
        auto t0 = clk::now();
        std::vector<std::vector<u8>> blobs(8);
        int k = 0;
        for (u32 cz = 0; cz < 2; ++cz) for (u32 cy = 0; cy < 2; ++cy) for (u32 cx = 0; cx < 2; ++cx)
            blobs[k++] = chunk::encode_chunk(sub128(vol, cz, cy, cx), q).serialize();
        auto t1 = clk::now();
        std::vector<u8> rec(C3D_VOXELS_PER_CHUNK);
        k = 0;
        for (u32 cz = 0; cz < 2; ++cz) for (u32 cy = 0; cy < 2; ++cy) for (u32 cx = 0; cx < 2; ++cx) {
            auto pl = chunk::Payload::deserialize(blobs[k++]);
            std::vector<u8> c(CHUNK_VOX); chunk::decode_chunk(pl, c);
            place128(rec, c, cz, cy, cx);
        }
        auto t2 = clk::now();
        size_t total = 0; for (auto& b : blobs) total += b.size();
        auto b = basket256(vol, rec);
        f64 ratio = double(C3D_VOXELS_PER_CHUNK) / total;
        std::printf("%-5s %-6.0f %7.1fx %8.2f %8.3f %8.3f %8.3f %9.3f %9.1f %9.1f\n",
                    "c4d", q, ratio, b.psnr_db, b.gmsd, b.ms_ssim, b.haarpsi, b.geomean(),
                    ms(t0,t1), ms(t1,t2));
    }
    return 0;
}
