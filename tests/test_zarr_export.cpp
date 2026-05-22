// End-to-end test for the zarr_to_c4d converter (tools/zarr_to_c4d.cpp):
// write a raw zarr v2 to a temp dir, run the built converter binary, then open
// the resulting .c4d archive and verify it decodes to the source voxels.
// Exercises both chunk-key separators ('.' flat and '/' nested) and that the
// converter re-tiles a source chunk shape != 64 into c4d's 64^3.
//
// The converter binary path is injected by CMake as C4D_ZARR_TO_C4D_BIN; if it
// isn't defined (e.g. building this test standalone) the test is skipped.
#include "c4d/archive.hpp"
#include "c4d/chunk.hpp"
#include "check.hpp"
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace c4d;

static void write_file(const fs::path& p, std::span<const u8> v) {
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()), (std::streamsize)v.size());
}

// Decode a c4d archive (intensity + region_tables members) to a flat volume.
static std::vector<u8> decode_archive(const std::vector<u8>& file, Coord3 shape) {
    archive::Reader r(file);
    size_t mi = r.find("intensity");
    size_t rti = r.find("region_tables");
    Coord3 g = chunk_grid(shape);
    Coord3 rg = region_grid(g);
    std::vector<u8> vol(shape.z * shape.y * shape.x, 0);
    std::vector<u8> cube(CHUNK_VOX);
    for (u64 ci = 0; ci < g.z * g.y * g.x; ++ci) {
        auto p = r.chunk_payload(mi, ci);
        if (p.empty()) continue;                                  // ABSENT -> zeros
        auto pl = chunk::Payload::deserialize(p);
        // region tables for this chunk's region
        Coord3 c = chunk_unravel(ci, g);
        Coord3 rc{c.z / REGION_CHUNKS, c.y / REGION_CHUNKS, c.x / REGION_CHUNKS};
        u64 ri = chunk_linear(rc, rg);
        auto tb = r.chunk_payload(rti, ri);
        size_t pos = 0;
        auto tables = chunk::SharedTables::deserialize(tb, pos);
        chunk::decode_chunk_shared(pl, &tables, cube);
        u64 z0 = c.z * CHUNK, y0 = c.y * CHUNK, x0 = c.x * CHUNK;
        for (u32 dz = 0; dz < CHUNK && z0+dz < shape.z; ++dz)
            for (u32 dy = 0; dy < CHUNK && y0+dy < shape.y; ++dy)
                for (u32 dx = 0; dx < CHUNK && x0+dx < shape.x; ++dx)
                    vol[((z0+dz)*shape.y + (y0+dy))*shape.x + (x0+dx)] = cube[vox_index(dz,dy,dx)];
    }
    return vol;
}

// Write a raw (uncompressed) zarr v2 of `vol` with chunk edge `ce` and the given
// key separator. Returns the zarr dir path.
static fs::path write_raw_zarr(const fs::path& base, const std::vector<u8>& vol,
                               Coord3 shape, u64 ce, char sep) {
    fs::create_directories(base);
    {
        std::ofstream za(base / ".zarray");
        za << "{\"shape\":[" << shape.z << "," << shape.y << "," << shape.x << "],"
           << "\"chunks\":[" << ce << "," << ce << "," << ce << "],"
           << "\"dtype\":\"|u1\",\"compressor\":null,\"fill_value\":0,"
           << "\"order\":\"C\",\"zarr_format\":2,\"filters\":null}";
    }
    u64 nz = (shape.z + ce - 1)/ce, ny = (shape.y + ce - 1)/ce, nx = (shape.x + ce - 1)/ce;
    for (u64 kz = 0; kz < nz; ++kz)
      for (u64 ky = 0; ky < ny; ++ky)
        for (u64 kx = 0; kx < nx; ++kx) {
            std::vector<u8> blk(ce*ce*ce, 0);
            for (u64 z = 0; z < ce && kz*ce+z < shape.z; ++z)
              for (u64 y = 0; y < ce && ky*ce+y < shape.y; ++y)
                for (u64 x = 0; x < ce && kx*ce+x < shape.x; ++x)
                    blk[(z*ce + y)*ce + x] = vol[((kz*ce+z)*shape.y + (ky*ce+y))*shape.x + (kx*ce+x)];
            fs::path cp;
            if (sep == '/') { fs::create_directories(base / std::to_string(kz) / std::to_string(ky));
                              cp = base / std::to_string(kz) / std::to_string(ky) / std::to_string(kx); }
            else cp = base / (std::to_string(kz) + "." + std::to_string(ky) + "." + std::to_string(kx));
            write_file(cp, blk);
        }
    return base;
}

static f64 psnr(const std::vector<u8>& a, const std::vector<u8>& b) {
    f64 se = 0; for (size_t i = 0; i < a.size(); ++i) { f64 d = f64(a[i])-b[i]; se += d*d; }
    f64 m = se / a.size(); return m <= 0 ? 999.0 : 10.0*std::log10(255.0*255.0/m);
}

int main() {
#ifndef C4D_ZARR_TO_C4D_BIN
    std::fprintf(stderr, "  zarr_to_c4d binary path not provided; skipping\n");
    RUN_TESTS_RETURN();
#else
    const char* bin = C4D_ZARR_TO_C4D_BIN;
    fs::path tmp = fs::temp_directory_path()
                 / ("c4d_zarr_test_" + std::to_string(std::random_device{}()));
    fs::create_directories(tmp);

    // a 128x128x128 CT-like volume (2x2x2 = 8 c4d chunks, fits one region)
    Coord3 shape{128, 128, 128};
    std::mt19937 g(99);
    std::normal_distribution<f32> n(0.f, 5.f);
    std::vector<u8> vol(shape.z*shape.y*shape.x);
    for (u64 z = 0; z < shape.z; ++z)
        for (u64 y = 0; y < shape.y; ++y)
            for (u64 x = 0; x < shape.x; ++x) {
                f32 v = 110 + 45*std::sin(0.05*x) + 30*std::cos(0.04*y) + 20*std::sin(0.06*z) + n(g);
                vol[(z*shape.y + y)*shape.x + x] = (u8)(v < 0 ? 0 : v > 255 ? 255 : v);
            }

    auto run = [&](char sep, u64 ce, const char* tag) {
        fs::path zdir = tmp / (std::string("z_") + tag);
        write_raw_zarr(zdir, vol, shape, ce, sep);
        fs::path out = tmp / (std::string("o_") + tag + ".c4d");
        std::string cmd = std::string(bin) + " " + zdir.string() + " " + out.string()
                        + " --q 16 2>/dev/null";
        int rc = std::system(cmd.c_str());
        CHECK(rc == 0);
        std::ifstream f(out, std::ios::binary | std::ios::ate);
        CHECK(static_cast<bool>(f));
        auto sz = f.tellg(); f.seekg(0);
        std::vector<u8> file((size_t)sz); f.read(reinterpret_cast<char*>(file.data()), sz);
        CHECK(file.size() > 0 && file.size() < vol.size());      // it compressed
        auto dec = decode_archive(file, shape);
        CHECK(dec.size() == vol.size());
        f64 p = psnr(vol, dec);
        std::fprintf(stderr, "  zarr(sep='%c', src-chunk=%llu) -> c4d: %.2f dB, %.1fx\n",
                     sep, (unsigned long long)ce, p, double(vol.size())/file.size());
        CHECK(p > 33.0);                                          // q16 quality holds
    };

    run('.', 128, "flat128");     // flat keys, source chunk != 64 (re-tile)
    run('/', 64,  "nested64");    // nested keys, source chunk == 64
    run('.', 96,  "flat96");      // unaligned source chunk (96 -> partial tiles)

    fs::remove_all(tmp);
    RUN_TESTS_RETURN();
#endif
}
