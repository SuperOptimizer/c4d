// zarr_to_c4d: convert a local zarr v2 u8 volume into a single .c4d archive.
//
// Modeled on volume-cartographer's vc_zarr_recompress, but self-contained and
// minimal: local filesystem only, u8 only, optional blosc. It re-tiles the
// source zarr (any chunk shape) into c4d's 64³ chunks, encodes each 256³ region
// as one shared-table group (the production path), and writes ONE footer-indexed
// .c4d archive (intensity member + region-tables member + optional validity
// mask). Identical archive layout to `c4d encode`, just sourced from zarr.
//
// A zarr v2 array is a directory with a `.zarray` JSON (shape / chunks / dtype /
// compressor / order / fill_value) and one file per chunk named "z.y.x" (or
// "z/y/x"). We read u8 ("|u1") arrays; blosc-compressed or raw. Multi-level
// pyramids: pass the level subdir (e.g. `vol.zarr/0`) as <zarr_dir>.
//
// Usage:
//   zarr_to_c4d <zarr_dir> <out.c4d> [--q N] [--mask-threshold T]
//
// Build (blosc optional):
//   g++ -std=c++26 -O3 -march=native -ffast-math -I include tools/zarr_to_c4d.cpp \
//       [-DC4D_HAVE_BLOSC -lblosc] -o zarr_to_c4d
#include "c4d/chunk.hpp"
#include "c4d/archive.hpp"
#include "c4d/mask.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>
#if __has_include(<blosc.h>) && defined(C4D_WITH_BLOSC)
#  include <blosc.h>
#  define C4D_HAVE_BLOSC 1
#endif

namespace fs = std::filesystem;
using namespace c4d;

// --- tiny helpers ----------------------------------------------------------
static std::vector<u8> read_bytes(const fs::path& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) return {};
    auto n = f.tellg(); f.seekg(0);
    std::vector<u8> v(static_cast<size_t>(n));
    f.read(reinterpret_cast<char*>(v.data()), n);
    return v;
}
static void write_bytes(const fs::path& p, std::span<const u8> v) {
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size()));
}

// Minimal JSON scalar/array extraction for the handful of .zarray fields we
// need (no nesting beyond flat arrays). Not a general JSON parser.
static std::string json_field(const std::string& js, const std::string& key) {
    auto k = js.find("\"" + key + "\"");
    if (k == std::string::npos) return {};
    auto c = js.find(':', k); if (c == std::string::npos) return {};
    size_t i = c + 1; while (i < js.size() && (js[i] == ' ' || js[i] == '\n' || js[i] == '\t')) ++i;
    if (i >= js.size()) return {};
    if (js[i] == '[') { auto e = js.find(']', i); return js.substr(i, e - i + 1); }
    if (js[i] == '"') { auto e = js.find('"', i + 1); return js.substr(i + 1, e - i - 1); }
    if (js[i] == 'n') return "null";
    size_t e = i; while (e < js.size() && js[e] != ',' && js[e] != '}' && js[e] != '\n') ++e;
    std::string s = js.substr(i, e - i);
    while (!s.empty() && (s.back() == ' ' || s.back() == '\r')) s.pop_back();
    return s;
}
static std::vector<u64> json_u64_array(const std::string& arr) {
    std::vector<u64> out; size_t i = 0;
    while (i < arr.size()) {
        while (i < arr.size() && !std::isdigit(static_cast<unsigned char>(arr[i]))) ++i;
        if (i >= arr.size()) break;
        out.push_back(std::strtoull(arr.c_str() + i, nullptr, 10));
        while (i < arr.size() && std::isdigit(static_cast<unsigned char>(arr[i]))) ++i;
    }
    return out;
}

// Decompress a zarr v2 chunk into `expected` raw u8 bytes. blosc if the header
// magic (byte 0 == 0x02) is present and blosc is compiled in; else raw/identity.
static std::vector<u8> decompress_chunk(const std::vector<u8>& comp, size_t expected,
                                        bool compressor) {
    if (!compressor || comp.empty()) {                 // raw store
        std::vector<u8> v = comp; v.resize(expected, 0); return v;
    }
#ifdef C4D_HAVE_BLOSC
    if (comp.size() >= 16 && comp[0] == 0x02) {
        u32 nbytes = 0; std::memcpy(&nbytes, comp.data() + 4, 4);
        std::vector<u8> out(nbytes);
        int ret = blosc_decompress_ctx(comp.data(), out.data(), nbytes, 1);
        if (ret < 0) { std::fprintf(stderr, "blosc_decompress failed (%d)\n", ret); std::exit(1); }
        out.resize(static_cast<size_t>(ret)); out.resize(expected, 0); return out;
    }
#endif
    if (!comp.empty() && comp.size() != expected && comp[0] == 0x02) {
        std::fprintf(stderr, "chunk is blosc-compressed but blosc support not built "
                             "(rebuild with -DC4D_WITH_BLOSC -lblosc)\n");
        std::exit(1);
    }
    std::vector<u8> v = comp; v.resize(expected, 0); return v;
}

// --- zarr v2 source --------------------------------------------------------
struct ZarrV2 {
    fs::path dir;
    u64 sz=0, sy=0, sx=0;             // shape (Z,Y,X)
    u64 cz=0, cy=0, cx=0;             // chunk shape
    bool compressor=false;
    char sep='.';                     // chunk-key separator ('.' v2-default or '/')
    u8 fill=0;

    static ZarrV2 open(const fs::path& d) {
        ZarrV2 z; z.dir = d;
        std::string js;
        { auto b = read_bytes(d / ".zarray");
          if (b.empty()) { std::fprintf(stderr, "no .zarray in %s\n", d.c_str()); std::exit(1); }
          js.assign(reinterpret_cast<char*>(b.data()), b.size()); }
        auto shape = json_u64_array(json_field(js, "shape"));
        auto chunks = json_u64_array(json_field(js, "chunks"));
        if (shape.size() != 3 || chunks.size() != 3) {
            std::fprintf(stderr, "need a 3D zarr (got shape ndim=%zu)\n", shape.size()); std::exit(1);
        }
        z.sz=shape[0]; z.sy=shape[1]; z.sx=shape[2];
        z.cz=chunks[0]; z.cy=chunks[1]; z.cx=chunks[2];
        std::string dt = json_field(js, "dtype");
        if (dt.find("u1") == std::string::npos && dt.find("i1") == std::string::npos) {
            std::fprintf(stderr, "c4d is u8-only; zarr dtype is '%s' (convert/window to u8 first)\n",
                         dt.c_str()); std::exit(1);
        }
        z.compressor = (json_field(js, "compressor") != "null" && !json_field(js,"compressor").empty());
        std::string fv = json_field(js, "fill_value");
        if (!fv.empty() && fv != "null") z.fill = static_cast<u8>(std::strtol(fv.c_str(), nullptr, 10));
        // detect separator: try a "0.0.0" file, else assume nested "0/0/0"
        if (!fs::exists(d / "0.0.0") && fs::exists(d / "0")) z.sep = '/';
        return z;
    }

    fs::path chunk_path(u64 kz, u64 ky, u64 kx) const {
        char buf[96];
        if (sep == '/') std::snprintf(buf, sizeof buf, "%llu/%llu/%llu",
                                      (unsigned long long)kz,(unsigned long long)ky,(unsigned long long)kx);
        else std::snprintf(buf, sizeof buf, "%llu.%llu.%llu",
                           (unsigned long long)kz,(unsigned long long)ky,(unsigned long long)kx);
        return dir / buf;
    }
};

// Cache the most-recently-read source chunk so a 64³ gather (which scans a
// contiguous run within one source chunk per row) doesn't re-read/-decompress.
struct ChunkCache {
    const ZarrV2& z;
    u64 ckz=~0ull, cky=~0ull, ckx=~0ull;
    std::vector<u8> buf;             // decompressed source chunk (cz*cy*cx)
    bool present=false;
    explicit ChunkCache(const ZarrV2& zz) : z(zz) {}

    // voxel (gz,gy,gx) in global coords; returns fill if the chunk is absent.
    u8 at(u64 gz, u64 gy, u64 gx) {
        u64 kz=gz/z.cz, ky=gy/z.cy, kx=gx/z.cx;
        if (kz!=ckz || ky!=cky || kx!=ckx) {
            auto comp = read_bytes(z.chunk_path(kz,ky,kx));
            present = !comp.empty();
            buf = present ? decompress_chunk(comp, z.cz*z.cy*z.cx, z.compressor)
                          : std::vector<u8>{};
            ckz=kz; cky=ky; ckx=kx;
        }
        if (!present) return z.fill;
        u64 lz=gz%z.cz, ly=gy%z.cy, lx=gx%z.cx;
        return buf[(lz*z.cy + ly)*z.cx + lx];
    }
};

int main(int argc, char** argv) {
    std::string in, out; f32 q = 16.f; int mask_thresh = -1;
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];
        if (a == "--q" && i+1 < argc) q = std::strtof(argv[++i], nullptr);
        else if (a == "--mask-threshold" && i+1 < argc) mask_thresh = std::atoi(argv[++i]);
        else pos.push_back(std::string(a));
    }
    if (pos.size() != 2) {
        std::fprintf(stderr, "usage: zarr_to_c4d <zarr_dir> <out.c4d> [--q N] [--mask-threshold T]\n");
        return 2;
    }
    in = pos[0]; out = pos[1];

    ZarrV2 z = ZarrV2::open(in);
    Coord3 shape{z.sz, z.sy, z.sx};
    std::fprintf(stderr, "zarr %s: shape %llux%llux%llu, chunks %llux%llux%llu, %s%s\n",
                 in.c_str(), (unsigned long long)z.sz,(unsigned long long)z.sy,(unsigned long long)z.sx,
                 (unsigned long long)z.cz,(unsigned long long)z.cy,(unsigned long long)z.cx,
                 z.compressor ? "blosc" : "raw", z.sep=='/' ? ", nested" : "");

    Coord3 g = chunk_grid(shape);
    u64 nchunks = g.z*g.y*g.x;
    Coord3 rg = region_grid(g);
    u64 nregions = rg.z*rg.y*rg.x;
    std::vector<std::vector<u8>> payloads(nchunks);
    std::vector<f32> qs(nchunks, q);
    std::vector<std::vector<u8>> region_tables(nregions);
    std::vector<f32> region_table_qs(nregions, 0.f);
    std::vector<std::vector<u8>> mask_payloads;
    std::vector<f32> mask_qs;
    if (mask_thresh >= 0) { mask_payloads.resize(nchunks); mask_qs.assign(nchunks, 0.f); }

    ChunkCache cache(z);
    // gather a 64³ c4d cube from the zarr; returns false if fully outside volume.
    auto gather = [&](Coord3 c, std::vector<u8>& cube) -> bool {
        std::fill(cube.begin(), cube.end(), u8(0));
        u64 z0=c.z*CHUNK, y0=c.y*CHUNK, x0=c.x*CHUNK;
        if (z0 >= shape.z || y0 >= shape.y || x0 >= shape.x) return false;
        for (u32 dz=0; dz<CHUNK && z0+dz<shape.z; ++dz)
            for (u32 dy=0; dy<CHUNK && y0+dy<shape.y; ++dy)
                for (u32 dx=0; dx<CHUNK && x0+dx<shape.x; ++dx)
                    cube[vox_index(dz,dy,dx)] = cache.at(z0+dz, y0+dy, x0+dx);
        return true;
    };

    u64 mask_total = 0;
    std::vector<std::vector<u8>> cubes;
    for (u64 ri = 0; ri < nregions; ++ri) {
        Coord3 rc = chunk_unravel(ri, rg);
        std::vector<u64> cidx; std::vector<std::span<const u8>> spans; cubes.clear();
        for (u32 dz=0; dz<REGION_CHUNKS; ++dz)
          for (u32 dy=0; dy<REGION_CHUNKS; ++dy)
            for (u32 dx=0; dx<REGION_CHUNKS; ++dx) {
                Coord3 c{rc.z*REGION_CHUNKS+dz, rc.y*REGION_CHUNKS+dy, rc.x*REGION_CHUNKS+dx};
                if (c.z>=g.z || c.y>=g.y || c.x>=g.x) continue;
                std::vector<u8> cube(CHUNK_VOX);
                if (!gather(c, cube)) { payloads[chunk_linear(c, g)] = {}; continue; }
                if (mask_thresh >= 0) {
                    std::vector<u8> valid(CHUNK_VOX);
                    for (u32 i=0;i<CHUNK_VOX;++i) valid[i] = (cube[i] > u8(mask_thresh)) ? 1 : 0;
                    auto me = mask::encode(valid);
                    mask_payloads[chunk_linear(c, g)] = me.bytes; mask_total += me.bytes.size();
                }
                cidx.push_back(chunk_linear(c, g));
                cubes.push_back(std::move(cube));
            }
        if (cubes.empty()) { region_tables[ri] = {}; continue; }
        for (auto& cb : cubes) spans.emplace_back(cb);
        auto ge = chunk::encode_group(spans, chunk::EncodeOpts{.q = q});
        std::vector<u8> tb; ge.tables.serialize(tb);
        region_tables[ri] = std::move(tb);
        for (size_t k=0;k<cidx.size();++k) payloads[cidx[k]] = ge.payloads[k].serialize();
    }

    archive::Writer w;
    w.add_member("intensity", archive::MemberType::Intensity, shape, payloads, qs);
    w.add_member("region_tables", archive::MemberType::Metadata,
                 Coord3{rg.z, rg.y, rg.x}, region_tables, region_table_qs);
    if (mask_thresh >= 0)
        w.add_member("mask", archive::MemberType::ValidityMask, shape, mask_payloads, mask_qs);
    char meta[192];
    std::snprintf(meta, sizeof meta, R"({"bbox":[%llu,%llu,%llu],"source":"zarr"%s})",
                  (unsigned long long)shape.z,(unsigned long long)shape.y,(unsigned long long)shape.x,
                  mask_thresh>=0 ? R"(,"mask":true)" : "");
    w.set_metadata(meta);
    auto file = w.finish();
    write_bytes(out, file);

    u64 vox = z.sz*z.sy*z.sx;
    std::fprintf(stderr, "encoded %llu chunks (%llu regions) -> %s (%zu bytes, %.1fx)\n",
                 (unsigned long long)nchunks, (unsigned long long)nregions, out.c_str(),
                 file.size(), double(vox) / file.size());
    if (mask_thresh >= 0)
        std::fprintf(stderr, "  + validity mask (thresh %d): %llu bytes\n",
                     mask_thresh, (unsigned long long)mask_total);
    return 0;
}
