// c4d CLI — encode / decode / info / compact / bench.
// `bench` runs the per-chunk codec across a directory of raw 128^3 u8 chunks
// and reports PSNR + compression ratio per quality knob (the start of the §7.1
// metric basket). encode/decode/compact are filled in as the archive lands.
#include "c4d/chunk.hpp"
#include "c4d/archive.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;
using namespace c4d;

static std::vector<u8> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    auto n = f.tellg(); f.seekg(0);
    std::vector<u8> buf(static_cast<size_t>(n));
    f.read(reinterpret_cast<char*>(buf.data()), n);
    return buf;
}

static f64 psnr_u8(std::span<const u8> a, std::span<const u8> b) {
    f64 se = 0;
    for (size_t i = 0; i < a.size(); ++i) { f64 d = f64(a[i]) - b[i]; se += d * d; }
    f64 mse = se / a.size();
    return mse <= 0 ? 999.0 : 10.0 * std::log10(255.0 * 255.0 / mse);
}

static int bench(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: c4d bench <dir-of-raw-chunks> [--denoise] [q1 q2 ...]\n"); return 2; }
    bool denoise = false;
    std::vector<f32> qs;
    for (int i = 3; i < argc; ++i) {
        std::string_view a = argv[i];
        if (a == "--denoise") denoise = true;
        else qs.push_back(std::strtof(argv[i], nullptr));
    }
    if (qs.empty()) qs = {4, 8, 16, 32, 64};

    std::vector<std::string> files;
    for (auto& e : fs::directory_iterator(argv[2]))
        if (e.path().extension() == ".raw") files.push_back(e.path().string());
    std::sort(files.begin(), files.end());
    if (files.empty()) { std::fprintf(stderr, "no .raw chunks in %s\n", argv[2]); return 1; }

    std::fprintf(stderr, "corpus: %zu chunks from %s%s\n\n", files.size(), argv[2],
                 denoise ? "  [noise-aware dead-zone]" : "");
    std::printf("%6s  %10s  %10s  %12s\n", "q", "PSNR(dB)", "ratio", "bits/voxel");
    for (f32 q : qs) {
        f64 sum_se = 0; u64 total_bytes = 0; u64 total_vox = 0;
        for (auto& f : files) {
            auto vox = read_file(f);
            if (vox.size() != CHUNK_VOX) continue;
            chunk::EncodeOpts opt{.q = q, .noise_aware = denoise};
            auto pl = chunk::encode_chunk(vox, opt);
            auto blob = pl.serialize();
            std::vector<u8> rec(CHUNK_VOX);
            chunk::decode_chunk(pl, rec);
            for (u32 i = 0; i < CHUNK_VOX; ++i) { f64 d = f64(vox[i]) - rec[i]; sum_se += d * d; }
            total_bytes += blob.size(); total_vox += CHUNK_VOX;
        }
        f64 mse = sum_se / total_vox;
        f64 psnr = mse <= 0 ? 999.0 : 10.0 * std::log10(255.0 * 255.0 / mse);
        f64 ratio = double(total_vox) / total_bytes;
        f64 bpv = 8.0 * total_bytes / total_vox;
        std::printf("%6.1f  %10.2f  %9.1fx  %12.4f\n", q, psnr, ratio, bpv);
    }
    return 0;
}

static void write_file(const std::string& path, std::span<const u8> bytes) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

// Slice the logical volume into padded 128^3 chunks (Z,Y,X), encode each, and
// write a single-member intensity archive. Edge chunks zero-padded (§2).
static int encode(int argc, char** argv) {
    Coord3 shape{}; f32 q = 16.f; std::string in, out;
    std::vector<std::string> pos;
    for (int i = 2; i < argc; ++i) {
        std::string_view a = argv[i];
        if (a == "--shape" && i + 1 < argc) {
            std::sscanf(argv[++i], "%llu,%llu,%llu",
                        (unsigned long long*)&shape.z, (unsigned long long*)&shape.y,
                        (unsigned long long*)&shape.x);
        } else if (a == "--q" && i + 1 < argc) {
            q = std::strtof(argv[++i], nullptr);
        } else pos.push_back(std::string(a));
    }
    if (pos.size() != 2 || shape.z == 0) {
        std::fprintf(stderr, "usage: c4d encode <raw_volume> <archive.c4d> --shape Z,Y,X [--q N]\n");
        return 2;
    }
    in = pos[0]; out = pos[1];
    auto vol = read_file(in);
    u64 expect = shape.z * shape.y * shape.x;
    if (vol.size() != expect) {
        std::fprintf(stderr, "input is %zu bytes, shape implies %llu\n", vol.size(),
                     (unsigned long long)expect);
        return 1;
    }
    Coord3 g = chunk_grid(shape);
    u64 nchunks = g.z * g.y * g.x;
    std::vector<std::vector<u8>> payloads(nchunks);
    std::vector<f32> qs(nchunks, q);
    std::vector<u8> cube(CHUNK_VOX);
    for (u64 ci = 0; ci < nchunks; ++ci) {
        Coord3 c = chunk_unravel(ci, g);
        // gather a zero-padded 128^3 cube from the volume at chunk coord c
        std::fill(cube.begin(), cube.end(), u8(0));
        u64 z0 = c.z * CHUNK, y0 = c.y * CHUNK, x0 = c.x * CHUNK;
        bool any = false;
        for (u32 dz = 0; dz < CHUNK && z0 + dz < shape.z; ++dz)
            for (u32 dy = 0; dy < CHUNK && y0 + dy < shape.y; ++dy) {
                u64 src = ((z0 + dz) * shape.y + (y0 + dy)) * shape.x + x0;
                u32 w = static_cast<u32>(std::min<u64>(CHUNK, shape.x - x0));
                std::memcpy(&cube[vox_index(dz, dy, 0)], &vol[src], w);
                any = true;
            }
        if (!any) { payloads[ci] = {}; continue; }   // fully outside -> ABSENT
        payloads[ci] = chunk::encode_chunk(cube, q).serialize();
    }
    archive::Writer w;
    w.add_member("intensity", archive::MemberType::Intensity, shape, payloads, qs);
    char meta[128];
    std::snprintf(meta, sizeof meta, R"({"bbox":[%llu,%llu,%llu]})",
                  (unsigned long long)shape.z, (unsigned long long)shape.y,
                  (unsigned long long)shape.x);
    w.set_metadata(meta);
    auto file = w.finish();
    write_file(out, file);
    std::fprintf(stderr, "encoded %llu chunks -> %s (%zu bytes, %.1fx)\n",
                 (unsigned long long)nchunks, out.c_str(), file.size(),
                 double(expect) / file.size());
    return 0;
}

static int decode(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: c4d decode <archive.c4d> <raw_volume>\n"); return 2; }
    auto file = read_file(argv[2]);
    archive::Reader r(file);
    size_t mi = r.find("intensity");
    if (mi == SIZE_MAX) { std::fprintf(stderr, "no intensity member\n"); return 1; }
    const archive::Member& m = r.member(mi);
    Coord3 shape = m.shape, g = chunk_grid(shape);
    std::vector<u8> vol(shape.z * shape.y * shape.x, 0);
    std::vector<u8> cube(CHUNK_VOX);
    for (u64 ci = 0; ci < m.index.size(); ++ci) {
        Coord3 c = chunk_unravel(ci, g);
        auto p = r.chunk_payload(mi, ci);
        if (p.empty()) std::fill(cube.begin(), cube.end(), u8(0));
        else { auto pl = chunk::Payload::deserialize(p); chunk::decode_chunk(pl, cube); }
        u64 z0 = c.z * CHUNK, y0 = c.y * CHUNK, x0 = c.x * CHUNK;
        for (u32 dz = 0; dz < CHUNK && z0 + dz < shape.z; ++dz)
            for (u32 dy = 0; dy < CHUNK && y0 + dy < shape.y; ++dy) {
                u64 dst = ((z0 + dz) * shape.y + (y0 + dy)) * shape.x + x0;
                u32 w = static_cast<u32>(std::min<u64>(CHUNK, shape.x - x0));
                std::memcpy(&vol[dst], &cube[vox_index(dz, dy, 0)], w);
            }
    }
    write_file(argv[3], vol);
    std::fprintf(stderr, "decoded %s -> %s (%zu bytes)\n", argv[2], argv[3], vol.size());
    return 0;
}

static int info(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: c4d info <archive.c4d>\n"); return 2; }
    auto file = read_file(argv[2]);
    archive::Reader r(file);
    std::printf("c4d archive: %zu bytes, %zu member(s)\n", file.size(), r.member_count());
    for (size_t i = 0; i < r.member_count(); ++i) {
        const auto& m = r.member(i);
        u64 absent = 0, stored = 0;
        for (auto& e : m.index) (e.flags & archive::IndexEntry::ABSENT) ? ++absent : ++stored;
        std::printf("  [%zu] %-12s type=%u shape=%llux%llux%llu chunks=%zu (stored=%llu absent=%llu)\n",
                    i, m.name.c_str(), unsigned(m.type),
                    (unsigned long long)m.shape.z, (unsigned long long)m.shape.y,
                    (unsigned long long)m.shape.x, m.index.size(),
                    (unsigned long long)stored, (unsigned long long)absent);
    }
    std::printf("  metadata: %.*s\n", int(r.metadata().size()), r.metadata().data());
    return 0;
}

static int usage() {
    std::fprintf(stderr,
        "c4d — compress4d archive tool\n"
        "usage:\n"
        "  c4d bench  <dir-of-raw-chunks> [q ...]   benchmark codec on 128^3 chunks\n"
        "  c4d info   <archive.c4d>\n"
        "  c4d encode <raw_volume> <archive.c4d> --shape Z,Y,X [--q N]\n"
        "  c4d decode <archive.c4d> <raw_volume>\n"
        "  c4d compact <in.c4d> <out.c4d>\n");
    return 2;
}

int main(int argc, char** argv) {
    if (argc < 2) return usage();
    std::string_view cmd = argv[1];
    if (cmd == "bench")  return bench(argc, argv);
    if (cmd == "encode") return encode(argc, argv);
    if (cmd == "decode") return decode(argc, argv);
    if (cmd == "info")   return info(argc, argv);
    if (cmd == "compact") {
        std::fprintf(stderr, "c4d compact: not yet implemented\n");
        return 1;
    }
    return usage();
}
