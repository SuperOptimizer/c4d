// c4d CLI — encode / decode / info / compact / bench.
// `bench` runs the per-chunk codec across a directory of raw 128^3 u8 chunks
// and reports PSNR + compression ratio per quality knob (the start of the §7.1
// metric basket). encode/decode/compact are filled in as the archive lands.
#include "c4d/chunk.hpp"
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
    if (argc < 3) { std::fprintf(stderr, "usage: c4d bench <dir-of-raw-chunks> [q1 q2 ...]\n"); return 2; }
    std::vector<f32> qs;
    for (int i = 3; i < argc; ++i) qs.push_back(std::strtof(argv[i], nullptr));
    if (qs.empty()) qs = {4, 8, 16, 32, 64};

    std::vector<std::string> files;
    for (auto& e : fs::directory_iterator(argv[2]))
        if (e.path().extension() == ".raw") files.push_back(e.path().string());
    std::sort(files.begin(), files.end());
    if (files.empty()) { std::fprintf(stderr, "no .raw chunks in %s\n", argv[2]); return 1; }

    std::fprintf(stderr, "corpus: %zu chunks from %s\n\n", files.size(), argv[2]);
    std::printf("%6s  %10s  %10s  %12s\n", "q", "PSNR(dB)", "ratio", "bits/voxel");
    for (f32 q : qs) {
        f64 sum_se = 0; u64 total_bytes = 0; u64 total_vox = 0;
        for (auto& f : files) {
            auto vox = read_file(f);
            if (vox.size() != CHUNK_VOX) continue;
            auto pl = chunk::encode_chunk(vox, q);
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
    if (cmd == "bench") return bench(argc, argv);
    if (cmd == "info" || cmd == "encode" || cmd == "decode" || cmd == "compact") {
        std::fprintf(stderr, "c4d %.*s: not yet implemented (archive container pending)\n",
                     static_cast<int>(cmd.size()), cmd.data());
        return 1;
    }
    return usage();
}
