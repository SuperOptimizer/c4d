// c4d CLI — encode / decode / info / compact / bench.
// `bench` runs the per-chunk codec across a directory of raw CHUNK^3 (64^3 by
// default) u8 chunks and reports PSNR + compression ratio per quality knob
// (the §7.1 metric basket). encode/decode build/read the .c4d archive.
#include "c4d/chunk.hpp"
#include "c4d/archive.hpp"
#include "c4d/metrics.hpp"
#include "c4d/mask.hpp"
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
    bool denoise = false, rdo = false; f32 tol = 0;
    std::vector<f32> qs;
    for (int i = 3; i < argc; ++i) {
        std::string_view a = argv[i];
        if (a == "--denoise") denoise = true;
        else if (a == "--rdo") rdo = true;
        else if (a == "--tolerance" && i + 1 < argc) tol = std::strtof(argv[++i], nullptr);
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
    // §7.1 basket: per-metric table across the q sweep x corpus.
    std::printf("%5s %7s %7s %7s %7s %7s %7s %7s %7s\n",
                "q", "ratio", "PSNR", "GMSD", "MSSSIM", "HaarPSI", "sharp", "EPI", "geomean");
    for (f32 q : qs) {
        u64 total_bytes = 0, total_vox = 0, nc = 0;
        f64 sp = 0, sg = 0, sm = 0, sh = 0, ss = 0, se = 0, sgeo = 0;
        for (auto& f : files) {
            auto vox = read_file(f);
            if (vox.size() != CHUNK_VOX) continue;
            chunk::EncodeOpts opt{.q = q, .noise_aware = denoise, .tolerance = tol,
                                  .perceptual_rdo = rdo};
            auto pl = chunk::encode_chunk(vox, opt);
            auto blob = pl.serialize();
            std::vector<u8> rec(CHUNK_VOX);
            chunk::decode_chunk(pl, rec);
            auto b = metrics::evaluate(vox, rec);
            sp += b.psnr_db; sg += b.gmsd; sm += b.ms_ssim; sh += b.haarpsi;
            ss += b.sharpness; se += b.epi; sgeo += b.geomean();
            total_bytes += blob.size(); total_vox += CHUNK_VOX; ++nc;
        }
        f64 ratio = double(total_vox) / total_bytes;
        std::printf("%5.1f %6.1fx %7.2f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f\n",
                    q, ratio, sp/nc, sg/nc, sm/nc, sh/nc, ss/nc, se/nc, sgeo/nc);
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
    int mask_thresh = -1;   // >=0 enables a validity mask: voxel <= thresh => invalid
    std::vector<std::string> pos;
    for (int i = 2; i < argc; ++i) {
        std::string_view a = argv[i];
        if (a == "--shape" && i + 1 < argc) {
            std::sscanf(argv[++i], "%llu,%llu,%llu",
                        (unsigned long long*)&shape.z, (unsigned long long*)&shape.y,
                        (unsigned long long*)&shape.x);
        } else if (a == "--q" && i + 1 < argc) {
            q = std::strtof(argv[++i], nullptr);
        } else if (a == "--mask-threshold" && i + 1 < argc) {
            mask_thresh = std::atoi(argv[++i]);
        } else pos.push_back(std::string(a));
    }
    if (pos.size() != 2 || shape.z == 0) {
        std::fprintf(stderr, "usage: c4d encode <raw_volume> <archive.c4d> --shape Z,Y,X "
                             "[--q N] [--mask-threshold T]\n");
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
    Coord3 rg = region_grid(g);
    u64 nregions = rg.z * rg.y * rg.x;
    std::vector<std::vector<u8>> payloads(nchunks);
    std::vector<f32> qs(nchunks, q);
    std::vector<std::vector<u8>> region_tables(nregions);   // shared tables/region
    std::vector<f32> region_table_qs(nregions, 0.f);
    std::vector<std::vector<u8>> mask_payloads;     // only if masking
    std::vector<f32> mask_qs;
    if (mask_thresh >= 0) { mask_payloads.resize(nchunks); mask_qs.assign(nchunks, 0.f); }

    auto gather = [&](Coord3 c, std::vector<u8>& cube) -> bool {
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
        return any;
    };

    // Encode REGION BY REGION: each 4×4×4-chunk region shares one entropy-table
    // set (the +10-14% no-independence win). Chunks fully outside the volume are
    // ABSENT. Region tables are stored as a separate member, indexed by region.
    u64 mask_total = 0;
    std::vector<std::vector<u8>> cubes;
    for (u64 ri = 0; ri < nregions; ++ri) {
        Coord3 rc = chunk_unravel(ri, rg);
        // collect the (up to) REGION_CHUNKS^3 present chunks of this region
        std::vector<u64> cidx; std::vector<std::span<const u8>> spans; cubes.clear();
        for (u32 dz = 0; dz < REGION_CHUNKS; ++dz)
          for (u32 dy = 0; dy < REGION_CHUNKS; ++dy)
            for (u32 dx = 0; dx < REGION_CHUNKS; ++dx) {
                Coord3 c{rc.z*REGION_CHUNKS+dz, rc.y*REGION_CHUNKS+dy, rc.x*REGION_CHUNKS+dx};
                if (c.z >= g.z || c.y >= g.y || c.x >= g.x) continue;
                std::vector<u8> cube(CHUNK_VOX);
                if (!gather(c, cube)) { payloads[chunk_linear(c, g)] = {}; continue; }
                if (mask_thresh >= 0) {
                    std::vector<u8> valid(CHUNK_VOX);
                    for (u32 i = 0; i < CHUNK_VOX; ++i) valid[i] = (cube[i] > u8(mask_thresh)) ? 1 : 0;
                    auto me = mask::encode(valid);
                    mask_payloads[chunk_linear(c, g)] = me.bytes; mask_total += me.bytes.size();
                }
                cidx.push_back(chunk_linear(c, g));
                cubes.push_back(std::move(cube));
            }
        if (cubes.empty()) { region_tables[ri] = {}; continue; }   // empty region
        for (auto& cb : cubes) spans.emplace_back(cb);
        auto ge = chunk::encode_group(spans, chunk::EncodeOpts{.q = q});
        std::vector<u8> tb; ge.tables.serialize(tb);
        region_tables[ri] = std::move(tb);
        for (size_t k = 0; k < cidx.size(); ++k)
            payloads[cidx[k]] = ge.payloads[k].serialize();
    }
    archive::Writer w;
    w.add_member("intensity", archive::MemberType::Intensity, shape, payloads, qs);
    w.add_member("region_tables", archive::MemberType::Metadata,
                 Coord3{rg.z, rg.y, rg.x}, region_tables, region_table_qs);
    if (mask_thresh >= 0)
        w.add_member("mask", archive::MemberType::ValidityMask, shape, mask_payloads, mask_qs);
    char meta[160];
    std::snprintf(meta, sizeof meta, R"({"bbox":[%llu,%llu,%llu]%s})",
                  (unsigned long long)shape.z, (unsigned long long)shape.y,
                  (unsigned long long)shape.x,
                  mask_thresh >= 0 ? R"(,"mask":true)" : "");
    w.set_metadata(meta);
    auto file = w.finish();
    write_file(out, file);
    std::fprintf(stderr, "encoded %llu chunks -> %s (%zu bytes, %.1fx)%s\n",
                 (unsigned long long)nchunks, out.c_str(), file.size(),
                 double(expect) / file.size(),
                 mask_thresh >= 0 ? "" : "");
    if (mask_thresh >= 0)
        std::fprintf(stderr, "  + validity mask (thresh %d): %llu bytes\n",
                     mask_thresh, (unsigned long long)mask_total);
    return 0;
}

static int decode(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: c4d decode <archive.c4d> <raw_volume>\n"); return 2; }
    auto file = read_file(argv[2]);
    archive::Reader r(file);
    size_t mi = r.find("intensity");
    if (mi == SIZE_MAX) { std::fprintf(stderr, "no intensity member\n"); return 1; }
    const archive::Member& m = r.member(mi);
    size_t mk = r.find("mask");                 // optional validity mask member
    size_t mt = r.find("region_tables");        // region-shared entropy tables
    Coord3 shape = m.shape, g = chunk_grid(shape), rg = region_grid(g);
    std::vector<u8> vol(shape.z * shape.y * shape.x, 0);
    std::vector<u8> cube(CHUNK_VOX), valid(CHUNK_VOX);
    // cache the most-recently-loaded region's tables (decode walks region-major-ish)
    u64 cached_region = ~0ull; chunk::SharedTables cached_tables;
    for (u64 ci = 0; ci < m.index.size(); ++ci) {
        Coord3 c = chunk_unravel(ci, g);
        auto p = r.chunk_payload(mi, ci);
        if (p.empty()) std::fill(cube.begin(), cube.end(), u8(0));
        else {
            auto pl = chunk::Payload::deserialize(p);
            if (pl.shared_tables && mt != SIZE_MAX) {
                u64 ri = region_linear(region_of_chunk(c), rg);
                if (ri != cached_region) {
                    auto tb = r.chunk_payload(mt, ri);
                    size_t pos = 0;
                    cached_tables = chunk::SharedTables::deserialize(tb, pos);
                    cached_region = ri;
                }
                chunk::decode_chunk_shared(pl, &cached_tables, cube);
            } else {
                chunk::decode_chunk(pl, cube);
            }
        }
        // apply the validity mask (§5.3): invalid voxels forced to 0.
        if (mk != SIZE_MAX) {
            auto mp = r.chunk_payload(mk, ci);
            if (mp.empty()) std::fill(cube.begin(), cube.end(), u8(0));  // no valid voxels
            else {
                mask::decode(mp, valid);
                for (u32 i = 0; i < CHUNK_VOX; ++i) if (!valid[i]) cube[i] = 0;
            }
        }
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
        "  c4d bench  <dir-of-raw-chunks> [q ...]   benchmark codec on 64^3 chunks\n"
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
        if (argc < 4) { std::fprintf(stderr, "usage: c4d compact <in.c4d> <out.c4d>\n"); return 2; }
        auto in = read_file(argv[2]);
        auto out = archive::compact(in);
        write_file(argv[3], out);
        std::fprintf(stderr, "compacted %s (%zu B) -> %s (%zu B)\n",
                     argv[2], in.size(), argv[3], out.size());
        return 0;
    }
    return usage();
}
