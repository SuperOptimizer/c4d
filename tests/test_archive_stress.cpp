// Thorough archive stress/property tests beyond the happy-path in
// test_archive / test_archive_ops: multi-append chaining, the dead-bytes
// invariant (old footer stays readable), exact range-GET byte correctness,
// edge cases (empty metadata, all-ABSENT member, single/many chunks), operation
// chains (append->append->compose->compact), checksum tamper detection on every
// checksummed region, and a randomized fuzz loop with full readback invariants.
#include "c4d/archive.hpp"
#include "check.hpp"
#include <random>
#include <string>
#include <vector>

using namespace c4d;
using namespace c4d::archive;

static std::vector<u8> rnd(std::mt19937& g, size_t n) {
    std::vector<u8> v(n);
    std::uniform_int_distribution<int> d(0, 255);
    for (auto& b : v) b = static_cast<u8>(d(g));
    return v;
}

// Build a one-member archive with `chunks` (empty blob => ABSENT).
static std::vector<u8> mk(const std::string& name, MemberType t, Coord3 shape,
                          std::vector<std::vector<u8>> chunks, std::string meta = "{}") {
    Writer w;
    std::vector<f32> qs(chunks.size(), 16.f);
    w.add_member(name, t, shape, chunks, qs);
    w.set_metadata(std::move(meta));
    return w.finish();
}

// Read every chunk of every member back and compare to expected payloads.
static bool readback_matches(std::span<const u8> file, const std::string& name,
                             const std::vector<std::vector<u8>>& expect) {
    Reader r(file);
    size_t mi = r.find(name);
    if (mi == SIZE_MAX) return false;
    const Member& m = r.member(mi);
    if (m.index.size() != expect.size()) return false;
    for (u64 ci = 0; ci < expect.size(); ++ci) {
        if (!r.chunk_ok(mi, ci)) return false;
        auto p = r.chunk_payload(mi, ci);
        if (std::vector<u8>(p.begin(), p.end()) != expect[ci]) return false;
    }
    return true;
}

int main() {
    std::mt19937 g(20260522);

    // --- 1. exact range-GET: each chunk's bytes are addressable & correct,
    //        and offsets are non-overlapping for the live (non-ABSENT) chunks.
    {
        Coord3 shape{192, 64, 64};            // 3x1x1 chunks
        std::vector<std::vector<u8>> c = { rnd(g, 333), {}, rnd(g, 1024) };  // mid ABSENT
        auto file = mk("intensity", MemberType::Intensity, shape, c);
        Reader r(file);
        size_t mi = r.find("intensity");
        CHECK(r.member(mi).index.size() == 3);
        CHECK(r.chunk_payload(mi, 1).empty());                 // ABSENT yields {}
        CHECK(r.member(mi).index[1].flags & IndexEntry::ABSENT);
        CHECK((std::vector<u8>(r.chunk_payload(mi,0).begin(), r.chunk_payload(mi,0).end()) == c[0]));
        CHECK((std::vector<u8>(r.chunk_payload(mi,2).begin(), r.chunk_payload(mi,2).end()) == c[2]));
        // live payloads occupy disjoint ranges
        auto e0 = r.member(mi).index[0]; auto e2 = r.member(mi).index[2];
        CHECK(e0.offset + e0.length <= e2.offset || e2.offset + e2.length <= e0.offset);
        CHECK(readback_matches(file, "intensity", c));
    }

    // --- 2. multi-append chaining: append twice; the newest trailer must still
    //        expose ALL members (original + both appends), each intact.
    {
        Coord3 sh{64, 64, 64};
        std::vector<std::vector<u8>> c0 = { rnd(g, 200) };
        auto f0 = mk("m0", MemberType::Intensity, sh, c0, R"({"gen":0})");

        std::vector<std::vector<u8>> c1 = { rnd(g, 150), rnd(g, 80) };
        auto f1 = append(f0, {{ "m1", MemberType::ValidityMask, {128,64,64}, c1, {0,0} }});

        std::vector<std::vector<u8>> c2 = { rnd(g, 300) };
        auto f2 = append(f1, {{ "m2", MemberType::Metadata, sh, c2, {0} }}, R"({"gen":2})");

        Reader r(f2);
        CHECK(r.member_count() == 3);
        CHECK(readback_matches(f2, "m0", c0));
        CHECK(readback_matches(f2, "m1", c1));
        CHECK(readback_matches(f2, "m2", c2));
        CHECK(r.metadata() == R"({"gen":2})");              // newest metadata wins
        CHECK(f2.size() > f1.size() && f1.size() > f0.size());  // monotone growth
    }

    // --- 3. dead-bytes invariant: after append, the ORIGINAL file bytes are
    //        unchanged (append never rewrites), so a reader over the old prefix
    //        length still parses the OLD archive (its trailer is intact).
    {
        Coord3 sh{64, 64, 64};
        std::vector<std::vector<u8>> c0 = { rnd(g, 256) };
        auto f0 = mk("only", MemberType::Intensity, sh, c0, R"({"old":1})");
        auto f1 = append(f0, {{ "added", MemberType::Metadata, sh, {{rnd(g,64)}}, {0} }});
        // f1's first f0.size() bytes are byte-identical to f0
        CHECK(f1.size() >= f0.size());
        CHECK(std::equal(f0.begin(), f0.end(), f1.begin()));
        // a reader given exactly the old prefix sees only the old member+meta
        Reader rold(std::span<const u8>(f1.data(), f0.size()));
        CHECK(rold.member_count() == 1);
        CHECK(rold.find("added") == SIZE_MAX);
        CHECK(rold.metadata() == R"({"old":1})");
    }

    // --- 4. edge cases ------------------------------------------------------
    {
        Coord3 sh{64, 64, 64};
        // empty metadata
        auto fe = mk("x", MemberType::Intensity, sh, {{ rnd(g, 10) }}, "");
        CHECK(Reader(fe).metadata().empty());
        // all-ABSENT member (every chunk empty) — no payload bytes, index intact
        std::vector<std::vector<u8>> allz = { {}, {}, {} };
        auto fz = mk("blank", MemberType::ValidityMask, {192,64,64}, allz);
        Reader rz(fz);
        size_t mz = rz.find("blank");
        CHECK(rz.member(mz).index.size() == 3);
        for (u64 ci = 0; ci < 3; ++ci) {
            CHECK(rz.member(mz).index[ci].flags & IndexEntry::ABSENT);
            CHECK(rz.chunk_payload(mz, ci).empty());
            CHECK(rz.chunk_ok(mz, ci));                     // ABSENT is always "ok"
        }
        // single chunk
        std::vector<std::vector<u8>> one = { rnd(g, 4096) };
        auto f1c = mk("solo", MemberType::Intensity, sh, one);
        CHECK(readback_matches(f1c, "solo", one));
        // many chunks (stresses index width / O(1) lookup)
        std::vector<std::vector<u8>> many;
        for (int i = 0; i < 257; ++i) many.push_back(rnd(g, 1 + (i % 13)));
        auto fm = mk("many", MemberType::Intensity, {257*64,64,64}, many);
        CHECK(readback_matches(fm, "many", many));
        CHECK(Reader(fm).member(0).index.size() == 257);
    }

    // --- 5. operation chain: append -> append -> compose with a fresh file ->
    //        compact, verifying integrity at the end of the whole pipeline.
    {
        Coord3 sh{64, 64, 64};
        std::vector<std::vector<u8>> a0 = { rnd(g, 100), {} , rnd(g, 200) };
        auto A = mk("A", MemberType::Intensity, {192,64,64}, a0, R"({"side":"A"})");
        std::vector<std::vector<u8>> a1 = { rnd(g, 90) };
        auto A2 = append(A, {{ "Amask", MemberType::ValidityMask, sh, a1, {0} }});

        std::vector<std::vector<u8>> b0 = { rnd(g, 175), rnd(g, 50) };
        auto B = mk("B", MemberType::Intensity, {128,64,64}, b0, R"({"side":"B"})");

        auto fused = compose(A2, B);                         // A wins metadata
        Reader rf(fused);
        CHECK(rf.member_count() == 3);
        CHECK(readback_matches(fused, "A", a0));
        CHECK(readback_matches(fused, "Amask", a1));
        CHECK(readback_matches(fused, "B", b0));
        CHECK(rf.metadata() == R"({"side":"A"})");

        auto packed = compact(fused);                        // drop dead bytes
        CHECK(packed.size() <= fused.size());
        CHECK(readback_matches(packed, "A", a0));
        CHECK(readback_matches(packed, "Amask", a1));
        CHECK(readback_matches(packed, "B", b0));
        // compact is idempotent on byte size (already minimal)
        CHECK(compact(packed).size() == packed.size());
    }

    // --- 6. checksum tamper detection: flipping a byte in any LIVE payload is
    //        caught by chunk_ok (the directory/index checksums are verified at
    //        parse time and would std::abort, so we only exercise chunk_ok here).
    {
        Coord3 sh{64, 64, 64};
        std::vector<std::vector<u8>> c = { rnd(g, 500), rnd(g, 600) };
        auto file = mk("intensity", MemberType::Intensity, {128,64,64}, c);
        Reader good(file);
        CHECK(good.chunk_ok(0, 0) && good.chunk_ok(0, 1));
        // corrupt one byte inside chunk 1's payload
        auto bad = file;
        u64 off = good.member(0).index[1].offset;
        bad[off] ^= 0xFF;
        Reader r2(bad);
        CHECK(!r2.chunk_ok(0, 1));                           // detected
        CHECK(r2.chunk_ok(0, 0));                            // untouched still ok
    }

    // --- 7. randomized fuzz: random op sequences, full readback invariant -----
    {
        struct Model { std::vector<u8> file; std::vector<std::pair<std::string,std::vector<std::vector<u8>>>> mem; };
        std::uniform_int_distribution<int> op(0, 2), nch(1, 5), sz(0, 800);
        Coord3 sh{64, 64, 64};
        int seq_ok = 0;
        for (int trial = 0; trial < 60; ++trial) {
            // start with a fresh 1-member archive
            std::string nm = "m" + std::to_string(trial) + "_0";
            std::vector<std::vector<u8>> ch;
            int n = nch(g);
            for (int i = 0; i < n; ++i) { int s = sz(g); ch.push_back(s ? rnd(g, s) : std::vector<u8>{}); }
            Model M; M.file = mk(nm, MemberType::Intensity, {u64(n)*64,64,64}, ch);
            M.mem.push_back({nm, ch});

            for (int step = 0; step < 6; ++step) {
                int o = op(g);
                if (o == 0) {  // append a new member
                    std::string an = "m" + std::to_string(trial) + "_" + std::to_string(step+1);
                    std::vector<std::vector<u8>> ach; int an2 = nch(g);
                    for (int i = 0; i < an2; ++i) { int s = sz(g); ach.push_back(s ? rnd(g,s):std::vector<u8>{}); }
                    M.file = append(M.file, {{ an, MemberType::Intensity, {u64(an2)*64,64,64}, ach,
                                               std::vector<f32>(an2,16.f) }});
                    M.mem.push_back({an, ach});
                } else if (o == 1) {  // compose with a fresh disjoint archive
                    std::string cn = "c" + std::to_string(trial) + "_" + std::to_string(step);
                    std::vector<std::vector<u8>> cch; int cn2 = nch(g);
                    for (int i = 0; i < cn2; ++i) { int s = sz(g); cch.push_back(s?rnd(g,s):std::vector<u8>{}); }
                    auto other = mk(cn, MemberType::Intensity, {u64(cn2)*64,64,64}, cch);
                    M.file = compose(M.file, other);
                    M.mem.push_back({cn, cch});
                } else {  // compact
                    M.file = compact(M.file);
                }
                // invariant: every modeled member reads back exactly, after every op
                for (auto& [name, chunks] : M.mem)
                    if (!readback_matches(M.file, name, chunks)) { CHECK(false); goto next_trial; }
            }
            ++seq_ok;
            next_trial:;
        }
        CHECK(seq_ok == 60);
        std::fprintf(stderr, "  archive fuzz: %d/60 random op-sequences held all invariants\n", seq_ok);
    }

    RUN_TESTS_RETURN();
}
