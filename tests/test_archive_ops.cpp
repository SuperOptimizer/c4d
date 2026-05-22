// Archive append (§5.5), disjoint compose (§5.7), compact (§5.6).
#include "c4d/archive.hpp"
#include "check.hpp"
#include <random>
#include <vector>

using namespace c4d;
using namespace c4d::archive;

static std::vector<u8> rnd_payload(std::mt19937& rng, size_t n) {
    std::vector<u8> v(n);
    std::uniform_int_distribution<int> d(0, 255);
    for (auto& b : v) b = static_cast<u8>(d(rng));
    return v;
}

int main() {
    std::mt19937 rng(17);

    // Base archive A: one "intensity" member, 2 chunks (shape 256x128x128 => 2x1x1).
    Coord3 shapeA{256, 128, 128};
    std::vector<std::vector<u8>> ca = { rnd_payload(rng, 500), rnd_payload(rng, 700) };
    std::vector<f32> qa = {16, 16};
    Writer wa;
    wa.add_member("intensity", MemberType::Intensity, shapeA, ca, qa);
    wa.set_metadata(R"({"v":1})");
    auto fileA = wa.finish();

    // --- APPEND a mask member ---
    std::vector<std::vector<u8>> cm = { rnd_payload(rng, 120), rnd_payload(rng, 90) };
    std::vector<NewMember> adds = {{ "mask", MemberType::ValidityMask, shapeA, cm, {0,0} }};
    auto fileB = append(fileA, adds);

    // file grew (append never shrinks), and both members read back intact
    CHECK(fileB.size() > fileA.size());
    {
        Reader r(fileB);
        CHECK(r.member_count() == 2);
        size_t mi = r.find("intensity"), mm = r.find("mask");
        CHECK(mi != SIZE_MAX && mm != SIZE_MAX);
        // original intensity payloads unchanged (same bytes, valid checksums)
        CHECK(r.chunk_ok(mi, 0) && r.chunk_ok(mi, 1));
        auto p0 = r.chunk_payload(mi, 0);
        CHECK(std::vector<u8>(p0.begin(), p0.end()) == ca[0]);
        // appended mask payloads present + valid
        CHECK(r.chunk_ok(mm, 0) && r.chunk_ok(mm, 1));
        auto m1 = r.chunk_payload(mm, 1);
        CHECK(std::vector<u8>(m1.begin(), m1.end()) == cm[1]);
    }
    // the old trailer is dead bytes inside fileB; reading the prefix as a
    // standalone archive still yields just A (newest-trailer-at-EOF model).
    {
        Reader rOld(fileA);
        CHECK(rOld.member_count() == 1);
    }

    // --- COMPOSE two disjoint archives ---
    Coord3 shapeC{128, 128, 128};
    std::vector<std::vector<u8>> cc = { rnd_payload(rng, 333) };
    Writer wc;
    wc.add_member("aux", MemberType::Intensity, shapeC, cc, {8});
    auto fileC = wc.finish();

    auto fused = compose(fileB, fileC);
    {
        Reader r(fused);
        CHECK(r.member_count() == 3);            // intensity + mask + aux
        size_t ma = r.find("aux");
        CHECK(ma != SIZE_MAX);
        CHECK(r.chunk_ok(ma, 0));
        auto pa = r.chunk_payload(ma, 0);
        CHECK(std::vector<u8>(pa.begin(), pa.end()) == cc[0]);   // remapped intact
        // and the carried-over members still verify
        CHECK(r.chunk_ok(r.find("intensity"), 0));
        CHECK(r.chunk_ok(r.find("mask"), 1));
    }

    // --- COMPACT reclaims the dead bytes from the append ---
    {
        auto compacted = compact(fileB);
        CHECK(compacted.size() < fileB.size());   // dead old-footer bytes dropped
        Reader r(compacted);
        CHECK(r.member_count() == 2);
        CHECK(r.chunk_ok(r.find("intensity"), 1));
        CHECK(r.chunk_ok(r.find("mask"), 0));
        std::fprintf(stderr, "  compact: %zu -> %zu bytes\n", fileB.size(), compacted.size());
    }

    RUN_TESTS_RETURN();
}
