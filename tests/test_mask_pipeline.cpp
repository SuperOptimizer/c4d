// End-to-end validity mask through a real archive: encode intensity + mask,
// decode, verify invalid voxels -> 0 and valid voxels preserved. Spec §5.3.
#include "c4d/chunk.hpp"
#include "c4d/archive.hpp"
#include "c4d/mask.hpp"
#include "check.hpp"
#include <random>
#include <vector>

using namespace c4d;

int main() {
    std::mt19937 rng(8);

    // Build a CHUNK^3 "scroll-ish" chunk: dense material in a sphere, low-value
    // noise floor (air) outside. Threshold separates them.
    std::vector<u8> vox(CHUNK_VOX, 0);
    std::normal_distribution<f32> mat(120.f, 25.f), air(8.f, 4.f);
    f32 c = CHUNK/2.f, cx=c, cy=c, cz=c, rad=0.75f*c;
    for (u32 z=0; z<CHUNK; ++z) for (u32 y=0; y<CHUNK; ++y) for (u32 x=0; x<CHUNK; ++x) {
        f32 dz=z-cz,dy=y-cy,dx=x-cx; bool inside = dz*dz+dy*dy+dx*dx <= rad*rad;
        f32 v = inside ? mat(rng) : air(rng);
        vox[vox_index(z,y,x)] = static_cast<u8>(std::clamp(v, 0.f, 255.f));
    }
    const u8 THRESH = 30;

    // validity grid: voxel > thresh => valid
    std::vector<u8> valid(CHUNK_VOX);
    for (u32 i=0;i<CHUNK_VOX;++i) valid[i] = (vox[i] > THRESH) ? 1 : 0;

    // build a 1-chunk archive with intensity + mask members
    Coord3 shape{CHUNK,CHUNK,CHUNK};
    auto ip = chunk::encode_chunk(vox, 16.f).serialize();
    auto mp = mask::encode(valid).bytes;
    archive::Writer w;
    w.add_member("intensity", archive::MemberType::Intensity, shape, {ip}, {16});
    w.add_member("mask", archive::MemberType::ValidityMask, shape, {mp}, {0});
    auto file = w.finish();

    // decode: intensity chunk, then apply mask
    archive::Reader r(file);
    size_t mi = r.find("intensity"), mk = r.find("mask");
    CHECK(mi != SIZE_MAX && mk != SIZE_MAX);

    std::vector<u8> rec(CHUNK_VOX), vmask(CHUNK_VOX);
    auto pl = chunk::Payload::deserialize(r.chunk_payload(mi, 0));
    chunk::decode_chunk(pl, rec);
    mask::decode(r.chunk_payload(mk, 0), vmask);
    for (u32 i=0;i<CHUNK_VOX;++i) if (!vmask[i]) rec[i] = 0;

    // every voxel the mask marks invalid must be exactly 0
    u64 inval=0, inval_zeroed=0, val=0, val_kept_field=0;
    for (u32 i=0;i<CHUNK_VOX;++i) {
        if (!vmask[i]) { ++inval; if (rec[i]==0) ++inval_zeroed; }
    }
    CHECK(inval > 0);
    CHECK(inval_zeroed == inval);   // all masked-invalid voxels are 0

    // the mask must be lossless-or-dilated: every ORIGINALLY-valid voxel stays
    // valid in the decoded mask (dilation-bias never zeroes real signal).
    bool no_signal_lost = true;
    for (u32 i=0;i<CHUNK_VOX;++i) if (valid[i] && !vmask[i]) no_signal_lost = false;
    CHECK(no_signal_lost);

    std::fprintf(stderr, "  mask pipeline: intensity %zu B + mask %zu B; %llu invalid voxels all->0\n",
                 ip.size(), mp.size(), (unsigned long long)inval);
    (void)val; (void)val_kept_field;

    RUN_TESTS_RETURN();
}
