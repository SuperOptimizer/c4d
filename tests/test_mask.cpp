// Validity-mask codec: lossless roundtrip (valid_frac=1) on uniform, half-split,
// spherical, and noisy masks; DAG collapses uniform regions; dilation-bias only
// grows valid. Spec §5.3.
#include "c4d/mask.hpp"
#include "check.hpp"
#include <cmath>
#include <random>
#include <vector>

using namespace c4d;

static u64 differ(std::span<const u8> a, std::span<const u8> b) {
    u64 d = 0; for (u32 i = 0; i < CHUNK_VOX; ++i) d += ((a[i]!=0) != (b[i]!=0)); return d;
}

int main() {
    std::mt19937 rng(42);

    // all-valid: must roundtrip exactly and be tiny (single shared leaf).
    {
        std::vector<u8> m(CHUNK_VOX, 1), rec(CHUNK_VOX, 9);
        auto e = mask::encode(m);
        mask::decode(e.bytes, rec);
        CHECK(differ(m, rec) == 0);
        CHECK(e.bytes.size() < 256);  // fixed overhead: 2 freq tables dominate a trivial DAG          // DAG collapses to ~nothing
        std::fprintf(stderr, "  all-valid: %zu bytes, %u nodes\n", e.bytes.size(), e.nnodes);
    }

    // all-invalid: exact + tiny.
    {
        std::vector<u8> m(CHUNK_VOX, 0), rec(CHUNK_VOX, 9);
        auto e = mask::encode(m);
        mask::decode(e.bytes, rec);
        CHECK(differ(m, rec) == 0);
        CHECK(e.bytes.size() < 256);  // fixed overhead: 2 freq tables dominate a trivial DAG
    }

    // half-split (valid where z<64): large uniform halves -> small DAG, exact.
    {
        std::vector<u8> m(CHUNK_VOX, 0), rec(CHUNK_VOX, 9);
        for (u32 z = 0; z < CHUNK/2; ++z) for (u32 y = 0; y < CHUNK; ++y) for (u32 x = 0; x < CHUNK; ++x)
            m[vox_index(z,y,x)] = 1;
        auto e = mask::encode(m);
        mask::decode(e.bytes, rec);
        CHECK(differ(m, rec) == 0);
        std::fprintf(stderr, "  half-split: %zu bytes, %u nodes\n", e.bytes.size(), e.nnodes);
    }

    // sphere (typical scroll-like single connected component): exact roundtrip.
    {
        std::vector<u8> m(CHUNK_VOX, 0), rec(CHUNK_VOX, 9);
        f32 c = CHUNK/2.f, cx = c, cy = c, cz = c, r = 0.78f*c;
        for (u32 z = 0; z < CHUNK; ++z) for (u32 y = 0; y < CHUNK; ++y) for (u32 x = 0; x < CHUNK; ++x) {
            f32 dz=z-cz, dy=y-cy, dx=x-cx;
            if (dz*dz+dy*dy+dx*dx <= r*r) m[vox_index(z,y,x)] = 1;
        }
        auto e = mask::encode(m);
        mask::decode(e.bytes, rec);
        CHECK(differ(m, rec) == 0);
        u64 nvalid = 0; for (u8 v : m) nvalid += v;
        std::fprintf(stderr, "  sphere: %zu bytes, %u nodes, %.1f%% valid\n",
                     e.bytes.size(), e.nnodes, 100.0*nvalid/CHUNK_VOX);
        // a smooth shape should compress far better than raw 1-bit (16KB).
        CHECK(e.bytes.size() < CHUNK_VOX / 8 / 4);
    }

    // random noise (worst case): still exact at valid_frac=1.
    {
        std::vector<u8> m(CHUNK_VOX), rec(CHUNK_VOX);
        std::bernoulli_distribution b(0.5);
        for (auto& v : m) v = b(rng);
        auto e = mask::encode(m);
        mask::decode(e.bytes, rec);
        CHECK(differ(m, rec) == 0);
    }

    // dilation bias: lossy termination only ADDS valid voxels, never removes.
    {
        std::vector<u8> m(CHUNK_VOX, 0), rec(CHUNK_VOX);
        f32 c = CHUNK/2.f, cx = c, cy = c, cz = c, r = 0.78f*c;
        for (u32 z = 0; z < CHUNK; ++z) for (u32 y = 0; y < CHUNK; ++y) for (u32 x = 0; x < CHUNK; ++x) {
            f32 dz=z-cz,dy=y-cy,dx=x-cx; if (dz*dz+dy*dy+dx*dx <= r*r) m[vox_index(z,y,x)]=1;
        }
        auto e = mask::encode(m, 0.6f);   // lossy: snap >=60%-valid nodes to valid
        mask::decode(e.bytes, rec);
        // every originally-valid voxel must remain valid (no real signal zeroed)
        bool no_loss = true;
        for (u32 i = 0; i < CHUNK_VOX; ++i) if (m[i] && !rec[i]) no_loss = false;
        CHECK(no_loss);
        std::fprintf(stderr, "  lossy sphere (vf=0.6): %zu bytes (vs %zu lossless)\n",
                     e.bytes.size(), mask::encode(m).bytes.size());
    }

    RUN_TESTS_RETURN();
}
