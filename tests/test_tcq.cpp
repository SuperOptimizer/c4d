// TCQ roundtrip sanity (the module is a measured reference, not wired into the
// chunk format — it lost at matched rate, see SPEC §7). Verify encode/decode
// reconstructs within the lattice tolerance so it doesn't bitrot.
#include "c4d/tcq.hpp"
#include "check.hpp"
#include <random>
#include <vector>
using namespace c4d;
int main() {
    std::mt19937 rng(4);
    std::uniform_real_distribution<f32> d(-300.f, 300.f);
    std::vector<f32> c(4096); for (auto& v : c) v = d(rng);
    f32 delta = 8.f;
    auto e = tcq::encode(c, delta);
    CHECK(e.coset.size() == c.size());
    CHECK(e.level.size() == c.size());
    std::vector<f32> rec(c.size());
    tcq::decode(e.coset, e.level, delta, rec);
    // each reconstruction is within one fine-lattice cell of the input
    f32 worst = 0; for (size_t i = 0; i < c.size(); ++i) worst = std::max(worst, std::fabs(c[i]-rec[i]));
    CHECK(worst <= 2.0f * delta);   // within a 2Δ coset cell
    RUN_TESTS_RETURN();
}
