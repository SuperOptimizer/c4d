// rANS: symbol roundtrip on skewed distributions, frequency-table (de)serialize,
// near-entropy coded size, and bypass bitstream roundtrip. Spec §4.7, §4.9.
#include "c4d/rans.hpp"
#include "check.hpp"
#include <cmath>
#include <random>
#include <vector>

using namespace c4d;

int main() {
    std::mt19937 rng(99);

    // --- symbol roundtrip on a skewed alphabet (zero-heavy, like HF tokens) ---
    {
        const u32 N = 200000, A = 16;
        std::vector<u32> syms(N);
        std::discrete_distribution<u32> d({800,120,40,20,8,4,2,1,1,1,1,1,1,1,1,1});
        for (auto& s : syms) s = d(rng);

        std::vector<u32> counts(A, 0);
        for (u32 s : syms) ++counts[s];
        auto tbl = rans::FreqTable::build(counts);

        rans::Encoder enc;
        for (i64 i = static_cast<i64>(N) - 1; i >= 0; --i) enc.put(tbl, syms[i]);  // reverse
        auto bytes = enc.finish();

        rans::Decoder dec(bytes);
        bool ok = true;
        for (u32 i = 0; i < N; ++i) if (dec.get(tbl) != syms[i]) { ok = false; break; }
        CHECK(ok);

        // coded size vs zeroth-order entropy: rANS within ~1% of H0.
        f64 H = 0;
        for (u32 c : counts) if (c) { f64 p = f64(c) / N; H -= p * std::log2(p); }
        f64 ideal_bytes = H * N / 8.0;
        f64 ratio = bytes.size() / ideal_bytes;
        std::fprintf(stderr, "  rANS bytes=%zu  ideal=%.0f  ratio=%.4f (H0=%.3f b/sym)\n",
                     bytes.size(), ideal_bytes, ratio, H);
        CHECK(ratio < 1.02);   // spec: rANS/H0 ~ 0.99
    }

    // --- frequency-table serialize/deserialize roundtrip ---
    {
        std::vector<u32> counts = {1000, 1, 0, 50, 7, 0, 3, 9};
        auto t = rans::FreqTable::build(counts);
        std::vector<u8> ser; t.serialize(ser);
        size_t pos = 0;
        auto t2 = rans::FreqTable::deserialize(ser, pos);
        CHECK(pos == ser.size());
        CHECK(t2.nsym == t.nsym);
        bool same = true;
        for (u32 s = 0; s < t.nsym; ++s) if (t2.freq[s] != t.freq[s]) same = false;
        CHECK(same);
        u32 sum = 0; for (u16 f : t2.freq) sum += f;
        CHECK(sum == rans::PROB_SCALE);
    }

    // --- bypass bitstream roundtrip (variable widths) ---
    {
        std::vector<std::pair<u32,u32>> vals;  // (value, nbits)
        std::uniform_int_distribution<u32> nb(1, 24);
        for (int i = 0; i < 50000; ++i) {
            u32 w = nb(rng);
            u32 v = std::uniform_int_distribution<u32>(0, (w >= 32 ? 0xffffffffu : (1u << w) - 1))(rng);
            vals.push_back({v, w});
        }
        rans::BitWriter bw;
        for (auto [v, w] : vals) bw.put(v, w);
        auto bytes = bw.finish();
        rans::BitReader br(bytes);
        bool ok = true;
        for (auto [v, w] : vals) if (br.get(w) != v) { ok = false; break; }
        CHECK(ok);
    }

    RUN_TESTS_RETURN();
}
