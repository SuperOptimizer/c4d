// Validity-mask codec (§5.3): a per-chunk 128^3 boolean "which voxels are valid"
// grid, coded as a region octree -> SVDAG (common-subtree merge) with rANS over
// the occupancy bytes. Large uniform valid/invalid regions collapse to single
// shared nodes; only the boundary shell carries detail. Lossy is allowed; the
// termination is dilation-biased toward valid (never wrongly zeroes real signal).
//
// This is the mask's OWN small codec — no wavelet, no shared state with the
// intensity pipeline. On decode, invalid voxels force the intensity to 0.
//
// Octree over 128^3: root covers 128^3, each node splits into 8 octants, depth 7
// reaches 1^3 leaves. A node is UNIFORM (all-valid or all-invalid) => it is a
// leaf with a 1-bit value; else it has an 8-bit child-occupancy mask and 8 child
// subtrees. The DAG merges isomorphic subtrees bottom-up.
#pragma once
#include "core.hpp"
#include "rans.hpp"
#include "hybrid_uint.hpp"
#include <algorithm>
#include <array>
#include <span>
#include <unordered_map>
#include <vector>

namespace c4d::mask {

inline constexpr u32 MASK_EDGE = CHUNK;       // 128
inline constexpr u32 MASK_DEPTH = 7;          // log2(128)

// --- octree node (uniform leaf or 8-way internal) --------------------------
// Encoded form: a post-order node list. Leaf nodes carry a value bit; internal
// nodes carry an 8-bit occupancy of which children are "all-valid leaves" vs
// "recurse". We dedup identical subtrees into a DAG by hashing.
struct Node {
    bool leaf = false;
    bool value = false;        // leaf: the uniform validity value
    std::array<u32, 8> child{};  // internal: indices into the node pool (DAG)
    friend bool operator==(const Node& a, const Node& b) {
        if (a.leaf != b.leaf) return false;
        if (a.leaf) return a.value == b.value;
        return a.child == b.child;
    }
};

struct NodeHash {
    size_t operator()(const Node& n) const {
        size_t h = n.leaf ? (n.value ? 0x9e37u : 0x1u) : 0xABCDu;
        if (!n.leaf) for (u32 c : n.child) h = h * 1099511628211ull + c + 0x100000001b3ull;
        return h;
    }
};

// --- DAG builder: bottom-up, hashing identical subtrees to shared ids -------
class DagBuilder {
public:
    std::vector<Node> pool;                       // node id -> Node (DAG)
    std::unordered_map<Node, u32, NodeHash> intern_;

    u32 intern(const Node& n) {
        auto it = intern_.find(n);
        if (it != intern_.end()) return it->second;
        u32 id = static_cast<u32>(pool.size());
        pool.push_back(n);
        intern_.emplace(n, id);
        return id;
    }

    // Build the octree-DAG from a 128^3 validity grid (1 byte/voxel: nonzero =
    // valid). Returns the root node id. Recurses; uniform subcubes collapse to
    // a shared leaf. Lossy `dilate`: if a node is >= valid_frac valid, snap it
    // to all-valid (dilation-biased — grows valid, never zeroes real signal).
    u32 build(std::span<const u8> valid, f32 valid_frac = 1.0f) {
        return rec(valid, 0, 0, 0, MASK_EDGE, valid_frac);
    }

private:
    // is the s^3 subcube at (z0,y0,x0) uniformly v? returns count of valid voxels.
    u32 rec(std::span<const u8> valid, u32 z0, u32 y0, u32 x0, u32 s, f32 vf) {
        // count valid in this subcube
        u64 nvalid = 0;
        for (u32 z = z0; z < z0 + s; ++z)
            for (u32 y = y0; y < y0 + s; ++y)
                for (u32 x = x0; x < x0 + s; ++x)
                    nvalid += (valid[vox_index(z, y, x)] != 0);
        u64 total = u64(s) * s * s;
        if (nvalid == 0)     return intern(Node{.leaf = true, .value = false});
        if (nvalid == total) return intern(Node{.leaf = true, .value = true});
        // lossy dilation: nearly-all-valid snaps to valid (never the reverse).
        if (vf < 1.0f && nvalid >= u64(vf * total))
            return intern(Node{.leaf = true, .value = true});
        if (s == 1)  // single voxel, not uniform shouldn't happen, but guard
            return intern(Node{.leaf = true, .value = (nvalid != 0)});
        // internal: recurse into 8 octants
        Node n; n.leaf = false;
        u32 h = s / 2; u32 k = 0;
        for (u32 dz = 0; dz < 2; ++dz)
            for (u32 dy = 0; dy < 2; ++dy)
                for (u32 dx = 0; dx < 2; ++dx)
                    n.child[k++] = rec(valid, z0 + dz*h, y0 + dy*h, x0 + dx*h, h, vf);
        return intern(n);
    }
};

// --- serialize the DAG: rANS over node descriptors -------------------------
// We emit nodes in id order (children always have smaller ids than parents,
// since they were interned first — bottom-up). Each node: 1 type+value symbol
// for leaves, or an "internal" symbol + 8 child-id references (delta from parent
// id, which is positive and small-ish, via HybridUint+bypass). The skew in
// child references (a few shared nodes referenced often) is what makes the DAG
// net-win on bytes.
//
// Symbol alphabet: 0 = leaf-invalid, 1 = leaf-valid, 2 = internal. Child refs go
// to a separate hybrid token+bypass stream.
struct Encoded {
    std::vector<u8> bytes;
    u32 nnodes = 0;
    u32 root = 0;
};

inline Encoded encode(std::span<const u8> valid, f32 valid_frac = 1.0f) {
    DagBuilder dag;
    u32 root = dag.build(valid, valid_frac);
    const auto& pool = dag.pool;

    // gather symbol counts
    std::vector<u32> counts(3, 0);
    rans::BitWriter refs;
    std::vector<u32> ref_counts(64, 0);    // hybrid tokens for child-id deltas
    std::vector<u32> type_syms, ref_syms;
    for (u32 id = 0; id < pool.size(); ++id) {
        const Node& n = pool[id];
        u32 sym = n.leaf ? (n.value ? 1u : 0u) : 2u;
        type_syms.push_back(sym); ++counts[sym];
        if (!n.leaf)
            for (u32 c : n.child) {
                // child id < id always; encode the gap (id - c) >= 1 via hybrid.
                u32 gap = id - c;
                auto s = hybrid::encode(gap);
                ref_syms.push_back(s.token); ++ref_counts[s.token];
                if (s.nbits) refs.put(s.raw, s.nbits);
            }
    }

    auto type_tbl = rans::FreqTable::build(counts);
    auto ref_tbl  = rans::FreqTable::build(ref_counts);
    rans::Encoder type_enc, ref_enc;
    for (i64 i = (i64)type_syms.size() - 1; i >= 0; --i) type_enc.put(type_tbl, type_syms[i]);
    for (i64 i = (i64)ref_syms.size()  - 1; i >= 0; --i) ref_enc.put(ref_tbl,  ref_syms[i]);
    auto type_bytes = type_enc.finish();
    auto ref_bytes  = ref_enc.finish();
    auto ref_bits   = refs.finish();

    Encoded e; e.nnodes = (u32)pool.size(); e.root = root;
    auto& b = e.bytes;
    auto push32 = [&](u32 v){ for(int i=0;i<4;++i) b.push_back(u8((v>>(8*i))&0xff)); };
    push32(e.nnodes); push32(e.root);
    push32((u32)type_syms.size()); push32((u32)ref_syms.size());
    type_tbl.serialize(b); ref_tbl.serialize(b);
    push32((u32)type_bytes.size()); push32((u32)ref_bytes.size()); push32((u32)ref_bits.size());
    b.insert(b.end(), type_bytes.begin(), type_bytes.end());
    b.insert(b.end(), ref_bytes.begin(),  ref_bytes.end());
    b.insert(b.end(), ref_bits.begin(),   ref_bits.end());
    return e;
}

// --- decode: rebuild the DAG node list, then rasterize the root to 128^3 -----
inline void decode(std::span<const u8> blob, std::span<u8> out_valid) {
    size_t pos = 0;
    auto rd32 = [&]{ u32 v=0; for(int i=0;i<4;++i) v|=u32(blob[pos++])<<(8*i); return v; };
    u32 nnodes = rd32(), root = rd32();
    u32 n_type = rd32(), n_ref = rd32();
    auto type_tbl = rans::FreqTable::deserialize(blob, pos);
    auto ref_tbl  = rans::FreqTable::deserialize(blob, pos);
    u32 tlen = rd32(), rlen = rd32(), blen = rd32();
    std::span<const u8> tb(blob.data()+pos, tlen); pos += tlen;
    std::span<const u8> rb(blob.data()+pos, rlen); pos += rlen;
    std::span<const u8> bb(blob.data()+pos, blen); pos += blen;
    rans::Decoder type_dec(tb), ref_dec(rb);
    rans::BitReader ref_br(bb);
    (void)n_type; (void)n_ref;

    std::vector<Node> pool(nnodes);
    for (u32 id = 0; id < nnodes; ++id) {
        u32 sym = type_dec.get(type_tbl);
        Node& n = pool[id];
        if (sym < 2) { n.leaf = true; n.value = (sym == 1); }
        else {
            n.leaf = false;
            for (u32 c = 0; c < 8; ++c) {
                u32 token = ref_dec.get(ref_tbl);
                u32 nb = hybrid::raw_bits_of(token);
                u32 raw = nb ? ref_br.get(nb) : 0;
                u32 gap = hybrid::decode(token, raw);
                n.child[c] = id - gap;
            }
        }
    }

    // rasterize root into the 128^3 validity grid
    std::fill(out_valid.begin(), out_valid.end(), u8(0));
    struct Frame { u32 id, z0, y0, x0, s; };
    std::vector<Frame> stk; stk.push_back({root, 0, 0, 0, MASK_EDGE});
    while (!stk.empty()) {
        Frame f = stk.back(); stk.pop_back();
        const Node& n = pool[f.id];
        if (n.leaf) {
            if (n.value)
                for (u32 z = f.z0; z < f.z0 + f.s; ++z)
                    for (u32 y = f.y0; y < f.y0 + f.s; ++y)
                        for (u32 x = f.x0; x < f.x0 + f.s; ++x)
                            out_valid[vox_index(z, y, x)] = 1;
        } else {
            u32 h = f.s / 2, k = 0;
            for (u32 dz = 0; dz < 2; ++dz)
                for (u32 dy = 0; dy < 2; ++dy)
                    for (u32 dx = 0; dx < 2; ++dx)
                        stk.push_back({n.child[k++], f.z0+dz*h, f.y0+dy*h, f.x0+dx*h, h});
        }
    }
}

} // namespace c4d::mask
