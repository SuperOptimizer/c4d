# compress4d (c4d) — Specification

**A CPU-only, SIMD, lossy compression codec and single-file archive format for
3D grayscale (u8) X-ray volumetric data.** Modern C++26, greenfield.

Status: design complete and research-grounded (mined from JPEG 2000/EBCOT,
JPEG XL, VVC/H.266, SPERR/ZFP/SZ, G-PCC/sparse-voxel-DAG, and modern rANS/SIMD
work). The architecture is **SPERR with rANS replacing SPECK** — a known-good
3D-volume codec with its serial bottleneck swapped for the fast path. Bitstream
internals (§4.3) are implementer-defined within the stated constraints; §4.7–4.10
give the recommended, research-backed recipe. Purpose-built for, and validated on,
the Herculaneum scroll corpus, but architecturally general for any 3D u8 X-ray
volume.

c4d is a ground-up rewrite (the spiritual successor to `c3d`, sharing none of its
code or format). It is **not** compatible with anything — no backward/forward
compatibility, no versioning machinery beyond a magic number. There is intended
to be **one** version of this format: design it, publish it, freeze it. Any
future change is itself a fresh ground-up rewrite.

> Note on the name: the data is 3D volumetric; "compress4d / c4d" is the chosen
> project name regardless.

---

## 0. Implementation stance (C++26)

- **Modern, blazing-fast C++26.** Not C. Lean on the standard library and modern
  language features where they don't cost performance.
- **Heavily `constexpr` / `consteval`.** Everything that can be computed at
  compile time is: wavelet lifting coefficients, quantization/dead-zone tables,
  subband geometry, scan orders, rANS table scaffolding, the chunk-grid
  arithmetic. Runtime does only the data-dependent work.
- **SIMD, CPU-only.** No GPU. Portable SIMD (std::simd where available, or
  intrinsics behind a thin abstraction). `-ffast-math` is acceptable
  (cross-platform byte-determinism is **not** required).
- **Single-threaded core, thread-safe (reentrant).** No internal threading;
  callers parallelize encode/decode across chunks via their own pools/OpenMP.
- **Happy-path optimized; fail fast.** Assert/abort on malformed input or
  internal invariant violations; no graceful-degradation or recovery machinery.

---

## 1. Scope

c4d encodes and decodes **one u8 volume + an optional validity mask** into a
single `.c4d` archive.

**In scope:** the lossy codec (transform + quantization + entropy coding of one
volume), the optional validity mask, and the archive container (layout, index,
append, compose).

**Out of scope (the producer's job, upstream of c4d):** pyramids / LODs,
normalization, contrast enhancement, air detection, downsampling, denoising,
reconstruction. c4d takes whatever u8 bytes it is given and stores them well; it
is agnostic to what those bytes *mean*.

### 1.1 Hard constraints

- **u8, single channel, grayscale, forever.** No 16-bit, no multichannel.
- **Lossy only, permanently.** No true-lossless mode. (The u8 input is already a
  lossy windowing of the source float32 reconstruction; "near-lossless" is served
  by high-quality / low-quantization settings.)
- **CPU-only, SIMD.** No GPU. Learned parameters are permitted only as fast
  artifacts (compile-time tables, baked constants, small per-class profiles) —
  never per-voxel neural inference.
- **9/7 float32 wavelet transform**, fixed. (Not CDF 5/3 — reversible/lossless.
  Not longer biorthogonal filters — not worth the speed cost.)
- **rANS entropy coding.** (Not SPIHT/EBCOT — full rewrite for ~5–7% gain that
  also costs speed; the headroom is reachable inside rANS via context modeling.)
- **Axis order Z, Y, X** everywhere, no exceptions. **Little-endian.**

### 1.2 The four evaluation axes

Every design choice is judged on a **balance** across: (1) compression quality,
(2) compression ratio, (3) encode speed, (4) decode speed. No single axis is
maximized at the others' expense.

---

## 2. Geometry

- **Chunk:** always **128³** voxels. The **only** structural unit — both the
  codec atom (one encode/decode call, decoded whole, never partially) **and** the
  random-access unit (range-GET any chunk by its index entry). 
- **Archive:** one `.c4d` file containing one or more members (§5), footer-indexed.
  This is the I/O unit.
- Edge chunks are **padded to full 128³** with **zero** fill. The codec and
  format always deal in full-size chunks.
- The **logical** volume extent is recorded as a **bounding box** in metadata
  (§5.4); padding may exist on any of the ±Z, ±Y, ±X faces. Cropping to the
  logical volume is a consumer operation using the bbox.

> **No shards.** Earlier drafts had a 2048³ "shard" as an S3-object grouping
> unit. The single-file archive + footer index (§5/§6) **is** the I/O unit, so
> the shard's only purpose — avoiding `LIST` storms over a million tiny
> chunk-files — no longer exists. There is exactly one structural unit (the
> chunk) plus the archive. Prefetch locality, if a consumer wants a big
> sequential read over a region, is a **write-time payload ordering hint**
> (concatenate chunks in Morton/Z-order so adjacent chunks are contiguous and
> the consumer coalesces their ranges) — not a format structure.

### 2.1 Why 128³ (settled)

Measured independent-unit compression cost vs. one-256³-unit, at matched quality:
128³ = +6%, 64³ = +16%, 32³ = +39%, 16³ = +86%. 256³→128³ captures ~90% of the
random-access benefit (~8× less decode-whole-to-use-part waste; ~10ms→~1ms per
chunk) at only +6% bytes. Two-tier transforms, adaptive-octree chunking, and
partial/sub-chunk decode were explored and **rejected** — a few more % for
permanent frozen-in complexity. Fine spatial access (e.g. a parametric-surface
sheet through a chunk) is served by decoding the whole 128³ (~2 MB, ~1 ms) and
slicing in memory. 16³ is a **post-decode in-memory cache** granularity, not a
codec unit.

---

## 3. Chunks are independent

Each 128³ chunk is **fully self-contained**: it decodes alone, no dependency on
neighbors. Required by the random-access model (range-GET any chunk); worth its
small ratio cost. No internal LODs, multi-resolution, progressive/embedded
coding, or resolution-ordered subbands. c4d encodes **one volume at one
resolution.** Pyramids are built **outside** c4d as separate archives/members
(OME-Zarr style); the LOD relationship is a consumer-side naming/metadata
convention. (Accepted cost: ~1.14× pyramid storage tax vs. progressive — parity
with OME-Zarr, bought for simplicity and S3-friendliness.)

---

## 4. Codec pipeline (per chunk)

```
u8 128³  ─▶  subtract DC/mean  ─▶  9/7 float32 DWT (mirror-extended, separable Z,Y,X)
         ─▶  (encoder-side: noise-aware dead-zone shrinkage)
         ─▶  dead-zone quantize (per-subband step, L2-weighted)
         ─▶  zero-run/EOB + rANS entropy code  ─▶  chunk payload bytes
```

Decode is the inverse; masked/padded/absent regions reconstruct to **0**.

**Lineage / validation.** c4d's architecture is essentially **SPERR**
(Lindstrom/NCAR: CDF 9/7 + coefficient coding + outlier pass for 3D scientific
volumes — the closest prior art) **with rANS replacing SPERR's SPECK.** SPECK is
SPERR's documented bottleneck — a serial, bit-oriented, branchy, non-vectorizable
tree walk that makes it several× slower than ZFP/SZ. Replacing it with interleaved
SIMD rANS over a fixed coefficient model is exactly c4d's speed advantage. This is
a known-good design with its slow part swapped out, not a clean-room gamble.

### 4.1 Fixed properties

- **Transform:** CDF 9/7 lifting, float32, separable Z then Y then X, ~5 levels
  (128 = 2⁷ permits up to 7; final count pinned by where compression saturates).
  **Unit-L2-normalized** (load-bearing — see §4.5).
- **Quantization:** scalar with a per-subband dead zone. The **decoder reads
  per-subband quantizer steps from the chunk** — it never recomputes them — so
  the encoder's choice of steps (and the objective they were tuned for) is **not**
  baked into the format. Step *values* can be retuned forever; only the
  *mechanism* (steps carried in the stream) is fixed.
- **Constant-quality, variable-rate by default.** Driven by a quality/quantizer
  knob (q), not a target byte-ratio; bytes vary with content; rate is an outcome,
  best-effort, not a target. (For a *hard* per-voxel quality guarantee, see the
  outlier pass, §4.6.)
- **Chunk-face boundary handling: SYMMETRIC (mirror) extension, NOT zero-pad.**
  See §4.4 — this is a correctness/quality requirement, not optional.

### 4.2 Per-chunk q field

The format carries a per-chunk q. The default encoder writes the **same global q
to every chunk**. Per-chunk variation is a *capability* the format permits;
future quality-equalization is then an encoder-policy change, never a format
change.

### 4.3 Bitstream internals — implementer-defined (within constraints)

The exact byte layout of an encoded chunk is left to the implementer, subject to
the above. The research-backed recipe (§4.7–4.9) is **recommended**, not all
mandated; what is *fixed* once published is whatever the shipped decoder requires.
The per-chunk header is **tiny** (per-chunk q, subband byte offsets, entropy
payload); everything constant is pushed to archive-level metadata (§5.4).

### 4.4 Chunk-face boundary handling (REQUIRED: mirror extension)

Independent 128³ chunks are a *tiling* of the volume, and tiling a wavelet codec
with **zero-padding at chunk faces** is the textbook recipe for JPEG2000-style
tiling artifacts: the zero step at each face injects a large artificial edge →
spurious high-frequency coefficients → wasted bits + ringing. **The 9/7 lifting
must use whole/half-sample symmetric (mirror) extension of the chunk's own data
at chunk faces** (exactly what JPEG2000 does per tile; CDF 9/7 is even
linear-phase and wants symmetric extension for perfect reconstruction on a finite
block). This is self-contained per chunk (no neighbor reads), preserves chunk
independence, and costs nothing at decode.

> Distinct concept: the **zero fill** of §2 applies to *absent/padded volume
> regions* (edge chunks beyond the logical bbox), and the all-zero reconstruction
> of masked/invalid voxels. That is a separate matter from the *transform's*
> internal boundary handling, which is mirror extension. Do not conflate them.

*Encoder-side overlap (HQ mode) — investigated, NOT shipped.* Inter-chunk seams
are real and measured (error-jump across a chunk face is ~1.5–2.2× the interior,
growing with q). But the obvious fixes don't work for an **independent-chunk
mirror-DWT** codec: (a) the forward/inverse boundary must stay *matched* for
perfect reconstruction, so the encoder cannot feed neighbor data into only the
forward boundary taps (it would not invert under the decoder's mirror inverse —
verified); (b) preconditioning the input by blending boundary voxels toward the
neighbor *worsens* the seam (it moves the stored value off the truth — measured,
q8 ratio 1.5→5.4). A genuine overlap/cropping halo is a **lapped transform**,
which requires a decoder-side inverse lap and so breaks the independent-chunk
random-access model (a hard requirement, §3). Conclusion: the ~1.5–2.2× seam
factor is the inherent, accepted cost of independent tiling (exactly JPEG2000's
per-tile behavior); mirror extension remains the right choice. Revisit only if
the random-access requirement is ever relaxed.

### 4.5 Unit-L2 normalization (REQUIRED)

The 9/7 transform must be normalized so the synthesis basis has unit L2 norm.
This makes **coefficient-domain L2 error ≈ reconstructed-domain L2 error**, which
(a) lets the encoder predict reconstruction quality from the quantizer step, and
(b) is what makes the outlier-pass error bound (§4.6) valid. Per-subband quant
steps are then weighted by each subband's synthesis L2 gain (a `constexpr` table)
so a single global q produces near-MSE-optimal allocation across subbands.

### 4.6 Optional outlier pass — hard point-wise-error / constant-quality

For a **guaranteed** per-voxel error bound (true constant-quality), c4d supports a
SPERR-style outlier-correction pass, which is **provably separable from the
coefficient coder** (it places zero requirement on how coefficients are coded):

1. Quantize coefficients with dead-zone step `q ≈ 1.5·t` (the cost-minimizing
   ratio; `t` = the point-wise error tolerance; data-dependent, sweep 1.4–1.8·t).
2. rANS-encode the coefficients (normal path).
3. **Internally** rANS-decode + inverse-9/7 (the encoder has the decoder anyway).
4. Diff against the original; collect `{position, original − reconstructed}` for
   every voxel where `|diff| > t`.
5. Code that sparse set as a **second stream** with c4d's own rANS sparse coder:
   delta-coded sorted positions + magnitudes quantized to multiples of `t` (almost
   always tiny integers) + signs. Target ≈ 10 bits/outlier.
6. On decode: reconstruct, then apply the sparse corrections → `|z − x| ≤ t`
   guaranteed (the math requires only that corrections are coded to ±t).

Cost: one extra internal decode at *encode* time; **decode is unaffected** beyond
applying a sparse correction list. Outlier fraction is typically ~0.6–4.5% of
voxels. (On u8 with `t = 1–2` the tuning differs from SPERR's float experiments —
re-measure `q`-vs-`t` and outlier fraction on scroll data.) This is the piece ZFP
lacks: ZFP fixed-accuracy is a *soft* L∞ bound; the outlier pass makes it *hard*.

### 4.7 Coefficient coding — model the zeros (recommended)

The dominant ratio lever, because rANS is already at the zeroth-order entropy
floor on magnitudes (measured: rANS/H0 ≈ 0.99) — **the headroom is in the zeros,
not the magnitudes.** Dead-zone-quantized HF subbands are 70–95% zeros; coding
each zero as its own rANS symbol wastes both bits and decode cycles.

- **EOB / all-zero-block + zero-run coding:** per sub-block, code an all-zero flag
  and an end-of-block position so the decoder skips the dead HF tail with *zero*
  rANS pulls; replace zero runs with run tokens. **10–30% on HF subbands, and
  faster decode** (fewer symbols). The single best ratio-per-cycle technique.
- **Subband-contiguous raster scan order.** The frequency-diagonal (radial,
  JPEG-zigzag-analog) within-band order was specced here to ~double run length,
  but **measured-and-rejected** (2026-05): on the dense scroll corpus it *shortens*
  runs (avg 11.2→10.1, corpus ratio 7.5×→7.3×) because the dense interior already
  clusters zeros spatially and the radial re-sort scatters that locality. Plain
  subband-contiguous raster order wins here. (Revisit if a future corpus differs.)
- **HybridUint token + raw-bits + sign split:** code each coefficient as a small
  magnitude-bucket *token* (rANS) + raw mantissa bits (bypass) + sign. Keeps the
  rANS alphabet small and histograms sharp.

### 4.8 Context modeling (recommended; measure win-vs-speed before committing)

Past the zeroth-order floor, ratio comes from conditioning the rANS model on a
small context. Use a **context map**: compute a rich context id per symbol, then
cluster to a small set (8–16) of **static** histograms via a signalled map — the
multi-context benefit without per-context table cost. Per-symbol distribution
switching is **cheap in rANS** (point at a different frequency table) but
expensive in tANS/FSE (table rebuild) — a key reason to keep rANS.

- **Context = subband id × coarse bucket of `max(|neighbours|, |parent|)`** —
  neighbour-significance (EBCOT) plus cross-subband parent (zerotree), as rANS
  contexts, *not* a SPIHT tree.
- Organize the symbol stream **by subband** so contexts change in block-aligned
  runs (cheap) rather than per-symbol (which forces a SIMD gather).
- Measured ceiling for the neighbour-significance axis alone ≈ 5–7%; full
  magnitude+parent+subband context is reportedly more (≈10–30% in wavelet coders).
  Each context axis costs ≈ 25% decode throughput — hence "measure before
  committing," and fund it with engine throughput (§4.9). If shipped, it is part
  of the frozen bitstream.

### 4.9 Entropy engine + transform implementation (speed)

- **Scalar 16-bit-renorm rANS** (32-bit state, 12-bit histograms). Interleaved
  N-way SIMD rANS was specced here for ~3× decode, but **measured-and-skipped**
  (2026-05): (1) the context model (§4.8) makes each token's context depend on
  the *previous* token's class, which serializes the decode loop — N independent
  lanes can't run because lane i+1's table depends on lane i's decoded symbol;
  interleaving would require dropping the prev-token context axis and ~3% of the
  §4.8 ratio gain. (2) On the real pipeline the rANS token decode is only ~18% of
  total decode time — the DWT (now SIMD, §4.9 lifting below) was the actual
  bottleneck. So the ~10% ratio win of the serial context model beats the bounded
  ~18%-of-decode speedup of breaking it. (A block-interleave variant — reset
  context at subband boundaries so lanes are independent within a block — remains
  a future option if the speed/ratio balance shifts.) Alias tables for O(1)
  symbol decode remain available if the context count grows.
- **SIMD multi-line lifting (shipped).** The Y/Z passes (the cache-hostile
  strided ones) lift `native-width` parallel lines at once — one SIMD lane per
  innermost-x offset, so consecutive x are contiguous aligned loads. Measured
  5.4× fwd / 3.9× inv DWT. The X pass stays scalar (already contiguous). The
  fully-fused single-loop "cube core" (one streaming pass over all three axes)
  remains a future option. Prime-stride layout (pad to 129/131 to dodge pow-2
  cache-set aliasing) is also deferred — the multi-line gather already mitigates
  most of it.
- **`constexpr`/`consteval`-bake** the lifting constants, per-subband L2 weights,
  scan-order permutation, default alias tables, and context→cluster map.

### 4.10 Encoder objective & noise-aware quantization (encoder-only, not frozen)

These tune the *step values* the decoder reads, so they are encoder policy and
freely improvable — they do not touch the frozen format. v1 (c3d) tuned its
params for MAE/MSE, which over-smooths and discards HF edge detail. c4d targets a
**balanced metric basket** (§7.1) and uses:

- **Noise-aware dead-zone (denoise-in-transform).** Dead-zone quantization and
  wavelet shrinkage are nearly the same operation. Estimate per-chunk noise
  `σ = median(|finest-HH coeffs|) / 0.6745` (robust MAD; the finest diagonal
  subband is ~pure noise in CT). Set the dead-zone width per subband from a
  BayesShrink-style threshold so noise-dominated subbands are suppressed and
  edge-bearing subbands are preserved. This improves **ratio and true-signal
  quality together** on noisy scans (the corpus-confirmed hard case), and it is
  *compression efficiency*, not denoising of the stored signal (c4d still stores
  what it is given; it just declines to spend bits coding sub-noise-floor
  coefficients). Bivariate (parent-child / Sendur-Selesnick) shrinkage was a
  candidate refinement here — **measured-and-skipped**: marginal shrinkage,
  perceptual RDO, *and* a fine-grid sweep all show the same thing — shrinkage
  variants slide ALONG the RD frontier on this corpus, they don't bend it (the
  unit-L2 weighting is already near-optimal, §4.5). Denoise's (ratio, geomean)
  points interpolate exactly onto the baseline curve. Bivariate is a refinement
  of the same shrinkage mechanism, so it would land on the same frontier — no
  gain to chase. (It IS off-by-default-useful for higher quality-at-fixed-q, as
  the shipped marginal noise-aware mode already provides.)
- **Perceptual / energy-preserving RDO.** Bias the per-subband step / per-chunk q
  from a cheap per-block weight: loosen quant in high-variance regions (masking),
  and add a per-subband **HF-energy-preservation** penalty that discourages
  zeroing detail coefficients that carry texture — directly countering "the
  wavelet throws away HF." **Gate the energy-preservation by the MAD noise floor**
  so it preserves signal HF, not noise (mask by gradient *coherence*, not raw
  variance, since noise is high-variance-but-incoherent). No per-candidate metric
  evaluation in the loop — just a weight from quantities already computed for rate
  control.

---

## 5. Archive format (`.c4d`)

A single file. Required properties (byte layout is implementer's discretion as
long as these hold):

- **S3 / remote-friendly:** open with one tail read; fetch any chunk by HTTP
  range-GET. No `LIST`, no millions of small objects.
- **Append-friendly:** add data without rewriting existing bytes.
- **Spinning-disk-friendly:** append-sequential writes; contiguous payload
  grouping; the index is one contiguous block (few seeks).
- **NVMe-friendly:** fixed-width index ⇒ O(1) random entry lookup ⇒ massively
  parallel random reads with no parsing bottleneck.
- **Network and local identical:** one access model (mmap+seek locally, range-GET
  remotely; one code path).
- **User-friendly:** one file to copy / `rsync` / `aws s3 cp` / share; one S3
  object; human-readable JSON metadata; identifiable by magic.

### 5.1 Layout

```
┌──────────────────────────────────────────────┐
│ chunk payloads (concatenated)                 │  ← intensity chunks, mask, …
├──────────────────────────────────────────────┤
│ per-member chunk indexes                      │
├──────────────────────────────────────────────┤
│ member directory                              │
├──────────────────────────────────────────────┤
│ metadata (JSON)                               │
├──────────────────────────────────────────────┤
│ fixed-size trailer (at EOF)                   │
└──────────────────────────────────────────────┘
```

**Read path:** read the last ~64 bytes → trailer → member directory → member →
its chunk index → range-GET the chunk(s).

### 5.2 Trailer, directory, index

- **Trailer (fixed, at EOF):** magic `C4D\0` (TBD), version (u32), directory
  offset+length (u64), directory checksum (xxhash u64), trailing magic. Found by a
  fixed tail read. Magic/version exist only to identify and reject — not for
  compatibility.
- **Member directory:** members, each: id/name, type
  (intensity-volume | validity-mask | metadata), codec tag, shape (3× u64, Z,Y,X),
  chunk size (u32, per-member), index offset+length (u64), index checksum.
  Optional members are simply absent (no mask ⇒ decode skips masking).
- **Chunk index (per member):** **fixed-width** entries in chunk-grid order,
  coordinate **implicit by position** (`entry i ↔ unravel(i)` over the padded
  grid), entry found by direct offset arithmetic. Each entry: payload offset
  (u64), payload length (u64), flags (u8, incl. **all-zero/absent sentinel** ⇒ no
  payload, decode yields zero), per-chunk q, per-chunk **xxhash** checksum. Small
  vs. payload; tail-fetch and hold resident.

### 5.3 Validity mask (optional member)

Carries "which voxels are valid." **Optional** (omitted when nothing is invalid),
**lossy** is fine. Encoded as a **region octree** (large uniform valid/invalid
regions collapse to leaves; detail only along the boundary shell — ideal for very
large to very small connected components). Lossy termination is **dilation-biased
toward valid** (never wrongly zero real signal). On decode, invalid voxels are
forced to **0**; exact 0-reproduction is not required. **Per-archive** (each LOD
archive carries its own). Its **own small codec** — no wavelet, no shared state
with the intensity pipeline. Octree parameters (max depth / leaf resolution)
deferred (§7).

Recommended coding recipe (from point-cloud / G-PCC / sparse-voxel-DAG research):

1. **Octree → DAG, bottom-up common-subtree merging** (the SVDAG idea): hash
   `(child-occupancy-mask, child pointers)` per level and merge isomorphic
   subtrees. Every all-valid / all-invalid region collapses to a single shared
   node, and repeated boundary motifs dedup — 1–3 orders of magnitude node
   reduction. This is the "large uniform → ~nothing" mechanism.
2. **rANS the DAG topology + skewed child-pointer references** (a few nodes are
   referenced often — entropy-code the child-id *gaps* rather than store raw
   pointers; this is what makes step 1 net-win on *bytes*, not just node count).
   **SHIPPED.**

> *Steps 2-3 (neighbor-occupancy context + planar-mode) — investigated, NOT
> shipped (measured 2026-05).* They target the per-node occupancy-byte cost, but
> the measurement says there's no headroom: (a) the shipped child-id-gap coding
> is already within **0.7%** of its entropy on a clean sphere mask — no slack to
> recover; (b) the two regimes that matter leave nothing to gain — a *clean*
> planar surface already collapses to ~nothing via the DAG (a tilted plane = 13
> nodes / 211 B), while a *noisy* fractal boundary (the 76%-valid 3.24µm chunk =
> 29551 nodes / 417 KB) is dominated by node *count*, which per-node context
> can't fix. The real lever for expensive masks is **lossy dilation** (`valid_frac`,
> already shipped): vf=0.85 cut that 417 KB to 193 KB, vf=0.70 to 178 B. So:
> ship the DAG + gap-rANS, expose dilation, skip the occupancy-byte refinements.

(Skip symmetry-aware merging — high encode cost, pointless for a lossy mask. No
neural occupancy models — CPU/constraint.)

### 5.4 Metadata (JSON member)

Free-form JSON (à la zarr `.zattrs`): logical bounding box (crop within the padded
grid) and consumer-side attributes (voxel spacing, energy, provenance, pyramid
convention, …). c4d is a pure pixel-grid codec and does not interpret physical
units.

### 5.5 Append

**Pure append.** New payloads, then a fresh per-member index, directory, metadata,
trailer — all at EOF. Existing bytes never rewritten; the old footer becomes dead
bytes; readers always read the newest trailer from EOF. (zip/parquet model.)

### 5.6 Compaction

Reclaimed by an **offline tool** (`c4d compact`, copies live members to a fresh
file). Append never compacts inline.

### 5.7 Compose / concat

**Disjoint only.** Stitch non-overlapping pieces (parallel-encoded z-ranges, or
adding a mask member to an intensity-only archive): append B's payloads after A's,
rewrite B's index offsets, merge directories, new trailer. Same chunk in both ⇒
**error**.

---

## 6. Why one file

Replaces zarr-v3-sharded storage (a `.zarr` is a directory tree of millions of
tiny files — painful `cp`/`rsync`/`scp`, S3 `LIST` storms + per-object overhead,
must tar to share, one lost file breaks a chunk). The single `.c4d` archive is
**one** file / **one** S3 object, self-contained, yet still **randomly accessible
without unpacking** (footer index + range-GET) — tar's one-file friendliness *and*
zarr's random chunk access at once, which neither alone provides.

---

## 7. Build priorities & deferred work

Container and codec fixed properties are settled. The rest is implementation,
encoder-side tuning (decoupled from the frozen format — the decoder reads
quantizer steps from the stream), or measurements. Build order, from the research
synthesis:

> **Implementation status (2026-05): T0, T1, the container, and the metric
> basket are DONE and validated on real scroll data.** c4d ties c3d v1 on quality
> and is ~2.5× faster single-thread at equal quality (see `bench/RESULTS.md`).
> Throughput ~70 MB/s enc / ~94 MB/s dec per core (memory-bandwidth bound, ~6.4×
> on 16 cores). The status notes below record what shipped vs. what measurement
> changed.

**T0 — Foundation (DONE):** unit-L2-normalized 9/7 (§4.5) with **mirror boundary
extension** (§4.4); dead-zone quant with per-subband L2-weighted steps (steps
carried in the chunk per §4.1); HybridUint token+raw-bits+sign framing; scalar
16-bit-renorm rANS (interleaved SIMD measured-and-skipped, §4.9); SIMD multi-line
lifting for the Y/Z passes (the fused single-loop variant remains a future
option); `constexpr`-baked tables.

**T1 — Biggest wins (DONE):** (a) **EOB / zero-run / significance coding** (§4.7)
— the standout; (b) within-band scan order — **frequency-diagonal measured WORSE
on dense scroll data, kept raster** (§4.7); (c) **MAD + BayesShrink noise-aware
dead-zone** (§4.10) — implemented, OFF by default (measured: loses on the §7.1
basket on *dense* interior where 'noise' is fiber texture; situational for
sparse/air-adjacent regions); (d) **outlier pass** (§4.6) — hard point-wise-error,
verified exact (t=1→max_err 1, ~0.15% outliers).

**T2 — Measure-then-add (all measured):**
- **static context map (§4.8) — SHIPPED.** 6 contexts (level-bucket × prev-class),
  ~10% smaller at equal quality, ~14% decode cost.
- **uniform-block fast-path — SHIPPED.** Constant chunks code to 2 bytes (tag +
  value) instead of ~1.7 KB of table overhead; lossless, q-independent.
- **perceptual/energy-preserving RDO (§4.10) — implemented, OFF by default.**
  Coherence-gated HF preservation; measured neutral-to-negative at matched rate
  (slides along the frontier, doesn't bend it).
- **bivariate parent-child shrinkage (§4.10) — measured-and-skipped.** A fine-grid
  RD sweep shows all shrinkage variants slide along the frontier here; bivariate
  is the same mechanism ⇒ same frontier ⇒ no gain.
- **TCQ — built as an experiment, NOT shipped.** The ceiling estimate said
  ~0.5-0.7 dB, but the realized 4-state coset implementation *loses* at matched
  rate (+0.2-0.35 dB for +0.3-0.8 bits/coeff — a net loss): the simplified
  coset-storage doesn't reproduce Marcellin's exact union-index coding where the
  trellis bit is ~0.5 bit and the level entropy drops below fine-scalar. The
  correct bitstream is a major frozen-format undertaking with uncertain payoff
  given this corpus's already-optimal frontier. `tcq.hpp` kept as a measured
  reference; revisit only with the exact index coding if a future corpus shows
  frontier slack.

**Other deferred:** mask neighbor-occupancy + planar-mode (§5.3 steps 2-3 —
**measured-and-skipped**: the gap-rANS DAG is already within 0.7% of entropy;
clean masks collapse via the DAG, noisy masks are node-count-bound and fixed by
lossy dilation, not per-node coding); fused-lifting + interleaved/block SIMD rANS
(the punted speed pass); whole-volume aggregate ratio + memory-footprint measurements.

> **Recurring lesson (5 measured-skip results: diagonal scan, noise-aware, RDO,
> bivariate, TCQ-as-built):** the baseline RD frontier is already near-optimal on
> the scroll corpus — the unit-L2 per-subband weighting (§4.5) plus zero-run +
> context coding capture the available gains. Encoder-side quantizer/step tweaks
> slide along the frontier rather than bend it. The wins that *stuck* were
> structural coding (zero-run §4.7, context §4.8) and overhead elimination
> (uniform fast-path), not quantization cleverness.

### 7.1 Encoder objective — the balanced metric basket

v1 (c3d) tuned its quantizer params for MAE/MSE, which over-smooths and discards
high-frequency edge detail. c4d instead optimizes a **balanced basket**, evaluated
across the scroll corpus. The deliverable is the **per-metric table** across a
parameter sweep × the corpus, sliced post-hoc with whichever aggregator answers
the question — simple/weighted average, **geometric mean** (punishes imbalance,
captures "good mix"), and **hard floors** for non-negotiables. The aggregator is
encoder-side instrumentation, **not** part of the frozen format; params are
encoder defaults, retunable forever.

Recommended basket: `geomean{ normalized-PSNR, MS-SSIM, 1−GMSD, HaarPSI }` with
hard floors on **MS-SSIM ≥ 0.90** and **Laplacian sharpness-retention ≥ ~0.85**.

- **GMSD** (gradient-magnitude-similarity *deviation*, std-pooled) — the primary
  edge term; its variance pooling specifically punishes "edges smoothed in some
  regions but not others" (v1's failure mode). ~10× cheaper than SSIM.
- **HaarPSI** — perceptual anchor and *wavelet-native* (defined on Haar
  coefficients), so it can be approximated from c4d's own coefficients cheaply.
- **MS-SSIM** — structural floor (cheap via integral-image local stats).
- **Laplacian sharpness-retention** — interpretable one-sided floor against gross
  over-smoothing. **EPI** is a useful CT-domain validation column. (Skip FSIM —
  too heavy; PFOM — threshold-fragile on noise.)

Two-tier evaluation: cheap **coefficient-domain proxies** (subband-energy ≈ SSIM
contrast by Parseval; HaarPSI-on-coefficients) drive in-loop quantization
decisions with no IDWT; **real spatial metrics** (GMSD/MS-SSIM/HaarPSI/EPI) are
computed after one IDWT per chunk for the offline sweep table.

> The HF-energy-preservation RDO (§4.10) and MS-SSIM pull in opposite directions
> (psy-detail vs. structural smoothness); the geomean basket is the referee —
> GMSD/HaarPSI reward recovered detail while the MS-SSIM floor stops it running
> away into noise.

### 7.2 Ratio/quality frontier — measured exhausted (2026-05)

A systematic ceiling analysis found the achievable RD frontier is reached:

- **Entropy coding is maxed.** c4d's shipped payload sits **within 0.25%** of the
  full per-coefficient causal-neighbor-significance context ceiling (240,666 B vs
  240,063 B on a q16 chunk). The clustered 6-context model (§4.8) already captures
  essentially all the conditioning gain a full EBCOT-style context could give. A
  separate significance-map + magnitude scheme is *worse* (325 KB) — confirming
  "model the zeros" via runs beats a sig-map here.
- **Cross-chunk redundancy is negligible.** 128³ chunks are large enough that the
  inter-chunk DC spread is tiny (~10/255 over 8 neighbors) and intra-chunk
  structure is already captured by the DWT — a cross-chunk predictor would save
  ~4 bytes per ~190 KB chunk.
- **Perceptual reweighting loses on its own basket.** Emphasizing HF bits (to feed
  GMSD/HaarPSI) at matched rate *lowers* geomean (0.871→0.865 at 8×) — the basket
  still penalizes the added noise. MSE-optimal L2 weighting wins even perceptually.
- **Longer wavelet — analyzed, declined.** The one untested structural lever.
  Likely +2–5% compaction on smooth regions, but offset by (a) more ringing on the
  sharp air/material boundaries that dominate the corpus, (b) worse chunk-face
  seams (longer filter contaminates more of the seam), (c) a direct hit on the
  already-bottlenecked DWT, and (d) it's a frozen-format change. Net upside
  marginal/negative on high-contrast scroll data; keep 9/7.

Conclusion: the gains came from **structural coding** (zero-run §4.7, context
§4.8) and **overhead elimination** (uniform fast-path); the quantizer/transform
frontier is at its practical optimum for this corpus. Further ratio/quality would
require either a different data regime or relaxing a hard constraint (e.g. the
independent-chunk requirement, which blocks lapped transforms and cross-chunk
prediction).

---

## Appendix: rejected alternatives (do not reopen)

- **CDF 5/3 / longer wavelets** — 5/3 is lossless; longer filters cost speed for
  negligible gain. Keep 9/7 float32.
- **SPIHT / SPECK / EBCOT / embedded bit-plane coder** — serial, bit-oriented,
  non-vectorizable (SPECK is SPERR's documented bottleneck); full rewrite for
  ~5–7%; rANS is at its entropy floor and the headroom is reachable via context
  modeling. Keep rANS. (Borrow SPERR's *outlier pass* and EBCOT's *context idea*,
  not their serial bit engines.)
- **tANS / FSE** — per-distribution table rebuild kills cheap multi-context (the
  whole ratio lever); even Oodle's fast tANS loses to its Huffman. Keep rANS.
- **CABAC / adaptive binary models** — force binarization, serialize, kill the
  SIMD interleaving that gives rANS its throughput (JPEG XL dropped CABAC for
  exactly this). Keep static-per-context multi-symbol rANS.
- **Full context mixing (PAQ-style)** — bit-wise, adaptive, ~10–100× slower
  decode. Its useful residue (neighbor/parent magnitude as context) is already in
  §4.8.
- **Internal/progressive LODs, `decode_lod`, resolution-first subband ordering** —
  failed v1 attempt to replace OME-Zarr levels. Dropped; LODs are separate
  archives.
- **Two-tier transform, adaptive-octree chunking, variable chunk sizes, spatial
  sub-chunk decode** — explored; permanent complexity for a few % over plain
  independent 128³ chunks. Rejected.
- **Zero-padding at chunk faces (for the transform)** — causes tiling artifacts;
  use mirror extension (§4.4). (Zero *fill* for absent volume regions / masked
  voxels is a different, retained concept.)
- **Adaptive-direction / adaptive-update lifting, secondary (LFNST-style)
  transforms, identity-transform skip** — change or augment the fixed 9/7
  transform. Out. (The only compatible lifting change is boundary handling, §4.4.)
- **Symmetry-aware DAG merging for the mask** — high encode cost, pointless for a
  lossy mask. LIDAR-specific G-PCC modes (angular/azimuthal/IDCM) — N/A to CT.
- **True-lossless mode** — non-goal; input is already lossy u8.
- **GPU / neural entropy models** — CPU-SIMD only.
- **Backward/forward compatibility, versioning, extensibility hooks** — one
  version, published and frozen.
