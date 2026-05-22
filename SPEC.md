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

*Optional encoder-side overlap (HQ mode):* to further suppress inter-chunk seams,
the encoder may transform each chunk from a small halo (e.g. 4–8 voxels/face) read
from neighbors and crop to the 128³ core. The decoder remains fully independent
(it discards nothing it didn't receive). Only the encoder needs neighbor voxels
(it has the whole volume). Small ratio tax; off by default.

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
- **Frequency-diagonal 3D scan order** (the 3D analog of JPEG zigzag: order each
  subband by radial frequency so significants come first, zeros collect at the
  tail): a `constexpr` permutation table, **zero runtime cost**, that ~doubles
  average run length and so multiplies the EOB/run-coding gain. (Replaces Morton
  ordering for the coefficient stream.)
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

- **Interleaved SIMD rANS, 16-bit renormalization**, ≥8-way (16/32-way on AVX2/
  AVX-512). ~3× decode throughput over scalar (≈1.4–2.1 GB/s class). Static
  per-chunk histograms (12–14-bit precision; *not* 16-bit on a 32-bit state).
  Alias tables for O(1) symbol decode once the context count grows.
- **3D single-loop fused "cube core" lifting** (one streaming pass over all three
  axes, not three separate passes with intermediate buffers): ~8× over naive,
  flat per-voxel cost. Diagonal vectorization; lane-count templated for
  NEON(4)/AVX2(8)/AVX-512(16).
- **Prime-stride chunk layout** (pad a 128³ chunk's row/slice stride to a prime,
  e.g. 129/131 floats) to avoid power-of-2 cache-set aliasing on the Y/Z passes.
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
  coefficients). Optionally add bivariate (parent-child) shrinkage for higher
  quality, reusing the §4.8 parent access.
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
2. **Neighbor-occupancy context for the residual boundary shell:** code each
   surviving node's 8-bit child-occupancy byte with an rANS context formed from
   the occupancy of already-coded neighbors (a reduced set — 6 face neighbors, or
   +12 edge bucketed — to keep the table small). Surfaces are locally planar, so
   neighbors are highly predictive. This codes the thin shell the DAG can't dedup.
3. **Planar-mode shortcut:** when a node's occupied children are coplanar, signal
   one plane position per axis instead of the full byte (CT surfaces are locally
   planar).
4. **rANS the DAG topology + skewed child-pointer references** (a few nodes are
   referenced often — entropy-code the skew rather than store raw pointers; this
   is what makes step 1 net-win on *bytes*, not just node count).

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

**T0 — Foundation (cheap, do first):** unit-L2-normalized 9/7 (§4.5) with **mirror
boundary extension** (§4.4); dead-zone quant with per-subband L2-weighted steps;
HybridUint token+raw-bits+sign coefficient framing; interleaved 16-bit-renorm SIMD
rANS (§4.9); 3D single-loop fused lifting + prime-stride layout; `constexpr`-baked
tables.

**T1 — Biggest wins:** (a) **EOB / zero-run / significance coding** (§4.7) —
10–30% on HF subbands *and* faster decode, the standout; (b) **frequency-diagonal
3D scan** replacing Morton — free, multiplies (a); (c) **MAD + BayesShrink
noise-aware dead-zone** (§4.10) — denoise-in-transform, helps ratio *and* quality
on the noisy scans that are the corpus-confirmed hard case; (d) **outlier pass**
(§4.6) — hard point-wise-error / constant-quality.

**T2 — Measure-then-add:** static context map keyed on neighbor/parent magnitude
(§4.8, ≈25%/axis decode cost — measure win vs. speed); bivariate parent-child
shrinkage; perceptual/energy-preserving RDO gated by the MAD noise floor (§4.10);
TCQ as an optional high-effort encode mode (free decode); uniform-block air
fast-path.

**Other deferred:** DWT level count (pin where compression saturates for 128³);
mask octree internals (§5.3, max depth / leaf resolution); parked measurements
(encode/decode speed baseline, whole-volume aggregate ratio, memory footprint).

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
