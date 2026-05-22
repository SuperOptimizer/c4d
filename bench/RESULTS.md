# c4d vs c3d v1 — benchmark

Head-to-head on real **256³ ESRF Paris4 scroll-interior** volumes (source v2 zarr,
level 0, ~100% nonzero). c4d encodes the 256³ as one region = **4×4×4 = 64 native
64³ chunks** sharing one entropy-table set (the production path, `encode_group`);
c3d encodes the 256³ as one native 256³ chunk. Same voxels, each codec's native
unit. Quality = §7.1 basket; speed = wall time on one AVX-512 box.

Build: `cmake -DC4D_BUILD_C3D_BENCH=ON -DC3D_DIR=~/c3d ...`; run
`compare_c3d <vol256.raw> [c4d_q...]` (c3d needs `-fopenmp`).

## Quality per byte — matched by ratio (two regions, consistent)

| ratio | codec | PSNR  | MS-SSIM | HaarPSI |
|------:|-------|------:|--------:|--------:|
| ~79×  | c3d   | 36.68 | 0.967   | 0.928   |
|       | c4d   | 35.31 | 0.955   | 0.907   |
| ~189× | c3d   | 32.86 | 0.930   | 0.887   |
|       | c4d   | 31.3  | 0.901   | 0.853   |

**c3d wins ~1.0–1.4 dB PSNR per byte at the shipped 64³.** This is a deliberate
tradeoff, not a defect: c4d's 64³ chunk is fixed for fine-grained random access
(any 64³ block is one range-GET). Smaller tiles mean more boundary/seam voxels
and less wavelet context per chunk than c3d's 256³.

**Diagnosed:** rebuilding c4d at 128³/256³ closes most of the gap — at 256³ c3d
wins only ~0.3–0.5 dB (e.g. 79.7×: c4d-256 36.37 vs c3d 36.68). DWT levels are
*not* the lever (64³ at 3/4/6 levels is identical). So ~1 dB of the gap is the
64³ random-access granularity; the residual ~0.3 dB is c3d's RD-allocator vs
c4d's per-subband-L2 step policy.

## Speed — c4d's advantage

c4d **single-threaded** vs c3d **OpenMP multi-threaded**, same box, matched ratio:

| | c4d enc | c3d enc | c4d dec | c3d dec |
|---|-------:|--------:|--------:|--------:|
| ~79× (q16/q32) | 46–63 ms | ~119 ms | 37–51 ms | ~91 ms |

**c4d is ~2–2.5× faster wall-clock despite being single-threaded** (c3d is using
all cores). Per-core the advantage is much larger; c4d's model is caller-parallel
across chunks (one thread per chunk, all thread-safe via per-thread scratch).

## Verdict

c4d trades **~1 dB quality-per-byte (the frozen-64³ random-access tradeoff)** for
**~2–2.5× throughput** (single- vs multi-threaded) plus capabilities c3d lacks:

- 64³ chunk granularity — any block is one range-GET from the single-file archive.
- Single-file `.c4d` archive: append / compose / compact, footer index.
- Optional validity-mask member; hard-L∞ near-lossless mode (outlier pass).

For a recompress / serving pipeline that fetches individual chunks, the speed and
random-access granularity outweigh the ~1 dB. c4d is **not** positioned as a
higher-quality-per-byte codec than c3d — it's the faster, range-addressable one.
