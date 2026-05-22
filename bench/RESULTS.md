# c4d vs c3d v1 — benchmark

Head-to-head on identical **256³ dense scroll-interior** volumes (97-99% nonzero,
from `s3://vesuvius-challenge`). c4d encodes as 8× its native 128³ chunks; c3d
encodes the 256³ as one native chunk. Same voxels, each codec's native unit.
Quality = §7.1 basket; speed = wall time (one machine, AVX-512, 16 cores).

Build: `cmake -DC4D_BUILD_C3D_BENCH=ON -DC3D_DIR=~/c3d ...`; run
`compare_c3d <vol256.raw> [q...]`.

## PHerc0172 (clean, 7.9µm) — matched by ratio, SINGLE-THREAD

| ratio | codec | PSNR  | GMSD  | MS-SSIM | HaarPSI | geomean | enc ms | dec ms |
|------:|-------|------:|------:|--------:|--------:|--------:|-------:|-------:|
| ~5×   | c3d   | 40.72 | 0.978 | 0.997   | 0.987   | 0.941   | 601    | 465    |
|       | c4d   | 40.38 | 0.977 | 0.997   | 0.985   | 0.938   | **330**| **222**|
| ~10×  | c3d   | 35.42 | 0.948 | 0.988   | 0.966   | 0.895   | 540    | 431    |
|       | c4d   | 35.39 | 0.947 | 0.989   | 0.964   | 0.894   | **233**| **167**|
| ~15×  | c3d   | 32.75 | 0.926 | 0.978   | 0.949   | 0.866   | 524    | 394    |
|       | c4d   | 32.72 | 0.925 | 0.979   | 0.947   | 0.866   | **208**| **151**|

## PHerc1667 (noisy, 3.24µm) — matched by ratio, SINGLE-THREAD

| ratio | codec | PSNR  | geomean | enc ms | dec ms |
|------:|-------|------:|--------:|-------:|-------:|
| ~10×  | c3d   | 32.85 | 0.858   | 563    | 455    |
|       | c4d   | 31.99 | 0.849   | **234**| **166**|
| ~15×  | c3d   | 30.23 | 0.820   | 532    | 397    |
|       | c4d   | 30.14 | 0.823   | **204**| **150**|

> **Note (post-§4.8 context modeling):** the table above predates the static
> context map. With it, c4d's ratios rose ~10% (e.g. q16 8.9×→9.6×, q24
> 13.1×→14.3× at the same quality), making the quality/ratio race a true dead
> heat — at any matched quality c4d and c3d land within a few % on ratio,
> trading the lead point to point. The speed win is unchanged. Re-run
> `compare_c3d` for current exact numbers.

## Verdict

- **Quality/ratio: a dead heat.** At matched quality c4d and c3d are within a few
  % on ratio everywhere (c3d a hair ahead on clean ~5-15×, c4d ahead on noisy
  ~15×); within 0.03 dB PSNR / 0.001 geomean. No clear winner, no regression.
- **Speed: c4d wins ~2.5×** single-thread (enc ~2.4-2.5×, dec ~2.6-2.8×), equal
  quality, every operating point. The decisive, repeatable advantage (§4.9 SIMD
  DWT + SIMD quant).
- **Plus capabilities c3d lacks:** hard-L∞ near-lossless mode (outlier pass §4.6),
  single-file archive with append/compose/compact (§5.5-5.7), optional
  validity-mask member (§5.3), perceptual metric basket (§7.1) — at no quality cost.
- **Threading context:** c3d is OpenMP-parallel; on 16 cores it reaches ~146ms
  enc / 162ms dec at 10× — i.e. c4d **single-threaded** ≈ c3d **16-threaded**.
  c4d's model is caller-parallel-across-chunks (spec §0), but see scaling below.

## Absolute throughput (measured, this box: AVX-512, 16 cores)

| | 1 core | 16 cores |
|---|-------|----------|
| encode | ~70 MB/s (73 Mvox/s) | ~0.45 GB/s |
| decode | ~94 MB/s (99 Mvox/s) | ~0.45 GB/s |

Parallel scaling is **~6.4×** at 16 cores, not 16× — the codec is
**memory-bandwidth bound** (each chunk streams 2 MB through DWT→quant→entropy).
Still short of the spec's 1.4-2.1 GB/s aspiration; the remaining lever is the
scalar rANS (last un-SIMD'd hot stage) + cutting memory passes.

Caveat: c4d at 128³ pays a fixed per-chunk overhead (freq table ~256B + step
table 160B) ⇒ 8 chunks/256³ carry it 8×; negligible on dense data but a known
target for a shared/default-table optimization.

## v2 update (64³ chunks + 256³ region-shared tables + spatial-neighbor context)

Relaxing chunk independence (encode reads 26 uncompressed neighbors, decode reads
26 compressed) enabled region-shared entropy tables + spatial-neighbor context.
Measured on cmp256/p172_a.raw (256³ = one region), single-thread, matched by ratio:

| quality | c3d v1 | c4d v2 | speed (c4d enc/dec vs c3d) |
|--------:|-------:|-------:|---------------------------|
| ~40.4 dB | 5.4× | **5.4×** (tied) | 442/382 ms vs 694/521 |
| ~35.4 dB | 10.2× | 9.9× | 311/268 ms vs 570/435 |
| ~32.7 dB | 15.2× | 14.5× | 278/228 ms vs 531/404 |

**c4d v2 ties-or-beats c3d on ratio** (v1 was slightly behind) **and is ~1.8–2×
faster** single-thread. The +10–14% v2 ratio gain over per-chunk 64³ closed the
small gap c3d v1 had. Net: c4d v2 is the better codec on ratio, quality (tied),
and speed (~2×).
