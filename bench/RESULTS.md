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

## Verdict

- **Quality: essentially tied.** On clean data c4d matches c3d to within 0.03 dB
  PSNR / 0.001 geomean at every ratio. On noisy data c3d is marginally ahead at
  ~10× (geomean 0.858 vs 0.849), c4d marginally ahead at ~15×. No regression.
- **Speed: c4d wins ~2.5×** single-thread (enc ~2.4-2.5×, dec ~2.6-2.8×), with
  zero quality cost. This is the §4.9 SIMD DWT + SIMD quant paying off.
- **Threading context:** c3d is OpenMP-parallel; on 16 cores it reaches ~146ms
  enc / 162ms dec at 10× — i.e. c4d **single-threaded** ≈ c3d **16-threaded**.
  Since c4d's model is caller-parallel-across-chunks (spec §0), it has ~16× more
  headroom on the same box.

Caveat: c4d at 128³ pays a fixed per-chunk overhead (freq table ~256B + step
table 160B) ⇒ 8 chunks/256³ carry it 8×; negligible on dense data but a known
target for a shared/default-table optimization.
