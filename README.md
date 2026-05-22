# c4d — compress4d

A CPU-only, SIMD, lossy compression codec and single-file archive format for
3D grayscale (`u8`) X-ray volumetric data. Modern C++26, header-only core.

c4d is purpose-built for the Herculaneum scroll CT corpus but is architecturally
general for any 3D `u8` X-ray volume. It is the spiritual successor to `c3d` —
a ground-up rewrite sharing none of its code or format. The format is meant to
be designed once, frozen, and never versioned; see [SPEC.md](SPEC.md) for the
full specification and design rationale.

## What it does

- **CDF 9/7 lifting wavelet** (separable 3D, float32, unit-L2 normalized) +
  **dead-zone scalar quantization** + **rANS** entropy coding with zero-run /
  EOB modeling and a small context model.
- **64³ chunks**, grouped into 256³ regions that share one entropy-table set.
- A single `.c4d` archive concatenates all chunks (and an optional validity
  mask) with a footer index, so any chunk is one range-GET.
- A single quality knob `q` sets the rate/quality trade-off (small `q` = high
  quality / low ratio; large `q` = high ratio). The decoder reads the
  per-subband steps from the stream, so the encoder's step policy is retunable
  without touching the format.
- **Single-threaded but thread-safe**: each call uses per-thread scratch, so
  callers parallelize by invoking the API from multiple threads (one per chunk).

## Build

Requires a C++26 compiler (GCC or Clang) and CMake ≥ 3.28.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build
ctest --test-dir build
```

The core (`include/c4d/`) is header-only — to use it as a library, add
`include/` to your include path and `#include "c4d/chunk.hpp"` (encode/decode)
or `c4d/archive.hpp` (the container). Build flags: `-O3 -march=native
-ffast-math -funroll-loops` with LTO (see `CMakeLists.txt`). The SIMD paths use
`std::experimental::simd`; an AVX-512 register transpose accelerates the DWT
with a scalar fallback for other ISAs.

## CLI

```
c4d encode  <raw_volume> <archive.c4d> --shape Z,Y,X [--q N] [--mask-threshold T]
c4d decode  <archive.c4d> <raw_volume>
c4d info    <archive.c4d>
c4d compact <in.c4d> <out.c4d>
c4d bench   <dir-of-raw-chunks> [q ...]      # PSNR + ratio per quality knob
```

`<raw_volume>` is row-major `u8` in Z,Y,X order. See [bench/RESULTS.md](bench/RESULTS.md)
for measured quality/ratio/throughput.

## Layout

- `include/c4d/` — the header-only codec (DWT, quant, rANS, chunk pipeline,
  archive container, validity mask).
- `cli/` — the `c4d` command-line tool.
- `tests/` — per-module correctness tests (run via `ctest`).
- `bench/` — the c3d head-to-head benchmark and a throughput microbench.

## License

See [LICENSE](LICENSE).
