# Tessera + OpenBLAS: Linnea-derived rules (paper Section "Case Study")

Files
- `tessera_openblas.h`: Tessera annotations on unmodified OpenBLAS (scipy-openblas32 0.3.34,
  symbols prefixed `scipy_`) and six Linnea-derived rules (L1-L6).
- `bench.c`: five programs (OLS = Linnea Ex.01, GLS = Ex.02, optimization = Ex.03,
  triangular, diagonal) with independent naive reference checks.
- `build.sh`: builds `bench_base` (no Tessera), `bench_tessera`, and controls
  `bench_norules` (-DTESSERA_NO_RULES) and `bench_nofacts` (-DNO_FACTS).
- `run_bench.sh`: sweep (5 programs x 4 sizes x {1,16} threads x 4 variants, median of 7);
  writes `results.csv`.
- `plot.py`: figure `openblas_speedup.png` and the run-time table rows.
- `posv_vs_gesv.c`: microbenchmark used to analyse the GLS 16-thread slowdown.

Setup used for the paper
- Compiler: Reactant `reactant-clang` built from `tessera/preserving-properties`
  with Enzyme-JAX `lchelini/tessera-linnea` (commit 4c0e811), see
  `examples/tessera/README.md`. Point `build.sh` at its `bazel-bin` with
  `TESSERA_BIN=/path/to/Reactant/enzyme/bazel-bin`.
- OpenBLAS: `pip download --no-deps scipy-openblas32`, unzip, link the
  `scipy_openblas32` directory as `./openblas`.
- Machine: AMD Ryzen 9 7950X, OpenBLAS pinned to cores 0-15 (taskset).
