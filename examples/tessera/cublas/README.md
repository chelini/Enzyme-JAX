# cuBLAS case study

Same method as `../openblas`: a header (`tessera_cublas.h`) re-declares cuBLAS
entry points with Tessera annotations, and `#pragma optimize` rules swap a
general routine for a cheaper specialized one. cuBLAS is not modified.

| file | role |
|---|---|
| `tessera_cublas.h` | annotated wrappers and 6 rewrite rules (SYRK, TRMM, DGMM) |
| `tl_cuda.cu` | the one device kernel the rules need (`tl_symmetrize_dev`) |
| `kernels.c` | four programs written with GEMM only: `gram`, `opt`, `tri`, `diag` |
| `bench.c` | harness: device allocation, timing, sampled-entry check |
| `build.sh`, `run_bench.sh` | build four variants, sweep, write `results.csv`, `environment.txt` |

## Differences from the OpenBLAS header

- cuBLAS takes `alpha` and `beta` by pointer, so a rule cannot read them. The
  `tl_dgemm`, `tl_dsyrk` and `tl_dtrmm` wrappers take them by value. Programs
  call the wrappers.
- cuBLAS is column-major only. No rule tests a layout argument.
- `cublasDdgmm` and the out-of-place `cublasDtrmm` exist, so the rules need
  neither a scaling helper nor a copy.
- A rule's last call replaces the matched call and must return the same type.
  `tl_symmetrize` therefore returns a `cublasStatus_t`.

## Variants

`base` compiles `kernels.c` with the Tessera pipeline off. `tessera` runs the
pipeline with rules and facts. Controls: `norules` (pipeline, no rules) and
`nofacts` (rules, no `tessera_guarantees`). Only `kernels.c` goes through the
pipeline. The harness is always built plain, because the pipeline's CPU backend
turns `cudaMalloc` into host `malloc`.

## Reproduce

Needs the `reactant-clang` toolchain (see `../openblas/README.md`), CUDA 12.6
and a GPU. Set `TESSERA_CUDA` (not `CUDA_HOME`, which often points elsewhere),
`REACTANT_BIN` and `CUDA_ARCH` (default `sm_86`).

```
./build.sh && ./run_bench.sh
```

`results.csv` holds median seconds of 7 runs (after one warm-up) and the
relative error of 256 sampled entries against host loops.
`environment.txt` records the GPU, CUDA, and Enzyme-JAX commit.
