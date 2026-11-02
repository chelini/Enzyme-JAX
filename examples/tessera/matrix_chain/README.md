# Matrix-chain planning: measured evaluation

Measures what `tessera-optimize-matrix-chain` buys on fixed-size matrix chains.

- **Baseline:** the chain written left to right, `((A0 A1) A2) ...`, compiled
  with the Tessera pipeline disabled (`DISABLE_REACTANT=1`).
- **Tessera:** the same source through the full pipeline.
- Both use the same `reactant-clang -O2`, the same kernels, and the same
  harness. The kernels are naive triple loops (`MATMUL` macro of `linnea.cpp`),
  so the speedup reflects the multiplication count, not a tuned BLAS.

For every case the script reports:

| column | meaning |
|---|---|
| `mults_ltr` | scalar multiplications of the left-to-right chain (computed) |
| `mults_opt` | optimum from an independent dynamic program (`run.py: dp`) |
| `mults_base_ll`, `mults_tessera_ll` | multiplications counted from the `@mm_M_K_N` calls in the emitted LLVM IR of each variant |
| `t_base_s`, `t_tessera_s` | median of 9 samples, one core (`taskset -c 2`) |
| `speedup` | `t_base_s / t_tessera_s` |
| `max_rel_diff` | max elementwise relative difference of the two outputs |

`mults_tessera_ll == mults_opt` checks that the pass found the optimum.

## Cases (`cases.py`)

- `clrs6`: the six-matrix instance of Cormen et al., dims `[30,35,15,5,10,20,25]`.
- `clrs6x4`: the same instance with every dimension multiplied by 4.
- `ex3`: the 3-matrix example of the paper, dims `[20,2,30,3]`.
- `rand{n}_{r}`: n = 4..8 matrices, dims drawn from `{8,16,...,128}` with
  `random.Random(2027)`; three chains per n.

Inputs are filled by a fixed LCG, so every run computes the same data.

## Reproduce

Requires the `reactant-clang` toolchain built as described in `../README.md`
and `taskset`.

```
git clone git@github.com:chelini/Enzyme-JAX.git && cd Enzyme-JAX
git checkout <commit>            # commit recorded in environment.txt
cd examples/tessera/matrix_chain
python3 run.py /path/to/Reactant/enzyme/bazel-bin /tmp/matrix-chain-out
```

`/tmp/matrix-chain-out/results.csv` holds the table and
`environment.txt` records the Enzyme-JAX commit, CPU, and compiler.

Counts (`mults_*`) are deterministic. Times are not: expect a few percent
run-to-run noise (see the reproducibility check in the paper repository).
