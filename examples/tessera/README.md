# Tessera and Linnea: builds, rewrites, and matrix-chain optimization

These examples integrate the pinned Tessera development branches below and
implement three transformations:

| Transformation | Implementation | Verified result |
| --- | --- | --- |
| Double transpose | One Tessera rule, `transpose(transpose(A)) -> A` | Both transpose calls disappear; bit patterns are preserved |
| Matrix-chain optimization | New `tessera-optimize-matrix-chain` pass | `(AB)C -> A(BC)` for dimensions `[20, 2, 30, 3]`, reducing scalar multiplication cost from 3000 to 300 |
| Linnea's symmetric update, `trick4` | Conditional Tessera rule plus a SYR2K-style kernel | `AᵀB + BᵀA + AᵀCA -> AᵀT + TᵀA`, `T = B + ½CA`, when `C` is symmetric |

The examples run through Reactant's C++ frontend and CPU pipeline. They use
small reference kernels so that the transformation and its numerical behavior
can be inspected without adding a BLAS dependency. Operation counts below are
not runtime benchmarks.

## Revisions and build status

| Repository | Development branch | Base revision |
| --- | --- | --- |
| [Reactant](https://github.com/EnzymeAD/Reactant/tree/tessera/preserving-properties) | `tessera/preserving-properties` | `932c166e7c1d5acaa7bf57be02bc4043f0e0cc4c` |
| [Enzyme-JAX](https://github.com/EnzymeAD/Enzyme-JAX/tree/jac/new_annotations) | `jac/new_annotations` | `52e9842aa9a8c5d381b798361bcaa8441c85e7f2` |
| [Linnea](https://github.com/HPAC/linnea) | Reference source | `c8fb5d1f64666bf63d35859484a5041ff75dbb90` |

Reactant here is the C++ compiler repository, not Reactant.jl. Its Bazel build
consumes Enzyme-JAX as `@enzyme_ad`; building the targets below builds code from
both branches in one compatible toolchain. This does not build a JAX Python
wheel or a Julia package.

The Enzyme-JAX changes are based on the pinned revision above. Reactant needs a
local Enzyme-JAX override to consume them, as described below. Linnea is used as
a source reference and does not need to be installed to run these examples.

Built successfully with Bazel 7.7.0 and the branch's LLVM/Clang 24 development
toolchain on Linux x86-64:

```text
Reactant/enzyme/bazel-bin/reactant-clang
Reactant/enzyme/bazel-bin/reactant-opt
Reactant/enzyme/bazel-bin/reactant-clang-resource
Reactant/enzyme/bazel-bin/external/enzyme_ad/enzymexlamlir-opt
Reactant/enzyme/bazel-bin/external/llvm-project/llvm/FileCheck
```

The reference build took about 47 minutes. The final incremental
build, adding CSE to the pipeline, took about 13 seconds. These are observations,
not resource requirements or performance guarantees.

## Run the examples

From this Enzyme-JAX checkout:

```bash
TESSERA_ROOT=/absolute/path/to/tessera-linnea
bash examples/tessera/run.sh \
  "$TESSERA_ROOT/Reactant/enzyme/bazel-bin" \
  "$TESSERA_ROOT/linnea-demo"
```

The runner compiles both examples with `-O2 -std=c++17 -fno-math-errno`, exports
the MLIR immediately after Tessera, checks the resulting calls with FileCheck,
links independent C++ numerical harnesses, and runs them. It enables
`REACTANT_VERIFY_EACH=1`. It does not enable `-ffast-math`.

Expected output, besides compiler diagnostics:

```text
PASS: bitwise double transpose and 1920 matrix-chain elements
PASS: 1440 symmetric-update elements (known, guarded, aliased inputs, nonsymmetric fallback)
Artifacts: /absolute/path/to/tessera-linnea/linnea-demo
```

Inspect `linnea.ll.mlir` and `symmetric_update.ll.mlir` for the transformed calls;
the corresponding `.ll` files contain the final LLVM IR. The executables are
`check_linnea` and `check_symmetric_update`.

The current LLVM round trip emits a non-fatal warning about different data-layout
strings when linking its converted module. This work does not fix that warning;
the IR verification and numerical checks above pass with it present.

## Reproduce the build

Place this Enzyme-JAX checkout (including the compiler changes and examples) and
a Reactant checkout side by side under a directory of your choice. Check out
Reactant at the pinned revision above. Set `TESSERA_ROOT` below to that directory.
Use Bazel 7.7.0 and a host Clang installation with `lld` available.

In `Reactant/enzyme/workspace.bzl`, set:

```python
OVERRIDE_ENZYMEXLA_PATH = "/absolute/path/to/tessera-linnea/Enzyme-JAX"
```

This replaces a machine-specific path already present on the Reactant branch.
Use your own absolute Enzyme-JAX checkout path if different. The local override
**copies a snapshot**; it is not a live symlink. After changing compiler source,
refresh it before building:

```bash
TESSERA_ROOT=/absolute/path/to/tessera-linnea
cd "$TESSERA_ROOT/Reactant/enzyme"
BAZEL=bazel
TESSERA_CC=/absolute/path/to/clang

"$BAZEL" --output_user_root="$TESSERA_ROOT/bazel" sync \
  --only=enzyme_ad --experimental_repo_remote_exec \
  --repo_env=CC="$TESSERA_CC" \
  --repo_env=HERMETIC_PYTHON_VERSION=3.11

"$BAZEL" --output_user_root="$TESSERA_ROOT/bazel" build \
  --config=public_cache \
  --repository_cache="$TESSERA_ROOT/repository-cache" \
  --disk_cache="$TESSERA_ROOT/bazel-disk" \
  --repo_env=CC="$TESSERA_CC" \
  --repo_env=HERMETIC_PYTHON_VERSION=3.11 \
  --noincompatible_enable_cc_toolchain_resolution \
  --jobs=24 --local_ram_resources=81920 \
  --copt=-fbracket-depth=1024 --host_copt=-fbracket-depth=1024 \
  --linkopt=-fuse-ld=lld --color=no \
  //:reactant-clang //:reactant-opt //:reactant-clang-resource \
  @enzyme_ad//:enzymexlamlir-opt @llvm-project//llvm:FileCheck
```

Adjust the Bazel, host Clang, cache paths, parallelism, and RAM budget for your
machine. `--experimental_repo_remote_exec` is needed on `sync` because this
branch's `.bazelrc` enables it only for `build`. Examples and tests run directly
from the source worktree and do not require another compiler build when only
their contents change.

## What Tessera represents

Tessera gives annotated library calls an explicit semantic identity inside MLIR.
It does not contain a built-in matrix algebra or automatically infer the meaning
of arbitrary C++ loops. The library author supplies the semantic names, argument
directions, purity, properties, and algebraic rules.

| Construct | Purpose |
| --- | --- |
| `tessera.define` | A library function and its semantic signature, argument modes, and properties |
| `tessera.call` | A call using that signature; lifted matrix data can flow as SSA values |
| `tessera.optimization` inside `tessera.optimizations` | A textual rewrite rule, optionally conditional |
| `tessera.guard` | Specialized and original computations selected by a condition |
| `tessera.yield` | Results of a guard branch |

For example:

```cpp
[[gnu::noinline,
  pure_tessera_op("su.ca(c:val=in, a:val=in, out:val=out)")]]
void su_ca(const FlatMatrix4 *c, const Matrix<4, 3> *a,
           Matrix<4, 3> *out);
```

The semantic operation is `su.ca`. `val=in` reads the pointee into an SSA input;
`val=out` produces an SSA result without reading uninitialized output memory.
`val=inout` both consumes and returns a value. Lowering reconstructs the original
pointer-based calls and stores the results. These examples compute into local
temporaries before writing outputs so their kernels have value semantics even
when input and output storage overlap.

`pure_tessera_op` is a trusted declaration. Incorrect purity or property
annotations can make transformations invalid. It also does not prevent early
inlining: keep `noinline` on semantic kernels until Tessera sees their calls.
For these frontend branches, use the demonstrated unscoped
`[[tessera_assumes("symmetric(c)")]]` spelling or its GNU equivalent. The scoped
`[[tessera::assumes(...)]]` spelling lost string arguments in the earlier probe.

### Properties and guards

* `tessera_assumes` declares facts true at function entry.
* `tessera_guarantees` declares facts established by a result or output.
* `tessera_preserves` declares how input facts imply an output fact, for example
  preserving `SPD` when both inputs of matrix addition are SPD.

Property propagation attaches facts to the SSA values passed to calls. It also
tracks supported handle-based patterns, invalidating facts when intervening
mutations are detected. Existing implications include `SPD` and
`positive_definite` implying one another, positive definiteness implying symmetry
and invertibility, and diagonal matrices implying symmetry and triangularity.
These are declarations and implication rules, not numerical proofs of an
arbitrary kernel implementation.

If a condition is known, the rewrite is unconditional. If a supported condition
is unknown, Tessera can emit a runtime guard. If the condition cannot be checked,
it retains the original computation. Property spellings matter: an unknown
`SPD(a)` is a declared-only property, whereas the built-in
`positive_definite(a)` has a conservative runtime checker. That checker uses
symmetry and strict positive diagonal dominance, not a general Cholesky test;
it can reject matrices that are actually SPD.

Runtime matrix checks currently unroll at most 64 elements by default. Their
layout resolver supports explicit `tessera.layout` and inference for flat square
arrays inside single-field structs. Nested arrays used by the matrix-chain
optimizer are not supported by this runtime element extractor. Accordingly,
the symmetric-update demo uses flat row-major storage for `C`. Symmetry is
unchanged by transposition, so the inferred storage order is sufficient for
this predicate. Order-dependent predicates require an explicit layout.

### Pipeline and rule matching

The relevant order in [raise.cpp](../../src/enzyme_ad/jax/raise.cpp) is:

```text
lift-tessera-annotations -> parse-optimization-rules -> ...
-> llvm-to-tessera -> polygeist-mem2reg -> canonicalize -> cse
-> tessera-optimize-matrix-chain -> tessera-propagate-properties
-> tessera-apply-pdl -> tessera-lower-guards -> tessera-to-llvm
```

The added `cse` step is necessary for the advanced C++ example: lifting each
pointer input can introduce another load of the same matrix. A repeated variable
such as `a` in a rule means the same SSA value. Standard MLIR CSE merges eligible
loads across pure Tessera calls; it does not merge them across intervening
writes. The mutation regression checks this boundary.

Rules become PDL patterns with benefit 1 and are applied greedily. They match the
specified call tree; there is no associative/commutative algebra matcher or
global search choosing among arbitrary rules. A `tessera.equivalent_to`
operation exists in these branches, but no implementation-selection consumer
was found. The matrix-chain pass added here provides a specific cost-based
choice independently of this annotation.

## 1. Double transpose

[linnea.cpp](linnea.cpp) reimplements the simplest identity from Linnea's
[transpose transformation](https://github.com/HPAC/linnea/blob/c8fb5d1f64666bf63d35859484a5041ff75dbb90/linnea/algebra/transformations.py#L389-L398):

```cpp
#pragma optimize "la.transpose(la.transpose(a)) -> a"
```

This data permutation does not require floating-point reassociation. The
FileCheck test confirms both calls disappear. [check.cpp](check.cpp) checks the
output bitwise, including signed zeros, infinities, a NaN payload, and a
subnormal, and also tests an in-place call. The IR regression includes a shared
intermediate that must remain for another consumer.

## 2. Matrix-chain optimization

Linnea has a
[matrix-chain solver](https://github.com/HPAC/linnea/blob/c8fb5d1f64666bf63d35859484a5041ff75dbb90/linnea/algorithm_generation/matrix_chain_solver.py).
The new Tessera pass implements the dense static-shape subset with dynamic
programming; it is not a port of Linnea's complete kernel and property search.

For dimensions `p[0], ..., p[n]`, the usual recurrence is:

```text
cost[i,i] = 0
cost[i,j] = min over k in [i,j):
              cost[i,k] + cost[k+1,j] + p[i] * p[k+1] * p[j+1]
```

Here each interval retains the best plan **per result type**. A combination is
available only when an existing marked kernel accepts the exact operand types
and returns that result type. This matters because C++ structs with equal
dimensions can have different ABI identities. Keeping one plan per shape alone
could discard the only plan a later kernel accepts. Candidate kernels are
indexed by input type pairs, with deterministic iteration through states.
For one representation per interval this is the usual cubic planner.

Opt in each usable product kernel explicitly:

```cpp
[[gnu::noinline, clang::annotate("tessera_reassociate_matmul"),
  pure_tessera_op("la.mm_20_2_30(a:val=in, b:val=in, out:val=out)")]]
void mm_20_2_30(const Matrix<20, 2> *, const Matrix<2, 30> *,
                Matrix<20, 30> *);
```

The marker becomes `tessera.reassociate_matmul` on the definition. It promises
an unconditional dense product and authorizes changed floating-point grouping.
The pass requires two full matrix values and one result, encoded as nonempty
row-major nested LLVM arrays of the same floating-point element type,
optionally wrapped in single-field structs. Descriptors with pointers are not
full matrix snapshots and are excluded.

The example supplies all four required kernels:

```text
Original:   (20×2 · 2×30) · 30×3  : 1200 + 1800 = 3000
Selected:   20×2 · (2×30 · 30×3)  :  180 +  120 =  300
```

The optimizer preserves operand order. Shared products and producers in other
blocks become leaves, and it never duplicates their computation. It requires a
strict cost improvement, preserves ties, respects `tessera.no_rewrite`, saturates
cost arithmetic to avoid overflow, and limits planning to 64 leaves. It does
not synthesize missing kernels. If no legal cheaper plan exists, the original
chain remains.

The six-matrix regression with dimensions `[30,35,15,5,10,20,25]` selects
`(A(BC))((DE)F)` with cost 15125. Other tests cover already optimal chains,
shared values, cross-block boundaries, absent kernels, incompatible ABI types,
malformed/impure declarations, ties, overflow, and the 64/65-leaf boundary.

## 3. Linnea's symmetric rank-2k transformation

[symmetric_update.cpp](symmetric_update.cpp) reimplements
[Linnea's `trick4`](https://github.com/HPAC/linnea/blob/c8fb5d1f64666bf63d35859484a5041ff75dbb90/linnea/algorithm_generation/tricks.py#L175-L204)
and targets its
[SYR2K kernel form](https://github.com/HPAC/linnea/blob/c8fb5d1f64666bf63d35859484a5041ff75dbb90/linnea/kernels/julia/kernels.py#L517-L547).
For real `A,B ∈ R^(m×n)` and symmetric `C ∈ R^(m×m)`, define
`T = B + ½CA`. Then:

```text
AᵀT + TᵀA
  = AᵀB + ½AᵀCA + BᵀA + ½AᵀCᵀA
  = AᵀB + BᵀA + AᵀCA, because Cᵀ = C.
```

The rule in the example is:

```cpp
#pragma optimize "if symmetric(c), su.add(su.add(su.mm(su.transpose(a), b), su.mm(su.transpose(b), a)), su.mm(su.transpose(a), su.ca(c, a))) -> su.syr2k(a, su.axpy(0.5, su.ca(c, a), b))"
```

`su.ca` computes `CA`; `su.axpy` forms `B + ½CA`. The RHS builds `T` once and
passes it to `su.syr2k`, which computes `AᵀT + TᵀA`. No temporary-binding syntax
or new algebra pass is needed. SYR2K computes the upper triangle and mirrors it
to produce the full matrix promised by this API. A production BLAS integration
would need to account for BLAS writing only the selected triangle.

The two entry points deliberately start from the same general computation:

* `symmetric_update_known` declares `symmetric(c)`. It becomes exactly `su_ca`,
  `su_axpy`, and `su_syr2k`, with no symmetry branch. Callers must honor that
  declaration.
* `symmetric_update_checked` declares no symmetry. It gets six equality
  comparisons for the off-diagonal pairs of the 4×4 `C`. Symmetric inputs use
  the optimized path; nonsymmetric inputs use the original expression.

This is stronger than the double-transpose example: it recognizes a shared
expression graph, uses a semantic property, introduces a shared temporary,
selects a structured kernel, and preserves a general fallback.

For these reference loops, with `m=4`, `n=3`, the source-level multiplication
counts are:

| Computation | Original | Rewritten |
| --- | ---: | ---: |
| `CA` | `m²n = 48` | `m²n = 48` |
| Three remaining dense products | `3mn² = 108` | 0 |
| Scaling in `T` | 0 | `mn = 12` |
| Upper-triangle SYR2K | 0 | `mn(n+1) = 48` |
| Total | **156** | **108** |

This excludes additions, transposes, copies, loads/stores, and the runtime
condition. Linnea's kernel model uses a leading-order FLOP estimate for SYR2K;
the table instead counts scalar multiplications in the actual small kernels.
The cost reduction alone does not establish a wall-clock speedup.

The user-supplied rule permits changed floating-point rounding. The tested
contract is finite real-valued data with a numerical tolerance, not strict IEEE
equivalence for NaNs, infinities, signed zeros, overflow, or underflow. Symmetry
checking is exact, not approximate; it does not prove those other conditions.
The replacement can also enforce bitwise symmetry where the original expression
could have slightly different rounding in its two triangles.

[check_symmetric_update.cpp](check_symmetric_update.cpp) compares against an
independent long-double evaluation of the original expression. Its 32 datasets
use symmetric **indefinite** `C`, establishing that SPD is unnecessary. It tests
both entry points, `A == B`, and nonsymmetric `C` whose output is demonstrably
nonsymmetric. There are 1440 checked output elements. The IR regression verifies
the shared temporary, the guard and fallback, and refusal to match an `A` that
changes between uses.

The rule matches the displayed parenthesization and operand order. Linnea's
pattern can additionally match within a larger commutative sum; that broader
algebra matching has not been added here.

## Earlier property-propagation probe

The initial investigation also demonstrated
`if SPD(A), inv(A) * b -> solve_spd(A,b)` with an explicit 2×2 Cholesky solver,
including propagation of SPD through addition of two SPD matrices. Its 48
residual checks passed during the investigation. That exploratory probe is not
included in this checkout or the reproducible test results below.

That probe illustrates declared properties and implementation substitution; it
does not implement general factorization planning or a full runtime SPD test.
The self-contained examples shipped in this directory are the three described
above.

## Tests and source map

The complete Tessera suite passed: **47/47 tests**. In addition, the C++ runner
passed all FileCheck assertions and both numerical harnesses. The earlier
matrix-chain harness covers 1920 output elements over 32 datasets, checked
against a long-double reference. Numerical comparisons use
`1e-11 * (1 + abs(reference))`; double transpose is checked bitwise.

Create a lit site configuration using the built tools and this checkout's
existing `test/lit.cfg.py`:

```bash
TESSERA_ROOT=/absolute/path/to/tessera-linnea
python3 - "$TESSERA_ROOT" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
source = root / "Enzyme-JAX"
build = root / "Reactant/enzyme/bazel-bin"
suite = root / "tessera-tests"
suite.mkdir(exist_ok=True)
(suite / "lit.site.cfg.py").write_text(
    f"config.llvm_tools_dir = {str(build / 'external/llvm-project/llvm')!r}\n"
    f"config.enzymexla_tools_dir = {str(build / 'external/enzyme_ad')!r}\n"
    f"lit_config.load_config(config, {str(source / 'test/lit.cfg.py')!r})\n"
    f"config.test_source_root = {str(source / 'test/lit_tests/tessera')!r}\n"
    f"config.test_exec_root = {str(suite)!r}\n"
)
PY
python3 "$TESSERA_ROOT/Reactant/enzyme/bazel-enzyme/external/llvm-project/llvm/utils/lit/lit.py" \
  -sv -j 8 "$TESSERA_ROOT/tessera-tests"
```

The Bazel execution-root symlink name depends on the workspace directory. If
`bazel-enzyme` is absent, locate the `external/llvm-project` directory under
`bazel info output_base` and use its `llvm/utils/lit/lit.py` instead.

| File | Responsibility |
| --- | --- |
| [Dialect/Tessera/Ops.td](../../src/enzyme_ad/jax/Dialect/Tessera/Ops.td), [Ops.cpp](../../src/enzyme_ad/jax/Dialect/Tessera/Ops.cpp) | Core IR; shared helper for lifted call operand types |
| [LiftTesseraAnnotations.cpp](../../src/enzyme_ad/jax/Passes/Tessera/LiftTesseraAnnotations.cpp) | Frontend metadata and the new reassociation marker |
| [LLVMToTessera.cpp](../../src/enzyme_ad/jax/Passes/Tessera/LLVMToTessera.cpp) | Semantic calls and pointer-to-value lifting |
| [RuleParser.cpp](../../src/enzyme_ad/jax/Passes/Tessera/RuleParser.cpp), [ParseOptimizationRules.cpp](../../src/enzyme_ad/jax/Passes/Tessera/ParseOptimizationRules.cpp) | Rule syntax and PDL construction |
| [OptimizeMatrixChain.cpp](../../src/enzyme_ad/jax/Passes/Tessera/OptimizeMatrixChain.cpp) | New dynamic-programming planner |
| [PropertyPropagation.cpp](../../src/enzyme_ad/jax/Passes/Tessera/PropertyPropagation.cpp) | Property declarations, implication, and transfer |
| [TesseraApplyPDL.cpp](../../src/enzyme_ad/jax/Passes/Tessera/TesseraApplyPDL.cpp) | Rule application, RHS validation, and guarded rewrites |
| [Predicates.cpp](../../src/enzyme_ad/jax/Passes/Tessera/Predicates.cpp), [LowerTesseraGuards.cpp](../../src/enzyme_ad/jax/Passes/Tessera/LowerTesseraGuards.cpp) | Runtime conditions and fallback lowering |
| [TesseraToLLVM.cpp](../../src/enzyme_ad/jax/Passes/Tessera/TesseraToLLVM.cpp) | Restore ABI-compatible calls |
| [Passes.td](../../src/enzyme_ad/jax/Passes/Tessera/Passes.td), [raise.cpp](../../src/enzyme_ad/jax/raise.cpp) | Pass contracts and frontend pipeline integration |
| [test/lit_tests/tessera](../../test/lit_tests/tessera) | IR regression tests, including the new transpose, chain, and symmetric-update cases |

## What remains outside this implementation

Tessera can express useful Linnea-style transformations today when a library
provides semantic calls and valid contracts. These examples add a static
matrix-chain planner and demonstrate a property-dependent multi-operation
rewrite through the existing machinery.

There is no general Linnea search graph, factorization planner, dynamic-shape
optimizer, kernel generator, or measured BLAS/GPU cost model here. Matrix-chain
planning and greedy algebraic rewriting run in a fixed order; they do not
jointly search for a global optimum across both transformation families. The
tests validate the stated static CPU contracts, not arbitrary shapes, backends,
numerical ranges, or incorrect annotations.
