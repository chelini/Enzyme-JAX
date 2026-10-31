#!/usr/bin/env bash
# Build bench twice with the same compiler: plain (DISABLE_REACTANT=1) and with
# the Tessera pipeline. Output: bench_base, bench_tessera, bench.ll.mlir.
set -euo pipefail
cd "$(dirname "$0")"
B=${TESSERA_BIN:-/tmp/tessera-linnea/Reactant/enzyme/bazel-bin}
O=$PWD/openblas
CC=("$B/reactant-clang" -O2 -std=gnu11 -I"$O/include" -resource-dir "$B/reactant-clang-resource")
LIBS=(-L"$O/lib" -lscipy_openblas -Wl,-rpath,"$O/lib" -lm)
DISABLE_REACTANT=1 "${CC[@]}" bench.c -o bench_base "${LIBS[@]}"
EXPORT_REACTANT=1 DISABLE_REACTANT=0 "${CC[@]}" -mllvm -reactant-backend=cpu \
  -S -emit-llvm bench.c -o bench.ll
DISABLE_REACTANT=1 "${CC[@]}" bench.ll -o bench_tessera "${LIBS[@]}"
# Controls: the Tessera pipeline without rules, and with rules but no facts.
for v in "norules -DTESSERA_NO_RULES" "nofacts -DNO_FACTS"; do
  set -- $v
  EXPORT_REACTANT=1 DISABLE_REACTANT=0 "${CC[@]}" $2 -mllvm -reactant-backend=cpu \
    -S -emit-llvm bench.c -o bench_$1.ll
  DISABLE_REACTANT=1 "${CC[@]}" bench_$1.ll -o bench_$1 "${LIBS[@]}"
done
