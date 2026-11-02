#!/usr/bin/env bash
# Build the kernels twice with the same compiler: plain (DISABLE_REACTANT=1) and
# through the Tessera pipeline; the harness is always built plain.
# Output: bench_base, bench_tessera, plus controls bench_norules, bench_nofacts.
set -euo pipefail
cd "$(dirname "$0")"
B=${REACTANT_BIN:-/tmp/tessera-linnea/Reactant/enzyme/bazel-bin}
CUDA=${TESSERA_CUDA:-/usr/local/cuda-12.6}   # not CUDA_HOME: often points elsewhere
ARCH=${CUDA_ARCH:-sm_86}
CC=("$B/reactant-clang" -O2 -std=gnu11 -I"$CUDA/include" -resource-dir "$B/reactant-clang-resource")
LIBS=(-L"$CUDA/lib64" -lcublas -lcudart -lstdc++ -lm -Wl,-rpath,"$CUDA/lib64")
"$CUDA/bin/nvcc" -O2 -arch="$ARCH" -c tl_cuda.cu -o tl_cuda.o
DISABLE_REACTANT=1 "${CC[@]}" -c bench.c -o bench.o
DISABLE_REACTANT=1 "${CC[@]}" -c kernels.c -o kernels_base.o
EXPORT_REACTANT=1 DISABLE_REACTANT=0 "${CC[@]}" -mllvm -reactant-backend=cpu \
  -S -emit-llvm kernels.c -o kernels.ll
DISABLE_REACTANT=1 "${CC[@]}" -c kernels.ll -o kernels_tessera.o
variants=(base tessera)
for v in "norules -DTESSERA_NO_RULES" "nofacts -DNO_FACTS"; do
  set -- $v
  EXPORT_REACTANT=1 DISABLE_REACTANT=0 "${CC[@]}" $2 -mllvm -reactant-backend=cpu \
    -S -emit-llvm kernels.c -o kernels_$1.ll
  DISABLE_REACTANT=1 "${CC[@]}" -c kernels_$1.ll -o kernels_$1.o
  variants+=($1)
done
for v in "${variants[@]}"; do
  DISABLE_REACTANT=1 "${CC[@]}" bench.o kernels_$v.o tl_cuda.o -o bench_$v "${LIBS[@]}"
done
