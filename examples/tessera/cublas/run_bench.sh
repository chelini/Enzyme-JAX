#!/usr/bin/env bash
# Sweep: kernel x size x variant. Writes results.csv and environment.txt.
cd "$(dirname "$0")"
REPS=${REPS:-7}
CUDA=${TESSERA_CUDA:-/usr/local/cuda-12.6}
echo "kernel,n,m,variant,median_s,err" > results.csv
cases=(
  "gram 1000 500" "gram 2000 1000" "gram 4000 2000" "gram 8000 4000"
  "opt 1000 250" "opt 2000 500" "opt 4000 1000" "opt 6000 1500"
  "tri 1000 500" "tri 2000 1000" "tri 4000 2000" "tri 8000 4000"
  "diag 1000 500" "diag 2000 1000" "diag 4000 2000" "diag 8000 4000"
)
for c in "${cases[@]}"; do
  for v in base norules nofacts tessera; do
    out=$(./bench_$v $c $REPS)
    set -- $c
    med=$(sed 's/.*median_s=\([^ ]*\).*/\1/' <<<"$out")
    err=$(sed 's/.*err=\([^ ]*\).*/\1/' <<<"$out")
    echo "$1,$2,$3,$v,$med,$err" | tee -a results.csv
  done
done
{
  echo "date: $(date -u +%FT%TZ)"
  echo "gpu: $(nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem --format=csv,noheader)"
  echo "nvcc: $("$CUDA/bin/nvcc" --version | tail -1)"
  echo "cublas: $(ls "$CUDA"/lib64/libcublas.so.* | head -1)"
  echo "cpu: $(lscpu | sed -n 's/^Model name: *//p')"
  echo "reactant-clang: ${REACTANT_BIN:-/tmp/tessera-linnea/Reactant/enzyme/bazel-bin}/reactant-clang"
  echo "enzyme-jax: $(git -C "${ENZYME_JAX:-/tmp/tessera-linnea/Enzyme-JAX}" rev-parse HEAD 2>/dev/null)"
} > environment.txt
