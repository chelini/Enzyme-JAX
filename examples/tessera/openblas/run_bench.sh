#!/usr/bin/env bash
# Sweep: kernel x size x variant x threads. Writes results.csv.
cd "$(dirname "$0")"
REPS=7
echo "kernel,n,m,threads,variant,median_s,err" > results.csv
cases=(
  "ols 500 250" "ols 1000 500" "ols 2000 1000" "ols 4000 2000"
  "gls 500 125" "gls 1000 250" "gls 2000 500" "gls 3000 750"
  "opt 500 250" "opt 1000 500" "opt 2000 1000" "opt 4000 2000"
  "tri 500 250" "tri 1000 500" "tri 2000 1000" "tri 4000 2000"
  "diag 500 250" "diag 1000 500" "diag 2000 1000" "diag 4000 2000"
)
for c in "${cases[@]}"; do
  for t in 1 16; do
    for v in base norules nofacts tessera; do
      out=$(OPENBLAS_NUM_THREADS=$t taskset -c 0-15 ./bench_$v $c $REPS)
      set -- $c
      med=$(sed 's/.*median_s=\([^ ]*\).*/\1/' <<<"$out")
      err=$(sed 's/.*err=\([^ ]*\).*/\1/' <<<"$out")
      echo "$1,$2,$3,$t,$v,$med,$err" | tee -a results.csv
    done
  done
done
