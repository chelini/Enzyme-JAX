#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash run.sh /path/to/Reactant/enzyme/bazel-bin [output-directory]" >&2
  exit 2
fi
build_bin=$(realpath "$1")
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
output_dir=${2:-$(mktemp -d)}
mkdir -p "$output_dir"
output_dir=$(realpath "$output_dir")
for example in linnea symmetric_update; do
  EXPORT_REACTANT=1 REACTANT_VERIFY_EACH=1 DISABLE_REACTANT=0 \
    "$build_bin/reactant-clang" -O2 -std=c++17 -fno-math-errno \
    -mllvm -reactant-backend=cpu \
    -resource-dir "$build_bin/reactant-clang-resource" \
    -S -emit-llvm "$source_dir/$example.cpp" -o "$output_dir/$example.ll"
  "$build_bin/external/llvm-project/llvm/FileCheck" "$source_dir/$example.cpp" \
    < "$output_dir/$example.ll.mlir"
  harness=check.cpp
  if [[ $example == symmetric_update ]]; then
    harness=check_symmetric_update.cpp
  fi
  DISABLE_REACTANT=1 "$build_bin/reactant-clang" --driver-mode=g++ \
    -O2 -std=c++17 -resource-dir "$build_bin/reactant-clang-resource" \
    "$source_dir/$harness" "$output_dir/$example.ll" \
    -o "$output_dir/check_$example"
  "$output_dir/check_$example"
done
printf 'Artifacts: %s\n' "$output_dir"
