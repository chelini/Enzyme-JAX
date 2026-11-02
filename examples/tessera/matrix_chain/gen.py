#!/usr/bin/env python3
"""Emit one matrix-chain case: <out>/<name>.cpp (library kernels + chain),
<out>/<name>.h (declaration) and <out>/<name>_bench.cpp (timing harness).
Usage: gen.py NAME OUTDIR p0 p1 ... pn   (matrix i is p_i x p_{i+1})"""
import sys, os

def main():
    name, out = sys.argv[1], sys.argv[2]
    p = [int(x) for x in sys.argv[3:]]
    n = len(p) - 1
    os.makedirs(out, exist_ok=True)
    kernels = sorted({(p[i], p[k], p[j]) for i in range(n + 1) for k in range(i + 1, n + 1)
                      for j in range(k + 1, n + 1)})
    args = ", ".join(f"const Matrix<{p[i]}, {p[i+1]}> *a{i}" for i in range(n))
    sig = f"void chain_{name}({args}, Matrix<{p[0]}, {p[n]}> *out)"
    with open(f"{out}/{name}.h", "w") as f:
        f.write('#pragma once\n#include "matrices.h"\n')
        f.write(f'extern "C" {sig};\n')
    with open(f"{out}/{name}.cpp", "w") as f:
        f.write(f'#include "{name}.h"\nextern "C" {{\n')
        for m, k, nn in kernels:
            f.write(f"""[[gnu::noinline, clang::annotate("tessera_reassociate_matmul"),
  pure_tessera_op("la.mm_{m}_{k}_{nn}(a:val=in, b:val=in, out:val=out)")]]
void mm_{m}_{k}_{nn}(const Matrix<{m}, {k}> *a, const Matrix<{k}, {nn}> *b, Matrix<{m}, {nn}> *out) {{
  Matrix<{m}, {nn}> result;
  for (int r = 0; r < {m}; ++r)
    for (int c = 0; c < {nn}; ++c) {{
      double sum = 0;
      for (int k = 0; k < {k}; ++k) sum += a->data[r][k] * b->data[k][c];
      result.data[r][c] = sum;
    }}
  *out = result;
}}
""")
        # left-to-right chain ((A0 A1) A2) ...
        f.write(f"[[gnu::noinline]]\n{sig} {{\n")
        prev = "a0"
        for i in range(1, n):
            dst = "out" if i == n - 1 else f"t{i}"
            if i != n - 1:
                f.write(f"  Matrix<{p[0]}, {p[i+1]}> t{i};\n")
            lhs = prev if i == 1 else f"&{prev}"
            dstp = "out" if i == n - 1 else f"&{dst}"
            f.write(f"  mm_{p[0]}_{p[i]}_{p[i+1]}({lhs}, a{i}, {dstp});\n")
            prev = dst
        f.write("}\n}\n")
    with open(f"{out}/{name}_bench.cpp", "w") as f:
        f.write(f'''#include "{name}.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int argc, char **argv) {{
  static ''')
        f.write("; static ".join(f"Matrix<{p[i]}, {p[i+1]}> a{i}" for i in range(n)))
        f.write(f"; static Matrix<{p[0]}, {p[n]}> out;\n  unsigned s = 12345;\n")
        for i in range(n):
            f.write(f"""  for (auto &r : a{i}.data) for (auto &v : r) {{ s = s * 1664525u + 1013904223u; v = (s >> 8) / 16777216.0 - 0.5; }}\n""")
        call = f"chain_{name}(" + ", ".join(f"&a{i}" for i in range(n)) + ", &out)"
        f.write(f"""  long reps = 1;
  auto run = [&](long r) {{
    auto t0 = std::chrono::steady_clock::now();
    for (long i = 0; i < r; ++i) {{ {call}; asm volatile("" ::: "memory"); }}
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  }};
  while (run(reps) < 0.05) reps *= 2;
  std::vector<double> t;
  for (int i = 0; i < 9; ++i) t.push_back(run(reps) / reps);
  std::sort(t.begin(), t.end());
  if (argc > 1) {{ FILE *f = fopen(argv[1], "wb"); fwrite(&out, sizeof(out), 1, f); fclose(f); }}
  printf("%.6e\\n", t[t.size() / 2]);
}}
""")

main()
