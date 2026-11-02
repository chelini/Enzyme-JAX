#!/usr/bin/env python3
"""Run matrix-chain experiments. Usage: run.py BAZEL_BIN OUTDIR [case-filter]
Per case: baseline = left-to-right chain compiled with the Tessera pipeline
disabled; tessera = same source through tessera-optimize-matrix-chain.
Reports multiplication counts (left-to-right, pass, DP optimum), median time
of 9 samples pinned to one core, and max relative output difference."""
import csv, os, re, subprocess, sys, struct
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.realpath(__file__)))
from cases import cases

bin_, out = os.path.realpath(sys.argv[1]), os.path.realpath(sys.argv[2])
flt = sys.argv[3] if len(sys.argv) > 3 else ""
here = os.path.dirname(os.path.realpath(__file__))
os.makedirs(out, exist_ok=True)
CC = f"{bin_}/reactant-clang"
COMMON = ["-O2", "-std=c++17", "-fno-math-errno", "-resource-dir", f"{bin_}/reactant-clang-resource", f"-I{here}", f"-I{here}/..", f"-I{out}"]

def sh(cmd, env=None, **kw):
    e = dict(os.environ, **(env or {}))
    return subprocess.run(cmd, env=e, check=True, capture_output=True, text=True, **kw)

def dp(p):
    n = len(p) - 1
    c = [[0] * n for _ in range(n)]
    for l in range(1, n):
        for i in range(n - l):
            j = i + l
            c[i][j] = min(c[i][k] + c[k+1][j] + p[i] * p[k+1] * p[j+1] for k in range(i, j))
    return c[0][n-1]

def ltr(p):
    return sum(p[0] * p[i] * p[i+1] for i in range(1, len(p) - 1))

def achieved(ll, name):
    m = re.search(rf"define[^\n]*@chain_{name}\(.*?\n}}", ll, re.S)
    body = m.group(0)
    return sum(int(a) * int(b) * int(c) for a, b, c in re.findall(r"@mm_(\d+)_(\d+)_(\d+)\(", body))

rows = []
for name, p in cases():
    if flt not in name:
        continue
    sh([sys.executable, "-I", f"{here}/gen.py", name, out] + [str(x) for x in p])
    d = {}
    for variant, env in (("base", {"DISABLE_REACTANT": "1"}),
                         ("tessera", {"EXPORT_REACTANT": "1", "REACTANT_VERIFY_EACH": "1", "DISABLE_REACTANT": "0"})):
        ll = f"{out}/{name}.{variant}.ll"
        sh([CC, *COMMON, "-mllvm", "-reactant-backend=cpu", "-S", "-emit-llvm", f"{out}/{name}.cpp", "-o", ll], env)
        exe = f"{out}/{name}.{variant}.exe"
        sh([CC, "--driver-mode=g++", *COMMON, f"{out}/{name}_bench.cpp", ll, "-o", exe], {"DISABLE_REACTANT": "1"})
        res = sh(["taskset", "-c", "2", exe, f"{out}/{name}.{variant}.bin"]).stdout
        d[variant] = (float(res), achieved(open(ll).read(), name))
    a = struct.unpack(f"{os.path.getsize(f'{out}/{name}.base.bin')//8}d", open(f"{out}/{name}.base.bin", "rb").read())
    b = struct.unpack(f"{len(a)}d", open(f"{out}/{name}.tessera.bin", "rb").read())
    err = max(abs(x - y) / max(abs(x), 1e-300) for x, y in zip(a, b))
    rows.append(dict(case=name, n=len(p) - 1, dims="x".join(map(str, p)), mults_ltr=ltr(p), mults_opt=dp(p),
                     mults_base_ll=d["base"][1], mults_tessera_ll=d["tessera"][1],
                     t_base_s=d["base"][0], t_tessera_s=d["tessera"][0],
                     speedup=d["base"][0] / d["tessera"][0], max_rel_diff=err))
    print(rows[-1], flush=True)
env = subprocess.run(f"git -C {here} rev-parse HEAD; lscpu | grep 'Model name'; {CC} --version | head -1; python3 --version",
                     shell=True, capture_output=True, text=True).stdout
open(f"{out}/environment.txt", "w").write(env)
with open(f"{out}/results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
