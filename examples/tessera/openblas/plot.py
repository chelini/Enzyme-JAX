"""Speedup chart and LaTeX table from results.csv (OpenBLAS + Tessera)."""
import csv
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, out_png, out_tex = sys.argv[1], sys.argv[2], sys.argv[3]
rows = list(csv.DictReader(open(src)))
t = {}
err = defaultdict(float)
for r in rows:
    key = (r["kernel"], int(r["n"]), int(r["m"]), int(r["threads"]))
    t[key + (r["variant"],)] = float(r["median_s"])
    err[r["variant"]] = max(err[r["variant"]], float(r["err"]))

kernels = ["ols", "gls", "opt", "tri", "diag"]
labels = {
    "ols": "OLS\n(Ex.\u202f01)",
    "gls": "GLS\n(Ex.\u202f02)",
    "opt": "Optimization\n(Ex.\u202f03)",
    "tri": "Triangular\nsolve+product",
    "diag": "Diagonal\nsolve",
}
sizes = {k: sorted({(n, m) for (kk, n, m, th, v) in t if kk == k}) for k in kernels}
ramp = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
ink, muted, grid = "#1f1f1e", "#6b6b66", "#e4e3df"

fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
for ax, th in zip(axes, [1, 16]):
    w = 0.19
    for si in range(4):
        xs, ys = [], []
        for ki, k in enumerate(kernels):
            n, m = sizes[k][si]
            s = t[(k, n, m, th, "base")] / t[(k, n, m, th, "tessera")]
            xs.append(ki + (si - 1.5) * (w + 0.01))
            ys.append(s)
        ax.bar(xs, ys, width=w, color=ramp[si], edgecolor="white", linewidth=0.8,
               label=f"size {si + 1}", zorder=3)
        for x, y in zip(xs, ys):
            ax.text(x, y * 1.06, f"{y:.1f}" if y < 10 else f"{y:.0f}",
                    ha="center", va="bottom", fontsize=6.5, color=muted,
                    rotation=90)
    ax.set_yscale("log")
    ax.axhline(1, color=muted, linewidth=0.8, linestyle="--", zorder=2)
    ax.set_xticks(range(len(kernels)))
    ax.set_xticklabels([labels[k] for k in kernels], fontsize=8, color=ink)
    ax.set_title(f"OpenBLAS, {th} thread{'s' if th > 1 else ''}", fontsize=9,
                 color=ink)
    ax.grid(axis="y", color=grid, linewidth=0.6, zorder=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(muted)
    ax.tick_params(colors=muted, labelsize=8)
axes[0].set_ylabel("Speedup over original program\n(log scale)", fontsize=8.5,
                   color=ink)
axes[0].set_ylim(0.3, 1500)
axes[1].legend(title="problem size (sizes in Table)", fontsize=7.5, title_fontsize=7.5,
               frameon=False, loc="upper left", ncol=2)
fig.tight_layout()
fig.savefig(out_png, dpi=300)

# LaTeX table: absolute times (ms) for every variant.
with open(out_tex, "w") as f:
    for k in kernels:
        for si, (n, m) in enumerate(sizes[k]):
            cells = []
            for th in [1, 16]:
                b = t[(k, n, m, th, "base")]
                cells += [f"{b * 1e3:.1f}",
                          f"{t[(k, n, m, th, 'norules')] / b:.2f}",
                          f"{t[(k, n, m, th, 'nofacts')] / b:.2f}",
                          f"{t[(k, n, m, th, 'tessera')] * 1e3:.2f}",
                          f"\\textbf{{{b / t[(k, n, m, th, 'tessera')]:.1f}}}$\\times$"]
            name = k.upper() if si == 0 else ""
            f.write(f"{name} & {n} & {m} & " + " & ".join(cells) + " \\\\\n")
        f.write("\\midrule\n" if k != kernels[-1] else "")
print("max err per variant:", dict(err))
for th in [1, 16]:
    sp = [t[(k, n, m, th, "base")] / t[(k, n, m, th, "tessera")]
          for k in kernels for (n, m) in sizes[k]]
    nr = [t[(k, n, m, th, "norules")] / t[(k, n, m, th, "base")]
          for k in kernels for (n, m) in sizes[k]]
    print(th, "speedup range", min(sp), max(sp), "norules ratio", min(nr), max(nr))
