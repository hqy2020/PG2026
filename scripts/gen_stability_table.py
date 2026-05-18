#!/usr/bin/env python3
"""Generate tab_stability.tex from stability results."""
import os

BASE = "/home/qyhu/Documents/r2_ours/PG2026"

data = {
    "Chest": {
        "R\\textsuperscript{2}-Gaussian": [26.165, 26.170, 26.273],
        "SPAGS": [27.502, 27.405, 27.488]
    },
    "Pancreas": {
        "R\\textsuperscript{2}-Gaussian": [28.647, 28.725, 28.796],
        "SPAGS": [29.375, 29.347, 29.317]
    }
}

tex = []
tex.append(r"\begin{table}[t]")
tex.append(r"\centering")
tex.append(r"\caption{Multi-seed stability on Chest and Pancreas 3-view. "
           r"Mean $\pm$ std across 3 random seeds.}")
tex.append(r"\label{tab:stability}")
tex.append(r"\begin{tabular}{lccccc}")
tex.append(r"\toprule")
tex.append(r"Organ & Method & Seed 0 & Seed 1 & Seed 2 & Mean $\pm$ Std \\")
tex.append(r"\midrule")

first = True
for organ in ["Chest", "Pancreas"]:
    for method, psnrs in data[organ].items():
        mu = sum(psnrs) / 3
        std = (sum((p-mu)**2 for p in psnrs) / 2)**0.5
        row = f" {organ} & {method} & {psnrs[0]:.2f} & {psnrs[1]:.2f} & {psnrs[2]:.2f} & ${mu:.2f}\\pm{std:.2f}$ \\\\"
        tex.append(row)
    if organ == "Chest":
        tex.append(r"\midrule")

tex.append(r"\bottomrule")
tex.append(r"\end{tabular}")
tex.append(r"\end{table}")

output = '\n'.join(tex) + '\n'
path = os.path.join(BASE, "tables/tab_stability.tex")
with open(path, "w") as f:
    f.write(output)
print(output)
print(f"✅ Written to {path}")
