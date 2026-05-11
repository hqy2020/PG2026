#!/usr/bin/env python3
"""
Generate tab_stability.tex from multi-seed training results.
Run after all 12 stability training runs complete + test.py evaluations.
"""
import os, sys, pickle, glob, yaml

BASE = "/home/qyhu/Documents/r2_ours/PG2026"

# Check what's available
for organ in ["chest", "pancreas"]:
    for method in ["r2", "spags"]:
        psnrs = []
        for seed in [0, 1, 2]:
            d = f"{BASE}/output/{organ}_stability_{method}_seed{seed}"
            for yml in [f"{d}/test/iter_30000/eval2d_render_test.yml",
                        f"{d}/eval/iter_030000/eval2d_render_test.yml"]:
                if os.path.exists(yml):
                    with open(yml) as f:
                        for line in f:
                            if line.startswith("psnr_2d:"):
                                psnrs.append(float(line.split(":")[1]))
                            break
                    break
        label = "R²-Gaussian" if method == "r2" else "SPAGS"
        if len(psnrs) >= 2:
            mu = sum(psnrs) / len(psnrs)
            std = (sum((p - mu)**2 for p in psnrs) / (len(psnrs) - 1))**0.5 if len(psnrs) > 1 else 0
            print(f"{organ} {label}: {mu:.2f}±{std:.2f} (n={len(psnrs)} seeds)")
        else:
            print(f"{organ} {label}: only {len(psnrs)} results available")
