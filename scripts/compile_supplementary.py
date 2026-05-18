#!/usr/bin/env python3
"""Compile all supplementary experiment results into a dedicated folder."""
import os, json, shutil, glob

BASE = "/home/qyhu/Documents/r2_ours/PG2026"
SUPP = os.path.join(BASE, "supplementary")

# Load results
with open(os.path.join(BASE, "output/comparison_234/results.json")) as f:
    results = json.load(f)

# Compute averages
for v in ['2', '3', '4']:
    r2_vals = []
    spags_vals = []
    for k, vdata in results.items():
        parts = k.split('/')
        if len(parts) == 3:
            method, organ, nv = parts
            if nv == v:
                if method == 'r2_gaussian':
                    r2_vals.append(vdata['psnr_2d'])
                elif method == 'spags':
                    spags_vals.append(vdata['psnr_2d'])
    r2_avg = sum(r2_vals) / len(r2_vals) if r2_vals else 0
    spags_avg = sum(spags_vals) / len(spags_vals) if spags_vals else 0
    print(f"{v}-view: R2={r2_avg:.2f} SPAGS={spags_avg:.2f} Δ={spags_avg-r2_avg:+.2f}")

# Print all results in a table
print("\n=== All results ===")
print(f"{'Method':15s} {'Organ':10s} {'View':5s} {'PSNR':8s} {'SSIM':8s}")
print("-"*50)
for k in sorted(results.keys()):
    method, organ, nv = k.split('/')
    v = results[k]
    print(f"{method:15s} {organ:10s} {nv+'v':5s} {v['psnr_2d']:>7.2f}  {v['ssim_2d']:.4f}")
