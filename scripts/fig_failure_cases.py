#!/usr/bin/env python3
"""P1-2: Failure Cases Figure using existing 2-view renders."""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

BASE = "/home/qyhu/Documents/r2_ours/PG2026"
FIGS_DIR = os.path.join(BASE, "figures")
os.makedirs(FIGS_DIR, exist_ok=True)


def load(run_dir, view, suffix="pred"):
    fp = f"{run_dir}/test/iter_30000/render_test/{view:05d}_{suffix}.png"
    if not os.path.exists(fp):
        return None
    img = np.array(Image.open(fp))
    if img.ndim == 3:
        img = img[:, :, 0]
    return img.astype(np.float32) / 255.0


def main():
    print("=== P1-2: Failure Cases ===")
    
    # Case 1: Foot 2-view
    foot_r2 = f"{BASE}/output/2026_05_01_foot_2views_r2_gaussian"
    foot_sp = f"{BASE}/output/2026_05_01_foot_2views_spags"
    
    # Case 2: Pancreas 2-view
    panc_r2 = f"{BASE}/output/2026_05_01_pancreas_2views_r2_gaussian"
    panc_sp = f"{BASE}/output/2026_05_02_pancreas_2views_spags_retry"
    
    # Fallbacks
    if not os.path.exists(f"{panc_r2}/test/iter_30000/render_test"):
        panc_r2 = f"{BASE}/output/2026_05_05_pancreas_2views_r2_gaussian"
    
    # Gather data
    CASES = []
    for name, r2dir, spdir, views in [
        ("Foot 2-view", foot_r2, foot_sp, [5, 30]),
        ("Pancreas 2-view", panc_r2, panc_sp, [5, 30]),
    ]:
        row = []
        for v in views:
            gt_sp = load(spdir, v, "gt")
            gt_r2 = load(r2dir, v, "gt")
            gt = gt_sp if gt_sp is not None else gt_r2
            r2 = load(r2dir, v, "pred")
            sp = load(spdir, v, "pred")
            if r2 is not None and sp is not None:
                row.append((gt, r2, sp))
        if row:
            CASES.append((name, row))
            print(f"  ✅ {name}: {len(row)} views")
    
    n_cases = len(CASES)
    if n_cases == 0:
        print("❌ No data available"); return
    
    # Simple layout: each case = 1 row, 5 columns (GT, R2, SP, |R2-GT|, |SP-GT|)
    fig, axes = plt.subplots(n_cases, 5, figsize=(7.5, 1.8 * n_cases))
    if n_cases == 1:
        axes = axes.reshape(1, 5)
    
    for ci, (name, views) in enumerate(CASES):
        # Use first view for display
        gt, r2, sp = views[0]
        
        # GT
        ax = axes[ci, 0]
        if gt is not None:
            ax.imshow(gt, cmap="gray", vmin=0, vmax=1, aspect="auto")
        ax.axis("off")
        if ci == 0: ax.set_title("GT", fontsize=7, pad=2)
        
        # R²-Gaussian
        ax = axes[ci, 1]
        if r2 is not None:
            ax.imshow(r2, cmap="gray", vmin=0, vmax=1, aspect="auto")
        ax.axis("off")
        if ci == 0: ax.set_title("R²-Gaussian", fontsize=7, pad=2)
        
        # SPAGS
        ax = axes[ci, 2]
        if sp is not None:
            ax.imshow(sp, cmap="gray", vmin=0, vmax=1, aspect="auto")
        ax.axis("off")
        if ci == 0: ax.set_title("SPAGS", fontsize=7, pad=2)
        
        # |R²-GT| residual
        ax = axes[ci, 3]
        if gt is not None and r2 is not None:
            res = np.abs(r2 - gt)
            ax.imshow(res, cmap="hot", vmin=0, vmax=0.15, aspect="auto")
        ax.axis("off")
        if ci == 0: ax.set_title("|R²−GT|", fontsize=7, pad=2)
        
        # |SPAGS-GT| residual
        ax = axes[ci, 4]
        if gt is not None and sp is not None:
            res = np.abs(sp - gt)
            ax.imshow(res, cmap="hot", vmin=0, vmax=0.15, aspect="auto")
        ax.axis("off")
        if ci == 0: ax.set_title("|SPAGS−GT|", fontsize=7, pad=2)
        
        # Case label
        axes[ci, 0].text(-0.1, 0.5, name, transform=axes[ci, 0].transAxes,
                        fontsize=7.5, fontweight="bold", va="center", ha="right",
                        rotation=90)
    
    plt.subplots_adjust(wspace=0.02, hspace=0.04)
    out = f"{FIGS_DIR}/fig_failure_cases.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print(f"  ✅ {out}")


if __name__ == "__main__":
    main()
