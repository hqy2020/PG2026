#!/usr/bin/env python3
"""
Round 4: Final revision — closing the last gaps.
Fixes:
1. fig_ablation_visual → 3-col: Baseline / +GAP / Full + residual insets
2. fig_consistency → visible row labels via fig.text
3. fig_efficiency_tradeoff → label position tweaks
"""
import os, sys, glob, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from PIL import Image

BASE = "/home/qyhu/Documents/r2_ours/PG2026"
FIGS_DIR = os.path.join(BASE, "figures")
os.makedirs(FIGS_DIR, exist_ok=True)


# ============ HELPERS ============

def _find_render(organ, method_key, view_idx):
    """Find a rendered projection image by iterating matching dirs."""
    dirs = sorted(glob.glob(f"{BASE}/output/*{organ}*3v*{method_key}*"))
    skip = ['opt_', 'adaptive', 'retry', 'pprune', 'spsv', 'gap_th']
    for d in dirs:
        dn = os.path.basename(d)
        if any(x in dn for x in skip):
            continue
        fp = f"{d}/test/iter_30000/render_test/{view_idx:05d}_pred.png"
        if os.path.exists(fp):
            img = np.array(Image.open(fp))
            if img.ndim == 3:
                img = img[:, :, 0]
            return img.astype(np.float32) / 255.0
    return None

def _find_render_by_dir(organ, method_key, view_idx, skip_prefixes=None):
    """Like _find_render but with configurable skip list for ablation dirs."""
    dirs = sorted(glob.glob(f"{BASE}/output/*{organ}*3v*{method_key}*"))
    skip = skip_prefixes or ['opt_', 'adaptive', 'retry', 'pprune', 'spsv', 'gap_th']
    for d in dirs:
        dn = os.path.basename(d)
        if any(x in dn for x in skip):
            continue
        fp = f"{d}/test/iter_30000/render_test/{view_idx:05d}_pred.png"
        if os.path.exists(fp):
            img = np.array(Image.open(fp))
            if img.ndim == 3:
                img = img[:, :, 0]
            return img.astype(np.float32) / 255.0
    return None

def _find_gt(organ, view_idx):
    """Find GT image."""
    dirs = sorted(glob.glob(f"{BASE}/output/*{organ}*3v*spags*"))
    for d in dirs:
        dn = os.path.basename(d)
        if any(x in dn for x in ['opt_', 'adaptive', 'retry', 'pprune', 'spsv', 'gap_th', 'adm_']):
            continue
        fp = f"{d}/test/iter_30000/render_test/{view_idx:05d}_gt.png"
        if os.path.exists(fp):
            img = np.array(Image.open(fp))
            if img.ndim == 3:
                img = img[:, :, 0]
            return img.astype(np.float32) / 255.0
    return None

def imshow_off(ax, img, vmin=0, vmax=1):
    if img is not None:
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=6,
                transform=ax.transAxes, color="gray")
        ax.set_facecolor("#f5f5f5")
    ax.axis("off")


# ================================================================
# 1. fig_ablation_visual — 3-col: Baseline / +GAP / Full + residuals
# ================================================================
def make_ablation():
    print("=== fig_ablation_visual (R4) ===")

    CONFIGS = [
        ("r2_gaussian", "Baseline"),
        ("gar_only", "+GAP"),
        ("spags", "Full"),
    ]

    # Two challenging cases from different organs
    CASES = [
        ("chest", 10, "Chest\nRib Border"),
        ("head", 25, "Head\nSkull Detail"),
    ]

    # ROI for each case in the full image
    ROIS = {
        "chest": (60, 170, 180, 270),   # rib border tight crop
        "head":  (30, 70, 150, 180),    # skull bone boundary
    }

    n_rows = len(CASES)
    n_cols = len(CONFIGS)

    fig = plt.figure(figsize=(7.1, 3.5))
    # 2 rows per case: main + residual
    gs = gridspec.GridSpec(n_rows * 2, n_cols + 1,
                           figure=fig,
                           width_ratios=[0.04] + [1]*n_cols,
                           wspace=0.04, hspace=0.15)

    for ci, (organ, view, cname) in enumerate(CASES):
        rx0, ry0, rx1, ry1 = ROIS[organ]
        gt_crop = None

        for col, (cfg_key, cfg_name) in enumerate(CONFIGS):
            # === Main crop row ===
            ax = fig.add_subplot(gs[ci * 2, col + 1])

            # Load render for this config
            if cfg_key in ("spags", "r2_gaussian"):
                pred = _find_render(organ, cfg_key, view)
            else:
                pred = _find_render_by_dir(organ, cfg_key, view, skip_prefixes=['opt_', 'pprune', 'retry', 'adaptive', 'spsv', 'gap_th'])

            if pred is not None:
                crop = pred[ry0:ry1, rx0:rx1]
                ax.imshow(crop, cmap="gray", vmin=0, vmax=1, aspect="auto")

                # Also load GT for residual
                gt = _find_gt(organ, view)
                if gt is not None and gt_crop is None:
                    gt_crop = gt[ry0:ry1, rx0:rx1]
            ax.axis("off")

            if ci == 0:
                ax.set_title(cfg_name, fontsize=7.5, pad=3, fontweight="bold")

            # Case label on far left
            if col == 0:
                ax_ylabel = fig.add_subplot(gs[ci * 2, 0])
                ax_ylabel.axis("off")
                ax_ylabel.text(0.5, 0.5, cname, fontsize=7.5, fontweight="bold",
                              ha="center", va="center", transform=ax_ylabel.transAxes)

            # === Residual row ===
            ax_res = fig.add_subplot(gs[ci * 2 + 1, col + 1])
            if pred is not None and gt is not None:
                gt = _find_gt(organ, view)
                if gt is not None:
                    gt_crop = gt[ry0:ry1, rx0:rx1]
                    residual = np.abs(crop - gt_crop)
                    # Normalize residual for visibility
                    res_vmax = max(residual.max(), 0.05)
                    ax_res.imshow(residual, cmap="hot", vmin=0, vmax=res_vmax, aspect="auto")
            ax_res.axis("off")

            if col == 0 and ci == 0:
                ax_res.text(-0.3, 0.5, "|pred−gt|", fontsize=6,
                          transform=ax_res.transAxes, ha="right", va="center",
                          rotation=90, color="gray")

    plt.savefig(f"{FIGS_DIR}/fig_ablation_visual.png", dpi=300,
                bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print("  ✅ fig_ablation_visual.png")


# ================================================================
# 2. fig_consistency — explicit row labels GT / R²-Gaussian / SPAGS
# ================================================================
def make_consistency():
    print("=== fig_consistency (R4) ===")

    organ = "chest"
    angles = [5, 25, 45]
    angle_labels = ["0°", "45°", "90°"]

    ROWS = [
        (None, "GT"),
        ("r2_gaussian", "R²-Gaussian"),
        ("spags", "SPAGS"),
    ]

    n_rows = len(ROWS)
    n_cols = len(angles)

    fig = plt.figure(figsize=(7.1, 4.8))
    gs = gridspec.GridSpec(n_rows, n_cols + 1,
                           figure=fig,
                           width_ratios=[0.1] + [1]*n_cols,
                           wspace=0.03, hspace=0.08)

    for row, (mkey, row_label) in enumerate(ROWS):
        for col, (angle, angle_label) in enumerate(zip(angles, angle_labels)):
            # Image cell: offset by 1 because column 0 is labels
            ax = fig.add_subplot(gs[row, col + 1])

            if mkey is None:
                img = _find_gt(organ, angle)
            else:
                img = _find_render(organ, mkey, angle)

            imshow_off(ax, img)

            # Column labels (top)
            if row == 0:
                ax.set_title(angle_label, fontsize=9, pad=4, fontweight="bold")

        # Row label using fig.text
        fig.text(0.01, 1.0 - (row + 0.5) / n_rows, row_label,
                fontsize=8, fontweight="bold", va="center", ha="left")

    plt.savefig(f"{FIGS_DIR}/fig_consistency.png", dpi=300,
                bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print("  ✅ fig_consistency.png")


# ================================================================
# 3. fig_efficiency_tradeoff — fine-tune labels, shrink legend
# ================================================================
def make_efficiency():
    print("=== fig_efficiency_tradeoff (R4) ===")

    data = [
        ("DN-Gaussian", 20.70, 85, 120000),
        ("CoR-GS", 22.03, 80, 110000),
        ("FSGS", 23.10, 75, 130000),
        ("X-Gaussian", 23.19, 70, 115000),
        ("R²-Gaussian", 27.83, 95, 105000),
        ("SPAGS", 28.22, 88, 90000),
    ]

    colors = ["#8f8f8f", "#a0a0a0", "#b0b0b0", "#c0c0c0", "#4a7bb5", "#c44e52"]

    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    for (method, psnr, fps, n_gauss), color in zip(data, colors):
        alpha = 0.9 if method == "SPAGS" else 0.45
        size = (n_gauss / 1000) * 15
        ax.scatter(fps, psnr, s=size, c=color, alpha=alpha,
                  edgecolors="white" if method == "SPAGS" else "none",
                  linewidths=0.5, zorder=5)

        # Manual label offsets — pushed further out to avoid overlap
        offsets = {
            "SPAGS": (-12, 5),          # left-down from default
            "R²-Gaussian": (10, -6),    # right-down
            "X-Gaussian": (12, 4),      # right-up
            "DN-Gaussian": (-14, 4),    # left-up
            "FSGS": (10, -6),           # right-down
            "CoR-GS": (-12, 5),         # left-up
        }
        off = offsets.get(method, (0, 0))
        ax.annotate(method, (fps, psnr), textcoords="offset points",
                   xytext=off, fontsize=5.5, ha="center",
                   arrowprops=dict(arrowstyle="-", color="gray", lw=0.3)
                   if off != (0, 0) else None)

    ax.set_xlabel("Inference Speed (FPS)", fontsize=8)
    ax.set_ylabel("Avg PSNR @ 3-view (dB)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Move legend to upper right (less crowded for left side)
    for n_gauss, label in [(90000, "90K"), (130000, "130K")]:
        ax.scatter([], [], s=(n_gauss/1000)*15, c="#bbbbbb",
                  alpha=0.6, edgecolors="none", label=f"{label} Gauss.")
    ax.legend(fontsize=5, loc="upper right", framealpha=0.7,
              handletextpad=0.3, markerscale=0.8)

    plt.tight_layout(pad=0.3)
    plt.savefig(f"{FIGS_DIR}/fig_efficiency_tradeoff.pdf", dpi=300,
                bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print("  ✅ fig_efficiency_tradeoff.pdf")


# ================================================================
# Main
# ================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Round 4: Final figure revisions")
    print("=" * 60)

    make_ablation()
    make_consistency()
    make_efficiency()

    print("\n✅ All R4 figures updated!")
    for f in sorted(os.listdir(FIGS_DIR)):
        fp = os.path.join(FIGS_DIR, f)
        if os.path.isfile(fp) and not f.startswith("draw_"):
            kb = os.path.getsize(fp) // 1024
            print(f"   {f} ({kb} KB)")
