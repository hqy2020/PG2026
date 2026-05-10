#!/usr/bin/env python3
"""
Round 3: Camera-ready figure revisions.
Fixes remaining issues after round 2 review.
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
    skip = ['opt_', 'adaptive', 'retry', 'pprune', 'spsv', 'gap_th', 'adm_', 'base']
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

def _find_checkpoint(organ, method_key):
    """Find checkpoint pickle and return data dict."""
    dirs = sorted(glob.glob(f"{BASE}/output/*{organ}*3v*{method_key}*"))
    skip = ['opt_', 'adaptive', 'retry', 'pprune', 'spsv', 'gap_th', 'adm_', 'base']
    for d in dirs:
        dn = os.path.basename(d)
        if any(x in dn for x in skip):
            continue
        ckpt = f"{d}/point_cloud/iteration_30000/point_cloud.pickle"
        if os.path.exists(ckpt):
            with open(ckpt, 'rb') as f:
                return pickle.load(f)
    return None

def imshow_off(ax, img, vmin=0, vmax=1):
    if img is not None:
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=6, transform=ax.transAxes, color="gray")
        ax.set_facecolor("#f5f5f5")
    ax.axis("off")


# ================================================================
# 1. fig_qual_main — add scene labels, 3-view note, input labels
# ================================================================
def make_qual_main():
    print("=== fig_qual_main (R3) ===")
    
    ORGANS = ["chest", "head", "pancreas"]
    DISP = ["Chest", "Head", "Pancreas"]
    VIEWS = [10, 25, 35]
    INPUT_VIEWS = {"chest": [0, 16, 33], "head": [0, 16, 33], "pancreas": [0, 16, 33]}
    
    METHODS = [
        ("dngaussian", "DN-Gaussian"),
        ("corgs", "CoR-GS"),
        ("fsgs", "FSGS"),
        ("xgaussian", "X-Gaussian"),
        ("r2_gaussian", "R²-Gaussian"),
        ("spags", "SPAGS"),
    ]
    
    n_cols = 1 + 1 + len(METHODS)  # Input + GT + 6 methods
    fig = plt.figure(figsize=(7.5, 4.5))
    gs = gridspec.GridSpec(len(ORGANS), n_cols, figure=fig, wspace=0.02, hspace=0.08)
    
    for row, (organ, disp, view) in enumerate(zip(ORGANS, DISP, VIEWS)):
        # === Input montage column ===
        ax = fig.add_subplot(gs[row, 0])
        invs = INPUT_VIEWS[organ]
        montages = [_find_gt(organ, iv) for iv in invs]
        valid_m = [m for m in montages if m is not None]
        if valid_m:
            h, w = valid_m[0].shape
            pad = 2
            total_h = h * len(valid_m) + pad * (len(valid_m) - 1)
            big = np.ones((total_h, w))
            for i, m in enumerate(valid_m):
                y0 = i * (h + pad)
                big[y0:y0+h, :] = m
            imshow_off(ax, big)
        else:
            imshow_off(ax, None)
        if row == 0:
            ax.set_title("Input Views", fontsize=7, pad=2)
        # Scene label on far left of this column
        ax.text(-0.15, 0.5, disp, transform=ax.transAxes,
               fontsize=8, fontweight="bold", va="center", ha="right",
               rotation=90)
        
        # === GT column ===
        ax = fig.add_subplot(gs[row, 1])
        gt = _find_gt(organ, view)
        imshow_off(ax, gt)
        if row == 0:
            ax.set_title("GT", fontsize=7, pad=2)
        
        # === Method columns ===
        for col, (mkey, mname) in enumerate(METHODS):
            ax = fig.add_subplot(gs[row, col + 2])
            pred = _find_render(organ, mkey, view)
            imshow_off(ax, pred)
            if row == 0:
                ax.set_title(mname, fontsize=6.5, pad=2)
    
    # 3-view note in top-right
    fig.text(0.99, 0.97, "3-view setting", fontsize=7, ha="right", va="top",
            style="italic", color="gray")
    
    plt.savefig(f"{FIGS_DIR}/fig_qual_main.png", dpi=300,
                bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print("  ✅ fig_qual_main.png")


# ================================================================
# 2. fig_qual_zoom — add Case A/B labels, ROI labels
# ================================================================
def make_qual_zoom():
    print("=== fig_qual_zoom (R3) ===")
    
    METHODS = [("spags", "SPAGS"), ("r2_gaussian", "R²-G"), ("fsgs", "FSGS"), ("corgs", "CoR-GS")]
    COL_NAMES = ["GT"] + [m[1] for m in METHODS]
    
    CASES = [
        ("chest", 10, 60, 170, 190, 280, "Case A\nRib Border"),
        ("head", 25, 30, 70, 150, 180, "Case B\nBone Detail"),
    ]
    
    n_rows_full = len(CASES) * 2  # full + zoom per case
    n_cols = 1 + len(METHODS)
    
    fig = plt.figure(figsize=(7.1, 4.2))
    gs = gridspec.GridSpec(n_rows_full, n_cols, figure=fig, wspace=0.03, hspace=0.25)
    
    for ci, (organ, view, rx0, ry0, rx1, ry1, cname) in enumerate(CASES):
        for col, mkey in enumerate([None] + [m[0] for m in METHODS]):
            # Full image row
            ax = fig.add_subplot(gs[ci * 2, col])
            img = _find_gt(organ, view) if mkey is None else _find_render(organ, mkey, view)
            if img is not None:
                ax.imshow(img, cmap="gray", vmin=0, vmax=1, aspect="auto")
                rect = Rectangle((rx0, ry0), rx1-rx0, ry1-ry0,
                               linewidth=1.0, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
            ax.axis("off")
            if ci == 0:
                ax.set_title(COL_NAMES[col], fontsize=7, pad=2)
            if col == 0:
                ax.set_ylabel(cname, fontsize=6.5, labelpad=2)
            
            # Zoom row
            ax = fig.add_subplot(gs[ci * 2 + 1, col])
            if img is not None:
                zoom = img[ry0:ry1, rx0:rx1]
                ax.imshow(zoom, cmap="gray", vmin=0, vmax=1, aspect="auto")
            # Add ROI label in corner
            ax.text(0.03, 0.97, f"ROI-{ci+1}", transform=ax.transAxes,
                   fontsize=5, color='r', va='top', ha='left')
            ax.axis("off")
    
    plt.savefig(f"{FIGS_DIR}/fig_qual_zoom.png", dpi=300,
                bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print("  ✅ fig_qual_zoom.png")


# ================================================================
# 3. fig_ablation_visual — add residual insets, choose harder ROI
# ================================================================
def make_ablation():
    print("=== fig_ablation_visual (R3) ===")
    
    CONFIGS = [
        ("r2_gaussian", "Baseline"),
        ("sps_only", "+SPS"),
        ("gar_only", "+GAP"),
        ("adm_only", "+ADM"),
        ("spags", "Full"),
    ]
    
    # Harder ROI: chest bone boundary area
    organ = "chest"
    roi_view = 10
    rx0, ry0, rx1, ry1 = 60, 170, 180, 270  # tight crop on rib border
    
    n_cases = 2
    n_cols = len(CONFIGS)
    
    fig = plt.figure(figsize=(7.1, 3.8))
    gs_main = gridspec.GridSpec(n_cases, n_cols, figure=fig, wspace=0.03, hspace=0.10)
    
    for case_idx in range(n_cases):
        # Use different views for variety
        v = [10, 25][case_idx]
        
        for col, (cfg_key, cfg_name) in enumerate(CONFIGS):
            ax = fig.add_subplot(gs_main[case_idx, col])
            
            # Load prediction
            pred = None
            if cfg_key in ("spags", "r2_gaussian"):
                pred = _find_render(organ, cfg_key, v)
            else:
                dirs = sorted(glob.glob(f"{BASE}/output/*{organ}*3v*{cfg_key}*"))
                for d_ab in dirs:
                    dn = os.path.basename(d_ab)
                    if any(x in dn for x in ['opt_', 'pprune', 'retry', 'adaptive', 'spsv', 'gap_th', 'adm_']):
                        continue
                    fp = f"{d_ab}/test/iter_30000/render_test/{v:05d}_pred.png"
                    if os.path.exists(fp):
                        img = np.array(Image.open(fp))
                        if img.ndim == 3:
                            img = img[:, :, 0]
                        pred = img.astype(np.float32) / 255.0
                        break
            
            if pred is not None:
                # Crop to ROI for better visibility
                crop = pred[ry0:ry1, rx0:rx1]
                ax.imshow(crop, cmap="gray", vmin=0, vmax=1, aspect="auto")
            ax.axis("off")
            
            if case_idx == 0:
                ax.set_title(cfg_name, fontsize=7, pad=2)
    
    plt.savefig(f"{FIGS_DIR}/fig_ablation_visual.png", dpi=300,
                bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print("  ✅ fig_ablation_visual.png")


# ================================================================
# 4. fig_efficiency_tradeoff — fix label overlaps, legend placement
# ================================================================
def make_efficiency():
    print("=== fig_efficiency_tradeoff (R3) ===")
    
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
        alpha = 0.9 if method == "SPAGS" else 0.5
        size = (n_gauss / 1000) * 15
        ax.scatter(fps, psnr, s=size, c=color, alpha=alpha,
                  edgecolors="white" if method == "SPAGS" else "none",
                  linewidths=0.5, zorder=5)
        
        # Manual label offsets to avoid overlap
        offsets = {
            "SPAGS": (8, 3),
            "R²-Gaussian": (0, -10),
            "X-Gaussian": (10, 0),
            "DN-Gaussian": (-10, 0),
            "FSGS": (8, 0),
            "CoR-GS": (-8, 0),
        }
        off = offsets.get(method, (0, 0))
        ax.annotate(method, (fps, psnr), textcoords="offset points",
                   xytext=off, fontsize=5.5, ha="center",
                   arrowprops=dict(arrowstyle="-", color="gray", lw=0.3) if off != (0,0) else None)
    
    ax.set_xlabel("Inference Speed (FPS)", fontsize=8)
    ax.set_ylabel("Avg PSNR @ 3-view (dB)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3, linestyle="--")
    
    # Legend in upper left corner (less crowded)
    for n_gauss, label in [(90000, "90K"), (130000, "130K")]:
        ax.scatter([], [], s=(n_gauss/1000)*15, c="#bbbbbb",
                  alpha=0.6, edgecolors="none", label=f"{label} Gauss.")
    ax.legend(fontsize=5, loc="upper left", framealpha=0.7, handletextpad=0.3)
    
    plt.tight_layout(pad=0.3)
    plt.savefig(f"{FIGS_DIR}/fig_efficiency_tradeoff.pdf", dpi=300,
                bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print("  ✅ fig_efficiency_tradeoff.pdf")


# ================================================================
# 5. fig_spatial_distribution — fix middle columns with real data
# Use ablation configs: R2=Uniform, sps_only=SPS, sps_gar=SPS+GAP, spags=Final
# ================================================================
def make_spatial():
    print("=== fig_spatial_distribution (R3) ===")
    
    organ = "chest"
    view_idx = 10
    
    # Map stages to configs for both structure and render
    stages = [
        ("r2_gaussian", "Baseline\n(Uniform Init)"),
        ("sps_only", "After SPS\nInit"),
        ("sps_gar", "After GAP\nPruning"),
        ("spags", "Final\nSPAGS"),
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(7.1, 3.5))
    
    for col, (cfg, title) in enumerate(stages):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]
        
        # === Top: Gaussian structure scatter ===
        ckpt = _find_checkpoint(organ, cfg)
        if ckpt is not None:
            xyz = ckpt.get('xyz')
            density = ckpt.get('density')
            if xyz is not None:
                xy = xyz.cpu().numpy() if hasattr(xyz, 'cpu') else xyz[:, :2]
                if density is not None:
                    d = density.cpu().numpy().ravel() if hasattr(density, 'cpu') else density.ravel()
                else:
                    d = None
                
                # Subsample for clarity
                n_max = 5000
                if len(xy) > n_max:
                    idx = np.random.choice(len(xy), n_max, replace=False)
                    xy = xy[idx]
                    d = d[idx] if d is not None else None
                
                if d is not None:
                    d_norm = (d - d.min()) / (d.max() - d.min() + 1e-8)
                    ax_top.scatter(xy[:, 0], xy[:, 1], c=d_norm, cmap="viridis",
                                  s=1.5, alpha=0.35, rasterized=True)
                else:
                    ax_top.scatter(xy[:, 0], xy[:, 1], c="#c44e52",
                                  s=1.5, alpha=0.35, rasterized=True)
                
                ax_top.set_xlabel(f"{len(xy):,} pts", fontsize=5.5)
        else:
            ax_top.text(0.5, 0.5, "No data", ha="center", va="center", 
                       transform=ax_top.transAxes, fontsize=7, color="gray")
        
        ax_top.set_title(title, fontsize=6.5)
        ax_top.axis("equal")
        ax_top.axis("off")
        
        # === Bottom: rendered projection ===
        pred = _find_render(organ, cfg, view_idx)
        imshow_off(ax_bot, pred)
    
    plt.subplots_adjust(wspace=0.05, hspace=0.08)
    plt.savefig(f"{FIGS_DIR}/fig_spatial_distribution.png", dpi=300,
                bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print("  ✅ fig_spatial_distribution.png")


# ================================================================
# 6. fig_consistency — explicit row/col labels
# ================================================================
def make_consistency():
    print("=== fig_consistency (R3) ===")
    
    organ = "chest"
    angles = [5, 25, 45]
    angle_labels = ["0°", "45°", "90°"]
    
    METHODS = [("r2_gaussian", "R²-Gaussian"), ("spags", "SPAGS")]
    ROW_NAMES = ["GT"] + [m[1] for m in METHODS]
    
    n_rows = 3
    n_cols = 3
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.1, 4.5))
    
    for col, (angle, angle_label) in enumerate(zip(angles, angle_labels)):
        for row, (mkey_or_none) in enumerate([None, "r2_gaussian", "spags"]):
            ax = axes[row, col]
            
            if mkey_or_none is None:
                img = _find_gt(organ, angle)
                label = "GT"
            else:
                img = _find_render(organ, mkey_or_none, angle)
                label = {"r2_gaussian": "R²-Gaussian", "spags": "SPAGS"}[mkey_or_none]
            
            imshow_off(ax, img)
            
            # Column labels (top)
            if row == 0:
                ax.set_title(angle_label, fontsize=8, pad=3, fontweight="bold")
            
            # Row labels (left)
            if col == 0:
                ax.set_ylabel(label, fontsize=7.5, labelpad=3, fontweight="bold")
    
    plt.subplots_adjust(wspace=0.02, hspace=0.06)
    plt.savefig(f"{FIGS_DIR}/fig_consistency.png", dpi=300,
                bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print("  ✅ fig_consistency.png")


# ================================================================
# 7. fig_convergence — reduce title font size
# ================================================================
def make_convergence():
    print("=== fig_convergence (R3) ===")
    
    def extract_curve(base_dir):
        iters, psnrs = [], []
        eval_dir = f"{base_dir}/eval"
        if os.path.isdir(eval_dir):
            for d in sorted(os.listdir(eval_dir)):
                if d.startswith("iter_"):
                    try:
                        iter_n = int(d.split("_")[1])
                    except:
                        try:
                            iter_n = int(d.split("_")[-1])
                        except:
                            continue
                    yml_path = f"{eval_dir}/{d}/eval2d_render_test.yml"
                    if os.path.exists(yml_path):
                        with open(yml_path) as f:
                            for line in f:
                                if line.startswith("psnr_2d:"):
                                    psnr = float(line.split(":")[1])
                                    iters.append(iter_n)
                                    psnrs.append(psnr)
                                    break
        return iters, psnrs
    
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.8))
    
    for ax, (title_text, xlabel) in zip(axes, [
        ("PSNR vs Iteration", "Iteration"),
        ("PSNR vs Time", "Time (seconds)"),
    ]):
        ax.set_title(title_text, fontsize=7)
        ax.set_xlabel(xlabel, fontsize=7)
        ax.set_ylabel("PSNR (dB)", fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.grid(True, alpha=0.3, linestyle="--")
    
    # SPAGS
    si, sp = extract_curve(f"{BASE}/output/2026_04_30_chest_3views_spags")
    if si:
        p = sorted(zip(si, sp))
        axes[0].plot([x[0] for x in p], [x[1] for x in p], '-o',
                    label="SPAGS", color="#c44e52", markersize=2.5, linewidth=1.0)
        axes[1].plot([x[0]/30000*1680 for x in p], [x[1] for x in p], '-o',
                    label="SPAGS", color="#c44e52", markersize=2.5, linewidth=1.0)
    
    # R2
    ri, rp = extract_curve(f"{BASE}/output/2026_04_30_chest_3views_r2_gaussian")
    if ri:
        p = sorted(zip(ri, rp))
        axes[0].plot([x[0] for x in p], [x[1] for x in p], '-o',
                    label="R²-Gaussian", color="#4a7bb5", markersize=2.5, linewidth=1.0)
        axes[1].plot([x[0]/30000*1560 for x in p], [x[1] for x in p], '-o',
                    label="R²-Gaussian", color="#4a7bb5", markersize=2.5, linewidth=1.0)
    
    axes[0].legend(fontsize=5.5)
    axes[1].legend(fontsize=5.5)
    
    plt.tight_layout(pad=0.3)
    plt.savefig(f"{FIGS_DIR}/fig_convergence.pdf", dpi=300,
                bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print("  ✅ fig_convergence.pdf")


# ================================================================
# Main
# ================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Round 3: Camera-ready figure revisions")
    print("=" * 60)
    
    make_qual_main()
    make_qual_zoom()
    make_ablation()
    make_efficiency()
    make_spatial()
    make_consistency()
    make_convergence()
    
    print("\n✅ All figures updated!")
    for f in sorted(os.listdir(FIGS_DIR)):
        if f.startswith(("fig_qual_main", "fig_qual_zoom", "fig_ablation", "fig_efficiency",
                        "fig_spatial", "fig_consistency", "fig_convergence", "fig_gaussian",
                        "fig_hparam")):
            kb = os.path.getsize(os.path.join(FIGS_DIR, f)) // 1024
            print(f"   {f} ({kb} KB)")
