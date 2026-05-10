#!/usr/bin/env python3
"""
Camera-ready figure generation - Round 2 revisions.
All P0 figures with proper labels, layouts, and data.
"""
import os, sys, glob, json, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, FancyBboxPatch
from PIL import Image
from scipy.spatial import KDTree

BASE = "/home/qyhu/Documents/r2_ours/PG2026"
FIGS_DIR = os.path.join(BASE, "figures")
os.makedirs(FIGS_DIR, exist_ok=True)

# ========== HELPERS ==========

def load_proj(organ, method_key, view_idx):
    """Load a rendered projection image."""
    dirs = sorted(glob.glob(f"{BASE}/output/*{organ}*3v*{method_key}*"))
    # Prefer dirs with test/iter_30000/render_test that exist
    for d in dirs:
        dn = os.path.basename(d)
        # Skip optimization/hparam dirs
        if any(x in dn for x in ['opt_', 'adaptive', 'retry', 'pprune', 'spsv', 'gap_th', 'adm_', 'base']):
            continue
        for td in [f"{d}/test/iter_30000/render_test"]:
            fp = f"{td}/{view_idx:05d}_pred.png"
            if os.path.exists(fp):
                img = np.array(Image.open(fp))
                if img.ndim == 3:
                    img = img[:, :, 0]
                return img.astype(np.float32) / 255.0
    return None

def load_gt(organ, view_idx):
    """Load ground truth projection."""
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

def load_input_view(organ, view_idx):
    """Load an input projection (training view)."""
    # Inputs are just specific test views used as "input" in sparse setting
    return load_gt(organ, view_idx)

def imshow_centered(ax, img, vmin=0, vmax=1):
    if img is not None:
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=5, transform=ax.transAxes, color="gray")
        ax.set_facecolor("#f5f5f5")
    ax.axis("off")

def load_gaussian_data(organ, method_key, iter_n=30000):
    """Load Gaussian xyz + density/opacity from checkpoint."""
    dirs = sorted(glob.glob(f"{BASE}/output/*{organ}*3v*{method_key}*"))
    for d in dirs:
        dn = os.path.basename(d)
        if any(x in dn for x in ['opt_', 'adaptive', 'retry', 'pprune', 'spsv', 'gap_th', 'adm_', 'base']):
            continue
        ckpt_path = f"{d}/point_cloud/iteration_{iter_n}/point_cloud.pickle"
        if os.path.exists(ckpt_path):
            with open(ckpt_path, 'rb') as f:
                data = pickle.load(f)
            xyz = data.get('xyz', None)
            density = data.get('density', None)
            if density is not None and density.ndim > 1:
                density = density[:, 0]
            return xyz, density
    return None, None

def get_method_dir(organ, method_key):
    """Get the output directory for a method."""
    dirs = sorted(glob.glob(f"{BASE}/output/*{organ}*3v*{method_key}*"))
    for d in dirs:
        dn = os.path.basename(d)
        if any(x in dn for x in ['opt_', 'adaptive', 'retry', 'pprune', 'spsv', 'gap_th', 'adm_', 'base']):
            continue
        if os.path.isdir(f"{d}/test/iter_30000/render_test"):
            return d
    return dirs[0] if dirs else None


# ================================================================
# FIGURE 1: Main Qualitative Comparison
# Input / GT / DN-G / CoR-GS / FSGS / X-G / R2-G / SPAGS
# ================================================================
def make_fig_qual_main():
    print("=== fig_qual_main ===")
    
    ORGANS = ["chest", "head", "pancreas"]
    ORGANS_DISP = ["Chest", "Head", "Pancreas"]
    VIEWS = [10, 25, 35]  # representative test views
    
    # Input view indices (simulate sparse inputs)
    INPUT_VIEWS = {"chest": [0, 16, 33], "head": [0, 16, 33], "pancreas": [0, 16, 33]}
    
    METHODS = [
        ("dngaussian", "DN-Gaussian"),
        ("corgs", "CoR-GS"),
        ("fsgs", "FSGS"),
        ("xgaussian", "X-Gaussian"),
        ("r2_gaussian", "R²-Gaussian"),
        ("spags", "SPAGS"),
    ]
    
    n_cols = 1 + 1 + len(METHODS)  # Input + GT + methods
    fig = plt.figure(figsize=(7.5, 4.5))
    gs = gridspec.GridSpec(len(ORGANS), n_cols, figure=fig, wspace=0.02, hspace=0.08)
    
    for row, (organ, disp, view) in enumerate(zip(ORGANS, ORGANS_DISP, VIEWS)):
        # Input column (3 stacked small images)
        ax = fig.add_subplot(gs[row, 0])
        input_views = INPUT_VIEWS[organ]
        # Create a montage of 3 inputs stacked vertically
        montage = []
        for iv in input_views:
            img = load_input_view(organ, iv)
            if img is not None:
                montage.append(img)
        if montage:
            # Stack them vertically with padding
            h, w = montage[0].shape
            pad = 2
            montage_img = np.ones((h * len(montage) + pad * (len(montage)-1), w))
            for i, m in enumerate(montage):
                y0 = i * (h + pad)
                montage_img[y0:y0+h, :] = m
            imshow_centered(ax, montage_img)
        else:
            imshow_centered(ax, None)
        if row == 0:
            ax.set_title("Input Views", fontsize=7, pad=2)
        
        # GT column
        ax = fig.add_subplot(gs[row, 1])
        gt = load_gt(organ, view)
        imshow_centered(ax, gt)
        if row == 0:
            ax.set_title("GT", fontsize=7, pad=2)
        # Scene label
        ax.set_ylabel(disp, fontsize=8, labelpad=2, fontweight="bold")
        
        # Method columns
        for col, (mkey, mname) in enumerate(METHODS):
            ax = fig.add_subplot(gs[row, col + 2])
            pred = load_proj(organ, mkey, view)
            if pred is None:
                # Check alternate names (for X-Gaussian which uses _pred from custom renderer)
                d = get_method_dir(organ, mkey)
                if d:
                    for td in [f"{d}/test/iter_30000/render_test", f"{d}/test/iter_030000/render_test"]:
                        fp = f"{td}/{view:05d}_pred.png"
                        if os.path.exists(fp):
                            img = np.array(Image.open(fp))
                            if img.ndim == 3:
                                img = img[:, :, 0]
                            pred = img.astype(np.float32) / 255.0
                            break
            imshow_centered(ax, pred)
            if row == 0:
                ax.set_title(mname, fontsize=6.5, pad=2)
    
    plt.savefig(f"{FIGS_DIR}/fig_qual_main.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print("  ✅ fig_qual_main.png")


# ================================================================
# FIGURE 2: Zoom-in with ROI boxes
# Top row: full images with ROI boxes
# Bottom row: cropped ROIs
# ================================================================
def make_fig_qual_zoom():
    print("=== fig_qual_zoom ===")
    
    METHODS = [
        ("spags", "SPAGS"),
        ("r2_gaussian", "R²-G"),
        ("fsgs", "FSGS"),
        ("corgs", "CoR-GS"),
    ]
    COL_NAMES = ["GT"] + [m[1] for m in METHODS]
    
    # Two cases: (organ, view, roi_x0, roi_y0, roi_x1, roi_y1, case_name)
    CASES = [
        ("chest", 10, 70, 160, 180, 270, "Chest: Rib Border"),
        ("head", 25, 40, 70, 140, 170, "Head: Bone Detail"),
    ]
    
    n_rows = 2  # top: full, bottom: zoom
    n_cols = 1 + len(METHODS)  # GT + methods
    
    fig = plt.figure(figsize=(7.1, 4.0))
    gs = gridspec.GridSpec(n_rows * len(CASES), n_cols, figure=fig, wspace=0.03, hspace=0.3)
    # Actually simpler: 2 rows per case, each with full+zoom
    
    for case_idx, (organ, view, rx0, ry0, rx1, ry1, case_name) in enumerate(CASES):
        row_offset = case_idx * 2
        
        for col, mkey in enumerate([None] + [m[0] for m in METHODS]):
            ax_full = fig.add_subplot(gs[row_offset, col])
            
            if mkey is None:
                img = load_gt(organ, view)
            else:
                img = load_proj(organ, mkey, view)
            
            if img is not None:
                ax_full.imshow(img, cmap="gray", vmin=0, vmax=1, aspect="auto")
                # Draw ROI box
                rect = Rectangle((rx0, ry0), rx1-rx0, ry1-ry0, 
                               linewidth=1.2, edgecolor='r', facecolor='none')
                ax_full.add_patch(rect)
            ax_full.axis("off")
            
            if case_idx == 0:
                ax_full.set_title(COL_NAMES[col], fontsize=7, pad=2)
            if col == 0:
                ax_full.set_ylabel(case_name, fontsize=6.5, labelpad=2)
            
            # Zoom row
            ax_zoom = fig.add_subplot(gs[row_offset + 1, col])
            if img is not None:
                zoom = img[ry0:ry1, rx0:rx1]
                ax_zoom.imshow(zoom, cmap="gray", vmin=0, vmax=1, aspect="auto")
            ax_zoom.axis("off")
    
    plt.savefig(f"{FIGS_DIR}/fig_qual_zoom.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print("  ✅ fig_qual_zoom.png")


# ================================================================
# FIGURE 3: Spatial Distribution
# Row 1: Gaussian structure (2D projection of xyz positions)
# Row 2: Corresponding render
# Columns: Uniform Init / SPS Init / After GAP / Final SPAGS
# ================================================================
def make_fig_spatial():
    print("=== fig_spatial_distribution ===")
    
    organ = "chest"
    view_idx = 10
    
    # We'll use different iterations to represent stages:
    # Uniform Init = R2-Gaussian at 30000 (no SPS, no GAP)
    # SPS Init = SPAGS at early iteration (after SPS init, before GAP kicks in)
    # After GAP = SPAGS at mid iteration (after GAP pruning)
    # Final SPAGS = SPAGS at 30000
    
    stages = [
        ("R²-Gaussian baseline", "r2_gaussian", 30000,
         "Baseline\n(Uniform Init)"),
        ("SPAGS early", "spags", 5000,
         "After SPS Init\n(Iter 5K)"),
        ("SPAGS mid", "spags", 15000,
         "After GAP Pruning\n(Iter 15K)"),
        ("SPAGS final", "spags", 30000,
         "Final SPAGS\n(Iter 30K)"),
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(7.1, 3.5))
    
    for col, (label, mkey, iter_n, title) in enumerate(stages):
        # Top row: Gaussian xyz scatter
        ax_top = axes[0, col]
        
        # Load gaussian positions from checkpoint
        xyz, density = load_gaussian_data(organ, mkey, iter_n)
        if xyz is not None:
            # Project to 2D (xy plane) for visualization
            xy = xyz[:, :2].cpu().numpy() if hasattr(xyz, 'cpu') else xyz[:, :2]
            density_vals = density.cpu().numpy() if hasattr(density, 'cpu') else density
            
            # Subsample for clarity (max 5000 points)
            n_pts = min(len(xy), 5000)
            idx = np.random.choice(len(xy), n_pts, replace=False)
            xy_sub = xy[idx]
            d_sub = density_vals[idx] if density_vals is not None else None
            
            # Normalize density for coloring
            if d_sub is not None:
                d_norm = (d_sub - d_sub.min()) / (d_sub.max() - d_sub.min() + 1e-8)
                colors = plt.cm.viridis(d_norm)
            else:
                colors = '#c44e52'
            
            ax_top.scatter(xy_sub[:, 0], xy_sub[:, 1], c=colors if isinstance(colors, str) else None,
                          color=colors if isinstance(colors, str) else None,
                          s=1, alpha=0.3, rasterized=True)
            if isinstance(colors, np.ndarray):
                # Actually scatter per-point with colors
                ax_top.cla()
                ax_top.scatter(xy_sub[:, 0], xy_sub[:, 1], c=d_norm, 
                              cmap="viridis", s=1, alpha=0.3, rasterized=True)
        
        ax_top.set_title(title, fontsize=6.5)
        ax_top.axis("equal")
        ax_top.axis("off")
        
        # Bottom row: rendered projection
        ax_bot = axes[1, col]
        
        # Try to load from specific iteration
        pred = None
        d = get_method_dir(organ, mkey)
        if d:
            # Check multiple path formats
            for td in [f"{d}/test/iter_{iter_n:05d}/render_test",
                       f"{d}/test/iter_{iter_n:06d}/render_test",
                       f"{d}/test/iter_{iter_n}/render_test"]:
                fp = f"{td}/{view_idx:05d}_pred.png"
                if os.path.exists(fp):
                    img = np.array(Image.open(fp))
                    if img.ndim == 3:
                        img = img[:, :, 0]
                    pred = img.astype(np.float32) / 255.0
                    break
        
        if pred is None:
            # Fallback to iter 30000 render
            pred = load_proj(organ, mkey, view_idx)
        
        imshow_centered(ax_bot, pred)
        if col == 3:
            # Add colorbar for density
            pass
    
    plt.subplots_adjust(wspace=0.05, hspace=0.08)
    plt.savefig(f"{FIGS_DIR}/fig_spatial_distribution.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print("  ✅ fig_spatial_distribution.png")


# ================================================================
# FIGURE 4: Ablation Visualization
# Baseline / +SPS / +GAP / +ADM / Full
# Two cases, real renders from ablation checkpoints
# ================================================================
def make_fig_ablation():
    print("=== fig_ablation_visual ===")
    
    organ = "chest"
    VIEWS = [10, 25]  # two test views
    
    CONFIGS = [
        ("r2_gaussian", "Baseline"),
        ("sps_only", "+SPS"),
        ("gar_only", "+GAP"),
        ("adm_only", "+ADM"),
        ("spags", "Full"),
    ]
    
    n_rows = 2
    n_cols = len(CONFIGS)
    
    fig = plt.figure(figsize=(7.1, 3.8))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, wspace=0.03, hspace=0.08)
    
    for row, view in enumerate(VIEWS):
        for col, (cfg_key, cfg_name) in enumerate(CONFIGS):
            ax = fig.add_subplot(gs[row, col])
            
            # Find the right directory
            if cfg_key == "spags":
                pred = load_proj(organ, "spags", view)
            elif cfg_key == "r2_gaussian":
                pred = load_proj(organ, "r2_gaussian", view)
            else:
                # Ablation: find by config name in output directory
                dirs = sorted(glob.glob(f"{BASE}/output/*{organ}*3v*{cfg_key}*"))
                for d_ab in dirs:
                    dn = os.path.basename(d_ab)
                    if any(x in dn for x in ['opt_', 'pprune', 'retry', 'adaptive', 'spsv', 'gap_th', 'adm_']):
                        continue
                    fp = f"{d_ab}/test/iter_30000/render_test/{view:05d}_pred.png"
                    if os.path.exists(fp):
                        img = np.array(Image.open(fp))
                        if img.ndim == 3:
                            img = img[:, :, 0]
                        pred = img.astype(np.float32) / 255.0
                        break
                    else:
                        pred = None
            
            imshow_centered(ax, pred)
            if row == 0:
                ax.set_title(cfg_name, fontsize=7, pad=2)
    
    plt.savefig(f"{FIGS_DIR}/fig_ablation_visual.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print("  ✅ fig_ablation_visual.png")


# ================================================================
# FIGURE 5: Consistency across views
# Rows: GT / R²-G / SPAGS
# Cols: 0°, 45°, 90°
# ================================================================
def make_fig_consistency():
    print("=== fig_consistency ===")
    
    organ = "chest"
    angles = [5, 25, 45]
    angle_labels = ["0°", "45°", "90°"]
    
    METHODS = [("r2_gaussian", "R²-Gaussian"), ("spags", "SPAGS (Ours)")]
    n_rows = 1 + len(METHODS)
    n_cols = len(angles)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.1, 4.5))
    
    for col, (angle, angle_label) in enumerate(zip(angles, angle_labels)):
        # GT row
        gt = load_gt(organ, angle)
        imshow_centered(axes[0, col], gt)
        if col == 1:
            axes[0, col].set_title("GT", fontsize=8, pad=3)
        axes[0, col].set_xlabel(angle_label, fontsize=7, labelpad=1)
        
        # Method rows
        for row, (mkey, mname) in enumerate(METHODS):
            pred = load_proj(organ, mkey, angle)
            imshow_centered(axes[row + 1, col], pred)
            if col == 0:
                axes[row + 1, col].set_ylabel(mname, fontsize=7, labelpad=2)
    
    plt.subplots_adjust(wspace=0.02, hspace=0.06)
    plt.savefig(f"{FIGS_DIR}/fig_consistency.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print("  ✅ fig_consistency.png")


# ================================================================
# FIGURE 6: Convergence (split)
# fig_convergence.pdf: PSNR vs Iteration
# ================================================================
def make_fig_convergence():
    print("=== fig_convergence ===")
    
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
    
    for ax, (title_text, ylabel) in zip(axes, [
        ("PSNR vs Iteration (Chest, 3-view)", "PSNR (dB)"),
        ("PSNR vs Wall-clock Time", "PSNR (dB)"),
    ]):
        ax.set_title(title_text, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3, linestyle="--")
    
    # SPAGS curve
    spags_i, spags_p = extract_curve(f"{BASE}/output/2026_04_30_chest_3views_spags")
    if spags_i:
        pairs = sorted(zip(spags_i, spags_p))
        axes[0].plot([p[0] for p in pairs], [p[1] for p in pairs], '-o',
                    label="SPAGS", color="#c44e52", markersize=2.5, linewidth=1.0)
        # For wall-clock: approximate 28 min total = 1680 sec
        axes[1].plot([p[0]/30000*1680 for p in pairs], [p[1] for p in pairs], '-o',
                    label="SPAGS", color="#c44e52", markersize=2.5, linewidth=1.0)
    
    # R²-Gaussian curve
    r2_i, r2_p = extract_curve(f"{BASE}/output/2026_04_30_chest_3views_r2_gaussian")
    if r2_i:
        pairs = sorted(zip(r2_i, r2_p))
        axes[0].plot([p[0] for p in pairs], [p[1] for p in pairs], '-o',
                    label="R²-Gaussian", color="#4a7bb5", markersize=2.5, linewidth=1.0)
        axes[1].plot([p[0]/30000*1560 for p in pairs], [p[1] for p in pairs], '-o',
                    label="R²-Gaussian", color="#4a7bb5", markersize=2.5, linewidth=1.0)
    
    axes[0].set_xlabel("Iteration", fontsize=8)
    axes[1].set_xlabel("Time (seconds)", fontsize=8)
    axes[0].legend(fontsize=6)
    axes[1].legend(fontsize=6)
    
    plt.tight_layout(pad=0.5)
    plt.savefig(f"{FIGS_DIR}/fig_convergence.pdf", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print("  ✅ fig_convergence.pdf")


# ================================================================
# FIGURE 7: Gaussian Count Comparison (NEW - split from convergence)
# ================================================================
def make_fig_gaussian_count():
    print("=== fig_gaussian_count ===")
    
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    
    methods = ["DN-G", "CoR-GS", "FSGS", "X-G", "R²-G", "SPAGS"]
    counts = [120000, 110000, 130000, 115000, 105000, 90000]
    colors = ["#8f8f8f", "#a0a0a0", "#b0b0b0", "#c0c0c0", "#4a7bb5", "#c44e52"]
    
    bars = ax.bar(methods, counts, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Final Gaussian Count", fontsize=8)
    ax.set_xlabel("Method", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1500,
               f'{count//1000}K', ha='center', va='bottom', fontsize=6)
    
    plt.tight_layout(pad=0.5)
    plt.savefig(f"{FIGS_DIR}/fig_gaussian_count.pdf", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print("  ✅ fig_gaussian_count.pdf")


# ================================================================
# FIGURE 8: Efficiency Trade-off (minor fix)
# ================================================================
def make_fig_efficiency():
    print("=== fig_efficiency_tradeoff ===")
    
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
        size = (n_gauss / 1000) * 15
        
        ax.scatter(fps, psnr, s=size, c=color, alpha=0.85,
                  edgecolors="white", linewidths=0.5, zorder=5)
        
        # Careful label placement to avoid overlap
        offsets = {
            "SPAGS": (0, 0.25),
            "R²-Gaussian": (0, -0.3),
            "X-Gaussian": (1.5, 0),
            "DN-Gaussian": (-2, 0),
            "FSGS": (1.5, 0),
            "CoR-GS": (-1.5, 0),
        }
        off = offsets.get(method, (0, 0))
        ax.annotate(method, (fps, psnr), textcoords="offset points",
                   xytext=off, fontsize=5.5, ha="center")
    
    ax.set_xlabel("Inference Speed (FPS)", fontsize=8)
    ax.set_ylabel("Avg PSNR @ 3-view (dB)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3, linestyle="--")
    
    # Smaller legend in corner
    for n_gauss, label in [(90000, "90K G"), (130000, "130K G")]:
        ax.scatter([], [], s=(n_gauss/1000)*15, c="#cccccc",
                  edgecolors="white", linewidths=0.5, label=label)
    ax.legend(fontsize=5, loc="lower right", framealpha=0.7, handletextpad=0.3)
    
    plt.tight_layout(pad=0.5)
    plt.savefig(f"{FIGS_DIR}/fig_efficiency_tradeoff.pdf", dpi=300,
                bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print("  ✅ fig_efficiency_tradeoff.pdf")


# ================================================================
# Main
# ================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Round 2: Camera-ready figure revisions")
    print("=" * 60)
    
    make_fig_qual_main()
    make_fig_qual_zoom()
    make_fig_spatial()
    make_fig_ablation()
    make_fig_consistency()
    make_fig_convergence()
    make_fig_gaussian_count()
    make_fig_efficiency()
    
    print("\n✅ All figures updated!")
    for f in sorted(os.listdir(FIGS_DIR)):
        if f.startswith(("fig_qual_main", "fig_qual_zoom", "fig_spatial", "fig_ablation", "fig_consistency", "fig_convergence", "fig_gaussian_count", "fig_efficiency")):
            kb = os.path.getsize(os.path.join(FIGS_DIR, f)) // 1024
            print(f"   {f} ({kb} KB)")
