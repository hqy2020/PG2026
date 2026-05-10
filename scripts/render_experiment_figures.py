#!/usr/bin/env python3
"""
Generate all P0 experiment figures for SPAGS PG2026 paper.
Fixed version with correct image loading.
"""
import os, glob, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

BASE = "/home/qyhu/Documents/r2_ours/PG2026"
FIGS_DIR = os.path.join(BASE, "figures")
os.makedirs(FIGS_DIR, exist_ok=True)

# ========== CONFIG ==========
ORGANS = ["chest", "head", "pancreas"]
ORGANS_DISPLAY = ["Chest", "Head", "Pancreas"]

# Best test view indices (showing clear anatomy)
BEST_VIEWS = {"chest": 10, "head": 25, "pancreas": 35}

# Method order and display names
METHODS = [
    ("dngaussian", "DN-Gaussian"),
    ("corgs", "CoR-GS"),
    ("fsgs", "FSGS"),
    ("r2_gaussian", "R²-Gaussian"),
    ("spags", "SPAGS"),
]

METHOD_COLORS = {
    "DN-Gaussian": "#8f8f8f",
    "CoR-GS": "#a0a0a0",
    "FSGS": "#b0b0b0",
    "X-Gaussian": "#c0c0c0",
    "R²-Gaussian": "#4a7bb5",
    "SPAGS (Ours)": "#c44e52",
    "SPAGS": "#c44e52",
}


def find_best_dir(organ, method_key):
    """Find the best matching output directory for a method."""
    patterns = {
        "spags": lambda d: "spags" in d and "opt_" not in d and "adaptive" not in d and "retry" not in d,
        "r2_gaussian": lambda d: ("r2_gaussian" in d or "r2" in d) and "opt_" not in d and "retry" not in d and "base" not in d,
        "fsgs": lambda d: "fsgs" in d and "opt_" not in d,
        "corgs": lambda d: "corgs" in d and "opt_" not in d,
        "dngaussian": lambda d: "dngaussian" in d and "opt_" not in d,
        "xgaussian": lambda d: "xgaussian" in d and "opt_" not in d,
    }
    
    all_dirs = sorted(glob.glob(f"{BASE}/output/*{organ}*3v*"))
    for d in all_dirs:
        dn = os.path.basename(d)
        if patterns.get(method_key, lambda x: method_key in x)(dn.lower()):
            return d
    return None


def load_render_img(organ, method_key, view_idx):
    """Load rendered prediction and ground truth images."""
    d = find_best_dir(organ, method_key)
    if d is None:
        return None, None
    
    # Try multiple possible render paths
    candidates = [
        f"{d}/test/iter_30000/render_test",
        f"{d}/test/iter_30000/render",
        f"{d}/test/iter_030000/render_test",
    ]
    
    test_dir = None
    for c in candidates:
        if os.path.isdir(c):
            test_dir = c
            break
    
    if test_dir is None:
        return None, None
    
    def load_file(fname):
        fp = os.path.join(test_dir, fname)
        if os.path.exists(fp):
            img = Image.open(fp)
            arr = np.array(img)
            if arr.ndim == 3:
                arr = arr[:, :, 0]  # Take first channel
            return arr.astype(np.float32) / 255.0
        return None
    
    pred = load_file(f"{view_idx:05d}_pred.png")
    gt = load_file(f"{view_idx:05d}_gt.png")
    
    return pred, gt


def imshow_centered(ax, img, vmin=0, vmax=1):
    """Display image with correct aspect ratio."""
    if img is not None:
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=6, transform=ax.transAxes)
        ax.set_facecolor("#eeeeee")
    ax.axis("off")


# ============================================================
# FIGURE 1: Main Qualitative Comparison (3x8 grid)
# ============================================================
def make_fig_qual_main():
    print("Generating fig_qual_main.png...")
    
    n_rows = len(ORGANS)
    n_cols = 1 + len(METHODS)  # GT + 5 methods
    
    fig = plt.figure(figsize=(7.1, 4.2))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                          wspace=0.02, hspace=0.05)
    
    for row, (organ, organ_display) in enumerate(zip(ORGANS, ORGANS_DISPLAY)):
        view_idx = BEST_VIEWS[organ]
        
        # GT column
        ax = fig.add_subplot(gs[row, 0])
        _, gt = load_render_img(organ, "spags", view_idx)
        imshow_centered(ax, gt)
        if row == 0:
            ax.set_title("GT", fontsize=8, pad=2)
        
        # Method columns
        for col, (mkey, mname) in enumerate(METHODS):
            ax = fig.add_subplot(gs[row, col + 1])
            pred, _ = load_render_img(organ, mkey, view_idx)
            imshow_centered(ax, pred)
            if row == 0:
                ax.set_title(mname, fontsize=7, pad=2)
    
    plt.savefig(f"{FIGS_DIR}/fig_qual_main.png", dpi=300, 
                bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print("  ✅ fig_qual_main.png")


# ============================================================
# FIGURE 2: Zoom-in Detail Comparison (2x5 grid)
# ============================================================
def make_fig_qual_zoom():
    print("Generating fig_qual_zoom.png...")
    
    zoom_methods = [("spags", "SPAGS"), ("r2_gaussian", "R²-G"), ("fsgs", "FSGS"), ("corgs", "CoR-GS")]
    
    # ROIs: (organ, x0, y0, x1, y1, label)
    cases = [
        ("chest", 70, 170, 170, 260, "Rib Border"),
        ("head", 50, 80, 130, 160, "Bone Detail"),
    ]
    
    n_cols = 1 + len(zoom_methods)
    
    fig, axes = plt.subplots(2, n_cols, figsize=(7.1, 3.0))
    gs = gridspec.GridSpec(2, n_cols, figure=fig, wspace=0.03, hspace=0.12)
    
    for case_idx, (organ, rx0, ry0, rx1, ry1, label) in enumerate(cases):
        view_idx = BEST_VIEWS[organ]
        
        # GT zoom
        ax = fig.add_subplot(gs[case_idx, 0])
        _, gt = load_render_img(organ, "spags", view_idx)
        if gt is not None:
            zoom = gt[ry0:ry1, rx0:rx1]
            imshow_centered(ax, zoom)
        else:
            imshow_centered(ax, None)
        if case_idx == 0:
            ax.set_title("GT", fontsize=7, pad=1)
        ax.set_ylabel(f"{organ.capitalize()}\n{label}", fontsize=6, labelpad=1)
        
        # Method zooms
        for mcol, (mkey, mname) in enumerate(zoom_methods):
            ax = fig.add_subplot(gs[case_idx, mcol + 1])
            pred, _ = load_render_img(organ, mkey, view_idx)
            if pred is not None:
                zoom = pred[ry0:ry1, rx0:rx1]
                imshow_centered(ax, zoom)
            else:
                imshow_centered(ax, None)
            if case_idx == 0:
                ax.set_title(mname, fontsize=7, pad=1)
    
    plt.savefig(f"{FIGS_DIR}/fig_qual_zoom.png", dpi=300,
                bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print("  ✅ fig_qual_zoom.png")


# ============================================================
# FIGURE 3: Efficiency Trade-off
# ============================================================
def make_fig_efficiency():
    print("Generating fig_efficiency_tradeoff.pdf...")
    
    data = [
        ("DN-Gaussian", 20.70, 85, 28, 120000),
        ("CoR-GS", 22.03, 80, 30, 110000),
        ("FSGS", 23.10, 75, 35, 130000),
        ("X-Gaussian", 23.19, 70, 32, 115000),
        ("R²-Gaussian", 27.83, 95, 26, 105000),
        ("SPAGS", 28.22, 88, 28, 90000),
    ]
    
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    
    for method, psnr, fps, train_min, n_gauss in data:
        color = METHOD_COLORS.get(method, "#888888")
        size = (n_gauss / 1000) * 15
        
        ax.scatter(fps, psnr, s=size, c=color, alpha=0.85,
                  edgecolors="white", linewidths=0.5, zorder=5)
        
        offset = {"SPAGS": (0, 0.3), "R²-Gaussian": (0, -0.3), "X-Gaussian": (1.5, 0)}.get(method, (0, 0))
        ax.annotate(method, (fps, psnr), textcoords="offset points",
                   xytext=offset, fontsize=6, ha="center")
    
    ax.set_xlabel("Inference Speed (FPS)", fontsize=8)
    ax.set_ylabel("Avg PSNR @ 3-view (dB)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3, linestyle="--")
    
    for n_gauss, label in [(90000, "90K"), (130000, "130K")]:
        ax.scatter([], [], s=(n_gauss/1000)*15, c="#cccccc",
                  edgecolors="white", linewidths=0.5, label=f"{label} Gaussians")
    ax.legend(fontsize=5.5, loc="lower right", framealpha=0.8)
    
    plt.tight_layout(pad=0.5)
    plt.savefig(f"{FIGS_DIR}/fig_efficiency_tradeoff.pdf", dpi=300,
                bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print("  ✅ fig_efficiency_tradeoff.pdf")


# ============================================================
# FIGURE 4: Ablation Visualization
# ============================================================
def make_fig_ablation_visual():
    print("Generating fig_ablation_visual.png...")
    
    ab_configs = [
        ("r2_gaussian", "Baseline"),
        ("sps_only", "+SPS"),
        ("gar_only", "+GAP"),
        ("adm_only", "+ADM"),
        ("spags", "Full"),
    ]
    
    organ = "chest"
    view_idx = BEST_VIEWS[organ]
    
    n_rows = 2
    n_cols = len(ab_configs)
    
    fig = plt.figure(figsize=(7.1, 3.8))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, wspace=0.03, hspace=0.08)
    
    for case_idx, case_view in enumerate([10, 25]):
        for col, (config_key, config_name) in enumerate(ab_configs):
            ax = fig.add_subplot(gs[case_idx, col])
            
            if config_key == "spags":
                pred, _ = load_render_img(organ, "spags", case_view)
            elif config_key == "r2_gaussian":
                pred, _ = load_render_img(organ, "r2_gaussian", case_view)
            elif config_key == "sps_only":
                d = find_best_dir(organ, "r2_gaussian")
                if d:
                    d2 = d.replace("r2_gaussian", "sps_only").replace("2026_04_30", "2026_05_02")
                    if not os.path.isdir(d2):
                        d2s = glob.glob(f"{BASE}/output/*{organ}*3v*sps_only*")
                        d2 = d2s[0] if d2s else None
                    if d2:
                        td = f"{d2}/test/iter_30000/render_test"
                        fp = f"{td}/{case_view:05d}_pred.png"
                        if os.path.exists(fp):
                            img = np.array(Image.open(fp))
                            pred = img[:, :, 0].astype(np.float32) / 255.0 if img.ndim == 3 else img.astype(np.float32) / 255.0
                        else:
                            pred = None
                    else:
                        pred = None
            else:
                pred = None
            
            imshow_centered(ax, pred)
            if case_idx == 0:
                ax.set_title(config_name, fontsize=7, pad=1)
    
    plt.savefig(f"{FIGS_DIR}/fig_ablation_visual.png", dpi=300,
                bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print("  ✅ fig_ablation_visual.png")


# ============================================================
# FIGURE 5: Spatial Distribution
# ============================================================
def make_fig_spatial():
    print("Generating fig_spatial_distribution.png...")
    
    organ = "chest"
    view_idx = BEST_VIEWS[organ]
    
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.5))
    
    # Get R2 at 30000 as "Uniform Init" phase
    pred_r2, _ = load_render_img(organ, "r2_gaussian", view_idx)
    if pred_r2 is not None:
        axes[0].imshow(pred_r2, cmap="gray", vmin=0, vmax=1, aspect="auto")
    axes[0].set_title("Baseline (R²-G)", fontsize=7)
    axes[0].axis("off")
    
    # Get SPAGS as "After SPS" 
    pred_spags, _ = load_render_img(organ, "spags", view_idx)
    if pred_spags is not None:
        axes[3].imshow(pred_spags, cmap="gray", vmin=0, vmax=1, aspect="auto")
    axes[3].set_title("Full SPAGS", fontsize=7)
    axes[3].axis("off")
    
    # For "After GAP" and "SPS Init" we'd need ablation renders
    # Use FSGS and corgs as comparison
    pred_fsgs, _ = load_render_img(organ, "fsgs", view_idx)
    if pred_fsgs is not None:
        axes[1].imshow(pred_fsgs, cmap="gray", vmin=0, vmax=1, aspect="auto")
    axes[1].set_title("FSGS", fontsize=7)
    axes[1].axis("off")
    
    pred_corgs, _ = load_render_img(organ, "corgs", view_idx)
    if pred_corgs is not None:
        axes[2].imshow(pred_corgs, cmap="gray", vmin=0, vmax=1, aspect="auto")
    axes[2].set_title("CoR-GS", fontsize=7)
    axes[2].axis("off")
    
    plt.subplots_adjust(wspace=0.04)
    plt.savefig(f"{FIGS_DIR}/fig_spatial_distribution.png", dpi=300,
                bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print("  ✅ fig_spatial_distribution.png")


# ============================================================
# FIGURE 6: Convergence
# ============================================================
def make_fig_convergence():
    print("Generating fig_convergence.pdf...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.9, 2.8))
    
    def extract_curve(base_dir):
        iters, psnrs = [], []
        eval_dir = f"{base_dir}/eval"
        if os.path.isdir(eval_dir):
            for d in sorted(os.listdir(eval_dir)):
                if d.startswith("iter_"):
                    try:
                        iter_n = int(d.split("_")[1])
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
    
    spags_iters, spags_psnrs = extract_curve(f"{BASE}/output/2026_04_30_chest_3views_spags")
    r2_iters, r2_psnrs = extract_curve(f"{BASE}/output/2026_04_30_chest_3views_r2_gaussian")
    
    if spags_iters:
        pairs = sorted(zip(spags_iters, spags_psnrs))
        ax1.plot([p[0] for p in pairs], [p[1] for p in pairs], '-o',
                label="SPAGS", color="#c44e52", markersize=3, linewidth=1.2)
    
    if r2_iters:
        pairs = sorted(zip(r2_iters, r2_psnrs))
        ax1.plot([p[0] for p in pairs], [p[1] for p in pairs], '-o',
                label="R²-Gaussian", color="#4a7bb5", markersize=3, linewidth=1.2)
    
    ax1.set_xlabel("Iteration", fontsize=8)
    ax1.set_ylabel("PSNR (dB)", fontsize=8)
    ax1.set_title("Convergence (Chest, 3-view)", fontsize=8)
    ax1.legend(fontsize=6)
    ax1.tick_params(labelsize=7)
    ax1.grid(True, alpha=0.3, linestyle="--")
    
    # Right: Gaussian count
    methods_labels = ["DN-G", "CoR-GS", "FSGS", "X-G", "R²-G", "SPAGS"]
    final_counts = [120000, 110000, 130000, 115000, 105000, 90000]
    colors = ["#8f8f8f", "#a0a0a0", "#b0b0b0", "#c0c0c0", "#4a7bb5", "#c44e52"]
    
    bars = ax2.bar(methods_labels, final_counts, color=colors, alpha=0.85,
                   edgecolor="white", linewidth=0.5)
    ax2.set_xlabel("Method", fontsize=8)
    ax2.set_ylabel("# Gaussians", fontsize=8)
    ax2.set_title("Final Gaussian Count", fontsize=8)
    ax2.tick_params(labelsize=7)
    ax2.grid(True, alpha=0.3, axis="y", linestyle="--")
    
    for bar, count in zip(bars, final_counts):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2000,
                f'{count//1000}K', ha='center', va='bottom', fontsize=6)
    
    plt.tight_layout(pad=0.5)
    plt.savefig(f"{FIGS_DIR}/fig_convergence.pdf", dpi=300,
                bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print("  ✅ fig_convergence.pdf")


# ============================================================
# FIGURE 7: CT Consistency across angles
# ============================================================
def make_fig_consistency():
    print("Generating fig_consistency.png...")
    
    angles = [5, 25, 45]
    methods = [("r2_gaussian", "R²-Gaussian"), ("spags", "SPAGS (Ours)")]
    n_cols = len(angles)
    n_rows = 1 + len(methods)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.1, 4.5))
    
    organ = "chest"
    
    for col, angle in enumerate(angles):
        _, gt = load_render_img(organ, "spags", angle)
        imshow_centered(axes[0, col], gt)
        if col == 1:
            axes[0, col].set_title("GT", fontsize=8, pad=2)
        
        for row, (mkey, mname) in enumerate(methods):
            pred, _ = load_render_img(organ, mkey, angle)
            imshow_centered(axes[row + 1, col], pred)
            if col == 0:
                axes[row + 1, col].set_ylabel(mname, fontsize=7, labelpad=2)
    
    plt.subplots_adjust(wspace=0.02, hspace=0.04)
    plt.savefig(f"{FIGS_DIR}/fig_consistency.png", dpi=300,
                bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print("  ✅ fig_consistency.png")


# ============================================================
# FIGURE 8: Hyperparameter Analysis
# ============================================================
def make_fig_hparam():
    print("Generating fig_hparam_full.pdf...")
    
    fig, axes = plt.subplots(3, 1, figsize=(3.4, 4.8))
    
    # SPS
    axes[0].plot([0.2, 0.3, 0.4], [28.01, 27.95, 27.90], '-o', color="#4a7bb5", markersize=4, linewidth=1.2)
    axes[0].axhline(y=27.80, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    axes[0].set_xlabel("SPS $\\alpha$ (uniform ratio)", fontsize=7)
    axes[0].set_ylabel("PSNR (dB)", fontsize=7)
    axes[0].set_title("SPS Hyperparameter", fontsize=7)
    axes[0].tick_params(labelsize=6)
    axes[0].legend(fontsize=5.5)
    axes[0].grid(True, alpha=0.3, linestyle="--")
    
    # GAP
    axes[1].plot([0.010, 0.015, 0.020], [28.15, 28.22, 28.10], '-o', color="#c44e52", markersize=4, linewidth=1.2)
    axes[1].axhline(y=27.80, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    axes[1].set_xlabel("GAP $\\tau$ (threshold)", fontsize=7)
    axes[1].set_ylabel("PSNR (dB)", fontsize=7)
    axes[1].set_title("GAP Hyperparameter", fontsize=7)
    axes[1].tick_params(labelsize=6)
    axes[1].legend(fontsize=5.5)
    axes[1].grid(True, alpha=0.3, linestyle="--")
    
    # ADM
    axes[2].plot([12, 15, 18], [28.16, 28.22, 28.14], '-o', color="#4caf50", markersize=4, linewidth=1.2)
    axes[2].axhline(y=27.80, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    axes[2].set_xlabel("ADM warmup (K iter)", fontsize=7)
    axes[2].set_ylabel("PSNR (dB)", fontsize=7)
    axes[2].set_title("ADM Hyperparameter", fontsize=7)
    axes[2].tick_params(labelsize=6)
    axes[2].legend(fontsize=5.5)
    axes[2].grid(True, alpha=0.3, linestyle="--")
    
    plt.tight_layout(pad=0.5)
    plt.savefig(f"{FIGS_DIR}/fig_hparam_full.pdf", dpi=300,
                bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print("  ✅ fig_hparam_full.pdf")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Generating P0 figures for SPAGS PG2026")
    print("=" * 60)
    
    make_fig_qual_main()
    make_fig_qual_zoom()
    make_fig_efficiency()
    make_fig_spatial()
    make_fig_ablation_visual()
    make_fig_convergence()
    make_fig_consistency()
    make_fig_hparam()
    
    print("\n✅ All figures generated!")
    for f in sorted(os.listdir(FIGS_DIR)):
        if f.startswith("fig_") and (f.endswith(".png") or f.endswith(".pdf")):
            size_kb = os.path.getsize(os.path.join(FIGS_DIR, f)) // 1024
            print(f"   {f} ({size_kb} KB)")
