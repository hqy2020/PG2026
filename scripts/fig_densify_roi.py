#!/usr/bin/env python3
"""
实验03: More-densify vs SPAGS ROI可视化对比图。
显示Chest和Pancreas 3-view的 GT / Baseline / More Densify / SPAGS 渲染对比。
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

BASE = "/home/qyhu/Documents/r2_ours/PG2026"
FIGS_DIR = os.path.join(BASE, "figures")
os.makedirs(FIGS_DIR, exist_ok=True)

ITERS = 30000

# Config for each organ
CONFIG = {
    "Chest": {
        "baseline": f"{BASE}/output/2026_05_02_chest_3views_r2_gaussian",
        "densify":  f"{BASE}/output/chest_densify_test",
        "spags":    f"{BASE}/output/2026_05_01_chest_3views_spags",
        "view": 25,
        "roi": (80, 130, 80, 130),  
    },
    "Pancreas": {
        "baseline": f"{BASE}/output/2026_04_30_pancreas_3views_r2_gaussian",
        "densify":  f"{BASE}/output/pancreas_3views_densify",
        "spags":    f"{BASE}/output/2026_04_30_pancreas_3views_spags",
        "view": 25,
        "roi": (60, 120, 100, 160),
    },
}

METHODS = [
    ("GT", None, "GT"),            
    ("R²", "baseline", "R²-Gaussian"),  
    ("More Densify", "densify", "More Densify"),
    ("SPAGS", "spags", "SPAGS"),
]

def load_render(dirpath, view, suffix="gt", iters=ITERS):
    fp = f"{dirpath}/test/iter_{iters}/render_test/{view:05d}_{suffix}.png"
    if not os.path.exists(fp):
        return None
    img = np.array(Image.open(fp))
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return img

def main():
    fig, axes = plt.subplots(4, 4, figsize=(8, 9.5))
    
    for oi, organ in enumerate(["Chest", "Pancreas"]):
        cfg = CONFIG[organ]
        view = cfg["view"]
        x1, x2, y1, y2 = cfg["roi"]
        
        for mi, (label, dir_key, title) in enumerate(METHODS):
            if label == "GT":
                img = load_render(cfg["baseline"], view, "gt")
            else:
                img = load_render(cfg[dir_key], view, "pred")
            
            if img is None:
                print(f"  WARNING: No render for {organ} {label}")
                continue
            
            # Full view
            ax_full = axes[mi, oi * 2]
            ax_full.imshow(img, cmap="gray")
            if mi == 0:
                ax_full.set_title(f"{organ} — Full View", fontsize=8, fontweight="bold")
            ax_full.axis("off")
            
            # Draw ROI rectangle
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False,
                                edgecolor="red", linewidth=0.8, linestyle="--")
            ax_full.add_patch(rect)
            
            # Zoom ROI
            ax_zoom = axes[mi, oi * 2 + 1]
            zoom_img = img[y1:y2, x1:x2] if img.ndim == 2 else img[y1:y2, x1:x2, :]
            ax_zoom.imshow(zoom_img, cmap="gray")
            if mi == 0:
                ax_zoom.set_title(f"ROI Zoom", fontsize=8, fontweight="bold")
            ax_zoom.axis("off")
            
            # Row label
            if oi == 0:
                axes[mi, 0].set_ylabel(label, fontsize=7, fontweight="bold")
    
    plt.tight_layout(pad=0.3, w_pad=0.2, h_pad=0.5)
    
    out_path = f"{FIGS_DIR}/fig_densify_roi.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"✅ {out_path}")

if __name__ == "__main__":
    main()
