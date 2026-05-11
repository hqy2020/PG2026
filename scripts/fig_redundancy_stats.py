#!/usr/bin/env python3
"""
P0-1: Structure Redundancy Evidence for GAP.
Loads point_cloud.pickle from R²-Gaussian / +GAP / SPAGS checkpoints,
computes KNN distance distributions, pruning heatmaps, and boundary stats.

Output: paper/figures/fig_redundancy_stats.pdf
"""
import os, sys, glob, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial import KDTree
from scipy.ndimage import zoom

BASE = "/home/qyhu/Documents/r2_ours/PG2026"
FIGS_DIR = os.path.join(BASE, "figures")
os.makedirs(FIGS_DIR, exist_ok=True)

ITER = 30000

# Use Apr 30 (data/369/) runs for all — they guarantee complete checkpoints
# This avoids data/234 vs data/369 inconsistency for the analysis
RUNS = {
    "chest": {
        "r2_gaussian": f"{BASE}/output/2026_04_30_chest_3views_r2_gaussian",
        "gar_only":   f"{BASE}/output/2026_05_02_chest_3views_gar_only",
        "spags":      f"{BASE}/output/2026_04_30_chest_3views_spags",
    },
    "head": {
        "r2_gaussian": f"{BASE}/output/2026_04_30_head_3views_r2_gaussian",
        "gar_only":   f"{BASE}/output/2026_05_02_head_3views_gar_only",
        "spags":      f"{BASE}/output/2026_04_30_head_3views_spags",
    },
    "pancreas": {
        "r2_gaussian": f"{BASE}/output/2026_04_30_pancreas_3views_r2_gaussian",
        "gar_only":   f"{BASE}/output/2026_05_02_pancreas_3views_gar_only",
        "spags":      f"{BASE}/output/2026_04_30_pancreas_3views_spags",
    },
}

CFGS = ["r2_gaussian", "gar_only", "spags"]
COLORS = {"r2_gaussian": "#4a7bb5", "gar_only": "#82b366", "spags": "#c44e52"}
LABELS = {"r2_gaussian": "R²-Gaussian", "gar_only": "+GAP", "spags": "Full SPAGS"}


def load_pc(organ, cfg):
    """Load point cloud pickle."""
    pkl = f"{RUNS[organ][cfg]}/point_cloud/iteration_{ITER}/point_cloud.pickle"
    if not os.path.exists(pkl):
        print(f"  ⚠ Missing: {pkl}")
        return None, None
    with open(pkl, 'rb') as f:
        data = pickle.load(f)
    xyz = data.get('xyz')
    if xyz is None:
        return None, None
    if hasattr(xyz, 'cpu'):
        xyz = xyz.cpu().numpy()
    density = data.get('density')
    if density is not None and hasattr(density, 'cpu'):
        density = density.cpu().numpy().ravel()
    return xyz, density


def knn_dists(xyz, k=5, sample=30000):
    """Compute mean KNN distances."""
    if len(xyz) > sample:
        idx = np.random.choice(len(xyz), sample, replace=False)
        xyz = xyz[idx]
    tree = KDTree(xyz)
    dists, _ = tree.query(xyz, k=k+1)
    return np.mean(dists[:, 1:], axis=1)


def compute_pruning_heatmap(xyz_before, xyz_after, shape=(64, 64, 64)):
    """Voxel count diff as pruning heatmap."""
    def to_vox(xyz, shape):
        mn, mx = xyz.min(axis=0), xyz.max(axis=0)
        idx = ((xyz - mn) / (mx - mn + 1e-8) * (np.array(shape) - 1)).astype(int)
        idx = np.clip(idx, 0, np.array(shape) - 1)
        return idx[:, 0], idx[:, 1], idx[:, 2]
    
    vol = np.zeros(shape, dtype=np.int32)
    xi, yi, zi = to_vox(xyz_before, shape)
    np.add.at(vol, (xi, yi, zi), 1)
    
    vol_after = np.zeros(shape, dtype=np.int32)
    xi, yi, zi = to_vox(xyz_after, shape)
    np.add.at(vol_after, (xi, yi, zi), 1)
    
    return vol - vol_after


def main():
    print("=" * 60)
    print("P0-1: Structure Redundancy Stats")
    print("=" * 60)
    
    ORGANS = ["chest", "head", "pancreas"]
    all_knn = {}
    
    # ==================== STATS ====================
    print("\n--- Computing KNN distributions ---")
    for organ in ORGANS:
        for cfg in CFGS:
            xyz, _ = load_pc(organ, cfg)
            if xyz is None:
                continue
            kd = knn_dists(xyz, k=5)
            all_knn[f"{organ}_{cfg}"] = kd
            print(f"  {organ}_{LABELS[cfg]}: n={len(kd):,}, "
                  f"μ={kd.mean():.4f}±{kd.std():.4f}, med={np.median(kd):.4f}")
    
    # Print Gaussian count comparison
    print("\n--- Gaussian Count Comparison ---")
    for organ in ORGANS:
        print(f"\n{organ.upper()}:")
        for cfg in CFGS:
            xyz, _ = load_pc(organ, cfg)
            if xyz is not None:
                print(f"  {LABELS[cfg]}: {len(xyz):,}")
        
        r2_xyz, _ = load_pc(organ, "r2_gaussian")
        gap_xyz, _ = load_pc(organ, "gar_only")
        full_xyz, _ = load_pc(organ, "spags")
        
        if r2_xyz is not None and gap_xyz is not None:
            pr = len(r2_xyz) - len(gap_xyz)
            print(f"  R²→+GAP: {len(r2_xyz):,} → {len(gap_xyz):,} "
                  f"(Δ={pr:+,d}, {pr/len(r2_xyz)*100:+.1f}%)")
        if gap_xyz is not None and full_xyz is not None:
            pr = len(gap_xyz) - len(full_xyz)
            print(f"  +GAP→Full: {len(gap_xyz):,} → {len(full_xyz):,} "
                  f"(Δ={pr:+,d}, {pr/len(gap_xyz)*100:+.1f}%)")
    
    # ==================== FIGURE ====================
    print("\n--- Generating fig_redundancy_stats.pdf ---")
    
    fig = plt.figure(figsize=(7.5, 8.0))
    gs = gridspec.GridSpec(4, 3, figure=fig, height_ratios=[1.2, 1, 1, 0.8],
                           wspace=0.20, hspace=0.25)
    
    # (a) KNN histograms — one column per organ
    for oi, organ in enumerate(ORGANS):
        ax = fig.add_subplot(gs[0, oi])
        for cfg in CFGS:
            key = f"{organ}_{cfg}"
            if key in all_knn:
                ax.hist(all_knn[key], bins=50, alpha=0.55, color=COLORS[cfg],
                       density=True, histtype='stepfilled')
        ax.set_title(f"{organ.capitalize()}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Mean KNN distance", fontsize=7)
        if oi == 0:
            ax.set_ylabel("Density", fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.grid(True, alpha=0.15, linestyle="--")
        if oi == 0:
            ax.legend([LABELS[c] for c in CFGS], fontsize=5.5, framealpha=0.7)
    fig.text(0.02, 0.93, "(a) KNN Distance Distribution", fontsize=9, fontweight="bold")
    
    # (b) Pruning heatmaps
    for oi, organ in enumerate(ORGANS):
        ax = fig.add_subplot(gs[1, oi])
        r2_xyz, _ = load_pc(organ, "r2_gaussian")
        gap_xyz, _ = load_pc(organ, "gar_only")
        if r2_xyz is not None and gap_xyz is not None:
            diff = compute_pruning_heatmap(r2_xyz, gap_xyz)
            diff_2d = diff.max(axis=2)
            im = ax.imshow(diff_2d, cmap="hot", aspect="auto")
            plt.colorbar(im, ax=ax, shrink=0.7, label="Pruned", pad=0.02)
        ax.set_title(f"{organ.capitalize()} (R²→+GAP)", fontsize=8)
        ax.tick_params(labelsize=6)
    fig.text(0.02, 0.69, "(b) Pruning Spatial Heatmap (max-projection)", fontsize=9, fontweight="bold")
    
    # (c) Boundary vs Interior density
    for oi, organ in enumerate(ORGANS):
        ax = fig.add_subplot(gs[2, oi])
        
        # Use SPAGS volume for gradient-based boundary classification
        vol_path = f"{RUNS[organ]['spags']}/test/iter_{ITER}/reconstruction/vol_gt.npy"
        if not os.path.exists(vol_path):
            vol_path = f"{RUNS[organ]['spags']}/eval/iter_030000/vol_gt.npy"
        if not os.path.exists(vol_path):
            alt_paths = glob.glob(f"{RUNS[organ]['spags']}/eval/iter_*/vol_gt.npy")
            vol_path = alt_paths[0] if alt_paths else None
        
        if vol_path and os.path.exists(vol_path):
            vol = np.load(vol_path)
            if vol.shape != (128, 128, 128):
                factors = [128.0/s for s in vol.shape]
                vol = zoom(vol, factors, order=1)
            grad = np.gradient(vol)
            grad_mag = np.sqrt(sum(g**2 for g in grad))
            thresh = np.percentile(grad_mag, 80)
            boundary = grad_mag > thresh
            
            x_pos = []
            for cfg in CFGS:
                xyz, _ = load_pc(organ, cfg)
                if xyz is None:
                    continue
                mn, mx = xyz.min(axis=0), xyz.max(axis=0)
                xi = ((xyz[:, 0] - mn[0]) / (mx[0] - mn[0] + 1e-8) * 127).astype(int)
                yi = ((xyz[:, 1] - mn[1]) / (mx[1] - mn[1] + 1e-8) * 127).astype(int)
                zi = ((xyz[:, 2] - mn[2]) / (mx[2] - mn[2] + 1e-8) * 127).astype(int)
                xi = np.clip(xi, 0, 127); yi = np.clip(yi, 0, 127); zi = np.clip(zi, 0, 127)
                
                is_b = boundary[xi, yi, zi].sum()
                is_i = len(xyz) - is_b
                db = is_b / boundary.sum()
                di = is_i / (~boundary).sum()
                
                p = len(x_pos)
                ax.bar(p - 0.15, db, 0.25, color=COLORS[cfg], alpha=0.8, label="Boundary" if p == 0 else "")
                ax.bar(p + 0.15, di, 0.25, color=COLORS[cfg], alpha=0.3, hatch="//", label="Interior" if p == 0 else "")
                x_pos.append(p)
            
            ax.set_xticks(range(len(x_pos)))
            ax.set_xticklabels([LABELS[c] for c in CFGS], fontsize=6, rotation=20)
            ax.set_title(f"{organ.capitalize()}", fontsize=8)
        else:
            ax.text(0.5, 0.5, "No volume data", ha="center", va="center", transform=ax.transAxes)
        
        ax.tick_params(labelsize=6.5)
        ax.grid(True, alpha=0.15, linestyle="--", axis="y")
        if oi == 0:
            ax.set_ylabel("GS per voxel", fontsize=7)
    
    fig.text(0.02, 0.44, "(c) Boundary vs Interior Gaussian Density (80% grad. percentile)", fontsize=9, fontweight="bold")
    
    # (d) Key stats table
    ax_tbl = fig.add_subplot(gs[3, :])
    ax_tbl.axis("off")
    
    rows = []
    for organ in ORGANS:
        for cfg in CFGS:
            xyz, _ = load_pc(organ, cfg)
            if xyz is None:
                continue
            key = f"{organ}_{cfg}"
            kd = all_knn.get(key)
            mean_knn = f"{kd.mean():.4f}" if kd is not None else "-"
            rows.append([organ.capitalize(), LABELS[cfg], f"{len(xyz):,}", mean_knn])
    
    table = ax_tbl.table(cellText=rows,
                         colLabels=["Organ", "Method", "#Gaussians", "KNN μ"],
                         loc='center', cellLoc='center',
                         colWidths=[0.08, 0.12, 0.08, 0.08])
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.scale(1, 1.3)
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_fontsize(7)
            cell.set_text_props(weight='bold')
    fig.text(0.02, 0.19, "(d) Gaussian Count & KNN Statistics", fontsize=9, fontweight="bold")
    
    plt.savefig(f"{FIGS_DIR}/fig_redundancy_stats.pdf", dpi=200,
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"\n✅ {FIGS_DIR}/fig_redundancy_stats.pdf")


if __name__ == "__main__":
    main()
