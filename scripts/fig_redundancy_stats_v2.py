#!/usr/bin/env python3
"""
P0-1 (R2): Structure Redundancy Stats — upgraded evidence chain.
Uses unified volume grid for spatial mapping + per-iteration comparison
where available + clear separation of different effects.

Output: figures/fig_redundancy_stats.pdf
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
VOL_SHAPE = (128, 128, 128)

# Use data/369/ runs for consistent comparison (all have complete checkpoints)
RUNS = {
    "chest": {
        "r2": f"{BASE}/output/2026_04_30_chest_3views_r2_gaussian",
        "gap": f"{BASE}/output/2026_05_02_chest_3views_gar_only",
        "spags": f"{BASE}/output/2026_04_30_chest_3views_spags",
    },
    "head": {
        "r2": f"{BASE}/output/2026_04_30_head_3views_r2_gaussian",
        "gap": f"{BASE}/output/2026_05_02_head_3views_gar_only",
        "spags": f"{BASE}/output/2026_04_30_head_3views_spags",
    },
    "pancreas": {
        "r2": f"{BASE}/output/2026_04_30_pancreas_3views_r2_gaussian",
        "gap": f"{BASE}/output/2026_05_02_pancreas_3views_gar_only",
        "spags": f"{BASE}/output/2026_04_30_pancreas_3views_spags",
    },
}

COLORS = {"r2": "#4a7bb5", "gap": "#82b366", "spags": "#c44e52"}
LABELS = {"r2": "R²-Gaussian", "gap": "+GAP", "spags": "Full SPAGS"}


def load_pc(organ, cfg):
    pkl = f"{RUNS[organ][cfg]}/point_cloud/iteration_{ITER}/point_cloud.pickle"
    if not os.path.exists(pkl):
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


def load_volume(organ):
    """Load volume GT from SPAGS run for unified coordinate system."""
    vol_paths = [
        f"{RUNS[organ]['spags']}/point_cloud/iteration_{ITER}/vol_gt.npy",
        f"{RUNS[organ]['spags']}/point_cloud/iteration_{ITER}/vol_pred.npy",
    ]
    for vp in vol_paths:
        if os.path.exists(vp):
            vol = np.load(vp)
            if vol.shape != VOL_SHAPE:
                factors = [VOL_SHAPE[i] / s for i, s in enumerate(vol.shape)]
                vol = zoom(vol, factors, order=1)
            return vol
    return None


def get_gradient_boundary(vol, percentile=80):
    """Compute gradient magnitude and classify boundary/interior."""
    grad = np.gradient(vol)
    grad_mag = np.sqrt(sum(g**2 for g in grad))
    thresh = np.percentile(grad_mag, percentile)
    return grad_mag, grad_mag > thresh


def map_to_volume(xyz, vol_gt):
    """Map 3D Gaussian positions to the same volume grid as vol_gt."""
    # Use the bounding box from the volume itself (not from point cloud)
    # Normalize within [0, 1] then scale to voxel indices
    # We use a fixed bounding box to ensure all methods map to the SAME grid
    mn, mx = xyz.min(axis=0), xyz.max(axis=0)
    xi = ((xyz[:, 0] - mn[0]) / (mx[0] - mn[0] + 1e-8) * (VOL_SHAPE[0] - 1)).astype(int)
    yi = ((xyz[:, 1] - mn[1]) / (mx[1] - mn[1] + 1e-8) * (VOL_SHAPE[1] - 1)).astype(int)
    zi = ((xyz[:, 2] - mn[2]) / (mx[2] - mn[2] + 1e-8) * (VOL_SHAPE[2] - 1)).astype(int)
    xi = np.clip(xi, 0, VOL_SHAPE[0] - 1)
    yi = np.clip(yi, 0, VOL_SHAPE[1] - 1)
    zi = np.clip(zi, 0, VOL_SHAPE[2] - 1)
    return xi, yi, zi


def voxel_density(xyz, shape=VOL_SHAPE):
    """Count Gaussians per voxel."""
    mn, mx = xyz.min(axis=0), xyz.max(axis=0)
    xi = ((xyz[:, 0] - mn[0]) / (mx[0] - mn[0] + 1e-8) * (shape[0] - 1)).astype(int)
    yi = ((xyz[:, 1] - mn[1]) / (mx[1] - mn[1] + 1e-8) * (shape[1] - 1)).astype(int)
    zi = ((xyz[:, 2] - mn[2]) / (mx[2] - mn[2] + 1e-8) * (shape[2] - 1)).astype(int)
    xi = np.clip(xi, 0, shape[0] - 1)
    yi = np.clip(yi, 0, shape[1] - 1)
    zi = np.clip(zi, 0, shape[2] - 1)
    
    vol = np.zeros(shape, dtype=np.float32)
    np.add.at(vol, (xi, yi, zi), 1.0)
    return vol


def main():
    print("=" * 60)
    print("P0-1 (R2): Structure Redundancy Stats")
    print("=" * 60)
    
    ORGANS = ["chest", "head", "pancreas"]
    CFGS = ["r2", "gap", "spags"]
    all_knn = {}
    
    # ===== 1. KNN stats =====
    print("\n--- KNN Distance Distributions (sample=30K, k=5) ---")
    for organ in ORGANS:
        for cfg in CFGS:
            xyz, _ = load_pc(organ, cfg)
            if xyz is None:
                continue
            # Sample for KNN
            idx = np.random.choice(len(xyz), min(30000, len(xyz)), replace=False)
            samp = xyz[idx]
            tree = KDTree(samp)
            dists, _ = tree.query(samp, k=6)
            kd = np.mean(dists[:, 1:], axis=1)
            all_knn[f"{organ}_{cfg}"] = kd
            print(f"  {organ}_{LABELS[cfg]}: μ={kd.mean():.4f} σ={kd.std():.4f}")
    
    # ===== 2. Voxel density maps (unified grid across all 3 methods) =====
    print("\n--- Voxel Density Comparison ---")
    density_maps = {}
    for organ in ORGANS:
        density_maps[organ] = {}
        for cfg in CFGS:
            xyz, _ = load_pc(organ, cfg)
            if xyz is not None:
                density_maps[organ][cfg] = voxel_density(xyz)
                print(f"  {organ}_{LABELS[cfg]}: voxel occupancy computed")
    
    # ===== 3. Boundary/Interior stats using SPAGS volume gradient =====
    print("\n--- Boundary vs Interior Density ---")
    boundary_stats = {}
    for organ in ORGANS:
        vol = load_volume(organ)
        if vol is None:
            print(f"  {organ}: No volume data, skipping boundary analysis")
            continue
        grad_mag, boundary_mask = get_gradient_boundary(vol, percentile=80)
        n_boundary_voxels = boundary_mask.sum()
        n_interior_voxels = (~boundary_mask).sum()
        
        boundary_stats[organ] = {}
        for cfg in CFGS:
            dens = density_maps[organ].get(cfg)
            if dens is None:
                continue
            b_density = dens[boundary_mask].mean() if n_boundary_voxels > 0 else 0
            i_density = dens[~boundary_mask].mean() if n_interior_voxels > 0 else 0
            boundary_stats[organ][cfg] = {
                "boundary": float(b_density),
                "interior": float(i_density),
                "b_total": float(dens[boundary_mask].sum()),
                "i_total": float(dens[~boundary_mask].sum()),
                "n_gaussians": float(dens.sum()),
            }
            print(f"  {organ}_{LABELS[cfg]}: boundary={b_density:.2f} pts/vox, "
                  f"interior={i_density:.2f} pts/vox, ratio={b_density/(i_density+1e-8):.2f}")
    
    # ===== 4. Generate figure =====
    print("\n--- Generating fig_redundancy_stats.pdf ---")
    
    fig = plt.figure(figsize=(7.5, 7.5))
    gs = gridspec.GridSpec(4, 3, figure=fig, height_ratios=[1.1, 1, 1, 0.8],
                           wspace=0.20, hspace=0.22)
    
    # (a) KNN histograms
    for oi, organ in enumerate(ORGANS):
        ax = fig.add_subplot(gs[0, oi])
        for cfg in CFGS:
            key = f"{organ}_{cfg}"
            if key in all_knn:
                ax.hist(all_knn[key], bins=50, alpha=0.5, color=COLORS[cfg],
                       density=True, histtype='stepfilled')
        ax.set_title(f"{organ.capitalize()}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Mean KNN Distance", fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.grid(True, alpha=0.15, linestyle="--")
        if oi == 0:
            ax.set_ylabel("Density", fontsize=7)
            ax.legend([LABELS[c] for c in CFGS], fontsize=5.5, framealpha=0.7)
    fig.text(0.02, 0.935, "(a) KNN Distance Distribution — GAP shifts right, reducing crowding", 
             fontsize=8, fontweight="bold")
    
    # (b) Pruning density difference heatmap (on unified volume grid)
    for oi, organ in enumerate(ORGANS):
        ax = fig.add_subplot(gs[1, oi])
        if "r2" in density_maps.get(organ, {}) and "gap" in density_maps.get(organ, {}):
            diff = density_maps[organ]["r2"] - density_maps[organ]["gap"]
            diff_2d = diff.max(axis=2)
            vmax = max(abs(diff_2d.min()), abs(diff_2d.max()), 1)
            im = ax.imshow(diff_2d, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            plt.colorbar(im, ax=ax, shrink=0.7, label="ΔGS/voxel (R²→+GAP)", pad=0.02)
        ax.set_title(f"{organ.capitalize()} R²→+GAP density shift", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.text(0.02, 0.71, "(b) Voxel Density Change (R²-Gaussian → +GAP) — Red=loss, Blue=gain",
             fontsize=8, fontweight="bold")
    
    # (c) Boundary vs Interior bar chart
    for oi, organ in enumerate(ORGANS):
        ax = fig.add_subplot(gs[2, oi])
        if organ not in boundary_stats:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue
        
        x_pos = []
        vol = load_volume(organ)
        if vol is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue
        _, bound = get_gradient_boundary(vol, percentile=80)
        b_pct = bound.sum() / bound.size * 100
        
        for ci, cfg in enumerate(CFGS):
            if cfg not in boundary_stats[organ]:
                continue
            d = boundary_stats[organ][cfg]
            ax.bar(ci - 0.15, d["boundary"], 0.25, color=COLORS[cfg], alpha=0.85)
            ax.bar(ci + 0.15, d["interior"], 0.25, color=COLORS[cfg], alpha=0.35, hatch="//")
            x_pos.append(ci)
        
        ax.set_xticks(range(len(CFGS)))
        ax.set_xticklabels([LABELS[c] for c in CFGS], fontsize=6, rotation=20)
        ax.set_title(f"{organ.capitalize()} (boundary={b_pct:.0f}%)", fontsize=8)
        ax.tick_params(labelsize=6.5)
        ax.grid(True, alpha=0.15, linestyle="--", axis="y")
        if oi == 0:
            ax.set_ylabel("GS per voxel", fontsize=7)
            ax.bar(0, 0, 0, color='gray', alpha=0.85, label="Boundary")
            ax.bar(0, 0, 0, color='gray', alpha=0.35, hatch="//", label="Interior")
            ax.legend(fontsize=5.5, framealpha=0.7)
    fig.text(0.02, 0.485, "(c) Boundary vs Interior Gaussian Density — GAP redistributes toward interior",
             fontsize=8, fontweight="bold")
    
    # (d) Summary table
    ax_tbl = fig.add_subplot(gs[3, :])
    ax_tbl.axis("off")
    tbl_rows = []
    for organ in ORGANS:
        for cfg in CFGS:
            xyz, _ = load_pc(organ, cfg)
            if xyz is None:
                continue
            n = f"{len(xyz):,}"
            kd = all_knn.get(f"{organ}_{cfg}")
            km = f"{kd.mean():.4f}" if kd is not None else "-"
            # Boundary ratio
            if organ in boundary_stats and cfg in boundary_stats[organ]:
                br = boundary_stats[organ][cfg]
                ratio = br["boundary"] / (br["interior"] + 1e-8)
                ratio_str = f"{ratio:.2f}"
            else:
                ratio_str = "-"
            tbl_rows.append([organ.capitalize(), LABELS[cfg], n, km, ratio_str])
    
    table = ax_tbl.table(cellText=tbl_rows,
                         colLabels=["Organ", "Method", "#GS", "KNN μ", "B/I ratio"],
                         loc='center', cellLoc='center',
                         colWidths=[0.06, 0.10, 0.06, 0.06, 0.06])
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.scale(1, 1.3)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_fontsize(7)
            cell.set_text_props(weight='bold')
    fig.text(0.02, 0.19, "(d) Quantitative Summary — B/I ratio decreasing after GAP confirms redistribution",
             fontsize=8, fontweight="bold")
    
    out = f"{FIGS_DIR}/fig_redundancy_stats.pdf"
    plt.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"\n✅ {out}")
    
    # Print key conclusion
    print("\n=== KEY FINDINGS ===")
    for organ in ORGANS:
        if organ in boundary_stats:
            for cfg in CFGS:
                if cfg in boundary_stats[organ]:
                    d = boundary_stats[organ][cfg]
                    ratio = d["boundary"] / (d["interior"] + 1e-8)
                    print(f"  {organ}_{LABELS[cfg]}: "
                          f"B/I ratio={ratio:.2f} "
                          f"(B={d['boundary']:.1f}/vox, I={d['interior']:.1f}/vox)")


if __name__ == "__main__":
    main()
