import os, glob, pickle
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import traceback

BASE = '/home/qyhu/Documents/r2_ours/PG2026'
FIGS_DIR = f'{BASE}/figures'

organs = ['chest', 'head', 'abdomen', 'foot', 'pancreas']
organ_labels = ['Chest', 'Head', 'Abdomen', 'Foot', 'Pancreas']
methods = ['dngaussian', 'corgs', 'fsgs', 'xgaussian', 'r2_gaussian', 'spags']
method_names = ['DN-Gaussian', 'CoR-GS', 'FSGS', 'X-Gaussian', 'R\u00b2-Gaussian', 'SPAGS']

print("Finding slices...")
slices_map = {}
for organ in organs:
    for method in methods:
        for pat in [f'{BASE}/output/*{organ}_4views_{method}*/eval/iter_030000/*slices*.png']:
            matches = sorted(glob.glob(pat))
            if matches:
                slices_map[(organ, method)] = matches[0]
                break
        if (organ, method) not in slices_map:
            # Broader search
            for pat in [f'{BASE}/output/*{organ}_4views*/eval/iter_030000/slices_*.png']:
                matches = sorted(glob.glob(pat))
                # Filter for method-specific file
                for m in matches:
                    fname = os.path.basename(m)
                    if method in fname:
                        slices_map[(organ, method)] = m
                        break
                if (organ, method) in slices_map:
                    break
print(f"Found {len(slices_map)}/{len(organs)*len(methods)} slices")

print("Loading GT volumes...")
gt_slices = {}
for organ in organs:
    pk = f'{BASE}/data/234/{organ}_50_4views.pickle'
    if not os.path.exists(pk):
        print(f"  {organ}: no pickle at {pk}")
        continue
    try:
        with open(pk, 'rb') as f:
            data = pickle.load(f)
        vol = None
        for key in ['vol_gt', 'gt', 'volume']:
            if key in data:
                vol = np.array(data[key])
                break
        if vol is None:
            for key in data:
                val = data[key]
                if isinstance(val, np.ndarray) and len(val.shape) == 3 and val.shape[-1] <= 256:
                    vol = val
                    break
        if vol is not None:
            z = vol.shape[0] // 2
            gt_slices[organ] = vol[z]
            print(f"  {organ}: GT shape {vol.shape}, slice {z}")
        else:
            print(f"  {organ}: no 3D volume in pickle (keys: {list(data.keys())[:5]})")
    except Exception as e:
        print(f"  {organ}: error - {e}")

print("Building figure...")
n_organs = len(organs)
n_cols = len(methods) + 1

fig = plt.figure(figsize=(14, 11))
gs = GridSpec(n_organs, n_cols, figure=fig, wspace=0.02, hspace=0.08)

# Load cell data
cell_data = {}
for i, organ in enumerate(organs):
    for j, method in enumerate(methods):
        sp = slices_map.get((organ, method))
        if sp and os.path.exists(sp):
            try:
                img = np.array(Image.open(sp))
                h, w = img.shape[:2]
                cell_h, cell_w = h // 5, w // 5
                center = img[2*cell_h:3*cell_h, 2*cell_w:3*cell_w]
                if center.ndim == 3:
                    center = np.mean(center, axis=2)
                cell_data[(i, j)] = center
            except:
                pass

# Compute global range from all pixels
all_pixels = np.concatenate([d.ravel()[::50] for d in cell_data.values()]) if cell_data else np.array([0, 255])
gmin, gmax = np.percentile(all_pixels, [2, 98])
print(f"Global range: [{gmin:.0f}, {gmax:.0f}]")

# Plot
for i, olabel in enumerate(organ_labels):
    for j, mname in enumerate(method_names):
        ax = fig.add_subplot(gs[i, j])
        if (i, j) in cell_data:
            ax.imshow(cell_data[(i, j)], cmap='gray', vmin=gmin, vmax=gmax)
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=7, transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if methods[j] == 'spags':
            for spine in ax.spines.values():
                spine.set_color('#cc0000'); spine.set_linewidth(2.5); spine.set_visible(True)
    
    # GT col
    ax_gt = fig.add_subplot(gs[i, n_cols-1])
    if organ in gt_slices:
        gt_img = gt_slices[organ]
        vmi = np.percentile(gt_img, 2); vma = np.percentile(gt_img, 98)
        ax_gt.imshow(gt_img, cmap='gray', vmin=vmi, vmax=vma)
    else:
        ax_gt.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=7, transform=ax_gt.transAxes)
    ax_gt.set_xticks([]); ax_gt.set_yticks([])
    for spine in ax_gt.spines.values():
        spine.set_color('green'); spine.set_linewidth(2.5); spine.set_visible(True)

# Column headers
for j, (mname, method) in enumerate(zip(method_names, methods)):
    c = '#cc0000' if method == 'spags' else 'black'
    fig.text((j + 0.5) / n_cols, 0.97, mname, ha='center', va='bottom',
             fontsize=8, fontweight='bold', color=c)
fig.text((n_cols - 0.5) / n_cols, 0.97, 'GT', ha='center', va='bottom',
         fontsize=8, fontweight='bold', color='green')

# Organ labels
for i, olabel in enumerate(organ_labels):
    fig.text(0.01, (n_organs - i - 0.5) / n_organs, olabel,
             ha='center', va='center', fontsize=9, fontweight='bold', rotation=90)

plt.tight_layout(rect=[0.04, 0, 1, 0.95])
fig.savefig(f'{FIGS_DIR}/fig_qualitative.pdf', bbox_inches='tight', dpi=200)
fig.savefig(f'{FIGS_DIR}/fig_qualitative.png', bbox_inches='tight', dpi=200)
print("DONE: fig_qualitative.pdf and .png")
