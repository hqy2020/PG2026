#!/usr/bin/env python3
"""
Update all CSV files with latest P1-optimized XRA-GS results.
"""
import os, json, yaml, re

BASE = "/home/qyhu/Documents/r2_ours/PG2026"

# =======================================================
# P1 实验最新结果 (May 30 tuning)
# =======================================================
p1_results = {
    "chest_2views_p1":  (21.2160, 0.7065),
    "chest_4views_p1":  (26.0341, 0.8586),
    "head_2views_p1":   (24.2580, 0.8694),
    "head_3views_p1":   (26.6472, 0.9176),
    "head_4views_p1":   (29.6833, 0.9518),
    "foot_3views_p1":   (28.6395, 0.8993),
    "foot_4views_p1":   (29.9532, 0.9149),
    "pancreas_4views_p1": (30.9536, 0.9361),
}

# map P1 name -> (organ, views)
p1_map = {
    "chest_2views_p1":  ("chest", 2),
    "chest_4views_p1":  ("chest", 4),
    "head_2views_p1":   ("head", 2),
    "head_3views_p1":   ("head", 3),
    "head_4views_p1":   ("head", 4),
    "foot_3views_p1":   ("foot", 3),
    "foot_4views_p1":   ("foot", 4),
    "pancreas_4views_p1": ("pancreas", 4),
}

# =======================================================
# 1. 读取原始 results.json (234-view sweep)
# =======================================================
with open(os.path.join(BASE, "output/comparison_234/results.json")) as f:
    original_results = json.load(f)

# Build lookup: (method, organ, views) -> (psnr, ssim)
orig_lookup = {}
for k, v in original_results.items():
    method, organ, nv = k.split('/')
    orig_lookup[(method, organ, int(nv))] = (v["psnr_2d"], v["ssim_2d"])

# =======================================================
# 2. Build updated XRA-GS values (P1 wins over original)
# =======================================================
def get_best_xrags(organ, views):
    """Get best XRA-GS result: P1 takes priority over original."""
    key = f"{organ}_{views}views_p1"
    if key in p1_results:
        psnr, ssim = p1_results[key]
        # Round to 4 decimal places
        return round(psnr, 4), round(ssim, 4)
    # Fall back to original
    if ("spags", organ, views) in orig_lookup:
        return orig_lookup[("spags", organ, views)]
    return (None, None)

# Verify which ones changed
print("=== XRA-GS Updated Values ===")
for organ in ["chest", "head", "abdomen", "foot", "pancreas"]:
    for views in [2, 3, 4]:
        new_psnr, new_ssim = get_best_xrags(organ, views)
        old_psnr, old_ssim = orig_lookup.get(("spags", organ, views), (None, None))
        if new_psnr != old_psnr or new_ssim != old_ssim:
            print(f"  {organ} {views}v: {old_psnr:.4f}/{old_ssim:.4f} → {new_psnr:.4f}/{new_ssim:.4f} {'⬆' if new_psnr > old_psnr else '⬇'}")

# =======================================================
# 3. Update results/all_methods_comparison.csv
# =======================================================
print("\n=== Updating results/all_methods_comparison.csv ===")
csv_path = os.path.join(BASE, "results/all_methods_comparison.csv")
lines = open(csv_path).readlines()
header = lines[0].strip()
updated = [header]

n_changed = 0
for line in lines[1:]:
    line = line.strip()
    if not line:
        continue
    parts = line.split(',')
    method, organ, views_str, psnr_str, ssim_str = parts[0], parts[1], parts[2], parts[3], parts[4]
    
    if method == "XRAGS":
        views = int(views_str)
        new_psnr, new_ssim = get_best_xrags(organ, views)
        if new_psnr is not None:
            # Round to 2 or 4 decimal places matching existing format
            new_line = f"XRAGS,{organ},{views},{new_psnr:.4f},{new_ssim:.4f}"
            if new_line != line:
                print(f"  {organ} {views}v: {line} → {new_line}")
                n_changed += 1
            updated.append(new_line)
            continue
    
    updated.append(line)

with open(csv_path, 'w') as f:
    f.write('\n'.join(updated) + '\n')
print(f"  Changed {n_changed} rows")

# =======================================================
# 4. Update data_visualization/comparison_psnr.csv
# =======================================================
print("\n=== Updating data_visualization/comparison_psnr.csv ===")
csv_path2 = os.path.join(BASE, "data_visualization/comparison_psnr.csv")
lines2 = open(csv_path2).readlines()
updated2 = [lines2[0].strip()]

n_changed2 = 0
for line in lines2[1:]:
    line = line.strip()
    if not line:
        continue
    parts = line.split(',')
    method, organ, views_str, psnr_str, ssim_str = parts[0], parts[1], parts[2], parts[3], parts[4]
    
    if method == "spags":
        views = int(views_str)
        new_psnr, new_ssim = get_best_xrags(organ, views)
        if new_psnr is not None:
            new_line = f"spags,{organ},{views},{new_psnr:.4f},{new_ssim:.4f}"
            if new_line != line:
                print(f"  {organ} {views}v: {line} → {new_line}")
                n_changed2 += 1
            updated2.append(new_line)
            continue
    
    updated2.append(line)

with open(csv_path2, 'w') as f:
    f.write('\n'.join(updated2) + '\n')
print(f"  Changed {n_changed2} rows")

# =======================================================
# 5. Update data_visualization/comparison_with_xfield.csv
# =======================================================
print("\n=== Updating data_visualization/comparison_with_xfield.csv ===")
csv_path3 = os.path.join(BASE, "data_visualization/comparison_with_xfield.csv")
lines3 = open(csv_path3).readlines()
updated3 = [lines3[0].strip()]

n_changed3 = 0
for line in lines3[1:]:
    line = line.strip()
    if not line:
        continue
    parts = line.split(',')
    method, organ, views_str, psnr_str, ssim_str = parts[0], parts[1], parts[2], parts[3], parts[4]
    
    if method == "spags":
        views = int(views_str)
        new_psnr, new_ssim = get_best_xrags(organ, views)
        if new_psnr is not None:
            new_line = f"spags,{organ},{views},{new_psnr:.4f},{new_ssim:.4f}"
            if new_line != line:
                print(f"  {organ} {views}v: {line} → {new_line}")
                n_changed3 += 1
            updated3.append(new_line)
            continue
    
    updated3.append(line)

with open(csv_path3, 'w') as f:
    f.write('\n'.join(updated3) + '\n')
print(f"  Changed {n_changed3} rows")

# =======================================================
# 6. efficiency.csv — no changes needed (no new efficiency runs)
# =======================================================
print("\n=== efficiency.csv ===")
print("  No changes (no new efficiency experiments from P1 tuning)")

# =======================================================
# 7. Summary
# =======================================================
print("\n" + "="*60)
print("FINAL UPDATED XRA-GS VALUES:")
print("="*60)
print(f"{'Organ':12s} {'Views':6s} {'PSNR':8s} {'SSIM':8s} {'vs Original':12s}")
print("-"*46)
for organ in ["chest", "head", "abdomen", "foot", "pancreas"]:
    for views in [2, 3, 4]:
        new_psnr, new_ssim = get_best_xrags(organ, views)
        old_psnr, old_ssim = orig_lookup.get(("spags", organ, views), (None, None))
        if old_psnr:
            delta = new_psnr - old_psnr
            delta_str = f"+{delta:.2f}dB" if delta > 0 else (f"{delta:.2f}dB" if delta < 0 else "same")
            print(f"{organ:12s} {views:<6d} {new_psnr:<8.4f} {new_ssim:<8.4f} {delta_str:>12s}")
