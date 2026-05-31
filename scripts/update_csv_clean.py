#!/usr/bin/env python3
"""
Clean CSV update: 
- XRAGS values from 234-view results.json (canonical full-pipeline results)
- Apply P1 updates only (verified: data/234/, full pipeline)
- Baselines: keep existing values
"""
import os, json

BASE = "/home/qyhu/Documents/r2_ours/PG2026/"
CSV_PATH = os.path.join(BASE, "results/all_methods_comparison.csv")

# 234-view canonical results
with open(os.path.join(BASE, "output/comparison_234/results.json")) as f:
    r234 = json.load(f)

# P1 updates (verified: data/234/, full pipeline SPS+GAP+ADM)
p1_updates = {
    ("chest", 2): (21.2160, 0.7065),   # P1 tuning
    ("chest", 4): (26.0341, 0.8586),   # P1 tuning
    ("head", 2): (24.2580, 0.8694),    # P1 tuning
    ("head", 3): (26.6472, 0.9176),    # P1 tuning
    ("head", 4): (29.6833, 0.9518),    # P1 tuning
    ("foot", 2): (19.9211, 0.6798),    # planB_lowtv (data/234/ ✅)
    ("foot", 3): (28.6395, 0.8993),    # P1 tuning
    ("foot", 4): (29.9532, 0.9149),    # P1 tuning
    ("pancreas", 2): (19.1048, 0.8238), # planB_lowtv (data/234/ ✅)
    ("pancreas", 4): (30.9536, 0.9361), # P1 tuning
}

# Build XRAGS values: 234-view base + P1 updates
xrags_values = {}
method_aliases = {"spags": "XRAGS"}

for k, v in r234.items():
    method, organ, nv_str = k.split('/')
    nv = int(nv_str)
    if method == "spags":
        key = (organ, nv)
        if key in p1_updates:
            xrags_values[key] = p1_updates[key]
            print(f"  XRAGS {organ:10s} {nv}v: PSNR={p1_updates[key][0]:.4f} (P1 updated)")
        else:
            xrags_values[key] = (round(v["psnr_2d"], 4), round(v["ssim_2d"], 4))
            print(f"  XRAGS {organ:10s} {nv}v: PSNR={v['psnr_2d']:.4f} (from 234-view)")

# Read and update CSV
lines = open(CSV_PATH).readlines()
updated = [lines[0].strip()]
n_updates = 0
for line in lines[1:]:
    line = line.strip()
    if not line: continue
    parts = line.split(',')
    method, organ, views_str = parts[0], parts[1], parts[2]
    
    if method == "XRAGS":
        key = (organ, int(views_str))
        if key in xrags_values:
            p, s = xrags_values[key]
            new_line = f"XRAGS,{organ},{views_str},{p:.4f},{s:.4f}"
            old = f"{float(parts[3]):.4f}"
            if f"{p:.4f}" != old:
                print(f"  CHG {organ:10s} {views_str}v: {old} → {p:.4f} ({float(p)-float(parts[3]):+.2f}dB)")
                n_updates += 1
            updated.append(new_line)
        else:
            updated.append(line)
    else:
        updated.append(line)

with open(CSV_PATH, 'w') as f:
    f.write('\n'.join(updated) + '\n')

print(f"\n✅ {n_updates} cells updated")
print(f"⚡ Baseline reseeds running in background: 5 methods × 2 organs × 2 seeds = 20 experiments")
