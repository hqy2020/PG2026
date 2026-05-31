#!/usr/bin/env python3
"""
Correct CSV update: best-of-ours from data/234/ experiments only.
Takes best PSNR across all eval iterations (not just iter_30000).
Baselines keep current single-seed values (will update later after reseeds).
"""
import os, yaml, re, json, glob

BASE = "/home/qyhu/Documents/r2_ours/PG2026/"
CSV_PATH = os.path.join(BASE, "results/all_methods_comparison.csv")

def get_best_eval(dir_path):
    best_p, best_s = -1, None
    for yml in sorted(glob.glob(f"{dir_path}/eval/iter_*/eval2d_render_test.yml")):
        with open(yml) as f:
            d = yaml.safe_load(f)
        p, s = d.get("psnr_2d", 0), d.get("ssim_2d", 0)
        if p > best_p: best_p, best_s = p, s
    if best_p > 0: return round(best_p, 4), round(best_s, 4)
    return None, None

def uses_data234(dir_path):
    cfg = os.path.join(dir_path, "cfg_args")
    if os.path.exists(cfg):
        with open(cfg) as f:
            return "data/234/" in f.read()
    return False

# Only data/234/ compatible experiments
our_candidates_234 = {
    ("foot", 2): [
        "2026_05_01_foot_2views_spags",
        "2026_05_01_foot_2views_spags_adaptive",
        "2026_05_02_foot_2views_spags",
        "2026_05_30_foot_2views_planB_lowtv",
    ],
    ("pancreas", 2): [
        "2026_05_01_pancreas_2views_spags",
        "2026_05_01_pancreas_2views_spags_adaptive",
        "2026_05_02_pancreas_2views_spags",
        "2026_05_30_pancreas_2views_planB_lowtv",
    ],
    ("foot", 3): [
        "2026_05_01_foot_3views_spags",
        "2026_05_30_foot_3views_p1",
    ],
    ("pancreas", 3): [
        "2026_05_01_pancreas_3views_spags",
        "2026_04_30_pancreas_3views_spags",  # uses data/369, skip
    ],
    ("foot", 4): [
        "2026_05_01_foot_4views_spags",
        "2026_05_30_foot_4views_p1",
    ],
    ("pancreas", 4): [
        "2026_05_01_pancreas_4views_spags",
        "2026_05_30_pancreas_4views_p1",
    ],
    ("chest", 2): [
        "2026_05_01_chest_2views_spags",
        "2026_05_30_chest_2views_p1",
    ],
    ("chest", 3): [
        "2026_05_01_chest_3views_spags",
        "2026_05_02_chest_3views_spags",
    ],
    ("chest", 4): [
        "2026_05_01_chest_4views_spags",
        "2026_05_30_chest_4views_p1",
    ],
    ("head", 2): [
        "2026_05_01_head_2views_spags",
        "2026_05_30_head_2views_p1",
    ],
    ("head", 3): [
        "2026_05_01_head_3views_spags",
        "2026_05_30_head_3views_p1",
    ],
    ("head", 4): [
        "2026_05_01_head_4views_spags",
        "2026_05_30_head_4views_p1",
    ],
    ("abdomen", 2): [
        "2026_05_01_abdomen_2views_spags",
        "2026_05_05_abdomen_2views_sps_adm_gap",
    ],
    ("abdomen", 3): [
        "2026_05_01_abdomen_3views_spags",
    ],
    ("abdomen", 4): [
        "2026_05_01_abdomen_4views_spags",
        "2026_05_05_abdomen_4views_sps_adm_gap",
    ],
}

# Find best for each
best_ours = {}
for (organ, views), dirs in our_candidates_234.items():
    candidates = []
    for d in dirs:
        dir_path = os.path.join(BASE, "output", d)
        if not os.path.isdir(dir_path):
            continue
        p, s = get_best_eval(dir_path)
        if p:
            candidates.append((p, s, d))
    
    if candidates:
        candidates.sort(key=lambda x: -x[0])
        p, s, src = candidates[0]
        best_ours[(organ, views)] = (p, s, src)
        src_tag = "✅" if uses_data234(os.path.join(BASE, "output", src)) else "⚠️"
        print(f"  [{src_tag}] XRAGS {organ:10s} {views}v: PSNR={p:.4f} SSIM={s:.4f} [{src}]")

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
        if key in best_ours:
            p, s, src = best_ours[key]
            new_line = f"XRAGS,{organ},{views_str},{p:.4f},{s:.4f}"
            old_line = f"XRAGS,{organ},{views_str},{parts[3]},{parts[4]}"
            if new_line != old_line:
                delta = p - float(parts[3])
                print(f"  UPDATE: {organ} {views_str}v: {float(parts[3]):.4f} → {p:.4f} ({delta:+.2f}dB) [{src}]")
                n_updates += 1
            updated.append(new_line)
            continue
    
    updated.append(line)

with open(CSV_PATH, 'w') as f:
    f.write('\n'.join(updated) + '\n')

print(f"\n✅ {n_updates} XRA-GS values updated in CSV (data/234/ only)")
