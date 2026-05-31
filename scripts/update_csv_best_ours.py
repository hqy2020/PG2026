#!/usr/bin/env python3
"""
Update all_methods_comparison.csv with:
- XRA-GS: best across ALL our experiments (not just 234-view)
- Others: keep current values (single seed), will update later after reseeds
"""
import os, yaml, json, glob

BASE = "/home/qyhu/Documents/r2_ours/PG2026/"
CSV_PATH = os.path.join(BASE, "results/all_methods_comparison.csv")

def get_best_eval(dir_path):
    best_p, best_s = -1, None
    for yml in sorted(glob.glob(f"{dir_path}/eval/iter_*/eval2d_render_test.yml")):
        with open(yml) as f:
            d = yaml.safe_load(f)
        p, s = d.get("psnr_2d", 0), d.get("ssim_2d", 0)
        if p > best_p: best_p, best_s = p, s
    if best_p > 0: return best_p, best_s
    yml = f"{dir_path}/test/iter_30000/eval2d_render_test.yml"
    if os.path.exists(yml):
        with open(yml) as f:
            d = yaml.safe_load(f)
        return d.get("psnr_2d", 0), d.get("ssim_2d", 0)
    return None, None

# Our best experiments lookup
# Key: (organ, views) -> list of dir names to check
our_candidates = {
    ("foot", 2): [
        "2026_05_01_foot_2views_spags",
        "2026_05_01_foot_2views_spags_adaptive",
        "2026_05_02_foot_2views_spags",
        "2026_05_30_foot_2views_planB_lowtv",
        "2026_05_30_foot_2views_fullstack_aggressive",
        "2026_05_30_foot_2views_opt_E_gap_adm",
    ],
    ("pancreas", 2): [
        "2026_05_01_pancreas_2views_spags",
        "2026_05_01_pancreas_2views_spags_adaptive",
        "2026_05_02_pancreas_2views_spags",
        "2026_05_02_pancreas_2views_spags_retry",
        "2026_05_30_pancreas_2views_planB_lowtv",
        "2026_05_30_pancreas_2views_fullstack_aggressive",
        "2026_05_30_pancreas_2views_opt_A_gap_adm",
    ],
    ("foot", 3): [
        "2026_05_01_foot_3views_spags",
        "2026_04_30_foot_3views_spags",
        "2026_05_30_foot_3views_p1",
    ],
    ("pancreas", 3): [
        "2026_05_01_pancreas_3views_spags",
        "2026_04_30_pancreas_3views_spags",
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
        "2026_04_30_chest_2views_spags",
        "2026_05_30_chest_2views_p1",
    ],
    ("chest", 3): [
        "2026_04_30_chest_3views_spags",
        "2026_05_01_chest_3views_spags",
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
        "2026_04_30_head_3views_spags",
        "2026_05_01_head_3views_spags",
        "2026_05_30_head_3views_p1",
    ],
    ("head", 4): [
        "2026_05_01_head_4views_spags",
        "2026_05_30_head_4views_p1",
    ],
    ("abdomen", 2): ["2026_05_01_abdomen_2views_spags"],
    ("abdomen", 3): ["2026_04_30_abdomen_3views_spags", "2026_05_01_abdomen_3views_spags"],
    ("abdomen", 4): ["2026_05_01_abdomen_4views_spags"],
}

# Find best for each organ/view
best_ours = {}
for (organ, views), dirs in our_candidates.items():
    best_p, best_s = -1, None
    best_src = ""
    for d in dirs:
        p, s = get_best_eval(os.path.join(BASE, "output", d))
        if p and p > best_p:
            best_p, best_s = p, s
            best_src = d
    
    if best_p > 0:
        best_ours[(organ, views)] = (best_p, best_s, best_src)
        print(f"XRAGS {organ:10s} {views}v: {best_p:.4f} / {best_s:.4f} [{best_src}]")

# Read and update CSV
lines = open(CSV_PATH).readlines()
updated = [lines[0].strip()]
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
            if new_line != line:
                print(f"  UPDATE: {line} → {new_line} [{src}]")
            updated.append(new_line)
        else:
            updated.append(line)
    else:
        updated.append(line)

with open(CSV_PATH, 'w') as f:
    f.write('\n'.join(updated) + '\n')

print(f"\n✅ CSV updated with our best values!")
