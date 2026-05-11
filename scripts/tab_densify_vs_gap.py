#!/usr/bin/env python3
"""Generate tab_densify_vs_gap.tex from existing data."""
import os, yaml, pickle

BASE = "/home/qyhu/Documents/r2_ours/PG2026"

def get_metrics(d):
    gs = None
    for p in [f"{d}/point_cloud/iteration_30000/point_cloud.pickle"]:
        if os.path.exists(p):
            with open(p, 'rb') as f:
                data = pickle.load(f)
            xyz = data.get('xyz')
            if xyz is not None: gs = len(xyz)
    psnr = ssim = None
    for yml in [f"{d}/test/iter_30000/eval2d_render_test.yml",
                f"{d}/eval/iter_030000/eval2d_render_test.yml"]:
        if os.path.exists(yml):
            with open(yml) as f:
                for line in f:
                    if line.startswith("psnr_2d:"): psnr = float(line.split(":")[1])
                    if line.startswith("ssim_2d:"): ssim = float(line.split(":")[1])
    return gs, psnr, ssim

data = [
    ("Baseline", f"{BASE}/output/2026_05_02_chest_3views_r2_gaussian"),
    ("More Densify", f"{BASE}/output/chest_densify_test"),
    ("+GAP", f"{BASE}/output/2026_05_02_chest_3views_gar_only"),
]

print("Densify vs GAP comparison (Chest 3-view):")
tex = []
tex.append(r"\begin{table}[t]")
tex.append(r"\centering")
tex.append(r"\caption{Densification vs GAP ablation on Chest 3-view. "
           r"More Densify = aggressive vanilla densification (interval=50, thresh=0.0001, cap=800K). "
           r"+GAP = FSGS proximity densifier + GAP pruning.}")
tex.append(r"\label{tab:densify_vs_gap}")
tex.append(r"\begin{tabular}{lccc}")
tex.append(r"\toprule")
tex.append(r" Config & \#Gaussians & PSNR (dB) & SSIM \\")
tex.append(r"\midrule")

for name, d in data:
    gs, psnr, ssim = get_metrics(d)
    gs_str = f"{gs:,}" if gs else "---"
    psnr_str = f"{psnr:.2f}" if psnr else "---"
    ssim_str = f"{ssim:.4f}" if ssim else "---"
    print(f"  {name:<20} GS={gs_str:>8}  PSNR={psnr_str:>6}  SSIM={ssim_str}")
    tex.append(f" {name} & {gs_str} & {psnr_str} & {ssim_str} \\\\")

tex.append(r"\bottomrule")
tex.append(r"\end{tabular}")
tex.append(r"\end{table}")

os.makedirs(f"{BASE}/tables", exist_ok=True)
with open(f"{BASE}/tables/tab_densify_vs_gap.tex", 'w') as f:
    f.write('\n'.join(tex) + '\n')
print(f"✅ tables/tab_densify_vs_gap.tex")
