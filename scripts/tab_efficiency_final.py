#!/usr/bin/env python3
"""
P0-2 (R2 final): Efficiency Table with actual timing data.
"""
import os, sys, yaml, pickle, glob

BASE = "/home/qyhu/Documents/r2_ours/PG2026"

def get_timing(run_dir):
    for path in [f"{run_dir}/test/iter_30000/timing_render_test.yml"]:
        if os.path.exists(path):
            with open(path) as f:
                return yaml.safe_load(f)
    return None

def get_gs(run_dir):
    for pkl in [f"{run_dir}/point_cloud/iteration_30000/point_cloud.pickle"]:
        if os.path.exists(pkl):
            with open(pkl, 'rb') as f:
                d = pickle.load(f)
            xyz = d.get('xyz')
            if xyz is not None: return len(xyz)
    return None

def get_psnr(run_dir):
    for yml in [f"{run_dir}/test/iter_30000/eval2d_render_test.yml",
                f"{run_dir}/eval/iter_030000/eval2d_render_test.yml"]:
        if os.path.exists(yml):
            with open(yml) as f:
                for line in f:
                    if line.startswith("psnr_2d:"):
                        return float(line.split(":")[1])
    return None

rows = [
    ("Baseline (R²-Gaussian)", f"{BASE}/output/2026_05_02_chest_3views_r2_gaussian"),
    ("More Densify", f"{BASE}/output/chest_densify_test"),
    ("+GAP", f"{BASE}/output/2026_05_02_chest_3views_gar_only"),
]

tex = []
tex.append(r"\begin{table}[t]")
tex.append(r"\centering")
tex.append(r"\caption{Efficiency and quality comparison on Chest 3-view (data/234/). "
           r"Inference timing measured with CUDA events at test time.}")
tex.append(r"\label{tab:efficiency}")
tex.append(r"\begin{tabular}{lccccc}")
tex.append(r"\toprule")
tex.append(r" Method & \#Gaussians & PSNR (dB) & SSIM & FPS & ms/view \\")
tex.append(r"\midrule")

for name, d in rows:
    gs = get_gs(d)
    psnr = get_psnr(d)
    timing = get_timing(d)
    fps = timing["fps"] if timing else "---"
    ms = f"{timing['avg_render_time_per_view_ms']:.1f}" if timing else "---"
    
    # Get SSIM from eval
    ssim = "---"
    for yml in [f"{d}/test/iter_30000/eval2d_render_test.yml",
                f"{d}/eval/iter_030000/eval2d_render_test.yml"]:
        if os.path.exists(yml):
            with open(yml) as f:
                for line in f:
                    if line.startswith("ssim_2d:"):
                        ssim = f"{float(line.split(':')[1]):.4f}"
    
    gs_str = f"{gs:,}" if gs else "---"
    psnr_str = f"{psnr:.2f}" if psnr else "---"
    tex.append(f" {name} & {gs_str} & {psnr_str} & {ssim} & {fps} & {ms} \\\\")

tex.append(r"\bottomrule")
tex.append(r"\end{tabular}")
tex.append(r"\end{table}")

with open(f"{BASE}/tables/tab_efficiency.tex", 'w') as f:
    f.write('\n'.join(tex) + '\n')

# Also print
for name, d in rows:
    gs = get_gs(d)
    psnr = get_psnr(d)
    timing = get_timing(d)
    fps = timing["fps"] if timing else "N/A"
    ms = f"{timing['avg_render_time_per_view_ms']:.1f}" if timing else "N/A"
    print(f"  {name:<30} GS={gs or 0:>7,}  PSNR={psnr or 0:>6.2f}  FPS={fps}  ms={ms}")

print("✅ tab_efficiency.tex generated")
