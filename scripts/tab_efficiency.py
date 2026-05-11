#!/usr/bin/env python3
"""
P0-2: Precise Efficiency Table.
Extract: Train time, Inference FPS, Gaussian count, Relative overhead.
Output: paper/tables/tab_efficiency.tex
"""
import os, sys, glob, re, yaml
import numpy as np

BASE = "/home/qyhu/Documents/r2_ours/PG2026"
TABLES_DIR = os.path.join(BASE, "tables")
os.makedirs(TABLES_DIR, exist_ok=True)

# Use data/369/ 3-view runs (which have complete test results)
ORGAN_RUNS_369 = {
    "chest": {
        "r2_gaussian": f"{BASE}/output/2026_04_30_chest_3views_r2_gaussian",
        "spags":       f"{BASE}/output/2026_04_30_chest_3views_spags",
    },
    "head": {
        "r2_gaussian": f"{BASE}/output/2026_04_30_head_3views_r2_gaussian",
        "spags":       f"{BASE}/output/2026_04_30_head_3views_spags",
    },
    "pancreas": {
        "r2_gaussian": f"{BASE}/output/2026_04_30_pancreas_3views_r2_gaussian",
        "spags":       f"{BASE}/output/2026_04_30_pancreas_3views_spags",
    },
    "abdomen": {
        "r2_gaussian": f"{BASE}/output/2026_04_30_abdomen_3views_r2_gaussian",
        "spags":       f"{BASE}/output/2026_04_30_abdomen_3views_spags",
    },
    "foot": {
        "r2_gaussian": f"{BASE}/output/2026_04_30_foot_3views_r2_gaussian",
        "spags":       f"{BASE}/output/2026_04_30_foot_3views_spags",
    },
}

def count_gaussians(run_dir):
    """Count Gaussians from point cloud pickle."""
    import pickle
    pkl = f"{run_dir}/point_cloud/iteration_30000/point_cloud.pickle"
    if not os.path.exists(pkl):
        # Try eval dirs
        pkls = glob.glob(f"{run_dir}/point_cloud/iteration_*/point_cloud.pickle")
        if pkls:
            pkl = sorted(pkls)[-1]
        else:
            return None
    with open(pkl, 'rb') as f:
        data = pickle.load(f)
    xyz = data.get('xyz')
    if xyz is None:
        return None
    return len(xyz)


def get_fps_from_eval(run_dir):
    """Estimate FPS from eval render timings."""
    # Check test dir first
    for yml_path in [
        f"{run_dir}/test/iter_30000/eval2d_render_test.yml",
        f"{run_dir}/eval/iter_030000/eval2d_render_test.yml",
    ]:
        if not os.path.exists(yml_path):
            continue
        with open(yml_path) as f:
            for line in f:
                if "render_time" in line.lower() or "fps" in line.lower():
                    parts = line.split(":")
                    if len(parts) == 2:
                        try:
                            val = float(parts[1].strip())
                            if "fps" in line.lower():
                                return val
                        except:
                            pass
    # If no timing in YAML, use the PSNR + view count to infer
    # Default: ~20ms per render view (50 FPS for 3GS)
    return None


def get_psnr_ssim(run_dir):
    """Extract PSNR and SSIM from test eval YAML."""
    for yml_path in [
        f"{run_dir}/test/iter_30000/eval2d_render_test.yml",
        f"{run_dir}/eval/iter_030000/eval2d_render_test.yml",
    ]:
        if os.path.exists(yml_path):
            with open(yml_path) as f:
                content = f.read()
            psnr = None; ssim = None
            for line in content.split('\n'):
                if line.startswith("psnr_2d:"):
                    psnr = float(line.split(":")[1].strip())
                if line.startswith("ssim_2d:"):
                    ssim = float(line.split(":")[1].strip())
            return psnr, ssim
    return None, None


def main():
    print("=" * 60)
    print("P0-2: Precise Efficiency Table")
    print("=" * 60)
    
    ORGANS = ["chest", "head", "pancreas", "abdomen", "foot"]
    results = {}
    
    for organ in ORGANS:
        print(f"\n--- {organ.capitalize()} ---")
        for cfg in ["r2_gaussian", "spags"]:
            run_dir = ORGAN_RUNS_369[organ][cfg]
            if not os.path.exists(run_dir):
                print(f"  {cfg}: ❌ missing")
                continue
            
            n_gauss = count_gaussians(run_dir)
            psnr, ssim = get_psnr_ssim(run_dir)
            
            print(f"  {cfg}: GS={n_gauss:,}, PSNR={psnr:.2f}, SSIM={ssim:.4f}")
            
            if organ not in results:
                results[organ] = {}
            results[organ][cfg] = {
                "n_gauss": n_gauss,
                "psnr": psnr,
                "ssim": ssim,
            }
    
    # Print summary table
    print("\n" + "=" * 60)
    print("EFFICIENCY TABLE")
    print("=" * 60)
    
    header = f"{'Organ':<12} {'Method':<16} {'#Gaussians':<12} {'PSNR':<8} {'SSIM':<8}"
    print(header)
    print("-" * 60)
    
    for organ in ORGANS:
        for cfg in ["r2_gaussian", "spags"]:
            if organ not in results or cfg not in results[organ]:
                continue
            r = results[organ][cfg]
            ng = f"{r['n_gauss']:,}" if r['n_gauss'] else "N/A"
            ps = f"{r['psnr']:.2f}" if r['psnr'] else "N/A"
            ss = f"{r['ssim']:.4f}" if r['ssim'] else "N/A"
            print(f"{organ.capitalize():<12} {cfg:<16} {ng:<12} {ps:<8} {ss:<8}")
    
    # Generate LaTeX table
    print("\n--- Generating tab_efficiency.tex ---")
    
    tex = []
    tex.append(r"\begin{table}[t]")
    tex.append(r"\centering")
    tex.append(r"\caption{Efficiency comparison between R\textsuperscript{2}-Gaussian and SPAGS.}")
    tex.append(r"\label{tab:efficiency}")
    tex.append(r"\begin{tabular}{lcccc}")
    tex.append(r"\toprule")
    tex.append(r" Organ & Method & \#Gaussians & PSNR (dB) & SSIM \\")
    tex.append(r"\midrule")
    
    for organ in ORGANS:
        for cfg in ["r2_gaussian", "spags"]:
            if organ not in results or cfg not in results[organ]:
                continue
            r = results[organ][cfg]
            ng = f"{r['n_gauss']:,}" if r['n_gauss'] else "---"
            ps = f"{r['psnr']:.2f}" if r['psnr'] else "---"
            ss = f"{r['ssim']:.4f}" if r['ssim'] else "---"
            method = r"\r2gaussian" if cfg == "r2_gaussian" else r"\spags"
            tex.append(f" {organ.capitalize()} & {method} & {ng} & {ps} & {ss} \\\\")
        tex.append(r"\cmidrule{1-5}")
    
    # Averages
    tex.append(r"\midrule")
    avg_r2_gs = np.mean([results[o]["r2_gaussian"]["n_gauss"] for o in ORGANS if o in results and "r2_gaussian" in results[o]])
    avg_sp_gs = np.mean([results[o]["spags"]["n_gauss"] for o in ORGANS if o in results and "spags" in results[o]])
    avg_r2_ps = np.mean([results[o]["r2_gaussian"]["psnr"] for o in ORGANS if o in results and "r2_gaussian" in results[o]])
    avg_sp_ps = np.mean([results[o]["spags"]["psnr"] for o in ORGANS if o in results and "spags" in results[o]])
    avg_r2_ss = np.mean([results[o]["r2_gaussian"]["ssim"] for o in ORGANS if o in results and "r2_gaussian" in results[o]])
    avg_sp_ss = np.mean([results[o]["spags"]["ssim"] for o in ORGANS if o in results and "spags" in results[o]])
    
    r2name = r"\r2gaussian"
    spname = r"\spags"
    tex.append(f" Avg & {r2name} & {avg_r2_gs:,.0f} & {avg_r2_ps:.2f} & {avg_r2_ss:.4f} \\\\")
    tex.append(f" Avg & {spname}      & {avg_sp_gs:,.0f} & {avg_sp_ps:.2f} & {avg_sp_ss:.4f} \\\\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    
    tex_path = f"{TABLES_DIR}/tab_efficiency.tex"
    with open(tex_path, 'w') as f:
        f.write('\n'.join(tex) + '\n')
    print(f"  ✅ {tex_path}")
    
    # Also print averages
    print(f"\n  Avg R²-Gaussian: GS={avg_r2_gs:,.0f}, PSNR={avg_r2_ps:.2f}, SSIM={avg_r2_ss:.4f}")
    print(f"  Avg SPAGS:      GS={avg_sp_gs:,.0f}, PSNR={avg_sp_ps:.2f}, SSIM={avg_sp_ss:.4f}")
    print(f"  PSNR gain: +{avg_sp_ps - avg_r2_ps:.2f} dB")
    print(f"  GS increase: {(avg_sp_gs - avg_r2_gs) / avg_r2_gs * 100:+.1f}%")


if __name__ == "__main__":
    main()
