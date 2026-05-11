#!/usr/bin/env python3
"""
P1-3: SSIM supplement table.
Collects SSIM from all existing test results across methods and organs.
Output: supp/tab_ssim_supp.tex
"""
import os, sys, glob

BASE = "/home/qyhu/Documents/r2_ours/PG2026"
SUPP_DIR = os.path.join(BASE, "supp")
os.makedirs(SUPP_DIR, exist_ok=True)

# Use data/369/ runs for main table
ORGANS = ["chest", "head", "pancreas", "abdomen", "foot"]
METHODS = ["dngaussian", "corgs", "fsgs", "xgaussian", "r2_gaussian", "spags"]
METHOD_NAMES = {
    "dngaussian": "DN-Gaussian",
    "corgs": "CoR-GS",
    "fsgs": "FSGS",
    "xgaussian": "X-Gaussian",
    "r2_gaussian": "R$^2$-Gaussian",
    "spags": "SPAGS",
}

def get_metrics(run_dir):
    for yml in [
        f"{run_dir}/test/iter_30000/eval2d_render_test.yml",
        f"{run_dir}/eval/iter_030000/eval2d_render_test.yml",
    ]:
        if os.path.exists(yml):
            psnr = ssim = None
            with open(yml) as f:
                for line in f:
                    if line.startswith("psnr_2d:"):
                        psnr = float(line.split(":")[1])
                    if line.startswith("ssim_2d:"):
                        ssim = float(line.split(":")[1])
            return psnr, ssim
    return None, None


def main():
    print("=== P1-3: SSIM Supplement ===")
    
    table = []
    header = ["Organ"] + [METHOD_NAMES[m] for m in METHODS]
    table.append(" & ".join(header) + r" \\")
    table.append(r"\midrule")
    
    for organ in ORGANS:
        row = [organ.capitalize()]
        for method in METHODS:
            dirs = sorted(glob.glob(f"{BASE}/output/*{organ}_3views_{method}*"))
            found = False
            for d in dirs:
                dn = os.path.basename(d)
                if any(x in dn for x in ['opt_', 'adaptive', 'retry', 'pprune', 'spsv', 'gap_th', 'adm_']):
                    continue
                psnr, ssim = get_metrics(d)
                if ssim is not None:
                    row.append(f"{ssim:.4f}")
                    found = True
                    break
            if not found:
                row.append("---")
        table.append(" & ".join(row) + r" \\")
    
    # Average
    avg_row = ["Avg"]
    for method in METHODS:
        vals = []
        for organ in ORGANS:
            dirs = sorted(glob.glob(f"{BASE}/output/*{organ}_3views_{method}*"))
            for d in dirs:
                dn = os.path.basename(d)
                if any(x in dn for x in ['opt_', 'adaptive', 'retry', 'pprune', 'spsv', 'gap_th', 'adm_']):
                    continue
                _, ssim = get_metrics(d)
                if ssim is not None:
                    vals.append(ssim)
                    break
        if vals:
            avg_row.append(f"{sum(vals)/len(vals):.4f}")
        else:
            avg_row.append("---")
    table.append(r"\midrule")
    table.append(" & ".join(avg_row) + r" \\")
    
    tex = []
    tex.append(r"\begin{table}[t]")
    tex.append(r"\centering")
    tex.append(r"\caption{Full SSIM comparison across all methods and organs (3-view setting).}")
    tex.append(r"\label{tab:ssim_full}")
    tex.append(r"\begin{tabular}{l" + "c" * len(METHODS) + "}")
    tex.append(r"\toprule")
    tex.extend(table)
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    
    out = f"{SUPP_DIR}/tab_ssim_supp.tex"
    with open(out, 'w') as f:
        f.write('\n'.join(tex) + '\n')
    print(f"  ✅ {out}")
    
    # Also print summary
    print("\n--- SSIM Summary ---")
    for m in METHODS:
        vals = []
        for o in ORGANS:
            dirs = sorted(glob.glob(f"{BASE}/output/*{o}_3views_{m}*"))
            for d in dirs:
                dn = os.path.basename(d)
                if any(x in dn for x in ['opt_', 'adaptive', 'retry', 'pprune', 'spsv', 'gap_th', 'adm_']):
                    continue
                _, s = get_metrics(d)
                if s is not None:
                    vals.append(s)
                    break
        if vals:
            print(f"  {METHOD_NAMES[m]:<15}: {sum(vals)/len(vals):.4f}")
    
    # Also print PSNR for completeness
    print("\n--- PSNR Summary ---")
    for m in METHODS:
        vals = []
        for o in ORGANS:
            dirs = sorted(glob.glob(f"{BASE}/output/*{o}_3views_{m}*"))
            for d in dirs:
                dn = os.path.basename(d)
                if any(x in dn for x in ['opt_', 'adaptive', 'retry', 'pprune', 'spsv', 'gap_th', 'adm_']):
                    continue
                p, _ = get_metrics(d)
                if p is not None:
                    vals.append(p)
                    break
        if vals:
            print(f"  {METHOD_NAMES[m]:<15}: {sum(vals)/len(vals):.2f}")


if __name__ == "__main__":
    main()
