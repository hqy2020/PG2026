#!/usr/bin/env python3
"""
Extract all ablation experiment data into CSV files for the ablation table.
"""
import os, re, yaml, json

BASE = "/home/qyhu/Documents/r2_ours/PG2026"
OUT = os.path.join(BASE, "data_visualization")

def get_best_eval(output_dir):
    """Find the best PSNR/SSIM across all eval iterations."""
    best_psnr = -1
    best_ssim = None
    best_iter = None
    
    eval_dir = os.path.join(output_dir, "eval")
    if not os.path.isdir(eval_dir):
        # Try test/iter_30000 as fallback
        yml_path = os.path.join(output_dir, "test/iter_30000/eval2d_render_test.yml")
        if os.path.exists(yml_path):
            with open(yml_path) as f:
                data = yaml.safe_load(f)
            return round(data.get("psnr_2d", 0), 4), round(data.get("ssim_2d", 0), 4), "iter_30000"
        return None, None, None
    
    for yml_path in sorted(glob.glob(f"{eval_dir}/iter_*/eval2d_render_test.yml")):
        m = re.search(r'iter_(\d+)', yml_path)
        it = m.group(1) if m else "?"
        with open(yml_path) as f:
            data = yaml.safe_load(f)
        psnr = data.get("psnr_2d", 0)
        ssim = data.get("ssim_2d", 0)
        if psnr > best_psnr:
            best_psnr = psnr
            best_ssim = ssim
            best_iter = f"iter_{it}"
    
    return round(best_psnr, 4), round(best_ssim, 4) if best_ssim else None, best_iter if best_psnr > 0 else None

def get_n_gaussians(output_dir):
    """Get Gaussian count from point cloud pickle."""
    pkl_path = os.path.join(output_dir, "point_cloud/iteration_30000/point_cloud.pickle")
    if not os.path.exists(pkl_path):
        return None
    try:
        import pickle
        data = pickle.load(open(pkl_path, "rb"))
        xyz = data.get("xyz")
        return len(xyz) if xyz is not None else None
    except:
        return None

import glob

# ============================================================
# 1. Component analysis: 5-organ avg 3-view
#    May 2 experiments have all 5 organs for each variant
# ============================================================
print("="*70)
print("1. COMPONENT ANALYSIS (5-organ 3-view)")
print("="*70)

organs = ["chest", "head", "abdomen", "foot", "pancreas"]
component_configs = [
    ("r2_gaussian",  "Baseline (R²-Gaussian)"),
    ("sps_only",     "B + SPS"),
    ("gar_only",     "B + GAP"),
    ("adm_only",     "B + ADM"),
    ("spags",        "Full XRA-GS"),
]

component_results = []
for suffix, label in component_configs:
    organ_results = []
    total_psnr = 0
    total_ssim = 0
    total_gs = 0
    n_organs = 0
    
    for organ in organs:
        dir_name = f"2026_05_02_{organ}_3views_{suffix}"
        dir_path = os.path.join(BASE, "output", dir_name)
        
        if not os.path.isdir(dir_path):
            # Try 2026_04_30 as fallback for spags
            dir_name2 = f"2026_04_30_{organ}_3views_{suffix}"
            dir_path2 = os.path.join(BASE, "output", dir_name2)
            if os.path.isdir(dir_path2):
                dir_path = dir_path2
                dir_name = dir_name2
            else:
                print(f"  ⚠ No dir for {organ} {label}")
                continue
        
        psnr, ssim, best_iter = get_best_eval(dir_path)
        n_gs = get_n_gaussians(dir_path)
        
        if psnr is not None:
            total_psnr += psnr
            total_ssim += ssim
            if n_gs:
                total_gs += n_gs
            n_organs += 1
            organ_results.append((organ, psnr, ssim, n_gs))
            print(f"  {label:25s} {organ:10s}: PSNR={psnr:.4f}, SSIM={ssim:.4f}, GS={n_gs} [{best_iter}]")
        else:
            print(f"  ⚠ No eval for {organ} {label}")
    
    if n_organs == 5:
        avg_psnr = round(total_psnr / 5, 4)
        avg_ssim = round(total_ssim / 5, 4)
        avg_gs = round(total_gs / 5)
        component_results.append({
            "label": label,
            "avg_psnr": avg_psnr,
            "avg_ssim": avg_ssim,
            "avg_gs": avg_gs,
            "details": organ_results
        })
        print(f"  {'─'*50}")
        print(f"  {label:25s} AVG(5): PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}, GS={avg_gs}")

# ============================================================
# 2. Hyperparameter sweeps (Chest 3v only)
# ============================================================

# 2a. SPS α - check cfg_args for sps_only variants
print("\n" + "="*70)
print("2a. SPS α sweep (Chest 3v)")
print("="*70)

# May 2 sps_only is the default (α=0.2)
# May 4 spsv*_gap experiments have SPS+GAP together, not SPS only
# Let's check cfg_args for SPS parameters
for d in sorted(os.listdir(os.path.join(BASE, "output"))):
    if "chest" in d and "3views" in d and ("sps" in d or "spags" in d):
        if "opt_" in d or "adaptive" in d or "retry" in d or "p1" in d:
            continue
        dir_path = os.path.join(BASE, "output", d)
        cfg_path = os.path.join(dir_path, "cfg_args")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = f.read()
            # Check for SPS mix_coef / alpha
            for key in ["mix_coef", "unif_ratio", "init_alpha", "alpha"]:
                for line in cfg.split("\n"):
                    if key in line.lower():
                        print(f"  {d:55s} | {line.strip()}")

print("\n--- Checking SPS unif_ratio in cfg_args ---")
for d in sorted(os.listdir(os.path.join(BASE, "output"))):
    if "chest" in d and "3views" in d and ("sps_only" in d or "spags" in d or "spsv" in d):
        if "opt_" in d or "adaptive" in d or "p1" in d:
            continue
        dir_path = os.path.join(BASE, "output", d)
        cfg_path = os.path.join(dir_path, "cfg_args")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = f.read()
            # Extract unif_ratio or similar
            m = re.search(r'unif_ratio[=\s]+([0-9.]+)', cfg)
            if m:
                print(f"  {d:55s} unif_ratio={m.group(1)}")
            else:
                m2 = re.search(r'mix_coef[=\s]+([0-9.]+)', cfg)
                if m2:
                    print(f"  {d:55s} mix_coef={m2.group(1)}")
                else:
                    print(f"  {d:55s} (no unif_ratio/mix_coef found)")

# 2b. GAP τ - extract from gap_th* experiments
print("\n" + "="*70)
print("2b. GAP τ sweep (Chest 3v)")
print("="*70)

gap_dirs = {}
for d in sorted(os.listdir(os.path.join(BASE, "output"))):
    if "chest" in d and "3views" in d and "gap_th" in d:
        dir_path = os.path.join(BASE, "output", d)
        psnr, ssim, best_iter = get_best_eval(dir_path)
        if psnr:
            # Parse τ from name
            m = re.search(r'th([0-9.p]+)', d)
            tau = m.group(1) if m else "?"
            m2 = re.search(r'_r(\d+)', d)
            r = m2.group(1) if m2 else "?"
            print(f"  {d:55s} τ={tau}, r={r}: PSNR={psnr:.4f}, SSIM={ssim:.4f} [{best_iter}]")
            gap_dirs[d] = {"tau": tau, "r": r, "psnr": psnr, "ssim": ssim}

# Also check gar_only for default (τ=0.015, k=5)
d = "2026_05_02_chest_3views_gar_only"
dir_path = os.path.join(BASE, "output", d)
if os.path.isdir(dir_path):
    psnr, ssim, best_iter = get_best_eval(dir_path)
    cfg_path = os.path.join(dir_path, "cfg_args")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = f.read()
        m = re.search(r'gar_threshold[=\s]+([0-9.]+)', cfg)
        tau = m.group(1) if m else "default?"
    print(f"  {d:55s} τ={tau}: PSNR={psnr:.4f}, SSIM={ssim:.4f} [{best_iter}]")

# 2c. ADM warmup iteration sweep
print("\n" + "="*70)
print("2c. ADM activation iteration sweep (Chest 3v)")
print("="*70)

adm_dirs = {}
for d in sorted(os.listdir(os.path.join(BASE, "output"))):
    if "chest" in d and "3views" in d and "adm_warm" in d:
        dir_path = os.path.join(BASE, "output", d)
        psnr, ssim, best_iter = get_best_eval(dir_path)
        if psnr:
            m = re.search(r'warmup(\d+)', d)
            warm = m.group(1) if m else "?"
            print(f"  {d:55s} warmup={warm}: PSNR={psnr:.4f}, SSIM={ssim:.4f} [{best_iter}]")
            adm_dirs[d] = {"warmup": warm, "psnr": psnr, "ssim": ssim}

# Also check adm_only for default (warmup=15K)
for d in ["2026_05_02_chest_3views_adm_only", "2026_05_04_chest_3views_adm_warm12k", "2026_05_04_chest_3views_adm_warm18k"]:
    dir_path = os.path.join(BASE, "output", d)
    if os.path.isdir(dir_path):
        psnr, ssim, best_iter = get_best_eval(dir_path)
        if psnr:
            cfg_path = os.path.join(dir_path, "cfg_args")
            warm = "?"
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    cfg = f.read()
                m = re.search(r'warm_up_iter[=\s]+(\d+)', cfg)
                if m:
                    warm = m.group(1)
                m2 = re.search(r'warmup_iter[=\s]+(\d+)', cfg)
                if m2:
                    warm = m2.group(1)
            print(f"  {d:55s} warmup={warm}: PSNR={psnr:.4f}, SSIM={ssim:.4f} [{best_iter}]")

# Now check for ADM warmup experiments from May 1 (full pipeline variant sweeps)
print("\n--- Full pipeline ADM warmup variants (May 1) ---")
for d in sorted(os.listdir(os.path.join(BASE, "output"))):
    if "chest" in d and "3views" in d and "adm_warmup" in d and "opt_" in d and "p1" not in d:
        dir_path = os.path.join(BASE, "output", d)
        psnr, ssim, best_iter = get_best_eval(dir_path)
        if psnr:
            m = re.search(r'warmup(\d+)', d)
            warm = m.group(1) if m else "?"
            # Check cfg for what modules are enabled
            cfg_path = os.path.join(dir_path, "cfg_args")
            has_sps = False; has_gap = False
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    cfg = f.read()
                has_sps = "enable_kplanes" in cfg or "unif_ratio" in cfg
                has_gap = "enable_gar" in cfg or "gar_threshold" in cfg
            modules = []
            if has_sps: modules.append("SPS")
            if has_gap: modules.append("GAP")
            tag = "+".join(modules) if modules else "?"
            print(f"  {d:55s} warmup={warm} [{tag}]: PSNR={psnr:.4f}, SSIM={ssim:.4f} [{best_iter}]")

# ============================================================
# 3. Write summary to CSV
# ============================================================
print("\n" + "="*70)
print("3. WRITING CSVs")
print("="*70)

# Component analysis CSV (5-organ avg)
with open(os.path.join(OUT, "ablation_component.csv"), "w") as f:
    f.write("configuration,psnr,ssim,n_gaussians\n")
    for cr in component_results:
        f.write(f"{cr['label']},{cr['avg_psnr']:.4f},{cr['avg_ssim']:.4f},{cr['avg_gs']}\n")
        print(f"  {cr['label']:25s}: {cr['avg_psnr']:.4f}, {cr['avg_ssim']:.4f}, {cr['avg_gs']}")

print(f"\n  Saved to {OUT}/ablation_component.csv")

# Check what configs were used for SPS α sweep
print("\n\nTo determine SPS α values, checking cfg_args for unif_ratio/sps_ratio...")
import subprocess
for d in sorted(os.listdir(os.path.join(BASE, "output"))):
    if "chest" in d and "3views" in d:
        is_sps = "spsv" in d or d.endswith("_sps_only") or d.endswith("_spags")
        if is_sps and "opt_" not in d and "adaptive" not in d and "p1" not in d and "smoke" not in d:
            dir_path = os.path.join(BASE, "output", d)
            cfg_path = os.path.join(dir_path, "cfg_args")
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    cfg = f.read()
                # Look for all sps-related params
                for line in cfg.split("\n"):
                    if any(k in line for k in ["unif", "ratio", "mix", "alpha", "sps", "kplane"]):
                        if "unif" in line or "ratio" in line or "mix" in line:
                            print(f"  {d:55s} | {line.strip()}")
