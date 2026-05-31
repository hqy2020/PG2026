#!/usr/bin/env python3
"""
Comprehensive ablation data extraction.
Reads cfg_args directly to determine enabled modules, then extracts eval data.
Outputs clean CSVs for the ablation table.
"""
import os, re, yaml, json, pickle, glob

BASE = "/home/qyhu/Documents/r2_ours/PG2026/"

def parse_cfg(cfg_path):
    """Parse cfg_args Namespace string to dict."""
    with open(cfg_path) as f:
        text = f.read()
    params = {}
    # Match key=value pairs
    for m in re.finditer(r'(\w+)=([^,)]+)(?:[,)]|$)', text):
        key = m.group(1)
        val = m.group(2).strip()
        params[key] = val
    return params

def get_best_eval(output_dir):
    """Get best PSNR/SSIM across eval iterations."""
    best_psnr = -1
    best_ssim = None
    
    for yml_path in sorted(glob.glob(f"{output_dir}/eval/iter_*/eval2d_render_test.yml")):
        with open(yml_path) as f:
            data = yaml.safe_load(f)
        psnr = data.get("psnr_2d", 0)
        ssim = data.get("ssim_2d", 0)
        if psnr > best_psnr:
            best_psnr = psnr
            best_ssim = ssim
    
    if best_psnr > 0:
        return round(best_psnr, 4), round(best_ssim, 4)
    
    # Fallback to test/iter_30000
    yml_path = f"{output_dir}/test/iter_30000/eval2d_render_test.yml"
    if os.path.exists(yml_path):
        with open(yml_path) as f:
            data = yaml.safe_load(f)
        return round(data.get("psnr_2d", 0), 4), round(data.get("ssim_2d", 0), 4)
    
    return None, None

def get_n_gaussians(output_dir):
    pkl_path = f"{output_dir}/point_cloud/iteration_30000/point_cloud.pickle"
    if not os.path.exists(pkl_path):
        return None
    try:
        data = pickle.load(open(pkl_path, "rb"))
        xyz = data.get("xyz")
        return len(xyz) if xyz is not None else None
    except:
        return None

# ==========================================================
# 1. Component Analysis — find ALL 5-organ experiments
# ==========================================================
print("="*70)
print("COMPONENT ANALYSIS — checking all May 2 experiments for enabled modules")
print("="*70)

organs = ["chest", "head", "abdomen", "foot", "pancreas"]
variants = ["r2_gaussian", "sps_only", "gar_only", "adm_only", "sps_adm", "sps_gar", "gar_adm", "spags"]

for suffix in variants:
    for organ in organs:
        dir_name = f"2026_05_02_{organ}_3views_{suffix}"
        dir_path = os.path.join(BASE, "output", dir_name)
        if not os.path.isdir(dir_path):
            continue
        cfg_path = os.path.join(dir_path, "cfg_args")
        params = parse_cfg(cfg_path) if os.path.exists(cfg_path) else {}
        
        modules = []
        if params.get("enable_kplanes") == "True":
            modules.append("KPlanes")
        if params.get("enable_fsgs_proximity") == "True":
            modules.append("FSGSprox")
        if params.get("enable_gar_proximity") == "True" or params.get("enable_gar") == "True":
            modules.append("GAR")
        if params.get("enable_gap") == "True":
            modules.append("GAP")
        if params.get("enable_adm") == "True":
            modules.append("ADM")
        
        psnr, ssim = get_best_eval(dir_path)
        ngs = get_n_gaussians(dir_path)
        
        mod_str = "+".join(modules) if modules else "NONE"
        print(f"  {dir_name:50s} | Modules: {mod_str:20s} | PSNR={psnr:.4f} SSIM={ssim} GS={ngs}")

# Check specific cases for ADM
print("\n\nChecking all cfg_args for enable_adm=True...")
for d in sorted(os.listdir(os.path.join(BASE, "output"))):
    cfg_path = os.path.join(BASE, "output", d, "cfg_args")
    if not os.path.exists(cfg_path):
        continue
    params = parse_cfg(cfg_path)
    if params.get("enable_adm") == "True":
        psnr, ssim = get_best_eval(os.path.join(BASE, "output", d))
        modules = []
        for k in ["enable_kplanes", "enable_fsgs_proximity", "enable_gar_proximity", "enable_gap", "enable_gar"]:
            if params.get(k) == "True":
                modules.append(k.replace("enable_",""))
        print(f"  ADM=True: {d:55s} | Modules: {','.join(modules)} | PSNR={psnr}")

print("\n\nChecking all cfg_args for enable_gap=True...")
for d in sorted(os.listdir(os.path.join(BASE, "output"))):
    cfg_path = os.path.join(BASE, "output", d, "cfg_args")
    if not os.path.exists(cfg_path):
        continue
    params = parse_cfg(cfg_path)
    if params.get("enable_gap") == "True":
        psnr, ssim = get_best_eval(os.path.join(BASE, "output", d))
        adm = params.get("enable_adm")
        kp = params.get("enable_kplanes")
        gap_th = params.get("gap_threshold")
        gap_k = params.get("gap_k")
        print(f"  GAP=True: {d:55s} | kplanes={kp} ADM={adm} | τ={gap_th} k={gap_k} | PSNR={psnr}")

print("\n\nChecking all cfg_args for enable_gar=True (old GAP)...")
for d in sorted(os.listdir(os.path.join(BASE, "output"))):
    cfg_path = os.path.join(BASE, "output", d, "cfg_args")
    if not os.path.exists(cfg_path):
        continue
    params = parse_cfg(cfg_path)
    if params.get("enable_gar") == "True" or params.get("enable_gar_proximity") == "True":
        psnr, ssim = get_best_eval(os.path.join(BASE, "output", d))
        kp = params.get("enable_kplanes")
        fsgs = params.get("enable_fsgs_proximity")
        gap_th = params.get("gar_proximity_threshold")
        gap_k = params.get("gar_proximity_k")
        print(f"  GAR=True: {d:55s} | kplanes={kp} fsgs={fsgs} | τ={gap_th} k={gap_k} | PSNR={psnr}")
