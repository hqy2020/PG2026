#!/usr/bin/env python3
"""
Comprehensive ablation data extraction for all 4 sections of the ablation table.
Outputs structured CSVs to data_visualization/ablation_*.csv
"""
import os, re, yaml, json, pickle, glob

BASE = "/home/qyhu/Documents/r2_ours/PG2026/"
OUT = os.path.join(BASE, "data_visualization")

def parse_cfg(cfg_path):
    with open(cfg_path) as f:
        text = f.read()
    params = {}
    for m in re.finditer(r'(\w+)=([^,)]+)(?:[,)]|$)', text):
        key = m.group(1)
        val = m.group(2).strip()
        params[key] = val
    return params

def get_best_eval(output_dir):
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

organs = ["chest", "head", "abdomen", "foot", "pancreas"]

# ============================================================
# SECTION 1: Component Analysis (5-organ avg, 3-view)
# Use May 2 experiments for clean isolation of each component
# ============================================================
print("="*70)
print("SECTION 1: COMPONENT ANALYSIS (5-organ avg, 3-view)")
print("="*70)

# Map directory suffix -> display name
# Note: sps_only uses SPS init (precomputed), gar_only uses FSGSprox(GAR),
# adm_only uses KPlanes, spags=gar_adm uses both KPlanes+FSGSprox
component_map = {
    "r2_gaussian": "Baseline (R²-Gaussian)",
    "sps_only": "B + SPS_init",
    "adm_only": "B + KPlanes",
    "gar_only": "B + GAR_proximity",
    "spags": "B + KPlanes+FSGSprox",
}

component_data = {}
for suffix, label in component_map.items():
    organs_data = []
    total_psnr = 0.0
    total_ssim = 0.0
    total_gs = 0.0
    n_valid = 0
    
    for organ in organs:
        dir_name = f"2026_05_02_{organ}_3views_{suffix}"
        dir_path = os.path.join(BASE, "output", dir_name)
        if not os.path.isdir(dir_path):
            continue
        psnr, ssim = get_best_eval(dir_path)
        ngs = get_n_gaussians(dir_path)
        if psnr:
            total_psnr += psnr
            total_ssim += ssim
            if ngs:
                total_gs += ngs
            n_valid += 1
            organs_data.append({"organ": organ, "psnr": psnr, "ssim": ssim, "ngs": ngs})
            print(f"  {label:30s} {organ:10s}: PSNR={psnr:.4f} SSIM={ssim} GS={ngs}")
    
    if n_valid == 5:
        avg_psnr = round(total_psnr / 5, 4)
        avg_ssim = round(total_ssim / 5, 4)
        avg_gs = round(total_gs / 5)
        component_data[label] = {"psnr": avg_psnr, "ssim": avg_ssim, "gs": avg_gs}
        print(f"  {'─'*55}")
        print(f"  {label:30s} AVG(5): PSNR={avg_psnr:.4f} SSIM={avg_ssim:.4f} GS≈{avg_gs}")

# Now add the NEW GAP module data (May 3 gap_th* experiments - includes both KPlanes+GAP)
# And the NEW full pipeline from P1
print("\n--- New GAP module (May 3: KPlanes+GAP, no ADM) ---")
for organ in organs:
    dir_name = f"2026_05_03_{organ}_3views_gap_th0p015_r5"
    dir_path = os.path.join(BASE, "output", dir_name)
    if os.path.isdir(dir_path):
        psnr, ssim = get_best_eval(dir_path)
        ngs = get_n_gaussians(dir_path)
        print(f"  B+KPlanes+GAP (new) {organ:10s}: PSNR={psnr:.4f} SSIM={ssim} GS={ngs}")

print("\n--- Full XRA-GS (from 234-view results.json) ---")
with open(os.path.join(BASE, "output/comparison_234/results.json")) as f:
    results234 = json.load(f)
total_psnr = 0
total_ssim = 0
for k, v in results234.items():
    method, organ, nv = k.split('/')
    if method == "spags" and int(nv) == 3:
        total_psnr += v["psnr_2d"]
        total_ssim += v["ssim_2d"]
        print(f"  XRA-GS {organ:10s} 3v: PSNR={v['psnr_2d']:.4f} SSIM={v['ssim_2d']:.4f}")
print(f"  XRA-GS 5-organ avg 3v: PSNR={total_psnr/5:.4f} SSIM={total_ssim/5:.4f}")

# ============================================================
# SECTION 2: SPS α — check which experiments vary unif_ratio
# ============================================================
print("\n" + "="*70)
print("SECTION 2: SPS α (unif_ratio) sweep")
print("="*70)
print("\n--- Checking spsv*_gap experiments (Chest 3v) for unif_ratio ---")
for d in sorted(os.listdir(os.path.join(BASE, "output"))):
    if "chest" in d and "3views" in d and "spsv" in d:
        dir_path = os.path.join(BASE, "output", d)
        cfg_path = os.path.join(dir_path, "cfg_args")
        if not os.path.exists(cfg_path):
            continue
        params = parse_cfg(cfg_path)
        # Look for SPS-related params
        for key in params:
            if any(x in key for x in ["unif", "ratio", "alpha", "mix"]):
                pass  # we'll check below
        
        gap_th = params.get("gap_threshold", "?")
        has_gap = params.get("enable_gap")
        has_kp = params.get("enable_kplanes")
        has_adm = params.get("enable_adm")
        psnr, ssim = get_best_eval(dir_path)
        
        print(f"  {d:50s} | kplanes={has_kp} gap={has_gap} adm={has_adm} | τ={gap_th} | PSNR={psnr:.4f}")

# ============================================================
# SECTION 3: GAP τ sweep — from May 3 gap_th* experiments
# ============================================================
print("\n" + "="*70)
print("SECTION 3: GAP τ sweep (Chest 3v)")
print("="*70)

gap_tau_data = []
for d in sorted(os.listdir(os.path.join(BASE, "output"))):
    if "chest" in d and "3views" in d and "gap_th" in d:
        dir_path = os.path.join(BASE, "output", d)
        cfg_path = os.path.join(dir_path, "cfg_args")
        params = parse_cfg(cfg_path) if os.path.exists(cfg_path) else {}
        
        tau = params.get("gap_threshold", "?")
        k = params.get("gap_k", "?")
        psnr, ssim = get_best_eval(dir_path)
        ngs = get_n_gaussians(dir_path)
        
        print(f"  τ={tau:6s} k={k:3s}: {d:50s} PSNR={psnr:.4f} SSIM={ssim} GS={ngs}")
        gap_tau_data.append({"tau": tau, "k": k, "psnr": psnr, "ssim": ssim, "gs": ngs})

# Also the default GAP from May 4 spsv*_gap
for d in sorted(os.listdir(os.path.join(BASE, "output"))):
    if "chest" in d and "3views" in d and "spsv2_gap" in d:
        dir_path = os.path.join(BASE, "output", d)
        cfg_path = os.path.join(dir_path, "cfg_args")
        params = parse_cfg(cfg_path) if os.path.exists(cfg_path) else {}
        tau = params.get("gap_threshold", "?")
        psnr, ssim = get_best_eval(dir_path)
        print(f"  τ={tau:6s} (default): {d:50s} PSNR={psnr:.4f} SSIM={ssim}")

# ============================================================
# SECTION 4: ADM activation iteration sweep
# ============================================================
print("\n" + "="*70)
print("SECTION 4: ADM activation iteration sweep (Chest 3v)")
print("="*70)

# Check May 4 adm_warm* experiments (which have GAP=True, KPlanes=True)
for warmup in [5000, 10000, 12000, 15000, 18000, 20000, 25000]:
    d = None
    if warmup == 5000:
        d = "2026_05_01_chest_3views_spags_opt_adm_warmup5000_3v"
    elif warmup == 10000:
        d = "2026_05_01_chest_3views_spags_opt_adm_warmup10000_3v"
    elif warmup == 12000:
        d = "2026_05_04_chest_3views_adm_warm12k"
    elif warmup == 15000:
        d = "2026_05_01_chest_3views_spags_opt_adm_warmup15000_3v"
        if not os.path.isdir(os.path.join(BASE, "output", d)):
            d = "2026_04_30_chest_3views_spags"  # default has warmup=15K
    elif warmup == 18000:
        d = "2026_05_04_chest_3views_adm_warm18k"
    elif warmup == 20000:
        d = "2026_05_01_chest_3views_spags_opt_adm_warmup20000_3v"
    else:
        continue
    
    if not d:
        continue
    dir_path = os.path.join(BASE, "output", d)
    if not os.path.isdir(dir_path):
        continue
    
    cfg_path = os.path.join(dir_path, "cfg_args")
    params = parse_cfg(cfg_path) if os.path.exists(cfg_path) else {}
    actual_warmup = params.get("adm_warmup_iters", "?")
    has_gap = params.get("enable_gap", params.get("enable_gar", "?"))
    has_kp = params.get("enable_kplanes", "?")
    has_adm = params.get("enable_adm", "?")
    
    psnr, ssim = get_best_eval(dir_path)
    print(f"  warmup={warmup:5d} (actual={actual_warmup:5s}): {d:55s} | kplanes={has_kp} gap={has_gap} adm={has_adm} | PSNR={psnr:.4f} SSIM={ssim}")

# Also check the May 4 adm_warm experiments in more detail
print("\n--- May 4 adm_warm* details (Chest 3v) ---")
for d in sorted(os.listdir(os.path.join(BASE, "output"))):
    if "chest" in d and "3views" in d and "adm_warm" in d:
        dir_path = os.path.join(BASE, "output", d)
        psnr, ssim = get_best_eval(dir_path)
        cfg_path = os.path.join(dir_path, "cfg_args")
        params = parse_cfg(cfg_path) if os.path.exists(cfg_path) else {}
        warm = params.get("adm_warmup_iters", "?")
        gap = params.get("enable_gap")
        kp = params.get("enable_kplanes")
        adm = params.get("enable_adm")
        print(f"  {d:50s} warmup={warm} gap={gap} kplanes={kp} adm={adm} | PSNR={psnr:.4f}")

print("\n\n=== SUMMARY ===")
print("Key findings for the ablation table:")
print("1. Component analysis: May 2 experiments have clean module isolation")
print("   BUT: ADM was never enabled (enable_adm=False for all)")
print("   AND: The new GAP module (enable_gap=True) only appears in May 3+4 experiments")
print("2. For clean B+GAP without SPS, we don't have clean data")
print("3. For SPS α: Need to check diff between spsv variants")
print("4. GAP τ: Available from May 3 (τ=0.01, 0.015, 0.02) — missing 0.005 and 0.030")
print("5. ADM warmup: Available from May 1 (warmup=5K,10K,15K,20K) and May 4 (12K,18K)")
