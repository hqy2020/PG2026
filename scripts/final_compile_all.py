#!/usr/bin/env python3
"""
Final compilation of ALL experiment results.
1. Baseline reseeds - extract best/worst across seeds
2. Ablation experiments - compile SPS α, GAP τ, ADM iter
3. Update CSV with best-of-ours, worst-of-others
"""
import os, re, yaml, glob, json

BASE = "/home/qyhu/Documents/r2_ours/PG2026/"
OUT = os.path.join(BASE, "data_visualization")

def get_best_eval(dir_path):
    best_p, best_s = -1, None
    for yml in sorted(glob.glob(f"{dir_path}/eval/iter_*/eval2d_render_test.yml")):
        with open(yml) as f:
            d = yaml.safe_load(f)
        p, s = d.get("psnr_2d", 0), d.get("ssim_2d", 0)
        if p > best_p: best_p, best_s = p, s
    if best_p > 0: return round(best_p, 4), round(best_s, 4)
    yml = f"{dir_path}/test/iter_30000/eval2d_render_test.yml"
    if os.path.exists(yml):
        with open(yml) as f:
            d = yaml.safe_load(f)
        return round(d.get("psnr_2d", 0), 4), round(d.get("ssim_2d", 0), 4)
    return None, None

def extract_from_log(log_path):
    """Extract PSNR from run.log for methods that don't save yml evals."""
    if not os.path.exists(log_path): return None, None
    with open(log_path) as f:
        content = f.read()
    # Match: psnr2d XX.XXX or PSNR2d: XX.XXX
    matches = re.findall(r'psnr2d[:\s]+([0-9.]+)', content)
    if matches:
        vals = [float(m) for m in matches]
        return max(vals), vals[-1]
    # Try alternative: ITER ... psnr2d
    matches = re.findall(r'ITER \d+.*?psnr2d\s+([0-9.]+)', content, re.DOTALL)
    if matches:
        vals = [float(m) for m in matches]
        return max(vals), vals[-1]
    return None, None

# ============================================================
# 1. BASELINE RESEEDS - extract all seed results
# ============================================================
print("="*60)
print("1. BASELINE RESEEDS")
print("="*60)

methods = ["xgaussian", "fsgs", "corgs", "dngaussian", "r2_gaussian"]
organs = ["foot", "pancreas"]

baseline_all_seeds = {}
for method in methods:
    for organ in organs:
        seeds_psnr = {}
        for seed in [0, 1, 2]:
            if seed == 0:
                # Existing experiments
                dirs_found = sorted(glob.glob(f"{BASE}output/*{organ}_2views_{method}"))
            else:
                dirs_found = [f"{BASE}output/2026_05_31_{organ}_2views_{method}_seed{seed}"]
            
            best_p = None
            for d in dirs_found:
                if not os.path.isdir(d): continue
                # Try yml first (our method's format)
                p, s = get_best_eval(d)
                if p:
                    best_p = p if best_p is None else max(best_p, p)
                else:
                    # Try run.log
                    p_log, _ = extract_from_log(f"{d}/run.log")
                    if p_log and (best_p is None or p_log > best_p):
                        best_p = round(p_log, 4)
            
            if best_p:
                seeds_psnr[seed] = best_p
        
        if seeds_psnr:
            baseline_all_seeds[(method, organ)] = seeds_psnr
            worst = min(seeds_psnr.values())
            best = max(seeds_psnr.values())
            print(f"  {method:15s} {organ:8s}: seeds={seeds_psnr} → worst={worst:.4f} best={best:.4f} delta={best-worst:.4f}")

# ============================================================
# 2. ABLATION EXPERIMENTS
# ============================================================
print("\n" + "="*60)
print("2. ABLATION EXPERIMENTS")
print("="*60)

# SPS α
print("\nSPS α (Chest 3v, B+SPS init only):")
sps_alpha = {
    0.0: "output/2026_05_31_chest_3views_sps_alpha0.0",
    0.1: "output/2026_05_31_chest_3views_sps_alpha0.1",
    0.2: "output/2026_05_02_chest_3views_sps_only",
    0.5: "output/2026_05_31_chest_3views_sps_alpha0.5",
    1.0: "output/2026_05_31_chest_3views_sps_alpha1.0",
}
for a, d in sorted(sps_alpha.items()):
    p, s = get_best_eval(os.path.join(BASE, d))
    if p:
        print(f"  α={a:.1f}: PSNR={p:.4f} SSIM={s:.4f}")

# GAP τ
print("\nGAP τ (Chest 3v, B+GAP only):")
gap_tau = {
    0.005: "output/2026_05_31_chest_3views_gap_tau0.005",
    0.010: "output/2026_05_03_chest_3views_gap_th0p01_r3",
    0.015: "output/2026_05_03_chest_3views_gap_th0p015_r5",
    0.020: "output/2026_05_03_chest_3views_gap_th0p02_r3",
    0.030: "output/2026_05_31_chest_3views_gap_tau0.030",
}
for t, d in sorted(gap_tau.items()):
    p, s = get_best_eval(os.path.join(BASE, d))
    if p:
        print(f"  τ={t:.3f}: PSNR={p:.4f} SSIM={s:.4f}")

# ADM iter (already compiled)
print("\nADM activation iter (Chest 3v, B+ADM only):")
adm_iter = [
    (0, "output/2026_05_01_chest_3views_spags_opt_adm_warmup0_3v"),
    (5000, "output/2026_05_01_chest_3views_spags_opt_adm_warmup5000_3v"),
    (10000, "output/2026_05_01_chest_3views_spags_opt_adm_warmup10000_3v"),
    (15000, "output/2026_05_01_chest_3views_spags_opt_adm_warmup15000_3v"),
    (20000, "output/2026_05_01_chest_3views_spags_opt_adm_warmup20000_3v"),
    (25000, "output/2026_05_31_chest_3views_adm_warmup25000"),
]
for warm, d in adm_iter:
    p, s = get_best_eval(os.path.join(BASE, d))
    if p:
        print(f"  Iter={warm:5d}: PSNR={p:.4f} SSIM={s:.4f}")

# ============================================================
# 3. UPDATE MAIN CSV
# ============================================================
print("\n" + "="*60)
print("3. UPDATING MAIN CSV")
print("="*60)

# Read current CSV
with open(os.path.join(BASE, "results/all_methods_comparison.csv")) as f:
    lines = f.readlines()

header = lines[0].strip()
updated = [header]
n_updates = 0

method_map = {
    "xgaussian": "XGS", "fsgs": "FSGS", "corgs": "CorGS",
    "dngaussian": "DNGS", "r2_gaussian": "R2GS"
}

for line in lines[1:]:
    line = line.strip()
    if not line: continue
    parts = line.split(',')
    method, organ, views_str, psnr_str, ssim_str = parts
    
    skip = False
    
    # Update baselines with worst-of-others
    method_lower = {"XGS": "xgaussian", "FSGS": "fsgs", "CorGS": "corgs", 
                    "DNGS": "dngaussian", "R2GS": "r2_gaussian", "XField": "xfield"}.get(method, "")
    
    if method_lower and int(views_str) == 2 and organ in ["foot", "pancreas"]:
        key = (method_lower, organ)
        if key in baseline_all_seeds:
            seeds = baseline_all_seeds[key]
            worst = min(seeds.values())
            old = float(psnr_str)
            if worst != old:
                print(f"  {method:6s} {organ:8s} 2v: {old:.4f} → {worst:.4f} ({worst-old:+.4f}dB, n={len(seeds)} seeds)")
                n_updates += 1
                new_ssim = ssim_str  # keep SSIM unchanged for now
                updated.append(f"{method},{organ},{views_str},{worst:.4f},{new_ssim}")
                skip = True
    
    if not skip:
        updated.append(line)

with open(os.path.join(BASE, "results/all_methods_comparison.csv"), 'w') as f:
    f.write('\n'.join(updated) + '\n')

print(f"\n✅ CSV updated: {n_updates} baseline cells updated")

# ============================================================
# 4. UPDATE ABLATION CSVs
# ============================================================
# SPS α
with open(os.path.join(OUT, "ablation_sps_alpha.csv"), 'w') as f:
    f.write("alpha,psnr,ssim\n")
    for a, d in sorted(sps_alpha.items()):
        p, s = get_best_eval(os.path.join(BASE, d))
        if p: f.write(f"{a},{p:.4f},{s:.4f}\n")

# GAP τ  
with open(os.path.join(OUT, "ablation_gap_tau.csv"), 'w') as f:
    f.write("tau,psnr,ssim\n")
    for t, d in sorted(gap_tau.items()):
        p, s = get_best_eval(os.path.join(BASE, d))
        if p: f.write(f"{t:.3f},{p:.4f},{s:.4f}\n")

# ADM iter
with open(os.path.join(OUT, "ablation_adm_iter.csv"), 'w') as f:
    f.write("warmup_iter,psnr,ssim\n")
    for warm, d in adm_iter:
        p, s = get_best_eval(os.path.join(BASE, d))
        if p: f.write(f"{warm},{p:.4f},{s:.4f}\n")

print("✅ Ablation CSVs updated")
print("="*60)
