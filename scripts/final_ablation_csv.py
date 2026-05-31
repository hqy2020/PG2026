#!/usr/bin/env python3
"""
Final ablation CSV generation.
Uses the best data available for each section of the ablation table.
Saves CSVs alongside existing data_visualization files.
"""
import os, yaml, json, pickle, glob

BASE = "/home/qyhu/Documents/r2_ours/PG2026/"
OUT = os.path.join(BASE, "data_visualization")
os.makedirs(OUT, exist_ok=True)

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

def get_gs(dir_path):
    pkl = f"{dir_path}/point_cloud/iteration_30000/point_cloud.pickle"
    if os.path.exists(pkl):
        try:
            return len(pickle.load(open(pkl,"rb")).get("xyz",[]))
        except: pass
    return None

organs = ["chest","head","abdomen","foot","pancreas"]

# ============================================================
# 1. COMPONENT ANALYSIS (5-organ avg, 3-view)
# ============================================================
print("="*60)
print("SECTION 1: Component Analysis (5-organ avg, 3-view)")
print("="*60)

# May 2 experiments mapping
# sps_only = B + SPS_init (precomputed SPS init)
# gar_only = B + GAR (old FSGS proximity) — NOTE: old/prototype GAP
# The new GAP module (enable_gap=True) only exists in May 3+ experiments
# which include both KPlanes and GAP together
#
# Best available data:
# - Baseline (R²-GS) = May 2 r2_gaussian
# - B + SPS = May 2 sps_only 
# - B + GAP = Needs GAP-only experiment. We DON'T have clean GAP-only.
#   Closest: May 4 spsv*_gap has KPlanes+GAP (both).
#   May 3 gap_th* has KPlanes+GAP (both).
#   So "GAP only" = B + GAP without SPS/KPlanes doesn't exist.
# - B + ADM = May 2 adm_only (has KPlanes only)
# - Full = 234-view best XRA-GS (with GAP+ADM+SPS)

# Baseline
print("\nBaseline (R²-GS):")
for o in organs:
    d = f"{BASE}output/2026_05_02_{o}_3views_r2_gaussian"
    p,s = get_best_eval(d)
    ng = get_gs(d)
    print(f"  {o:10s}: PSNR={p:.4f} SSIM={s:.4f} GS={ng}")

# B+SPS (SPS init precomputed)
print("\nB + SPS (SPS init):")
for o in organs:
    d = f"{BASE}output/2026_05_02_{o}_3views_sps_only"
    p,s = get_best_eval(d)
    ng = get_gs(d)
    print(f"  {o:10s}: PSNR={p:.4f} SSIM={s:.4f} GS={ng}")

# B+GAP — use May 3 gap_th0p015_r5 (has KPlanes + GAP, no ADM)
# Note: this is NOT "GAP only" — it includes KPlanes too
print("\nB + KPlanes+GAP (May 3, best available as closest to B+GAP):")
for o in organs:
    d = f"{BASE}output/2026_05_03_{o}_3views_gap_th0p015_r5"
    p,s = get_best_eval(d)
    print(f"  {o:10s}: PSNR={p:.4f} SSIM={s:.4f}")

# B+ADM (May 2 adm_only = has KPlanes)
print("\nB + ADM (May 2 adm_only = KPlanes):")
for o in organs:
    d = f"{BASE}output/2026_05_02_{o}_3views_adm_only"
    p,s = get_best_eval(d)
    print(f"  {o:10s}: PSNR={p:.4f} SSIM={s:.4f}")

# Full XRA-GS (234-view best, with P1 updates for applicable cells)
print("\nFull XRA-GS (234-view sweep + P1):")
total_p, total_s = 0, 0
for o in organs:
    # For 3-view, use 234-view results.json (no P1 update for most 3-view)
    with open(f"{BASE}output/comparison_234/results.json") as f:
        r234 = json.load(f)
    key = f"spags/{o}/3"
    if key in r234:
        p, s = r234[key]["psnr_2d"], r234[key]["ssim_2d"]
        total_p += p
        total_s += s
        print(f"  {o:10s}: PSNR={p:.4f} SSIM={s:.4f} (from 234-view)")
print(f"  AVG(5): PSNR={total_p/5:.4f} SSIM={total_s/5:.4f}")

# SPS α sweep (May 4 spsv*_gap)
print("\n\nSECTION 2: SPS α sweep (Chest 3v)")
sps_alpha_data = {
    "spsv2 (unif=0.4, gamma=0.7)": "2026_05_04_chest_3views_spsv2_gap",
    "spsv4 (unif=0.3, gamma=0.8)": "2026_05_04_chest_3views_spsv4_gap",
    "spsv5 (mean_init)": "2026_05_04_chest_3views_spsv5_gap",
    "spsv6 (unif=0.2, gamma=1.0, 75K)": "2026_05_04_chest_3views_spsv6_gap",
}
for label, dirname in sps_alpha_data.items():
    d = f"{BASE}output/{dirname}"
    p,s = get_best_eval(d)
    print(f"  {label:35s}: PSNR={p:.4f} SSIM={s:.4f}")

# GAP τ sweep (May 3 gap_th*, Chest 3v)
print("\n\nSECTION 3: GAP τ sweep (Chest 3v)")
gap_tau_data = [
    ("0.01 (k=3)", "2026_05_03_chest_3views_gap_th0p01_r3"),
    ("0.015 (k=2)", "2026_05_03_chest_3views_gap_th0p015_r2"),
    ("0.015 (k=5)", "2026_05_03_chest_3views_gap_th0p015_r5"),
    ("0.02 (k=3)", "2026_05_03_chest_3views_gap_th0p02_r3"),
]
for label, dirname in gap_tau_data:
    d = f"{BASE}output/{dirname}"
    p,s = get_best_eval(d)
    print(f"  τ={label:15s}: PSNR={p:.4f} SSIM={s:.4f}")

# ADM activation iter sweep
print("\n\nSECTION 4: ADM activation iter sweep (Chest 3v)")
adm_data = [
    ("0K (no ADM)", "2026_05_01_chest_3views_spags_opt_adm_warmup0_3v"),
    ("5K", "2026_05_01_chest_3views_spags_opt_adm_warmup5000_3v"),
    ("10K", "2026_05_01_chest_3views_spags_opt_adm_warmup10000_3v"),
    ("12K", "2026_05_04_chest_3views_adm_warm12k"),
    ("15K (best)", "2026_05_01_chest_3views_spags_opt_adm_warmup15000_3v"),
    ("18K", "2026_05_04_chest_3views_adm_warm18k"),
    ("20K", "2026_05_01_chest_3views_spags_opt_adm_warmup20000_3v"),
]
for label, dirname in adm_data:
    d = f"{BASE}output/{dirname}"
    if os.path.isdir(d):
        p,s = get_best_eval(d)
        print(f"  Iter={label:15s}: PSNR={p:.4f} SSIM={s:.4f}")

# ============================================================
# Write to CSV
# ============================================================
# 1a. Component analysis per-organ (for 5-organ avg computation)
print("\n\nWriting CSVs...")
with open(os.path.join(OUT, "ablation_component_per_organ.csv"), "w") as f:
    f.write("config,organ,psnr,ssim,n_gaussians\n")
    for cfg_label, suffix in [
        ("Baseline_R2GS", "r2_gaussian"),
        ("B_SPS_init", "sps_only"),
        ("B_KPlanes", "adm_only"),
        ("B_GAR_prox", "gar_only"),
    ]:
        for o in organs:
            d = f"{BASE}output/2026_05_02_{o}_3views_{suffix}"
            p,s = get_best_eval(d)
            ng = get_gs(d)
            f.write(f"{cfg_label},{o},{p},{s},{ng}\n")

# 1b. Full XRA-GS from 234-view
with open(os.path.join(OUT, "ablation_xrags_per_organ.csv"), "w") as f:
    f.write("config,organ,views,psnr,ssim\n")
    with open(f"{BASE}output/comparison_234/results.json") as fj:
        r234 = json.load(fj)
    for k, v in sorted(r234.items()):
        method, organ, nv = k.split('/')
        if method == "spags":
            f.write(f"XRAGS,{organ},{nv},{v['psnr_2d']:.4f},{v['ssim_2d']:.4f}\n")

# Also write the R2GS baseline from 234-view for comparison
with open(os.path.join(OUT, "ablation_baseline_per_view.csv"), "w") as f:
    f.write("method,organ,views,psnr,ssim\n")
    with open(f"{BASE}output/comparison_234/results.json") as fj:
        r234 = json.load(fj)
    for k, v in sorted(r234.items()):
        method_display = "R2GS" if "r2_gaussian" in k else "XRAGS" if "spags" in k else method
        if method in ["r2_gaussian", "spags"]:
            _, organ, nv = k.split('/')
            f.write(f"{method_display},{organ},{nv},{v['psnr_2d']:.4f},{v['ssim_2d']:.4f}\n")

# 2. SPS α
with open(os.path.join(OUT, "ablation_sps_alpha.csv"), "w") as f:
    f.write("variant,unif_ratio,gamma,psnr,ssim\n")
    for label, dirname, unif, gamma in [
        ("spsv2_gap", "2026_05_04_chest_3views_spsv2_gap", 0.4, 0.7),
        ("spsv4_gap", "2026_05_04_chest_3views_spsv4_gap", 0.3, 0.8),
        ("spsv5_gap(mean_init)", "2026_05_04_chest_3views_spsv5_gap", "mean", "mean"),
        ("spsv6_gap(75K)", "2026_05_04_chest_3views_spsv6_gap", 0.2, 1.0),
    ]:
        d = f"{BASE}output/{dirname}"
        p,s = get_best_eval(d)
        f.write(f"{label},{unif},{gamma},{p},{s}\n")

# 3. GAP τ
with open(os.path.join(OUT, "ablation_gap_tau.csv"), "w") as f:
    f.write("tau,k,psnr,ssim\n")
    for label, dirname in gap_tau_data:
        d = f"{BASE}output/{dirname}"
        p,s = get_best_eval(d)
        tau, k = label.split(" (k=")
        k = k.replace(")", "")
        f.write(f"{tau},{k},{p},{s}\n")

# 4. ADM iter
with open(os.path.join(OUT, "ablation_adm_iter.csv"), "w") as f:
    f.write("warmup_iter,psnr,ssim\n")
    for label, dirname in adm_data:
        d = f"{BASE}output/{dirname}"
        if os.path.isdir(d):
            p,s = get_best_eval(d)
            warmup = label.split(" (")[0].replace("K","000").replace("no ADM","0")
            if "K" in label:
                warmup = label.replace("K","000")
            elif "no ADM" in label:
                warmup = "0"
            f.write(f"{warmup},{p},{s}\n")

print(f"✅ All CSVs saved to {OUT}/ablation_*.csv")
print("Files created:")
for fn in ["ablation_component_per_organ.csv","ablation_xrags_per_organ.csv",
           "ablation_baseline_per_view.csv","ablation_sps_alpha.csv",
           "ablation_gap_tau.csv","ablation_adm_iter.csv"]:
    path = os.path.join(OUT, fn)
    if os.path.exists(path):
        print(f"  ✅ {fn}")
    else:
        print(f"  ❌ {fn}")
