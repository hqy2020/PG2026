#!/usr/bin/env python3
"""Build a single comprehensive CSV with ALL experiments."""
import csv

rows = []

# ============================================================
# 1. MAIN COMPARISON (7 methods × 5 organs × 3 views = 105)
# ============================================================
# Source: data_visualization/comparison_all_105.csv (already consolidated)
with open("data_visualization/comparison_all_105.csv") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append({
            "experiment_type": "main_comparison",
            "method": r["method"],
            "config": r["method"],
            "organ": r["organ"],
            "views": int(r["views"]),
            "psnr": float(r["psnr"]),
            "ssim": float(r["ssim"]),
            "n_gaussians": "",
        })

# ============================================================
# 2. COMPONENT ABLATION (5 configs × 5 organs, all 3-view)
# ============================================================
# Source: data_visualization/ablation_component_per_organ.csv
with open("data_visualization/ablation_component_per_organ.csv") as f:
    reader = csv.DictReader(f)
    for r in reader:
        config_short = r["config"]
        # Map config names
        if config_short == "Baseline_R2GS":
            config_name = "Baseline (R²-Gaussian)"
            method_label = "r2_gaussian"
        elif config_short == "B_SPS_init":
            config_name = "B + SPS"
            method_label = "xrags+SPS"
        elif config_short == "B_KPlanes":
            config_name = "B + ADM"
            method_label = "xrags+ADM"
        elif config_short == "B_GAR_prox":
            config_name = "B + GAP"
            method_label = "xrags+GAP"
        elif config_short == "Full_XRAGS":
            config_name = "Full XRA-GS"
            method_label = "xrags"
        else:
            config_name = config_short
            method_label = config_short

        n_gauss = r.get("n_gaussians", "").strip()
        rows.append({
            "experiment_type": "component_ablation",
            "method": method_label,
            "config": config_name,
            "organ": r["organ"],
            "views": 3,
            "psnr": float(r["psnr"]),
            "ssim": float(r["ssim"]),
            "n_gaussians": n_gauss,
        })

# ============================================================
# 3. SPS α SWEEP (5 α × 5 organs, all 3-view)
# ============================================================
# Source: data_visualization/ablation_sps_alpha.csv (wide format)
with open("data_visualization/ablation_sps_alpha.csv") as f:
    reader = csv.DictReader(f)
    for r in reader:
        org = r["org"]
        if org == "avg5":
            continue  # skip avg row
        alphas = ["0.0", "0.1", "0.2", "0.5", "1.0"]
        for a in alphas:
            psnr_key = f"alpha_{a}_psnr"
            ssim_key = f"alpha_{a}_ssim"
            if psnr_key in r and r[psnr_key]:
                psnr_val = float(r[psnr_key])
                ssim_val = float(r[ssim_key]) if ssim_key in r and r[ssim_key] else ""
                rows.append({
                    "experiment_type": "sps_alpha_sweep",
                    "method": "xrags+SPS",
                    "config": f"SPS α={a}",
                    "organ": org,
                    "views": 3,
                    "psnr": psnr_val,
                    "ssim": ssim_val,
                    "n_gaussians": "",
                })

# ============================================================
# 4. GAP τ SWEEP (5 τ × 5 organs, all 3-view)
# ============================================================
# Source: data_visualization/ablation_gap_tau.csv (wide format)
with open("data_visualization/ablation_gap_tau.csv") as f:
    reader = csv.DictReader(f)
    for r in reader:
        org = r["org"]
        if org == "avg5":
            continue
        taus = ["0.005", "0.010", "0.015", "0.020", "0.030"]
        for t in taus:
            psnr_key = f"tau_{t}_psnr"
            ssim_key = f"tau_{t}_ssim"
            if psnr_key in r and r[psnr_key]:
                psnr_val = float(r[psnr_key])
                ssim_val = float(r[ssim_key]) if ssim_key in r and r[ssim_key] else ""
                rows.append({
                    "experiment_type": "gap_tau_sweep",
                    "method": "xrags+GAP",
                    "config": f"GAP τ={t}",
                    "organ": org,
                    "views": 3,
                    "psnr": psnr_val,
                    "ssim": ssim_val,
                    "n_gaussians": "",
                })

# ============================================================
# 5. ADM ITER SWEEP (6 warmup × 5 organs, all 3-view)
# ============================================================
# Source: data_visualization/ablation_adm_iter.csv (wide format)
with open("data_visualization/ablation_adm_iter.csv") as f:
    reader = csv.DictReader(f)
    for r in reader:
        org = r["org"]
        if org == "avg5":
            continue
        warmups = ["0", "5000", "10000", "15000", "20000", "25000"]
        for w in warmups:
            psnr_key = f"warmup_{w}_psnr"
            ssim_key = f"warmup_{w}_ssim"
            if psnr_key in r and r[psnr_key]:
                psnr_val = float(r[psnr_key])
                ssim_val = float(r[ssim_key]) if ssim_key in r and r[ssim_key] else ""
                rows.append({
                    "experiment_type": "adm_iter_sweep",
                    "method": "xrags",
                    "config": f"ADM warmup={w}",
                    "organ": org,
                    "views": 3,
                    "psnr": psnr_val,
                    "ssim": ssim_val,
                    "n_gaussians": "",
                })

# ============================================================
# Write unified CSV
# ============================================================
fieldnames = ["experiment_type", "method", "config", "organ", "views", "psnr", "ssim", "n_gaussians"]
outpath = "data_visualization/all_experiments_unified.csv"
with open(outpath, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print(f"Wrote {len(rows)} rows to {outpath}")

# Summary by type
from collections import Counter
types = Counter(r["experiment_type"] for r in rows)
print("\n=== Summary by experiment type ===")
for t, cnt in sorted(types.items()):
    print(f"  {t}: {cnt}")
print(f"  TOTAL: {sum(types.values())}")
