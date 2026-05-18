#!/usr/bin/env python3
"""Fix ablation data and add more efficiency data."""
import os, json, pickle, yaml, glob

BASE = "/home/qyhu/Documents/r2_ours/PG2026"
OUT = os.path.join(BASE, "data_visualization")

# ============================
# Fix ablation mapping
# ============================
# The 2026_05_02 Chest 3-view ablation series uses data/234/ split
# But the SPAGS labeled dirs don't have proper eval (no test.py run)
# Let's use the Apr 30 and May 1 eval data which was run with test.py

ablation_fixed = [
    # (dir, label, psnr_source)
    # Chest 3v from 2026_04_30 (data/369/) - has eval data
    ("2026_04_30_chest_3views_r2_gaussian", "Baseline (R²-Gaussian)", "eval"),
    ("2026_04_30_chest_3views_spags", "SPAGS (full 369)", "eval"),
    
    # Chest 3v from 2026_05_02 (data/234/) - ablation series
    ("2026_05_02_chest_3views_r2_gaussian", "R²-Gaussian (234)", "eval"),
    ("2026_05_02_chest_3views_sps_only", "SPS only", "eval"),
    ("2026_05_02_chest_3views_adm_only", "ADM only", "eval"),
    ("2026_05_02_chest_3views_gar_only", "GAP only", "eval"),
    ("2026_05_02_chest_3views_sps_adm", "SPS+ADM", "eval"),
    ("2026_05_02_chest_3views_sps_gar", "SPS+GAP", "eval"),
    ("2026_05_02_chest_3views_gar_adm", "ADM+GAP", "eval"),
    
    # Chest 3v from 2026_05_01 (data/234/) - has test.py eval
    ("2026_05_01_chest_3views_r2_gaussian", "R²-Gaussian (234 test)", "test"),
    ("2026_05_01_chest_3views_spags", "SPAGS (234 test)", "test"),
    
    # Chest stability (data/369/) - has test.py with timing
    ("chest_stability_r2_seed0", "R²-Gaussian (stability)", "test"),
    ("chest_stability_spags_seed0", "SPAGS (stability)", "test"),
]

ablation_data = []
for d, label, src in ablation_fixed:
    dirpath = os.path.join(BASE, "output", d)
    
    # Try test/ then eval/  
    for sub in ["test/iter_30000", "eval/iter_030000", "eval/iter_030000"]:
        yml = os.path.join(dirpath, sub, "eval2d_render_test.yml")
        if os.path.exists(yml):
            try:
                with open(yml) as f:
                    data = yaml.safe_load(f)
                psnr = data.get("psnr_2d")
                ssim = data.get("ssim_2d")
            except:
                # Manual parse for yaml with numpy tags
                psnr = None
                ssim = None
                with open(yml) as f:
                    for line in f:
                        if line.startswith("psnr_2d:"):
                            psnr = float(line.split(":")[1].strip())
                        if line.startswith("ssim_2d:"):
                            ssim = float(line.split(":")[1].strip())
            
            if psnr is not None:
                # Get GS count
                pkl = os.path.join(dirpath, "point_cloud/iteration_30000/point_cloud.pickle")
                n_gs = None
                if os.path.exists(pkl):
                    try:
                        pd = pickle.load(open(pkl, "rb"))
                        xyz = pd.get("xyz")
                        if xyz is not None:
                            n_gs = len(xyz)
                    except:
                        pass
                
                entry = {"label": label, "dir": d, "psnr_2d": psnr, "ssim_2d": ssim, "n_gaussians": n_gs}
                ablation_data.append(entry)
                print(f"  {label:35s} PSNR={psnr:.2f}  SSIM={ssim:.4f}  GS={n_gs}")
            break

with open(os.path.join(OUT, "ablation_fixed.json"), "w") as f:
    json.dump(ablation_data, f, indent=2)

# ============================
# 2-3-4 view comparsion with GS counts from 2026_05_01 data
# ============================
print("\n=== 5-organ × 2/3/4-view GS counts (2026_05_01) ===")
gs_234 = {}
organs = ["chest", "head", "abdomen", "pancreas", "foot"]
for organ in organs:
    for nv in ["2", "3", "4"]:
        for method in ["r2_gaussian", "spags"]:
            d = f"2026_05_01_{organ}_{nv}views_{method}"
            pkl = os.path.join(BASE, "output", d, "point_cloud/iteration_30000/point_cloud.pickle")
            if os.path.exists(pkl):
                try:
                    pd = pickle.load(open(pkl, "rb"))
                    xyz = pd.get("xyz")
                    if xyz is not None:
                        gs_234[f"{method}/{organ}/{nv}"] = len(xyz)
                except:
                    pass

with open(os.path.join(OUT, "gs_counts_234.json"), "w") as f:
    json.dump(gs_234, f, indent=2)

for k, v in sorted(gs_234.items()):
    print(f"  {k:30s} GS={v:>6,}")

# ============================
# Comprehensive combined data for visualization
# ============================
print("\n=== Writing master data file ===")

# Load results.json
with open(os.path.join(BASE, "output/comparison_234/results.json")) as f:
    results = json.load(f)

master = {
    "comparison": {},
    "gs_counts": gs_234,
    "xfield": {},
    "ablation": ablation_data,
    "stability": {},
    "densify": {}
}

# Main comparison with GS counts
for organ in organs:
    for nv in ["2", "3", "4"]:
        key_base = f"{organ}_{nv}v"
        for method in ["r2_gaussian", "spags", "xgaussian", "fsgs", "corgs", "dngaussian"]:
            rk = f"{method}/{organ}/{nv}"
            if rk in results:
                entry = {
                    "psnr_2d": results[rk]["psnr_2d"],
                    "ssim_2d": results[rk]["ssim_2d"],
                    "n_gaussians": gs_234.get(rk)
                }
                master["comparison"][rk] = entry

# X-Field
for organ in organs:
    for nv in ["2", "3", "4"]:
        d = os.path.join(BASE, "output/xfield", f"{organ}_{nv}views_xfield")
        for yml_path in sorted(glob.glob(f"{d}/eval/iter_*/eval2d_render_test.yml")):
            pass  # just get latest
        ymls = sorted(glob.glob(f"{d}/eval/iter_*/eval2d_render_test.yml"))
        if not ymls:
            ymls = sorted(glob.glob(f"{d}/test/iter_*/eval2d_render_test.yml"))
        if ymls:
            with open(ymls[-1]) as f:
                data = yaml.safe_load(f)
            master["xfield"][f"xfield/{organ}/{nv}"] = {
                "psnr_2d": data.get("psnr_2d"),
                "ssim_2d": data.get("ssim_2d"),
                "fps": data.get("fps")
            }

# Stability
for organ in ["chest", "pancreas"]:
    for method, mlabel in [("r2", "R²-Gaussian"), ("spags", "SPAGS")]:
        seeds_data = []
        for seed in [0, 1, 2]:
            d = os.path.join(BASE, "output", f"{organ}_stability_{method}_seed{seed}")
            for yml_path in [f"{d}/test/iter_30000/eval2d_render_test.yml",
                             f"{d}/eval/iter_030000/eval2d_render_test.yml"]:
                if os.path.exists(yml_path):
                    with open(yml_path) as f:
                        raw = f.read()
                    psnr = [l for l in raw.split('\n') if 'psnr_2d:' in l][0].split(':')[1].strip()
                    ssim = [l for l in raw.split('\n') if 'ssim_2d:' in l][0].split(':')[1].strip()
                    seeds_data.append({"seed": seed, "psnr_2d": float(psnr), "ssim_2d": float(ssim)})
                    break
        if seeds_data:
            psnrs = [s["psnr_2d"] for s in seeds_data]
            mu = sum(psnrs) / len(psnrs)
            std = (sum((p-mu)**2 for p in psnrs)/(len(psnrs)-1))**0.5 if len(psnrs)>1 else 0
            master["stability"][f"{organ}/{mlabel}"] = {
                "seeds": seeds_data,
                "mean": mu,
                "std": std
            }

# More-densify
for d, label in [("output/pancreas_3views_densify", "Pancreas"), ("output/chest_densify_test", "Chest")]:
    yml = os.path.join(BASE, d, "test/iter_30000/eval2d_render_test.yml")
    if os.path.exists(yml):
        with open(yml) as f:
            data = yaml.safe_load(f)
        master["densify"][label] = {
            "psnr_2d": data.get("psnr_2d"),
            "ssim_2d": data.get("ssim_2d")
        }

with open(os.path.join(OUT, "master_data.json"), "w") as f:
    json.dump(master, f, indent=2, default=str)

print("✅ Master data written to data_visualization/master_data.json")
print(f"  Comparison entries: {len(master['comparison'])}")
print(f"  X-Field entries: {len(master['xfield'])}")
print(f"  Ablation entries: {len(master['ablation'])}")
print(f"  Stability entries: {len(master['stability'])}")
print(f"  GS count entries: {len(master['gs_counts'])}")
