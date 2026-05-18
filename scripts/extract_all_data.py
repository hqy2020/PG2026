#!/usr/bin/env python3
"""
PG2026 综合数据提取脚本
提取所有实验的结构化数据到 JSON 文件，供 matplotlib 可视化使用

输出:
  data_visualization/all_results.json  — 全部定量数据
  data_visualization/efficiency.csv    — 效率数据
  data_visualization/ablation.csv      — 消融数据
  data_visualization/comparison.csv    — 主对比表
"""
import os, sys, json, pickle, glob, yaml

BASE = "/home/qyhu/Documents/r2_ours/PG2026"
OUT = os.path.join(BASE, "data_visualization")
os.makedirs(OUT, exist_ok=True)

# ============================
# 1. 主对比数据 (results.json)
# ============================
with open(os.path.join(BASE, "output/comparison_234/results.json")) as f:
    results = json.load(f)

comparison_data = []
for k, v in sorted(results.items()):
    method, organ, nv = k.split('/')
    entry = {
        "method": method,
        "organ": organ,
        "views": int(nv),
        "psnr_2d": v["psnr_2d"],
        "ssim_2d": v["ssim_2d"]
    }
    comparison_data.append(entry)

# Write comparison CSV
with open(os.path.join(OUT, "comparison.json"), "w") as f:
    json.dump(comparison_data, f, indent=2)

methods_pretty = {
    "r2_gaussian": "R²-Gaussian",
    "spags": "SPAGS",
    "xgaussian": "X-Gaussian",
    "fsgs": "FSGS",
    "corgs": "CoR-GS",
    "dngaussian": "DN-Gaussian"
}

# ============================
# 2. GS 数量提取 (从 pickle)
# ============================
print("=== Extracting GS counts from pickles ===")
gs_data = []

# Find output dirs with point cloud pickles
for d in sorted(os.listdir(os.path.join(BASE, "output"))):
    pkl_path = os.path.join(BASE, "output", d, "point_cloud/iteration_30000/point_cloud.pickle")
    if os.path.exists(pkl_path):
        try:
            data = pickle.load(open(pkl_path, "rb"))
            xyz = data.get("xyz")
            if xyz is not None:
                n_gs = len(xyz)
                gs_data.append({"dir": d, "n_gaussians": n_gs})
                print(f"  {d:55s}  GS={n_gs:>6,}")
        except:
            pass

with open(os.path.join(OUT, "gs_counts.json"), "w") as f:
    json.dump(gs_data, f, indent=2)

# ============================
# 3. 效率数据 (timing yml)
# ============================
print("\n=== Extracting timing data ===")
timing_data = []
for d in sorted(os.listdir(os.path.join(BASE, "output"))):
    tyml = os.path.join(BASE, "output", d, "test/iter_30000/timing_render_test.yml")
    if os.path.exists(tyml):
        try:
            with open(tyml) as f:
                t = yaml.safe_load(f)
            timing_data.append({
                "dir": d,
                "fps": t.get("fps"),
                "ms_per_view": t.get("avg_render_time_per_view_ms"),
                "total_ms": t.get("total_render_time_ms"),
                "n_views": t.get("num_views")
            })
            print(f"  {d:55s}  FPS={t.get('fps','N/A')}")
        except:
            pass

with open(os.path.join(OUT, "timing.json"), "w") as f:
    json.dump(timing_data, f, indent=2)

# ============================
# 4. X-Field 评估数据
# ============================
print("\n=== Extracting X-Field eval data ===")
xfield_data = []
organs = ["chest", "head", "abdomen", "pancreas", "foot"]
for organ in organs:
    for nv in ["2", "3", "4"]:
        d = os.path.join(BASE, "output/xfield", f"{organ}_{nv}views_xfield")
        # Try eval/ first (proper evaluation), then test/
        ymls = sorted(glob.glob(f"{d}/eval/iter_*/eval2d_render_test.yml"))
        if not ymls:
            ymls = sorted(glob.glob(f"{d}/test/iter_*/eval2d_render_test.yml"))
        if ymls:
            with open(ymls[-1]) as f:
                data = yaml.safe_load(f)
            entry = {
                "organ": organ,
                "views": int(nv),
                "psnr_2d": data.get("psnr_2d"),
                "ssim_2d": data.get("ssim_2d"),
                "fps": data.get("fps"),
                "method": "xfield"
            }
            xfield_data.append(entry)
            print(f"  {organ:10s} {nv}v: PSNR={data.get('psnr_2d','N/A'):.2f}")

with open(os.path.join(OUT, "xfield_eval.json"), "w") as f:
    json.dump(xfield_data, f, indent=2)

# ============================
# 5. 稳定性数据
# ============================
print("\n=== Compiling stability data ===")
stability_data = []
for organ in ["chest", "pancreas"]:
    for method, mlabel in [("r2", "R²-Gaussian"), ("spags", "SPAGS")]:
        psnrs = []
        for seed in [0, 1, 2]:
            d = os.path.join(BASE, "output", f"{organ}_stability_{method}_seed{seed}")
            for yml_path in [f"{d}/test/iter_30000/eval2d_render_test.yml",
                             f"{d}/eval/iter_030000/eval2d_render_test.yml"]:
                if os.path.exists(yml_path):
                    with open(yml_path) as f:
                        data = yaml.safe_load(f)
                    psnr = data.get("psnr_2d")
                    ssim = data.get("ssim_2d")
                    psnrs.append(psnr)
                    stability_data.append({
                        "organ": organ,
                        "method": mlabel,
                        "seed": seed,
                        "psnr_2d": psnr,
                        "ssim_2d": ssim
                    })
                    break
        
        if psnrs:
            mu = sum(psnrs) / len(psnrs)
            std = (sum((p - mu)**2 for p in psnrs) / (len(psnrs) - 1))**0.5 if len(psnrs) > 1 else 0
            print(f"  {organ:10s} {mlabel:15s}: {mu:.2f} ± {std:.3f} (n={len(psnrs)})")

with open(os.path.join(OUT, "stability.json"), "w") as f:
    json.dump(stability_data, f, indent=2)

# ============================
# 6. More-densify 数据
# ============================
print("\n=== More-densify data ===")
densify_data = {}
for d, label in [("output/pancreas_3views_densify", "More Densify (Pancreas)"),
                  ("output/chest_densify_test", "More Densify (Chest)")]:
    yml = os.path.join(BASE, d, "test/iter_30000/eval2d_render_test.yml")
    if os.path.exists(yml):
        with open(yml) as f:
            data = yaml.safe_load(f)
        densify_data[label] = {
            "psnr_2d": data.get("psnr_2d"),
            "ssim_2d": data.get("ssim_2d")
        }
        print(f"  {label}: PSNR={data.get('psnr_2d'):.2f}")

with open(os.path.join(OUT, "densify.json"), "w") as f:
    json.dump(densify_data, f, indent=2)

# ============================
# 7. 消融数据 (Chest 3v)
# ============================
print("\n=== Ablation data ===")
ablation_map = {
    "2026_05_02_chest_3views_r2_gaussian": "Baseline (R²-Gaussian)",
    "2026_05_02_chest_3views_sps_only": "SPS only",
    "2026_05_02_chest_3views_adm_only": "ADM only",
    "2026_05_02_chest_3views_gar_only": "GAP only",
    "2026_05_02_chest_3views_sps_adm": "SPS+ADM",
    "2026_05_02_chest_3views_sps_gar": "SPS+GAP",
    "2026_05_02_chest_3views_gar_adm": "ADM+GAP",
    # Full SPAGS - need to find the right dir
}

# Find spags_full directories
for d in os.listdir(os.path.join(BASE, "output")):
    if "chest" in d and "spags" in d and "opt" not in d and "adaptive" not in d and "retry" not in d:
        if d not in ["2026_05_01_chest_3views_spags", "2026_04_30_chest_3views_spags"]:
            # Check if it has FSGS proximity + kplanes (SPAGS-like)
            cfg = os.path.join(BASE, "output", d, "cfg_args")
            if os.path.exists(cfg):
                with open(cfg) as f:
                    cfg_str = f.read()
                if "enable_kplanes=True" in cfg_str and ("enable_fsgs_proximity=True" in cfg_str or "enable_gar=True" in cfg_str):
                    ablation_map[d] = "SPAGS (full)"

ablation_data = []
for d, label in sorted(ablation_map.items()):
    yml = os.path.join(BASE, "output", d, "test/iter_30000/eval2d_render_test.yml")
    if not os.path.exists(yml):
        yml = os.path.join(BASE, "output", d, "eval/iter_030000/eval2d_render_test.yml")
    if os.path.exists(yml):
        with open(yml) as f:
            data = yaml.safe_load(f)
        ablation_data.append({
            "label": label,
            "dir": d,
            "psnr_2d_30k": data.get("psnr_2d"),
            "ssim_2d_30k": data.get("ssim_2d")
        })
        print(f"  {label:30s}: PSNR={data.get('psnr_2d'):.2f}")

with open(os.path.join(OUT, "ablation.json"), "w") as f:
    json.dump(ablation_data, f, indent=2)

# ============================
# 8. 撰写综合 CSVs
# ============================
# Main comparison table (methods × organs × views)
print("\n=== Writing CSVs ===")
methods_of_interest = ["r2_gaussian", "spags", "xgaussian", "fsgs", "corgs", "dngaussian"]
organs = ["chest", "head", "abdomen", "pancreas", "foot"]

with open(os.path.join(OUT, "comparison_psnr.csv"), "w") as f:
    f.write("method,organ,views,psnr,ssim\n")
    for entry in comparison_data:
        if entry["method"] in methods_of_interest:
            f.write(f"{entry['method']},{entry['organ']},{entry['views']},{entry['psnr_2d']:.4f},{entry['ssim_2d']:.4f}\n")

# With X-Field
with open(os.path.join(OUT, "comparison_with_xfield.csv"), "w") as f:
    f.write("method,organ,views,psnr,ssim\n")
    for entry in comparison_data:
        if entry["method"] in methods_of_interest:
            f.write(f"{entry['method']},{entry['organ']},{entry['views']},{entry['psnr_2d']:.4f},{entry['ssim_2d']:.4f}\n")
    for entry in xfield_data:
        f.write(f"xfield,{entry['organ']},{entry['views']},{entry['psnr_2d']:.4f},{entry['ssim_2d']:.4f}\n")

# Efficiency: merge GS + timing
print("\n=== Merging GS + Timing for efficiency table ===")
gs_lookup = {g["dir"]: g["n_gaussians"] for g in gs_data}
timing_lookup = {t["dir"]: t for t in timing_data}

# Key runs for efficiency
efficiency_runs = [
    # (dir_method_organ, method_name, organ)
    ("2026_05_02_chest_3views_r2_gaussian", "R²-Gaussian", "Chest"),
    ("2026_05_02_chest_3views_sps_only", "SPS only", "Chest"),
    ("2026_05_02_chest_3views_adm_only", "ADM only", "Chest"),
    ("2026_05_02_chest_3views_gar_only", "GAP only", "Chest"),
    ("2026_05_02_chest_3views_sps_adm", "SPS+ADM", "Chest"),
    ("2026_05_02_chest_3views_sps_gar", "SPS+GAP", "Chest"),
    ("2026_05_02_chest_3views_gar_adm", "ADM+GAP", "Chest"),
    ("chest_densify_test", "More Densify", "Chest"),
    ("pancreas_3views_densify", "More Densify", "Pancreas"),
]

# Also add all R² and SPAGS 3-view from 2026_04_30 (has checkpoints)
for organ in ["chest", "head", "abdomen", "pancreas", "foot"]:
    for method, mlabel in [("r2_gaussian", "R²-Gaussian"), ("spags", "SPAGS")]:
        d = f"2026_04_30_{organ}_3views_{method}"
        if d in gs_lookup:
            efficiency_runs.append((d, mlabel, organ.capitalize()))

with open(os.path.join(OUT, "efficiency.csv"), "w") as f:
    f.write("dir,method,organ,n_gaussians,fps,ms_per_view\n")
    for d, method, organ in efficiency_runs:
        n_gs = gs_lookup.get(d, "N/A")
        timing = timing_lookup.get(d, {})
        fps = timing.get("fps", "N/A")
        ms = timing.get("ms_per_view", "N/A")
        f.write(f"{d},{method},{organ},{n_gs},{fps},{ms}\n")
        print(f"  {method:20s} {organ:10s} GS={n_gs} FPS={fps}")

print(f"\n✅ All data written to {OUT}/")
print(f"  Files: comparison.json, gs_counts.json, timing.json, xfield_eval.json,")
print(f"         stability.json, densify.json, ablation.json, efficiency.csv")
