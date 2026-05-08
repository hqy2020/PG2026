#!/usr/bin/env python3
"""
SPAGS 消融实验 — 3 views × 5 organs × 3 消融 = 15 组
"""
import os, sys, subprocess, time, concurrent.futures
from pathlib import Path

PROJECT = Path(os.path.dirname(os.path.abspath(__file__))).parent
os.chdir(str(PROJECT))
PYTHON = "/home/qyhu/anaconda3/envs/r2_gaussian_new/bin/python"
DATA_DIR = "data/234"
SPS_DIR = "data/234-sps"
OUTPUT_DIR = "output"
DATE_STR = time.strftime("%Y_%m_%d")

ORGANS = ["chest", "head", "abdomen", "foot", "pancreas"]
VIEWS = [3]  # 只用 3 views
ITERS = 30000
TEST_ITERS = [5000, 10000, 15000, 20000, 25000, 30000]

# SPAGS base args
SPAGS_BASE = [
    "--enable_fsgs_proximity", "--gar_proximity_threshold", "0.05",
    "--gar_proximity_k", "5", "--no_gar_adaptive_threshold",
    "--no_gar_progressive_decay", "--gar_new_per_source", "1", "--gar_max_candidates", "2000",
    "--enable_kplanes", "--adm_resolution", "64", "--adm_feature_dim", "32",
    "--adm_decoder_hidden", "128", "--adm_decoder_layers", "3",
    "--kplanes_lr_init", "0.005", "--lambda_plane_tv", "0.0005",
    "--adm_warmup_iters", "15000", "--adm_max_range", "0.3",
    "--adm_view_adaptive", "--adm_zero_mean", "--adm_zero_mean_mode", "density_confidence",
]

ABLATIONS = {
    "spags_wogar": {  # w/o GAR: only ADM
        "args": [],
        "label": "SPAGS w/o GAR",
    },
    "spags_woadm": {  # w/o ADM: only GAR
        "args": [],
        "label": "SPAGS w/o ADM",
    },
    "spags_wotv": {  # w/o TV loss
        "args": [],
        "label": "SPAGS w/o TV",
    },
}

# 构建各消融的参数列表
for i, (key, val) in enumerate(zip(SPAGS_BASE[::2], SPAGS_BASE[1::2])):
    # w/o GAR: 去掉所有 --gar_* 和 --enable_fsgs_proximity
    if key not in ["--enable_fsgs_proximity"] and not key.startswith("--gar_"):
        ABLATIONS["spags_wogar"]["args"].extend([key, val])
    
    # w/o ADM: 去掉所有 --adm_*, --kplanes_*, --enable_kplanes, --lambda_plane_tv
    if not key.startswith("--adm_") and not key.startswith("--kplanes_") \
       and key not in ["--enable_kplanes", "--lambda_plane_tv"]:
        ABLATIONS["spags_woadm"]["args"].extend([key, val])
    
    # w/o TV: 所有保留，但 lambda_plane_tv = 0
    if key == "--lambda_plane_tv":
        ABLATIONS["spags_wotv"]["args"].extend([key, "0"])
    else:
        ABLATIONS["spags_wotv"]["args"].extend([key, val])

def run_one(ablation, organ, views, gpu):
    cfg = ABLATIONS[ablation]
    output_dir = f"{OUTPUT_DIR}/{DATE_STR}_{organ}_{views}views_{ablation}"
    data_path = f"{DATA_DIR}/{organ}_50_{views}views.pickle"
    ply_path = f"{SPS_DIR}/init_{organ}_50_{views}views.npy"
    
    if os.path.exists(f"{output_dir}/eval/iter_030000/eval2d_render_test.yml"):
        return f"⏭️  {cfg['label']} | {organ} {views}v — 已存在"
    
    os.makedirs(output_dir, exist_ok=True)
    log_file = f"{output_dir}/run.log"
    
    cmd = [PYTHON, "train.py", "--method", "r2_gaussian",
           "-s", data_path, "-m", output_dir,
           "--iterations", str(ITERS),
           "--test_iterations"] + [str(t) for t in TEST_ITERS] + \
           ["--save_iterations", "30000", "--ply_path", ply_path] + cfg["args"]
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    
    start = time.time()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=7200)
    with open(log_file, "w") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n=== STDERR ===\n" + result.stderr)
    
    duration = time.time() - start
    if os.path.exists(f"{output_dir}/eval/iter_030000/eval2d_render_test.yml"):
        import yaml
        with open(f"{output_dir}/eval/iter_030000/eval2d_render_test.yml") as f:
            eval_data = yaml.safe_load(f)
        psnr = eval_data.get("psnr_2d", 0)
        return f"✅ {cfg['label']} | {organ} {views}v | PSNR={psnr:.2f} | {duration:.0f}s"
    else:
        err = result.stderr[-300:] if result.stderr else "Unknown"
        return f"❌ {cfg['label']} | {organ} {views}v | FAILED: {err[:100]}"

# Run
print("=" * 60)
print("SPAGS 消融实验 (3 views × 5 organs × 3 ablations = 15 组)")
print("=" * 60)

experiments = []
for ablation in ABLATIONS:
    for organ in ORGANS:
        experiments.append((ablation, organ, 3))

print(f"总计 {len(experiments)} 组，GPU 0/1 并行")
gpus = [0, 1] * (len(experiments) // 2 + 1)

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    futures = {}
    for i, (ablation, organ, views) in enumerate(experiments):
        gpu = gpus[i]
        future = executor.submit(run_one, ablation, organ, views, gpu)
        futures[future] = f"[{i+1}/{len(experiments)}] {ABLATIONS[ablation]['label']}/{organ}"
    
    for future in concurrent.futures.as_completed(futures):
        label = futures[future]
        try:
            print(f"  {future.result()}")
        except Exception as e:
            print(f"  ❌ {label} — {e}")

print("\n✅ 消融实验全部完成!")
