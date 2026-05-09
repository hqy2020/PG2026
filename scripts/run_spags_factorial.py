#!/usr/bin/env python3
"""
SPAGS 2³ 析因消融实验
8 configs × 5 organs × 3 views = 40 组 (已有 10 组, 新跑 30 组)
"""
import os, sys, subprocess, time, concurrent.futures
from pathlib import Path

PROJECT = Path(os.path.dirname(os.path.abspath(__file__))).parent
os.chdir(str(PROJECT))
PY = "/home/qyhu/anaconda3/envs/r2_gaussian_new/bin/python"
D = "data/234"
SD = "data/234-sps"
OUT = "output"
DS = time.strftime("%Y_%m_%d")

ORGANS = ["chest", "head", "abdomen", "foot", "pancreas"]
VIEW = 3
ITERS = 30000
TITERS = [5000, 10000, 15000, 20000, 25000, 30000]

# GAP args
GAP_ARGS = ["--enable_fsgs_proximity", "--gap_proximity_threshold", "0.05",
            "--gap_proximity_k", "5", "--no_gap_adaptive_threshold",
            "--no_gap_progressive_decay", "--gap_new_per_source", "1", "--gap_max_candidates", "2000"]

# ADM args
ADM_ARGS = ["--enable_kplanes", "--adm_resolution", "64", "--adm_feature_dim", "32",
            "--adm_decoder_hidden", "128", "--adm_decoder_layers", "3",
            "--kplanes_lr_init", "0.005", "--lambda_plane_tv", "0.0005",
            "--adm_warmup_iters", "15000", "--adm_max_range", "0.3",
            "--adm_view_adaptive", "--adm_zero_mean", "--adm_zero_mean_mode", "density_confidence"]

# 8 个配置
CONFIGS = [
    {"name": "r2_gaussian", "ply": f"{D}/init",        "gap": [],          "adm": [],          "label": "R²-Gaussian (基线)"},
    {"name": "sps_only",    "ply": f"{SD}/init",       "gap": [],          "adm": [],          "label": "+SPS"},
    {"name": "gap_only",    "ply": f"{D}/init",        "gap": GAP_ARGS,    "adm": [],          "label": "+GAP"},
    {"name": "adm_only",    "ply": f"{D}/init",        "gap": [],          "adm": ADM_ARGS,    "label": "+ADM"},
    {"name": "sps_gap",     "ply": f"{SD}/init",       "gap": GAP_ARGS,    "adm": [],          "label": "+SPS+GAP"},
    {"name": "sps_adm",     "ply": f"{SD}/init",       "gap": [],          "adm": ADM_ARGS,    "label": "+SPS+ADM"},
    {"name": "gap_adm",     "ply": f"{D}/init",        "gap": GAP_ARGS,    "adm": ADM_ARGS,    "label": "+GAP+ADM"},
    {"name": "spags",       "ply": f"{SD}/init",       "gap": GAP_ARGS,    "adm": ADM_ARGS,    "label": "Full SPAGS (完整)"},
]

def check_existing(name, organ):
    """Check if this experiment already exists from any date."""
    from glob import glob
    pattern = f"{OUT}/????_??_??_{organ}_{VIEW}views_{name}"
    for d in sorted(glob(pattern), reverse=True):
        eval_file = f"{d}/eval/iter_030000/eval2d_render_test.yml"
        if os.path.exists(eval_file):
            import yaml
            with open(eval_file) as f:
                ed = yaml.safe_load(f)
            return d, ed.get('psnr_2d', 0)
    return None, None

def run_one(cfg, organ, gpu):
    name = cfg["name"]
    ply_path = f"{cfg['ply']}_{organ}_50_{VIEW}views.npy"
    out_dir = f"{OUT}/{DS}_{organ}_{VIEW}views_{name}"
    data_path = f"{D}/{organ}_50_{VIEW}views.pickle"
    
    # 检查是否已有结果（任意日期）
    existing_dir, existing_psnr = check_existing(name, organ)
    if existing_dir is not None:
        return f"⏭️  {cfg['label']:20s} | {organ:8s} | PSNR={existing_psnr:.2f} (已有)"
    
    os.makedirs(out_dir, exist_ok=True)
    log = f"{out_dir}/run.log"
    
    cmd = [PY, "train.py", "--method", "r2_gaussian",
           "-s", data_path, "-m", out_dir,
           "--iterations", str(ITERS),
           "--test_iterations"] + [str(t) for t in TITERS] + \
           ["--save_iterations", "30000", "--ply_path", ply_path] + \
           cfg["gap"] + cfg["adm"]
    
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    
    t0 = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=7200)
    with open(log, "w") as f:
        f.write(r.stdout)
        if r.stderr: f.write("\n=== STDERR ===\n" + r.stderr)
    
    dur = time.time() - t0
    if os.path.exists(f"{out_dir}/eval/iter_030000/eval2d_render_test.yml"):
        import yaml
        with open(f"{out_dir}/eval/iter_030000/eval2d_render_test.yml") as f:
            ed = yaml.safe_load(f)
        return f"✅ {cfg['label']:20s} | {organ:8s} | PSNR={ed['psnr_2d']:.2f} | {dur:.0f}s"
    else:
        return f"❌ {cfg['label']:20s} | {organ:8s} | FAIL | {dur:.0f}s"

# 收集需要跑和已存在的
all_results = {}
for cfg in CONFIGS:
    psnrs = []
    for organ in ORGANS:
        existing_dir, existing_psnr = check_existing(cfg["name"], organ)
        if existing_dir is not None:
            all_results[(cfg["name"], organ)] = existing_psnr
            psnrs.append(existing_psnr)
    if psnrs:
        avg = sum(psnrs)/len(psnrs)
        print(f"  📊 {cfg['label']:20s}: {', '.join(f'{p:.2f}' for p in psnrs)} | avg={avg:.2f}")

todo = [(cfg, organ) for cfg in CONFIGS for organ in ORGANS 
        if (cfg["name"], organ) not in all_results]
done_count = 40 - len(todo)
print(f"\n总计: 40 组 | 已有: {done_count} | 新跑: {len(todo)}")

print(f"\n开始运行 {len(todo)} 组...")
gpus = [0, 1] * (len(todo) // 2 + 1)

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
    fs = {}
    for i, (cfg, organ) in enumerate(todo):
        gpu = gpus[i]
        f = ex.submit(run_one, cfg, organ, gpu)
        fs[f] = f"[{i+1}/{len(todo)}] {cfg['label']}/{organ}"
    
    for f in concurrent.futures.as_completed(fs):
        print(f"  {f.result()}")

print(f"\n✅ 消融实验完成! 输出至 {OUT}/{DS}_*")
