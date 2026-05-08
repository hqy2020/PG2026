#!/usr/bin/env python3
"""
SPAGS 改进实验: SPS 调优 × ADM 调优
8 configs × 5 organs = 40 组 (已有SPAGS基线10组, 新跑30组)
设计: SPS-v2/v3 + ADM-v2/v3 交叉组合

SPS变体:
  v1 (original): adaptive, uniform_ratio=0.2, gamma=1.0, init_mode=raw
  v2:            adaptive, uniform_ratio=0.4, gamma=0.7, init_mode=raw
  v3:            adaptive, uniform_ratio=0.5, gamma=0.5, init_mode=match_valid_mean

ADM变体:
  v1 (original): res=64, warmup=15000, max_range=0.3
  v2:            res=128, warmup=8000,  max_range=0.5
  v3:            res=128, warmup=10000, max_range=0.5
"""
import os, sys, subprocess, time, concurrent.futures
from pathlib import Path
import numpy as np

PROJECT = Path(os.path.dirname(os.path.abspath(__file__))).parent
os.chdir(str(PROJECT))
PY = "/home/qyhu/anaconda3/envs/r2_gaussian_new/bin/python"
D = "data/234"
OUT = "output"
DS = time.strftime("%Y_%m_%d")

ORGANS = ["chest", "head", "abdomen", "foot", "pancreas"]
VIEW = 3
ITERS = 30000
TITERS = [5000, 10000, 15000, 20000, 25000, 30000]

# ── SPS 变体定义 (参数不同 -> 不同 init 目录) ──
SPS_VARIANTS = {
    "spsv1": {
        "label": "SPS-v1(orig)",
        "init_dir": f"{D}-sps",          # 已有的 original SPS init
        "init_args": [],
        "already": True,                  # 已有文件,不用重新生成
    },
    "spsv2": {
        "label": "SPS-v2(unif0.4/gamma0.7)",
        "init_dir": f"{D}-sps-v2",
        "init_args": ["--sps_uniform_ratio", "0.4", "--sps_density_gamma", "0.7"],
        "already": False,
    },
    "spsv3": {
        "label": "SPS-v3(unif0.5/gamma0.5/mean)",
        "init_dir": f"{D}-sps-v3",
        "init_args": ["--sps_uniform_ratio", "0.5", "--sps_density_gamma", "0.5",
                       "--sps_density_init_mode", "match_valid_mean"],
        "already": False,
    },
}

# ── ADM 变体定义 ──
ADM_BASE = ["--enable_kplanes", "--adm_feature_dim", "32",
            "--adm_decoder_hidden", "128", "--adm_decoder_layers", "3",
            "--kplanes_lr_init", "0.005", "--lambda_plane_tv", "0.0005",
            "--adm_view_adaptive", "--adm_zero_mean", "--adm_zero_mean_mode", "density_confidence"]

ADM_VARIANTS = {
    "admv1": {
        "label": "ADM-v1(orig)",
        "args": ADM_BASE + ["--adm_resolution", "64", "--adm_warmup_iters", "15000", "--adm_max_range", "0.3"],
    },
    "admv2": {
        "label": "ADM-v2(res128/warm8k/range0.5)",
        "args": ADM_BASE + ["--adm_resolution", "128", "--adm_warmup_iters", "8000", "--adm_max_range", "0.5"],
    },
    "admv3": {
        "label": "ADM-v3(res128/warm10k/range0.5)",
        "args": ADM_BASE + ["--adm_resolution", "128", "--adm_warmup_iters", "10000", "--adm_max_range", "0.5"],
    },
}

# ── 实验配置组合 ──
EXPS = [
    # SPS-v2 + ADM-original → SPS improvement only
    {"name": "spsv2_admv1", "label": "SPS-v2 + ADM-v1", "sps": "spsv2", "adm": "admv1"},
    # SPS-v3 + ADM-original → more aggressive SPS
    {"name": "spsv3_admv1", "label": "SPS-v3 + ADM-v1", "sps": "spsv3", "adm": "admv1"},
    # SPS-original + ADM-v2 → ADM improvement only
    {"name": "spsv1_admv2", "label": "SPS-v1 + ADM-v2", "sps": "spsv1", "adm": "admv2"},
    # SPS-original + ADM-v3 → ADM moderate
    {"name": "spsv1_admv3", "label": "SPS-v1 + ADM-v3", "sps": "spsv1", "adm": "admv3"},
    # SPS-v2 + ADM-v2 → best combo
    {"name": "spsv2_admv2", "label": "SPS-v2 + ADM-v2", "sps": "spsv2", "adm": "admv2"},
]

# ── 1. 生成新的 SPS 初始化文件 ──
print("╔═══════════════════════════════════════════╗")
print("║  Phase 1: Generate SPS init files         ║")
print("╚═══════════════════════════════════════════╝")

for sps_key, sps_cfg in SPS_VARIANTS.items():
    if sps_cfg["already"]:
        print(f"  ✅ {sps_cfg['label']}: already exists, skip")
        continue
    init_dir = sps_cfg["init_dir"]
    os.makedirs(init_dir, exist_ok=True)
    print(f"  🏗️  {sps_cfg['label']}: generating...")
    for organ in ORGANS:
        out_path = f"{init_dir}/init_{organ}_50_{VIEW}views.npy"
        if os.path.exists(out_path):
            d = np.load(out_path)
            print(f"    ✅ {organ}: already exists ({d.shape[0]} pts)")
            continue
        data_path = f"{D}/{organ}_50_{VIEW}views.pickle"
        cmd = [PY, "initialize_pcd.py",
               "--data", data_path,
               "--enable_sps",
               "--sps_strategy", "adaptive",
               "--output", out_path] + sps_cfg["init_args"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            print(f"    ❌ {organ}: {r.stderr[:200]}")
        else:
            d = np.load(out_path)
            print(f"    ✅ {organ}: {d.shape[0]} pts")
    print()

# ── 2. 训练实验 ──
print("╔═══════════════════════════════════════════╗")
print("║  Phase 2: Run training experiments        ║")
print("╚═══════════════════════════════════════════╝")

# 收集已有结果
def check_existing(name, organ):
    from glob import glob
    import yaml
    pattern = f"{OUT}/????_??_??_{organ}_{VIEW}views_{name}"
    for d in sorted(glob(pattern), reverse=True):
        eval_file = f"{d}/eval/iter_030000/eval2d_render_test.yml"
        if os.path.exists(eval_file):
            with open(eval_file) as f:
                ed = yaml.safe_load(f)
            return d, ed.get('psnr_2d', 0)
    return None, None

all_results = {}
for exp in EXPS:
    psnrs = []
    for organ in ORGANS:
        existing_dir, existing_psnr = check_existing(exp["name"], organ)
        if existing_dir is not None:
            all_results[(exp["name"], organ)] = existing_psnr
            psnrs.append(existing_psnr)
    if psnrs:
        avg = sum(psnrs)/len(psnrs)
        print(f"  📊 {exp['label']:30s}: {', '.join(f'{p:.2f}' for p in psnrs)} | avg={avg:.2f}")

todo = [(exp, organ) for exp in EXPS for organ in ORGANS
        if (exp["name"], organ) not in all_results]
done_count = len(EXPS) * len(ORGANS) - len(todo)
print(f"\n总计: {len(EXPS)*len(ORGANS)} 组 | 已有: {done_count} | 新跑: {len(todo)}")

def run_one(exp, organ, gpu):
    name = exp["name"]
    sps_key = exp["sps"]
    adm_key = exp["adm"]
    sps_cfg = SPS_VARIANTS[sps_key]
    adm_cfg = ADM_VARIANTS[adm_key]

    # 检查是否已有结果
    existing_dir, existing_psnr = check_existing(name, organ)
    if existing_dir is not None:
        return f"⏭️  {exp['label']:30s} | {organ:8s} | PSNR={existing_psnr:.2f} (已有)"

    ply_path = f"{sps_cfg['init_dir']}/init_{organ}_50_{VIEW}views.npy"
    out_dir = f"{OUT}/{DS}_{organ}_{VIEW}views_{name}"
    data_path = f"{D}/{organ}_50_{VIEW}views.pickle"

    os.makedirs(out_dir, exist_ok=True)
    log = f"{out_dir}/run.log"

    cmd = [PY, "train.py", "--method", "r2_gaussian",
           "-s", data_path, "-m", out_dir,
           "--iterations", str(ITERS),
           "--test_iterations"] + [str(t) for t in TITERS] + \
           ["--save_iterations", "30000", "--ply_path", ply_path,
            "--no_enable_gar"] + adm_cfg["args"]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    t0 = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=7200)
    with open(log, "w") as f:
        f.write(r.stdout)
        if r.stderr:
            f.write("\n=== STDERR ===\n" + r.stderr)

    dur = time.time() - t0
    if os.path.exists(f"{out_dir}/eval/iter_030000/eval2d_render_test.yml"):
        import yaml
        with open(f"{out_dir}/eval/iter_030000/eval2d_render_test.yml") as f:
            ed = yaml.safe_load(f)
        return f"✅ {exp['label']:30s} | {organ:8s} | PSNR={ed['psnr_2d']:.2f} | {dur:.0f}s"
    else:
        return f"❌ {exp['label']:30s} | {organ:8s} | FAIL | {dur:.0f}s"

print(f"\n开始运行 {len(todo)} 组...")
gpus = [0, 1] * (len(todo) // 2 + 1)

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
    fs = {}
    for i, (exp, organ) in enumerate(todo):
        gpu = gpus[i]
        f = ex.submit(run_one, exp, organ, gpu)
        fs[f] = f"[{i+1}/{len(todo)}] {exp['label']}/{organ}"

    for f in concurrent.futures.as_completed(fs):
        print(f"  {f.result()}")

print(f"\n✅ 改进实验完成! 输出至 {OUT}/{DS}_*")
print(f"\n{'='*60}")
print("  SPS+ADM 原始              | avg=28.09")
print(f"  预计完成后可以对比改进幅度")
print(f"{'='*60}")
