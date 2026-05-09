#!/usr/bin/env python3
"""SPS 超参数扫描（带GAP）"""
import os, sys, subprocess, time, concurrent.futures
from pathlib import Path
import numpy as np

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

# ADM 固定最优参数
ADM_ARGS = ["--enable_kplanes", "--adm_resolution", "64", "--adm_feature_dim", "32",
            "--adm_decoder_hidden", "128", "--adm_decoder_layers", "3",
            "--kplanes_lr_init", "0.005", "--lambda_plane_tv", "0.0005",
            "--adm_warmup_iters", "15000", "--adm_max_range", "0.3",
            "--adm_view_adaptive", "--adm_zero_mean", "--adm_zero_mean_mode", "density_confidence"]

# GAP 固定最优参数
GAP_ARGS = ["--enable_gap", "--gap_threshold", "0.015", "--gap_max_ratio", "0.02",
            "--gap_start_iter", "2000", "--gap_until_iter", "20000",
            "--gap_interval", "500", "--gap_k", "5",
            "--gap_gradient_aware", "--gap_gradient_threshold", "0.0002"]

# SPS 变体
SPS_VARIANTS = [
    # (name, label, init_dir, init_args)
    ("spsv1_gap", "SPS-v1(orig)+GAP", SD, []),  # 已有, 直接引用
    ("spsv2_gap", "SPS-v2(unif0.4/g0.7)+GAP", f"{D}-sps-v2-gap",
     ["--sps_uniform_ratio", "0.4", "--sps_density_gamma", "0.7"]),
    ("spsv4_gap", "SPS-v4(unif0.3/g0.8)+GAP", f"{D}-sps-v4-gap",
     ["--sps_uniform_ratio", "0.3", "--sps_density_gamma", "0.8"]),
    ("spsv5_gap", "SPS-v5(mean_init)+GAP", f"{D}-sps-v5-gap",
     ["--sps_density_init_mode", "match_valid_mean"]),
    ("spsv6_gap", "SPS-v6(75Kpts)+GAP", f"{D}-sps-v6-gap",
     ["--sps_uniform_ratio", "0.2", "--sps_density_gamma", "1.0", "--n_points", "75000"]),
]

def check_existing(name, organ):
    from glob import glob
    import yaml
    for d in sorted(glob(f"{OUT}/????_??_??_{organ}_{VIEW}views_{name}"), reverse=True):
        ef = f"{d}/eval/iter_030000/eval2d_render_test.yml"
        if os.path.exists(ef):
            with open(ef) as f:
                return d, yaml.safe_load(f).get('psnr_2d', 0)
    return None, None

# ── Phase 1: 生成初始化文件 ──
print("="*60)
print("Phase 1: 生成 SPS 初始化文件")
print("="*60)
for name, label, init_dir, init_args in SPS_VARIANTS:
    if init_dir == SD:
        print(f"  ⏭️  {label}: 使用已有 {init_dir}")
        continue
    os.makedirs(init_dir, exist_ok=True)
    print(f"  🏗️  {label}")
    for organ in ORGANS:
        out_path = f"{init_dir}/init_{organ}_50_{VIEW}views.npy"
        if os.path.exists(out_path):
            d = np.load(out_path)
            print(f"    ✅ {organ}: 已有 ({d.shape[0]} pts)")
            continue
        cmd = [PY, "initialize_pcd.py", "--data", f"{D}/{organ}_50_{VIEW}views.pickle",
               "--enable_sps", "--sps_strategy", "adaptive", "--output", out_path] + init_args
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            print(f"    ❌ {organ}: {r.stderr[:150]}")
        else:
            d = np.load(out_path)
            print(f"    ✅ {organ}: {d.shape[0]} pts")

# ── Phase 2: 训练 ──
print("\n" + "="*60)
print("Phase 2: 训练")
print("="*60)

# 统计已有（跳过 spsv1_gap，已有 gap_th0p015_r2 = 28.22）
for name, label, init_dir, _ in SPS_VARIANTS:
    if name == "spsv1_gap":
        print(f"  📊 {label:30s}: avg=28.22 (已有)")
        continue

# 构建 todo，排除 spsv1_gap
todo = []
for name, label, init_dir, _ in SPS_VARIANTS:
    if name == "spsv1_gap":
        continue
    for organ in ORGANS:
        if check_existing(name, organ)[0] is None:
            todo.append((name, label, init_dir, organ))

done_count = len(SPS_VARIANTS)*len(ORGANS) - len(todo)
print(f"\n总计: {len(SPS_VARIANTS)*len(ORGANS)} 组 | 已有: {done_count} | 新跑: {len(todo)}")

def run_one(name, label, init_dir, organ, gpu):
    d, _ = check_existing(name, organ)
    if d is not None:
        return f"⏭️  {label:30s} | {organ:8s} | PSNR={_:.2f} (已有)"

    ply_path = f"{init_dir}/init_{organ}_50_{VIEW}views.npy"
    out_dir = f"{OUT}/{DS}_{organ}_{VIEW}views_{name}"
    data_path = f"{D}/{organ}_50_{VIEW}views.pickle"
    os.makedirs(out_dir, exist_ok=True)
    log = f"{out_dir}/run.log"

    cmd = [PY, "train.py", "--method", "r2_gaussian",
           "-s", data_path, "-m", out_dir,
           "--iterations", str(ITERS),
           "--test_iterations"] + [str(t) for t in TITERS] + \
           ["--save_iterations", "30000", "--ply_path", ply_path,
            "--no_enable_gap"] + ADM_ARGS + GAP_ARGS

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
        return f"✅ {label:30s} | {organ:8s} | PSNR={ed['psnr_2d']:.2f} | {dur:.0f}s"
    else:
        return f"❌ {label:30s} | {organ:8s} | FAIL | {dur:.0f}s"

if len(todo) == 0:
    print("全部已有，无需运行！")
    sys.exit(0)

print(f"\n开始运行 {len(todo)} 组...")
gpus = [0, 1] * (len(todo)//2 + 1)

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
    fs = {}
    for i, (name, label, init_dir, organ) in enumerate(todo):
        f = ex.submit(run_one, name, label, init_dir, organ, gpus[i])
        fs[f] = f"[{i+1}/{len(todo)}] {label}/{organ}"
    for f in concurrent.futures.as_completed(fs):
        print(f"  {f.result()}")

print(f"\n✅ SPS 超参扫描完成! 输出至 {OUT}/{DS}_*")
