#!/usr/bin/env python3
"""ADM 超参数扫描（带 SPS + GAP 最优）"""
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

# GAP 固定最优
GAP_ARGS = ["--enable_gap", "--gap_threshold", "0.015", "--gap_max_ratio", "0.02",
            "--gap_start_iter", "2000", "--gap_until_iter", "20000",
            "--gap_interval", "500", "--gap_k", "5",
            "--gap_gradient_aware", "--gap_gradient_threshold", "0.0002"]

# ADM 变体（基于原始最优改单个参数）
ADM_BASE = ["--enable_kplanes", "--adm_feature_dim", "32",
            "--adm_decoder_hidden", "128", "--adm_decoder_layers", "3",
            "--kplanes_lr_init", "0.005", "--lambda_plane_tv", "0.0005",
            "--adm_view_adaptive", "--adm_zero_mean", "--adm_zero_mean_mode", "density_confidence"]

ADM_GRID = [
    # warmup 扫描
    {"name": "adm_warm12k", "label": "ADM(warm=12K, r=0.3, res=64)",
     "args": ADM_BASE + ["--adm_resolution","64","--adm_warmup_iters","12000","--adm_max_range","0.3"]},
    {"name": "adm_warm18k", "label": "ADM(warm=18K, r=0.3, res=64)",
     "args": ADM_BASE + ["--adm_resolution","64","--adm_warmup_iters","18000","--adm_max_range","0.3"]},
    # max_range 扫描
    {"name": "adm_range0p2",  "label": "ADM(warm=15K, r=0.2, res=64)",
     "args": ADM_BASE + ["--adm_resolution","64","--adm_warmup_iters","15000","--adm_max_range","0.2"]},
    {"name": "adm_range0p4",  "label": "ADM(warm=15K, r=0.4, res=64)",
     "args": ADM_BASE + ["--adm_resolution","64","--adm_warmup_iters","15000","--adm_max_range","0.4"]},
    # resolution 重测（带 GAP）
    {"name": "adm_res128",    "label": "ADM(warm=15K, r=0.3, res=128)",
     "args": ADM_BASE + ["--adm_resolution","128","--adm_warmup_iters","15000","--adm_max_range","0.3"]},
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

# 统计已有
for cfg in ADM_GRID:
    psnrs = []
    for organ in ORGANS:
        d, p = check_existing(cfg["name"], organ)
        if d is not None:
            psnrs.append(p)
    if psnrs:
        print(f"  📊 {cfg['label']:35s}: avg={sum(psnrs)/len(psnrs):.2f} ({len(psnrs)}/5)")

todo = [(cfg, organ) for cfg in ADM_GRID for organ in ORGANS
        if check_existing(cfg["name"], organ)[0] is None]
total = len(ADM_GRID) * len(ORGANS)
done = total - len(todo)
print(f"\n总计: {total} 组 | 已有: {done} | 新跑: {len(todo)}")

def run_one(cfg, organ, gpu):
    d, _ = check_existing(cfg["name"], organ)
    if d is not None:
        return f"⏭️  {cfg['label']:35s} | {organ:8s} | 已有"

    ply_path = f"{SD}/init_{organ}_50_{VIEW}views.npy"
    out_dir = f"{OUT}/{DS}_{organ}_{VIEW}views_{cfg['name']}"
    data_path = f"{D}/{organ}_50_{VIEW}views.pickle"
    os.makedirs(out_dir, exist_ok=True)
    log = f"{out_dir}/run.log"

    cmd = [PY, "train.py", "--method", "r2_gaussian",
           "-s", data_path, "-m", out_dir,
           "--iterations", str(ITERS),
           "--test_iterations"] + [str(t) for t in TITERS] + \
           ["--save_iterations", "30000", "--ply_path", ply_path,
            "--no_enable_gap"] + cfg["args"] + GAP_ARGS

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
        return f"✅ {cfg['label']:35s} | {organ:8s} | PSNR={ed['psnr_2d']:.2f} | {dur:.0f}s"
    else:
        return f"❌ {cfg['label']:35s} | {organ:8s} | FAIL | {dur:.0f}s"

if len(todo) == 0:
    print("全部已有，无需运行！")
    sys.exit(0)

print(f"\n开始运行 {len(todo)} 组...")
gpus = [0, 1] * (len(todo)//2 + 1)

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
    fs = {}
    for i, (cfg, organ) in enumerate(todo):
        f = ex.submit(run_one, cfg, organ, gpus[i])
        fs[f] = f"[{i+1}/{len(todo)}] {cfg['label']}/{organ}"
    for f in concurrent.futures.as_completed(fs):
        print(f"  {f.result()}")

print(f"\n✅ ADM 超参扫描完成! 输出至 {OUT}/{DS}_*")
