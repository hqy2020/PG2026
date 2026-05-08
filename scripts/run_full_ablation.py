#!/usr/bin/env python3
"""全部消融实验：8 configs × 3 views × 5 organs，只跑缺失的"""
import os, sys, subprocess, time, concurrent.futures
from pathlib import Path
from glob import glob
import yaml

PROJECT = Path(os.path.dirname(os.path.abspath(__file__))).parent
os.chdir(str(PROJECT))
PY = "/home/qyhu/anaconda3/envs/r2_gaussian_new/bin/python"
SD = "data"
D = f"{SD}/234"
SPS = f"{SD}/234-sps"
OUT = "output"
DS = time.strftime("%Y_%m_%d")

VIEWS = [2, 3, 4]
ORGANS = ["chest", "head", "abdomen", "foot", "pancreas"]
ITERS = 30000
TITERS = [5000, 10000, 15000, 20000, 25000, 30000]

# ADM 最优参数
ADM_ARGS = ["--enable_kplanes", "--adm_resolution", "64", "--adm_feature_dim", "32",
            "--adm_decoder_hidden", "128", "--adm_decoder_layers", "3",
            "--kplanes_lr_init", "0.005", "--lambda_plane_tv", "0.0005",
            "--adm_warmup_iters", "15000", "--adm_max_range", "0.3",
            "--adm_view_adaptive", "--adm_zero_mean", "--adm_zero_mean_mode", "density_confidence"]

# GAP 最优参数
GAP_ARGS = ["--enable_gap", "--gap_threshold", "0.015", "--gap_max_ratio", "0.02",
            "--gap_start_iter", "2000", "--gap_until_iter", "20000",
            "--gap_interval", "500", "--gap_k", "5",
            "--gap_gradient_aware", "--gap_gradient_threshold", "0.0002"]

def find_result(name, organ, views):
    for d in sorted(glob(f"{OUT}/????_??_??_{organ}_{views}views_{name}"), reverse=True):
        ef = f"{d}/eval/iter_030000/eval2d_render_test.yml"
        if os.path.exists(ef):
            with open(ef) as f:
                try:
                    return d, yaml.safe_load(f).get('psnr_2d', 0)
                except:
                    pass
    # 检查别名目录
    for alias in ["gap_th0p015_r2", "sps_adm_pprune_v1", "sps_adm_gap"]:
        d = f"{OUT}/2026_05_03_{organ}_{views}views_{alias}"
        ef = f"{d}/eval/iter_030000/eval2d_render_test.yml"
        if os.path.exists(ef):
            with open(ef) as f:
                try:
                    return d, yaml.safe_load(f).get('psnr_2d', 0)
                except:
                    pass
    return None, None

# 8个配置
CONFIGS = [
    {"name": "r2_gaussian",  "label": "R²-Gaussian",  "sps": False, "adm": False, "gap": False},
    {"name": "sps_only",     "label": "+SPS",          "sps": True,  "adm": False, "gap": False},
    {"name": "adm_only",     "label": "+ADM",          "sps": False, "adm": True,  "gap": False},
    {"name": "gap_only",     "label": "+GAP",          "sps": False, "adm": False, "gap": True},
    {"name": "sps_adm",      "label": "SPS+ADM",       "sps": True,  "adm": True,  "gap": False},
    {"name": "sps_gap",      "label": "SPS+GAP",       "sps": True,  "adm": False, "gap": True},
    {"name": "adm_gap",      "label": "ADM+GAP",       "sps": False, "adm": True,  "gap": True},
    {"name": "sps_adm_gap",  "label": "SPS+ADM+GAP",   "sps": True,  "adm": True,  "gap": True},
]

# 收集 todo
todo = []
for cfg in CONFIGS:
    for v in VIEWS:
        for organ in ORGANS:
            d, p = find_result(cfg["name"], organ, v)
            if d is None:
                todo.append((cfg, organ, v))

total = len(CONFIGS) * len(VIEWS) * len(ORGANS)
done = total - len(todo)
print(f"总计: {total} 组 | 已有: {done} | 新跑: {len(todo)}")
if done > 0:
    print(f"\n已有结果摘要:")
    for cfg in CONFIGS:
        for v in VIEWS:
            psnrs = []
            for organ in ORGANS:
                d, p = find_result(cfg["name"], organ, v)
                if d:
                    psnrs.append(p)
            if len(psnrs) == 5:
                print(f"  {cfg['label']:20s} | {v}v | avg={sum(psnrs)/5:.2f}")
            elif len(psnrs) > 0 and len(psnrs) < 5:
                print(f"  {cfg['label']:20s} | {v}v | 部分({len(psnrs)}/5) avg={sum(psnrs)/len(psnrs):.2f}")

def run_one(cfg, organ, view, gpu):
    name = cfg["name"]
    init_std = f"{D}/init_{organ}_50_{view}views.npy"
    init_sps = f"{SPS}/init_{organ}_50_{view}views.npy"
    ply_path = init_sps if cfg["sps"] else init_std
    out_dir = f"{OUT}/{DS}_{organ}_{view}views_{name}"
    data_path = f"{D}/{organ}_50_{view}views.pickle"
    os.makedirs(out_dir, exist_ok=True)
    log = f"{out_dir}/run.log"

    cmd = [PY, "train.py", "--method", "r2_gaussian",
           "-s", data_path, "-m", out_dir,
           "--iterations", str(ITERS),
           "--test_iterations"] + [str(t) for t in TITERS] + \
           ["--save_iterations", "30000", "--ply_path", ply_path]

    if cfg["adm"]:
        cmd += ADM_ARGS
    if cfg["gap"]:
        cmd += GAP_ARGS

    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    t0 = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=7200)
    with open(log, "w") as f:
        f.write(r.stdout)
        if r.stderr: f.write("\n=== STDERR ===\n" + r.stderr)
    dur = time.time() - t0
    ef = f"{out_dir}/eval/iter_030000/eval2d_render_test.yml"
    if os.path.exists(ef):
        with open(ef) as f:
            ed = yaml.safe_load(f)
        return f"✅ {cfg['label']:20s} | {organ:8s} | {view}v | PSNR={ed['psnr_2d']:.2f} | {dur:.0f}s"
    else:
        err = open(log).read()[-300:] if os.path.exists(log) else ""
        return f"❌ {cfg['label']:20s} | {organ:8s} | {view}v | FAIL | {dur:.0f}s"

if len(todo) == 0:
    print("✅ 全部已有，无需运行！")
    sys.exit(0)

print(f"\n开始运行 {len(todo)} 组 (2 GPU 并行)...")
gpus = [0, 1] * (len(todo)//2 + 1)

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
    fs = {}
    for i, (cfg, organ, view) in enumerate(todo):
        f = ex.submit(run_one, cfg, organ, view, gpus[i])
        fs[f] = f
    for f in concurrent.futures.as_completed(fs):
        print(f"  {f.result()}")

print(f"\n✅ 全部消融实验完成! 输出至 {OUT}/{DS}_*")
