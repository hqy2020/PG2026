#!/usr/bin/env python3
"""
PPrune (邻近剪枝) 实验
替代GAR，在SPS+ADM基础上添加邻近剪枝

设计：
  baseline: SPS+ADM（已有数据）
  pprune-v1: 保守剪枝 (threshold=0.015, ratio=0.03, start=3K, until=15K)
  pprune-v2: 中等剪枝 (threshold=0.025, ratio=0.05, start=2K, until=20K)
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

# ADM 参数（使用已验证的最优值）
ADM_ARGS = ["--enable_kplanes", "--adm_resolution", "64", "--adm_feature_dim", "32",
            "--adm_decoder_hidden", "128", "--adm_decoder_layers", "3",
            "--kplanes_lr_init", "0.005", "--lambda_plane_tv", "0.0005",
            "--adm_warmup_iters", "15000", "--adm_max_range", "0.3",
            "--adm_view_adaptive", "--adm_zero_mean", "--adm_zero_mean_mode", "density_confidence"]

# PPrune 变体
PPRUNE_VARIANTS = {
    "pprune_v1": {
        "label": "SPS+ADM+PPrune-v1(保守)",
        "args": ["--enable_prune", "--prune_threshold", "0.015", "--prune_max_ratio", "0.03",
                 "--prune_start_iter", "3000", "--prune_until_iter", "15000",
                 "--prune_interval", "500", "--prune_k", "5",
                 "--prune_gradient_aware", "--prune_gradient_threshold", "0.0002"],
    },
    "pprune_v2": {
        "label": "SPS+ADM+PPrune-v2(中等)",
        "args": ["--enable_prune", "--prune_threshold", "0.025", "--prune_max_ratio", "0.05",
                 "--prune_start_iter", "2000", "--prune_until_iter", "20000",
                 "--prune_interval", "500", "--prune_k", "5",
                 "--prune_gradient_aware", "--prune_gradient_threshold", "0.0002"],
    },
}

EXPS = [
    {"name": "sps_adm_pprune_v1", "pprune": "pprune_v1"},
    {"name": "sps_adm_pprune_v2", "pprune": "pprune_v2"},
]

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

# 统计已有结果
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
        print(f"  📊 {exp['name']:30s}: {', '.join(f'{p:.2f}' for p in psnrs)} | avg={avg:.2f}")

todo = [(exp, organ) for exp in EXPS for organ in ORGANS
        if (exp["name"], organ) not in all_results]
done_count = len(EXPS) * len(ORGANS) - len(todo)
print(f"\n总计: {len(EXPS)*len(ORGANS)} 组 | 已有: {done_count} | 新跑: {len(todo)}")

def run_one(exp, organ, gpu):
    name = exp["name"]
    pprune_key = exp["pprune"]
    pprune_cfg = PPRUNE_VARIANTS[pprune_key]

    existing_dir, existing_psnr = check_existing(name, organ)
    if existing_dir is not None:
        return f"⏭️  {pprune_cfg['label']:30s} | {organ:8s} | PSNR={existing_psnr:.2f} (已有)"

    ply_path = f"{SD}/init_{organ}_50_{VIEW}views.npy"
    out_dir = f"{OUT}/{DS}_{organ}_{VIEW}views_{name}"
    data_path = f"{D}/{organ}_50_{VIEW}views.pickle"
    os.makedirs(out_dir, exist_ok=True)
    log = f"{out_dir}/run.log"

    cmd = [PY, "train.py", "--method", "r2_gaussian",
           "-s", data_path, "-m", out_dir,
           "--iterations", str(ITERS),
           "--test_iterations"] + [str(t) for t in TITERS] + \
           ["--save_iterations", "30000", "--ply_path", ply_path,
            "--no_enable_gar"] + ADM_ARGS + pprune_cfg["args"]

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
        return f"✅ {pprune_cfg['label']:30s} | {organ:8s} | PSNR={ed['psnr_2d']:.2f} | {dur:.0f}s"
    else:
        return f"❌ {pprune_cfg['label']:30s} | {organ:8s} | FAIL | {dur:.0f}s"

if len(todo) == 0:
    print("全部已有，无需运行！")
    sys.exit(0)

print(f"\n开始运行 {len(todo)} 组...")
gpus = [0, 1] * (len(todo) // 2 + 1)

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
    fs = {}
    for i, (exp, organ) in enumerate(todo):
        gpu = gpus[i]
        f = ex.submit(run_one, exp, organ, gpu)
        fs[f] = f"[{i+1}/{len(todo)}] {PPRUNE_VARIANTS[exp['pprune']]['label']}/{organ}"

    for f in concurrent.futures.as_completed(fs):
        print(f"  {f.result()}")

print(f"\n✅ PPrune 实验完成! 输出至 {OUT}/{DS}_*")
