#!/usr/bin/env python3
"""GAP 超参数扫描：围绕阈值 0.015 和比例 3% 做网格搜索"""
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

ADM_ARGS = ["--enable_kplanes", "--adm_resolution", "64", "--adm_feature_dim", "32",
            "--adm_decoder_hidden", "128", "--adm_decoder_layers", "3",
            "--kplanes_lr_init", "0.005", "--lambda_plane_tv", "0.0005",
            "--adm_warmup_iters", "15000", "--adm_max_range", "0.3",
            "--adm_view_adaptive", "--adm_zero_mean", "--adm_zero_mean_mode", "density_confidence"]

GAP_GRID = [
    {"name": "gap_th0p01_r3",  "label": "GAP(t=0.01,r=3%)",  "args": ["--enable_gap", "--gap_threshold","0.01","--gap_max_ratio","0.03","--gap_start_iter","2000","--gap_until_iter","20000","--gap_interval","500","--gap_k","5","--gap_gradient_aware","--gap_gradient_threshold","0.0002"]},
    {"name": "gap_th0p02_r3",  "label": "GAP(t=0.02,r=3%)",  "args": ["--enable_gap", "--gap_threshold","0.02","--gap_max_ratio","0.03","--gap_start_iter","2000","--gap_until_iter","20000","--gap_interval","500","--gap_k","5","--gap_gradient_aware","--gap_gradient_threshold","0.0002"]},
    {"name": "gap_th0p015_r2", "label": "GAP(t=0.015,r=2%)","args": ["--enable_gap", "--gap_threshold","0.015","--gap_max_ratio","0.02","--gap_start_iter","2000","--gap_until_iter","20000","--gap_interval","500","--gap_k","5","--gap_gradient_aware","--gap_gradient_threshold","0.0002"]},
    {"name": "gap_th0p015_r5", "label": "GAP(t=0.015,r=5%)","args": ["--enable_gap", "--gap_threshold","0.015","--gap_max_ratio","0.05","--gap_start_iter","2000","--gap_until_iter","20000","--gap_interval","500","--gap_k","5","--gap_gradient_aware","--gap_gradient_threshold","0.0002"]},
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

all_results = {}
for cfg in GAP_GRID:
    psnrs = []
    for organ in ORGANS:
        d, p = check_existing(cfg["name"], organ)
        if d is not None:
            all_results[(cfg["name"], organ)] = p
            psnrs.append(p)
    if psnrs:
        avg = sum(psnrs)/len(psnrs)
        print(f"  📊 {cfg['label']:25s}: {', '.join(f'{v:.2f}' for v in psnrs)} | avg={avg:.2f}")

todo = [(cfg, organ) for cfg in GAP_GRID for organ in ORGANS if (cfg["name"], organ) not in all_results]
total = len(GAP_GRID)*len(ORGANS)
done = total - len(todo)
print(f"\n总计: {total} 组 | 已有: {done} | 新跑: {len(todo)}")

def run_one(cfg, organ, gpu):
    d, _ = check_existing(cfg["name"], organ)
    if d is not None:
        return f"⏭️  {cfg['label']:25s} | {organ:8s} | 已有"

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
            "--no_enable_gap"] + ADM_ARGS + cfg["args"]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
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
        return f"✅ {cfg['label']:25s} | {organ:8s} | PSNR={ed['psnr_2d']:.2f} | {dur:.0f}s"
    else:
        return f"❌ {cfg['label']:25s} | {organ:8s} | FAIL | {dur:.0f}s"

if len(todo) == 0:
    print("全部已有，无需运行！")
    sys.exit(0)

print(f"\n开始运行 {len(todo)} 组...")
gpus = [0, 1] * (len(todo)//2 + 1)

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
    fs = {}
    for i, (cfg, organ) in enumerate(todo):
        gpu = gpus[i]
        f = ex.submit(run_one, cfg, organ, gpu)
        fs[f] = f"[{i+1}/{len(todo)}] {cfg['label']}/{organ}"
    for f in concurrent.futures.as_completed(fs):
        print(f"  {f.result()}")

print(f"\n✅ GAP 超参扫描完成! 输出至 {OUT}/{DS}_*")
