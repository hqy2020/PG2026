#!/usr/bin/env python3
"""
SPAGS 性能优化实验 — 基于 ARIS 分析
========================================
目标：在 2/3/4 views 设定上压榨 SPAGS 的最佳性能

ARIS (MiniMax + DeepSeek V4 Pro) 分析结论：
- GAR threshold 是最关键的超参数
- ADM warmup/zero-mean 次之
- K-planes TV 影响最小
"""
import os, sys, time, json, yaml, argparse, subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__))).parent
os.chdir(str(PROJECT_ROOT))

DATA_DIR = "data/234"
SPS_DIR = "data/234-sps"

PYTHON_BIN = "/home/qyhu/anaconda3/envs/r2_gaussian_new/bin/python"
ITERATIONS = 30000
TEST_ITERATIONS = [5000, 10000, 15000, 20000, 25000, 30000]
DATE_STR = datetime.now().strftime("%Y_%m_%d")
OUTPUT_DIR = PROJECT_ROOT / "output"
RESULTS_DIR = PROJECT_ROOT / "results"

# ========== 实验设计 ==========

# Batch 1: GAR threshold 扫描 (最关键)
# 根据 ARIS 分析：proximity_threshold=0.05 是稀疏视角定制的，
# 不同视角密度需要不同阈值
GAR_THRESHOLDS = [0.02, 0.05, 0.08, 0.12]

# Batch 2: ADM warmup 调整
# 不同 warmup 迭代数对性能的影响
ADM_WARMUPS = [0, 5000, 10000, 15000, 20000]

# Batch 3: ADM zero-mean 影响
# zero-mean 约束可能过正则化
ADM_ZERO_MEAN_MODES = ["density_confidence", "none"]

# 默认 SPAGS 参数
DEFAULT_SPAGS_ARGS = [
    "--enable_fsgs_proximity", "--gar_proximity_threshold", "0.05",
    "--gar_proximity_k", "5",
    "--no_gar_adaptive_threshold", "--no_gar_progressive_decay",
    "--gar_new_per_source", "1", "--gar_max_candidates", "2000",
    "--enable_kplanes", "--adm_resolution", "64", "--adm_feature_dim", "32",
    "--adm_decoder_hidden", "128", "--adm_decoder_layers", "3",
    "--kplanes_lr_init", "0.005", "--lambda_plane_tv", "0.0005",
    "--adm_warmup_iters", "15000", "--adm_max_range", "0.3",
    "--adm_view_adaptive", "--adm_zero_mean", "--adm_zero_mean_mode", "density_confidence",
]

# ========== 实验批次定义 ==========

EXPERIMENTS = []

# ---- Batch A: GAR 阈值扫描 (Chest, 最具代表性) ----
# 问题：不同视角密度需要不同的 GAR 阈值
for v in [2, 3, 4]:
    for thr in GAR_THRESHOLDS:
        name = f"gar_thr{thr}_{v}v"
        args = list(DEFAULT_SPAGS_ARGS)
        # 替换默认 threshold
        idx = args.index("--gar_proximity_threshold") if "--gar_proximity_threshold" in args else -1
        if idx >= 0:
            args[idx+1] = str(thr)
        else:
            args += ["--gar_proximity_threshold", str(thr)]
        EXPERIMENTS.append({
            "name": name,
            "organ": "chest",
            "views": v,
            "args": args,
            "batch": "A_GAR_threshold",
            "desc": f"GAR proximity_threshold={thr}",
        })

# ---- Batch B: ADM warmup 扫描 (Chest, 3v, 用 Batch A 最优 threshold) ----
# 注意：这里先用默认 threshold=0.05，后续可替换
for wp in ADM_WARMUPS:
    name = f"adm_warmup{wp}_3v"
    args = list(DEFAULT_SPAGS_ARGS)
    idx = args.index("--adm_warmup_iters") if "--adm_warmup_iters" in args else -1
    if idx >= 0:
        args[idx+1] = str(wp)
    EXPERIMENTS.append({
        "name": name,
        "organ": "chest",
        "views": 3,
        "args": args,
        "batch": "B_ADM_warmup",
        "desc": f"ADM warmup_iters={wp}",
    })

# ---- Batch C: ADM zero-mean 变体 (Chest, 3v) ----
for zm in ADM_ZERO_MEAN_MODES:
    name = f"adm_zm_{zm}_3v"
    args = list(DEFAULT_SPAGS_ARGS)
    if zm == "none":
        # 去掉 --adm_zero_mean 相关参数（含值）
        args = [a for a in args if a not in ["--adm_zero_mean", "--adm_zero_mean_mode", "density_confidence"]]
    else:
        idx = args.index("--adm_zero_mean_mode") if "--adm_zero_mean_mode" in args else -1
        if idx >= 0:
            args[idx+1] = zm
    EXPERIMENTS.append({
        "name": name,
        "organ": "chest",
        "views": 3,
        "args": args,
        "batch": "C_ADM_zeromean",
        "desc": f"ADM zero_mean_mode={zm}",
    })

# ---- Batch D: 最优组合 × 全部器官 ----
# 用优化后的 GAR threshold + ADM warmup 在所有器官上复现
BEST_THRESHOLD_BY_VIEW = {2: 0.05, 3: 0.05, 4: 0.05}  # 会被 Batch A 结果更新
for organ in ["head", "abdomen", "foot", "pancreas"]:
    for v in [2, 3, 4]:
        name = f"optimized_{organ}_{v}v"
        args = list(DEFAULT_SPAGS_ARGS)
        thr = BEST_THRESHOLD_BY_VIEW[v]
        idx = args.index("--gar_proximity_threshold")
        args[idx+1] = str(thr)
        EXPERIMENTS.append({
            "name": name,
            "organ": organ,
            "views": v,
            "args": args,
            "batch": "D_optimized_all",
            "desc": f"优化配置 (GAR thr={thr})",
        })


def build_command(organ, views, extra_args, gpu):
    output = OUTPUT_DIR / f"{DATE_STR}_{organ}_{views}views_spags_optimized"
    cmd = [
        PYTHON_BIN, "train.py",
        "--method", "r2_gaussian",
        "-s", f"{DATA_DIR}/{organ}_50_{views}views.pickle",
        "-m", str(output),
        "--iterations", str(ITERATIONS),
        "--test_iterations", *map(str, TEST_ITERATIONS),
        "--save_iterations", "30000",
        "--ply_path", f"{SPS_DIR}/init_{organ}_50_{views}views.npy",
    ] + extra_args
    return cmd, output


def run_experiment(exp, gpu):
    output_dir = OUTPUT_DIR / f"{DATE_STR}_{exp['organ']}_{exp['views']}views_spags_opt_{exp['name']}"
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd, _ = build_command(exp["organ"], exp["views"], exp["args"], gpu)
    cmd[cmd.index("-m") + 1] = str(output_dir)
    cmd[cmd.index("--ply_path") + 1] = f"{SPS_DIR}/init_{exp['organ']}_50_{exp['views']}views.npy"
    cmd[cmd.index("-s") + 1] = f"{DATA_DIR}/{exp['organ']}_50_{exp['views']}views.pickle"
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    
    result = {
        "name": exp["name"],
        "organ": exp["organ"],
        "views": exp["views"],
        "batch": exp["batch"],
        "desc": exp["desc"],
        "gpu": gpu,
        "output_dir": str(output_dir),
        "status": "running",
        "start_time": datetime.now().isoformat(),
    }
    
    log_path = output_dir / "run.log"
    start_time = time.time()
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=7200)
        with open(log_path, "w") as f:
            f.write(proc.stdout)
            if proc.stderr:
                f.write("\n\n=== STDERR ===\n")
                f.write(proc.stderr)
        result["status"] = "completed" if proc.returncode == 0 else "failed"
        result["error"] = proc.stderr[-500:] if proc.stderr and proc.returncode != 0 else None
    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
    
    result["duration_s"] = time.time() - start_time
    result["end_time"] = datetime.now().isoformat()
    return result


def get_psnr(output_dir):
    eval_dir = Path(output_dir) / "eval" / "iter_030000"
    if not eval_dir.exists():
        return None
    files = list(eval_dir.glob("eval2d_*.yml"))
    if not files:
        return None
    with open(files[0]) as f:
        data = yaml.safe_load(f)
    return data.get("psnr_2d"), data.get("ssim_2d")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--batch", type=str, default=None, 
                        choices=["A_GAR_threshold", "B_ADM_warmup", "C_ADM_zeromean", "D_optimized_all", None])
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    
    scope = [e for e in EXPERIMENTS if args.batch is None or e["batch"] == args.batch]
    
    if args.summarize_only:
        print(f"\n{'='*70}")
        print(f"SPAGS 优化实验结果汇总 ({len(scope)} 组)")
        print(f"{'='*70}")
        for exp in scope:
            out_dir = OUTPUT_DIR / f"{DATE_STR}_{exp['organ']}_{exp['views']}views_spags_opt_{exp['name']}"
            psnr_ssim = get_psnr(out_dir)
            if psnr_ssim:
                print(f"  ✅ {exp['name']:30s} PSNR={psnr_ssim[0]:.2f}  SSIM={psnr_ssim[1]:.4f}")
            else:
                print(f"  ⏳ {exp['name']:30s} (running or not started)")
        return
    
    print(f"SPAGS 优化实验 ({len(scope)} 组)")
    print(f"GPU: {args.gpus}")
    print(f"批次: {args.batch or '全部'}\n")
    
    pending = scope
    gpu_cycle = iter(args.gpus * (len(pending) // len(args.gpus) + 1))
    
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = {}
        for i, exp in enumerate(pending):
            gpu = next(gpu_cycle)
            label = f"[{i+1}/{len(pending)}] {exp['name']} | {exp['organ']} | {exp['views']}v | GPU {gpu}"
            print(f"  > 启动: {label}")
            future = executor.submit(run_experiment, exp, gpu)
            futures[future] = label
        
        for future in as_completed(futures):
            label = futures[future]
            try:
                result = future.result()
                if result["status"] == "completed":
                    psnr_ssim = get_psnr(result["output_dir"])
                    psnr_str = f"{psnr_ssim[0]:.2f}" if psnr_ssim else "?"
                    print(f"  ✓ {label} | PSNR={psnr_str} | {result['duration_s']:.0f}s")
                else:
                    err = (result.get("error") or "?")[:80]
                    print(f"  ✗ {label} | FAILED: {err}")
            except Exception as e:
                print(f"  ✗ {label} | EXCEPTION: {e}")


if __name__ == "__main__":
    main()
