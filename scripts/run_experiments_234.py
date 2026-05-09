#!/usr/bin/env python3
"""
PG2026 实验编排系统 — 2/3/4 views 版本
========================================
管理 6 方法 × 3 视角 × 5 器官 = 90 个实验的完整生命周期。
使用 data/234/ 和 data/234-sps/ 作为数据源。

用法:
  python scripts/run_experiments_234.py --gpus 0 1

  # 只汇总结果（不跑实验）
  python scripts/run_experiments_234.py --summarize-only
"""

import os
import sys
import time
import json
import yaml
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─────────── 配置 ───────────
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__))).parent
os.chdir(str(PROJECT_ROOT))

DATA_DIR = "data/234"
SPS_DIR = "data/234-sps"

# 方法定义
METHODS = {
    "r2_gaussian": {
        "cli": "r2_gaussian",
        "args": ["--ply_path", f"{DATA_DIR}/init_{{organ}}_50_{{views}}views.npy"],
        "label": "R²-Gaussian",
        "group": "baseline",
    },
    "spags": {
        "cli": "r2_gaussian",
        "args": [
            f"--ply_path", f"{SPS_DIR}/init_{{organ}}_50_{{views}}views.npy",
            "--enable_fsgs_proximity", "--gap_proximity_threshold", "0.05",
            "--gap_proximity_k", "5", "--no_gap_adaptive_threshold",
            "--no_gap_progressive_decay", "--gap_new_per_source", "1", "--gap_max_candidates", "2000",
            "--enable_kplanes", "--adm_resolution", "64", "--adm_feature_dim", "32",
            "--adm_decoder_hidden", "128", "--adm_decoder_layers", "3",
            "--kplanes_lr_init", "0.005", "--lambda_plane_tv", "0.0005",
            "--adm_warmup_iters", "15000", "--adm_max_range", "0.3",
            "--adm_view_adaptive", "--adm_zero_mean", "--adm_zero_mean_mode", "density_confidence",
        ],
        "label": "SPAGS",
        "group": "ours",
    },
    "xgaussian": {
        "cli": "xgaussian",
        "args": ["--ply_path", f"{DATA_DIR}/init_{{organ}}_50_{{views}}views.npy"],
        "label": "X-Gaussian",
        "group": "comparison",
    },
    "fsgs": {
        "cli": "fsgs",
        "args": ["--ply_path", f"{DATA_DIR}/init_{{organ}}_50_{{views}}views.npy"],
        "label": "FSGS",
        "group": "comparison",
    },
    "corgs": {
        "cli": "corgs",
        "args": ["--ply_path", f"{DATA_DIR}/init_{{organ}}_50_{{views}}views.npy"],
        "label": "CoR-GS",
        "group": "comparison",
    },
    "dngaussian": {
        "cli": "dngaussian",
        "args": ["--ply_path", f"{DATA_DIR}/init_{{organ}}_50_{{views}}views.npy"],
        "label": "DN-Gaussian",
        "group": "comparison",
    },
}

# 器官和视角
ORGANS = ["chest", "head", "abdomen", "foot", "pancreas"]
ORGAN_LABELS = {"chest": "Chest", "head": "Head", "abdomen": "Abdomen",
                "foot": "Foot", "pancreas": "Pancreas"}
VIEWS = [2, 3, 4]

# 训练参数
# 使用完整 Python 路径（确保 conda 环境正确）
PYTHON_BIN = "/home/qyhu/anaconda3/envs/r2_gaussian_new/bin/python"

# 训练参数
ITERATIONS = 30000
TEST_ITERATIONS = [5000, 10000, 15000, 20000, 25000, 30000]

# 输出
DATE_STR = datetime.now().strftime("%Y_%m_%d")
OUTPUT_DIR = PROJECT_ROOT / "output"
RESULTS_DIR = PROJECT_ROOT / "results"


def make_output_dir(method: str, organ: str, views: int) -> Path:
    return OUTPUT_DIR / f"{DATE_STR}_{organ}_{views}views_{method}"


def build_command(method: str, organ: str, views: int, gpu: int) -> List[str]:
    cfg = METHODS[method]
    output = make_output_dir(method, organ, views)
    cmd = [
        PYTHON_BIN, "train.py",
        "--method", cfg["cli"],
        "-s", f"{DATA_DIR}/{organ}_50_{views}views.pickle",
        "-m", str(output),
        "--iterations", str(ITERATIONS),
        "--test_iterations", *map(str, TEST_ITERATIONS),
        "--save_iterations", "30000",
    ]
    for arg in cfg["args"]:
        cmd.append(arg.format(organ=organ, views=views))
    return cmd


def run_experiment(method: str, organ: str, views: int, gpu: int) -> Dict:
    output_dir = make_output_dir(method, organ, views)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_command(method, organ, views, gpu)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    start_time = time.time()
    result = {
        "method": method, "organ": organ, "views": views,
        "gpu": gpu, "output_dir": str(output_dir),
        "status": "running",
        "start_time": datetime.now().isoformat(),
    }

    log_path = output_dir / "run.log"
    try:
        proc = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=3600,
        )
        with open(log_path, "w") as f:
            f.write(proc.stdout)
            if proc.stderr:
                f.write("\n\n=== STDERR ===\n")
                f.write(proc.stderr)

        if proc.returncode == 0:
            result["status"] = "completed"
        else:
            result["status"] = "failed"
            result["error"] = proc.stderr[-500:] if proc.stderr else "Unknown error"
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = "Experiment timed out (3600s)"
    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)

    result["duration_s"] = time.time() - start_time
    result["end_time"] = datetime.now().isoformat()
    return result


def get_experiment_results(output_dir: Path) -> Optional[Dict]:
    eval_dir = output_dir / "eval" / "iter_030000"
    if not eval_dir.exists():
        return None

    # 不同方法使用不同的 eval 文件名
    eval_files = list(eval_dir.glob("eval2d_*.yml"))
    if not eval_files:
        return None

    eval_file = eval_files[0]
    with open(eval_file) as f:
        data = yaml.safe_load(f)

    result = {
        "psnr_2d": data.get("psnr_2d", None),
        "ssim_2d": data.get("ssim_2d", None),
    }
    return result


def collect_all_results(methods: Optional[Dict] = None) -> Dict:
    if methods is None:
        methods = METHODS
    all_results = {}
    for method in methods:
        for organ in ORGANS:
            for views in VIEWS:
                output_dir = make_output_dir(method, organ, views)
                key = f"{method}/{organ}/{views}"
                results = get_experiment_results(output_dir)
                if results:
                    all_results[key] = results
                else:
                    if output_dir.exists() and (output_dir / "run.log").exists():
                        all_results[key] = {"status": "no_results"}
                    else:
                        all_results[key] = {"status": "not_run"}
    return all_results


def print_results_table(results: Dict, methods: Optional[Dict] = None):
    if methods is None:
        methods = METHODS
    print("\n" + "="*100)
    print(f"PG2026 实验结果汇总 (2/3/4 views) — 日期: {DATE_STR}")
    print("="*100)

    for views in VIEWS:
        print(f"\n{'─'*100}")
        print(f"【{views} Views】PSNR_2D / SSIM_2D")
        print(f"{'─'*100}")
        header = f"{'器官':<12}"
        for m in methods:
            header += f"  {methods[m]['label']:<18}"
        print(header)
        print("-"*100)

        for organ in ORGANS:
            row = f"{ORGAN_LABELS[organ]:<12}"
            for m in methods:
                key = f"{m}/{organ}/{views}"
                r = results.get(key, {})
                if r.get("psnr_2d") is not None:
                    row += f"  {r['psnr_2d']:>6.2f} / {r['ssim_2d']:.4f}"
                elif r.get("status") == "no_results":
                    row += f"  {'RUN':>8} / {'--':>8}"
                else:
                    row += f"  {'--':>8} / {'--':>8}"
            print(row)

    # 按方法/视角的 PSNR 平均值
    print(f"\n{'─'*100}")
    print("【按视角平均 PSNR (跨 5 器官)】")
    print(f"{'─'*100}")
    for views in VIEWS:
        print(f"\n  {views} Views:")
        for m in methods:
            psnrs = []
            for organ in ORGANS:
                key = f"{m}/{organ}/{views}"
                r = results.get(key, {})
                if r.get("psnr_2d") is not None:
                    psnrs.append(r["psnr_2d"])
            if psnrs:
                avg = sum(psnrs) / len(psnrs)
                print(f"    {methods[m]['label']:<18}: {avg:.4f} (n={len(psnrs)})")
            else:
                print(f"    {methods[m]['label']:<18}: --")

    print(f"\n{'─'*100}")
    print("【总平均 PSNR (跨所有)】")
    print(f"{'─'*100}")
    for m in methods:
        psnrs = []
        for organ in ORGANS:
            for views in VIEWS:
                key = f"{m}/{organ}/{views}"
                r = results.get(key, {})
                if r.get("psnr_2d") is not None:
                    psnrs.append(r["psnr_2d"])
        if psnrs:
            avg = sum(psnrs) / len(psnrs)
            print(f"  {methods[m]['label']:<18}: {avg:.4f} (n={len(psnrs)})")
        else:
            print(f"  {methods[m]['label']:<18}: -- (0/15)")


def export_results_json(results: Dict, methods: Optional[Dict] = None):
    if methods is None:
        methods = METHODS
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output = {}
    for key, r in results.items():
        if r.get("psnr_2d") is not None:
            output[key] = {
                "psnr_2d": r["psnr_2d"],
                "ssim_2d": r["ssim_2d"],
            }

    suffix = "234"
    json_path = RESULTS_DIR / f"results_{DATE_STR}_{suffix}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n结果导出至: {json_path}")

    md_path = RESULTS_DIR / f"results_{DATE_STR}_{suffix}.md"
    with open(md_path, "w") as f:
        f.write(f"# PG2026 实验结果 (2/3/4 Views, {DATE_STR})\n\n")
        for views in VIEWS:
            f.write(f"## {views} Views\n\n")
            f.write("| 器官 |")
            for m in methods:
                f.write(f" {methods[m]['label']} PSNR | {methods[m]['label']} SSIM |")
            f.write("\n|------|")
            for _ in methods:
                f.write("------:|------:|")
            f.write("\n")
            for organ in ORGANS:
                f.write(f"| {ORGAN_LABELS[organ]} |")
                for m in methods:
                    key = f"{m}/{organ}/{views}"
                    r = results.get(key, {})
                    if r.get("psnr_2d") is not None:
                        f.write(f" {r['psnr_2d']:.2f} | {r['ssim_2d']:.4f} |")
                    else:
                        f.write(" -- | -- |")
                f.write("\n")
            f.write("\n")
    print(f"Markdown 表格导出至: {md_path}")


def run_all_experiments(gpus: List[int]):
    all_experiments = []
    for method in METHODS:
        for organ in ORGANS:
            for views in VIEWS:
                all_experiments.append((method, organ, views))

    total = len(all_experiments)
    completed_before = sum(
        1 for m, o, v in all_experiments
        if get_experiment_results(make_output_dir(m, o, v)) is not None
    )

    print(f"总计 {total} 个实验，已完成 {completed_before}，还需运行 {total - completed_before}")
    print(f"使用 GPU: {gpus}")

    pending = [(m, o, v) for m, o, v in all_experiments
               if get_experiment_results(make_output_dir(m, o, v)) is None]

    if not pending:
        print("所有实验已完成！")
        return

    print(f"还需运行 {len(pending)} 个实验...\n")
    gpu_cycle = iter(gpus * (len(pending) // len(gpus) + 1))

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {}
        for i, (method, organ, views) in enumerate(pending):
            gpu = next(gpu_cycle)
            label = f"[{i+1}/{len(pending)}] {METHODS[method]['label']} | {ORGAN_LABELS[organ]} | {views}views | GPU {gpu}"
            print(f"  > 启动: {label}")
            future = executor.submit(run_experiment, method, organ, views, gpu)
            futures[future] = label

        for future in as_completed(futures):
            label = futures[future]
            try:
                result = future.result()
                status = result["status"]
                duration = result.get("duration_s", 0)
                if status == "completed":
                    output_dir = Path(result["output_dir"])
                    metrics = get_experiment_results(output_dir)
                    psnr_str = f"{metrics['psnr_2d']:.2f}" if metrics and metrics.get('psnr_2d') else "?"
                    print(f"  ✓ {label} | PSNR={psnr_str} | {duration:.0f}s")
                elif status == "failed":
                    err = (result.get("error") or "Unknown")[:100]
                    print(f"  ✗ {label} | FAILED: {err}")
                else:
                    print(f"  ✗ {label} | {status}")
            except Exception as e:
                print(f"  ✗ {label} | EXCEPTION: {e}")


def main():
    parser = argparse.ArgumentParser(description="PG2026 实验编排系统 (2/3/4 views)")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--methods", nargs="+",
                        choices=list(METHODS.keys()) + ["all"],
                        default=["all"])
    args = parser.parse_args()

    scope = METHODS if args.methods == ["all"] else \
        {k: v for k, v in METHODS.items() if k in args.methods}

    if args.summarize_only:
        results = collect_all_results(scope)
        print_results_table(results, scope)
        export_results_json(results, scope)
        return

    run_all_experiments(args.gpus)

    print("\n\n== 汇总结果 ==")
    results = collect_all_results(scope)
    print_results_table(results, scope)
    export_results_json(results, scope)


if __name__ == "__main__":
    main()
