#!/usr/bin/env python3
"""
PG2026 实验编排系统 v2
========================
管理 6 方法 × 3 视角 × 5 器官 = 90 个实验的完整生命周期。

方法:
  - r2_gaussian (baseline): R²-Gaussian，不加 SPAGS 特征
  - xgaussian, fsgs, corgs, dngaussian: 对比基线
  - spags: 我们的完整方法 (SPS + GAR + ADM)

用法:
  # 运行所有实验
  python scripts/run_experiments_v2.py --gpus 0 1

  # 只运行指定方法
  python scripts/run_experiments_v2.py --gpus 0 1 --methods r2_gaussian xgaussian

  # 只汇总结果（不跑实验）
  python scripts/run_experiments_v2.py --summarize-only
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
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─────────── 配置 ───────────
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__))).parent
os.chdir(str(PROJECT_ROOT))

# 方法定义
METHODS = {
    "r2_gaussian": {
        "cli": "r2_gaussian",
        "args": ["--ply_path", "data/369/init_{organ}_50_{views}views.npy"],
        "label": "R²-Gaussian",
        "group": "baseline",
    },
    "spags": {
        "cli": "r2_gaussian",
        "args": [
            "--ply_path", "data/369-sps/init_{organ}_50_{views}views.npy",
            "--enable_fsgs_proximity", "--gar_proximity_threshold", "0.05",
            "--gar_proximity_k", "5", "--no_gar_adaptive_threshold",
            "--no_gar_progressive_decay", "--gar_new_per_source", "1", "--gar_max_candidates", "2000",
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
        "args": ["--ply_path", "data/369/init_{organ}_50_{views}views.npy"],
        "label": "X-Gaussian",
        "group": "comparison",
    },
    "fsgs": {
        "cli": "fsgs",
        "args": ["--ply_path", "data/369/init_{organ}_50_{views}views.npy"],
        "label": "FSGS",
        "group": "comparison",
    },
    "corgs": {
        "cli": "corgs",
        "args": ["--ply_path", "data/369/init_{organ}_50_{views}views.npy"],
        "label": "CoR-GS",
        "group": "comparison",
    },
    "dngaussian": {
        "cli": "dngaussian",
        "args": ["--ply_path", "data/369/init_{organ}_50_{views}views.npy"],
        "label": "DN-Gaussian",
        "group": "comparison",
    },
}

# 器官和视角
ORGANS = ["chest", "head", "abdomen", "foot", "pancreas"]
ORGAN_LABELS = {"chest": "Chest", "head": "Head", "abdomen": "Abdomen",
                "foot": "Foot", "pancreas": "Pancreas"}
VIEWS = [3, 6, 9]

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
        "python", "train.py",
        "--method", cfg["cli"],
        "-s", f"data/369/{organ}_50_{views}views.pickle",
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
    eval_file = output_dir / "eval" / "iter_030000" / "eval2d_render_test.yml"
    eval3d_file = output_dir / "eval" / "iter_030000" / "eval3d.yml"
    has_volume = (output_dir / "point_cloud" / "iteration_30000" / "vol_pred.npy").exists()

    if not eval_file.exists():
        return None

    with open(eval_file) as f:
        data = yaml.safe_load(f)

    result = {
        "psnr_2d": data.get("psnr_2d", None),
        "ssim_2d": data.get("ssim_2d", None),
    }

    if eval3d_file.exists():
        with open(eval3d_file) as f:
            data3d = yaml.safe_load(f)
        result["psnr_3d"] = data3d.get("psnr_3d", None)
        result["ssim_3d"] = data3d.get("ssim_3d", None)

    result["has_volume"] = has_volume
    result["volume_path"] = str(output_dir / "point_cloud" / "iteration_30000") if has_volume else None
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
    print(f"PG2026 实验结果汇总 — 日期: {DATE_STR}")
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

    print(f"\n{'─'*100}")
    print("【平均 PSNR (跨所有器官和视角)】")
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
                "psnr_3d": r.get("psnr_3d"),
                "ssim_3d": r.get("ssim_3d"),
                "has_volume": r.get("has_volume", False),
            }

    json_path = RESULTS_DIR / f"results_{DATE_STR}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n结果导出至: {json_path}")

    md_path = RESULTS_DIR / f"results_{DATE_STR}.md"
    with open(md_path, "w") as f:
        f.write(f"# PG2026 实验结果 ({DATE_STR})\n\n")
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
    parser = argparse.ArgumentParser(description="PG2026 实验编排系统")
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
