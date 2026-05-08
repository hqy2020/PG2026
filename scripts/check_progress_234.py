#!/usr/bin/env python3
"""检查 PG2026 2/3/4-views 实验进度，输出文本报告"""
import os
import sys
import yaml
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path("/home/qyhu/Documents/r2_ours/PG2026")
OUTPUT_DIR = PROJECT_ROOT / "output"
RESULTS_DIR = PROJECT_ROOT / "results"

METHODS = {
    "r2_gaussian": "R²-Gaussian",
    "spags": "SPAGS",
    "xgaussian": "X-Gaussian",
    "fsgs": "FSGS",
    "corgs": "CoR-GS",
    "dngaussian": "DN-Gaussian",
}

ORGANS = ["chest", "head", "abdomen", "foot", "pancreas"]
ORGAN_LABELS = {"chest": "Chest", "head": "Head", "abdomen": "Abdomen", "foot": "Foot", "pancreas": "Pancreas"}
VIEWS = [2, 3, 4]

DATE_STR = datetime.now().strftime("%Y_%m_%d")

def get_experiment_results(output_dir):
    """获取实验的 eval 结果（如果有 iter_030000）"""
    eval_dir = output_dir / "eval" / "iter_030000"
    if not eval_dir.exists():
        return None
    eval_files = list(eval_dir.glob("eval2d_*.yml"))
    if not eval_files:
        return None
    with open(eval_files[0]) as f:
        data = yaml.safe_load(f)
    return {
        "psnr_2d": data.get("psnr_2d"),
        "ssim_2d": data.get("ssim_2d"),
    }

def get_max_eval_iter(output_dir):
    """获取当前最大 eval iter"""
    eval_root = output_dir / "eval"
    if not eval_root.exists():
        return 0
    iters = []
    for d in eval_root.iterdir():
        if d.name.startswith("iter_"):
            try:
                iters.append(int(d.name.split("_")[1]))
            except:
                pass
    return max(iters) if iters else 0

def check_progress():
    completed = []
    running = []
    not_started = []
    
    for method_key, method_label in METHODS.items():
        for organ in ORGANS:
            for views in VIEWS:
                output_dir = OUTPUT_DIR / f"{DATE_STR}_{organ}_{views}views_{method_key}"
                key = f"{method_label}/{ORGAN_LABELS[organ]}/{views}views"
                
                if output_dir.exists():
                    max_iter = get_max_eval_iter(output_dir)
                    results = get_experiment_results(output_dir)
                    if results:
                        completed.append((key, results["psnr_2d"], results["ssim_2d"], method_key))
                    else:
                        running.append((key, max_iter, method_key))
                else:
                    not_started.append((key, method_key))
    
    return completed, running, not_started

def main():
    completed, running, not_started = check_progress()
    
    total = len(METHODS) * len(ORGANS) * len(VIEWS)
    print(f"╔══ PG2026 2/3/4-views 实验进度报告 ══╗")
    print(f"║ 时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"║ 完成: {len(completed)}/{total}  ({len(completed)/total*100:.0f}%)")
    print(f"║ 运行中: {len(running)}")
    print(f"║ 未开始: {len(not_started)}")
    print(f"╚{'═'*40}╝")
    
    if running:
        print(f"\n▶ 运行中 ({len(running)}):")
        running.sort(key=lambda x: x[1], reverse=True)
        for key, max_iter, mk in running:
            pct = min(max_iter / 30000 * 100, 99)
            bar = "█" * (int(pct/5)) + "░" * (20 - int(pct/5))
            print(f"  {key:<35} [{bar}] {max_iter/30000*100:.0f}% ({max_iter}/30000)")
    
    if completed:
        print(f"\n✅ 已完成 ({len(completed)}):")
        # 按方法/视角分组
        by_method = {}
        for key, psnr, ssim, mk in completed:
            by_method.setdefault(mk, []).append((key, psnr, ssim))
        
        for mk in ["r2_gaussian", "spags", "xgaussian", "fsgs", "corgs", "dngaussian"]:
            if mk in by_method:
                items = by_method[mk]
                avg_psnr = sum(x[1] for x in items) / len(items)
                print(f"  {METHODS[mk]}: avg PSNR={avg_psnr:.2f}")
                for key, psnr, ssim in sorted(items):
                    print(f"    {key:<35} PSNR={psnr:.2f}  SSIM={ssim:.4f}")
    
    if not_started:
        print(f"\n⏳ 待运行 ({len(not_started)}):")
        by_method = {}
        for key, mk in not_started:
            by_method.setdefault(mk, []).append(key)
        for mk in METHODS:
            if mk in by_method:
                print(f"  {METHODS[mk]}: {len(by_method[mk])} 个")
    
    # 按视角统计已完成的 PSNR
    if completed:
        print(f"\n📊 按视角平均 PSNR:")
        for views in VIEWS:
            psnrs = [x[1] for x in completed if f"{views}views" in x[0]]
            if psnrs:
                print(f"  {views} Views: {sum(psnrs)/len(psnrs):.2f} (n={len(psnrs)})")

if __name__ == "__main__":
    main()
