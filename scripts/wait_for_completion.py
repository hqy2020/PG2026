#!/usr/bin/env python3
"""
实验完成监控 — 当全部 90 个实验跑完后，生成最终报告并写入 /tmp/experiments_done.txt
"""
import os, sys, json, yaml, time, subprocess
from pathlib import Path

PROJECT = Path("/home/qyhu/Documents/r2_ours/PG2026")
os.chdir(str(PROJECT))
DONE_FLAG = Path("/tmp/experiments_done_flag.txt")

def count_completed():
    output_dir = PROJECT / "output"
    count = 0
    for d in output_dir.glob("2026_04_30_*/"):
        eval_files = list(d.glob("eval/iter_030000/eval2d_*.yml"))
        if eval_files:
            count += 1
    return count

def generate_final_report():
    """生成最终汇总报告"""
    output_dir = PROJECT / "output"
    results = {}
    for d in sorted(output_dir.glob("2026_04_30_*/")):
        name = d.name
        rest = name[len("2026_04_30_"):]
        organ, rest2 = rest.split("_", 1)
        views_str, method = rest2.split("views_", 1)
        eval_files = list(d.glob("eval/iter_030000/eval2d_*.yml"))
        if eval_files:
            with open(eval_files[0]) as f:
                data = yaml.safe_load(f)
            results[name] = {
                "organ": organ, "views": int(views_str), "method": method,
                "psnr": round(data.get("psnr_2d", 0), 2),
                "ssim": round(data.get("ssim_2d", 0), 2),
            }

    # 按 di rname 排序输出
    lines = []
    for name in sorted(results.keys()):
        r = results[name]
        lines.append(f"{name:<50} | PSNR={r['psnr']:<6.2f} | SSIM={r['ssim']:.4f}")

    # 保存
    with open(PROJECT / "results" / "final_summary.txt", "w") as f:
        f.write("=== 90 实验最终结果 ===\n")
        f.write(f"实验目录名 {'':>35} PSNR     SSIM\n")
        f.write("="*80 + "\n")
        for line in lines:
            f.write(line + "\n")

    # 按方法平均
    from collections import defaultdict
    by_method = defaultdict(list)
    for r in results.values():
        by_method[r["method"]].append(r["psnr"])

    with open(PROJECT / "results" / "final_summary.txt", "a") as f:
        f.write("\n\n=== 按方法平均 PSNR ===\n")
        method_labels = {"r2_gaussian":"R²-Gaussian","spags":"SPAGS","xgaussian":"X-Gaussian","fsgs":"FSGS","corgs":"CoR-GS","dngaussian":"DN-Gaussian"}
        for m in ["r2_gaussian","spags","xgaussian","fsgs","corgs","dngaussian"]:
            psnrs = by_method.get(m, [])
            if psnrs:
                f.write(f"  {method_labels.get(m,m):<18}: avg={sum(psnrs)/len(psnrs):.2f}dB (n={len(psnrs)})\n")

    print(f"Final report saved to results/final_summary.txt")
    return lines, results

while True:
    n = count_completed()
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] Completed: {n}/90")

    if n >= 90:
        print("\n=== ALL 90 EXPERIMENTS COMPLETE! ===")
        generate_final_report()
        DONE_FLAG.write_text("done")
        break

    time.sleep(120)
