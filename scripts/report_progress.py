#!/usr/bin/env python3
"""
实验进度汇报脚本 — 由 cronjob 每 30 分钟调用
"""
import os, sys, json, subprocess
from pathlib import Path
from datetime import datetime

PROJECT = Path("/home/qyhu/Documents/r2_ours/PG2026")
os.chdir(str(PROJECT))

# 检查主进程是否还在运行
main_proc = subprocess.run(
    ["ps", "aux", "--no-headers"],
    capture_output=True, text=True
)
running = [l for l in main_proc.stdout.split("\n") if "train.py" in l and "grep" not in l]
n_running = len(running)

# 检查完整结果
completed = list(Path("output").glob("2026_04_30_*/eval/iter_030000/eval2d_render_test.yml"))
n_completed = len(completed)

# 收集完成实验的信息
if n_completed > 0:
    import yaml
    results_by_method = {}
    for f in completed:
        parts = f.parts
        # output/2026_04_30_chest_3views_r2_gaussian/eval/iter_030000/eval2d_render_test.yml
        dirname = f.parent.parent.parent.name  # the experiment dir
        # parse: 2026_04_30_{organ}_{views}views_{method}
        # e.g. 2026_04_30_chest_3views_r2_gaussian
        try:
            rest = dirname[len("2026_04_30_"):]  # chest_3views_r2_gaussian
            organ, rest = rest.split("_", 1)  # organ=chest, rest=3views_r2_gaussian
            views_str, method = rest.split("views_", 1)
            views = int(views_str)
        except:
            continue
        with open(f) as fh:
            data = yaml.safe_load(fh)
        psnr = data.get("psnr_2d", 0)
        key = f"{method}/{organ}/{views}"
        results_by_method.setdefault(method, []).append(psnr)

    # 计算各方法平均
    avgs = {}
    for m, psnrs in results_by_method.items():
        avgs[m] = sum(psnrs) / len(psnrs)

    report = "\n".join(
        f"  {m:<18}: {avg:.2f} dB (n={len(results_by_method[m])})"
        for m, avg in avgs.items()
    )
else:
    report = "  （暂无完成实验）"

# 时间
now = datetime.now().strftime("%H:%M")

# 检查现有输出目录数量
n_dirs = len(list(Path("output").glob("2026_04_30_*/")))

print(f"📊 **实验进度报告 ({now})**")
print(f"")
print(f"| 指标 | 数值 |")
print(f"|:-----|:----:|")
print(f"| 总实验数 | 90 |")
print(f"| 已完成 | **{n_completed}** |")
print(f"| 运行中 | {n_running} |")
print(f"| 有输出目录 | {n_dirs} |")
print(f"")
if n_running > 0:
    print(f"🔵 训练仍在进行中...")
elif n_completed >= 90:
    print(f"✅ **全部 90 个实验已完成！**")
else:
    print(f"⏸️ 训练进程已结束")
