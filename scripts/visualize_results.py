#!/usr/bin/env python3
"""
PG2026 新视角合成可视化 + 实验报告系统

为每个完成的实验运行 test.py 生成 2D 渲染图，
创建 GT vs Pred 对比图，并整合成进度看板。
"""
import os, sys, json, yaml, subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(os.path.dirname(os.path.abspath(__file__))).parent
os.chdir(str(PROJECT))

OUTPUT = PROJECT / "output"
RESULTS = PROJECT / "results"
PROGRESS_FILE = RESULTS / ".render_progress.json"
CONDA_ENV = "r2_gaussian_new"
CONDA_SH = os.path.expanduser("~/anaconda3/etc/profile.d/conda.sh")

METHOD_LABELS = {
    "r2_gaussian": "R²-Gaussian", "spags": "SPAGS",
    "xgaussian": "X-Gaussian", "fsgs": "FSGS",
    "corgs": "CoR-GS", "dngaussian": "DN-Gaussian",
}
ORGAN_LABELS = {
    "chest": "Chest", "head": "Head", "abdomen": "Abdomen",
    "foot": "Foot", "pancreas": "Pancreas",
}
METHOD_COLORS = {
    "r2_gaussian": "#4CAF50", "spags": "#2196F3",
    "xgaussian": "#FF9800", "fsgs": "#E91E63",
    "corgs": "#9C27B0", "dngaussian": "#607D8B",
}

RESULTS.mkdir(parents=True, exist_ok=True)


def parse_experiment_dir(dirname):
    """解析实验目录名"""
    rest = dirname[len("2026_04_30_"):]
    try:
        organ, rest2 = rest.split("_", 1)
        views_str, method = rest2.split("views_", 1)
        return {"organ": organ, "views": int(views_str), "method": method}
    except:
        return None


def find_completed():
    """找到所有完成的实验"""
    completed = []
    for f in sorted(OUTPUT.glob("2026_04_30_*/eval/iter_030000/eval2d_render_test.yml")):
        dirname = f.parent.parent.parent.name
        info = parse_experiment_dir(dirname)
        if info is None:
            continue
        info["dirname"] = dirname
        info["path"] = str(f.parent.parent.parent)
        with open(f) as fh:
            data = yaml.safe_load(fh)
        info["psnr_2d"] = data.get("psnr_2d", 0)
        info["ssim_2d"] = data.get("ssim_2d", 0)
        info["has_test_render"] = (f.parent.parent.parent / "test" / "iter_30000" / "render_test").exists()
        completed.append(info)
    return completed


def run_test(exp_dir):
    """对单个实验运行 test.py 生成渲染图"""
    cmd = (
        f"source {CONDA_SH} && conda activate {CONDA_ENV} && "
        f"CUDA_VISIBLE_DEVICES=0 python test.py "
        f"-m {exp_dir} --iteration 30000 "
        f"--skip_recon 2>/dev/null"
    )
    try:
        subprocess.run(cmd, shell=True, timeout=120,
                       capture_output=True, text=True)
        return True
    except subprocess.TimeoutExpired:
        return False
    except Exception as e:
        return False


def ensure_test_renders(completed, max_new=2):
    """对未渲染的实验运行 test.py，最多处理 max_new 个"""
    new_processed = 0
    for exp in reversed(completed):
        if new_processed >= max_new:
            break
        if not exp["has_test_render"]:
            print(f"  > 渲染: {exp['dirname']}")
            ok = run_test(exp["path"])
            if ok:
                exp["has_test_render"] = True
                new_processed += 1
            else:
                print(f"  ✗ 渲染失败: {exp['dirname']}")
    return new_processed


def create_comparison_montage(exp, output_path, n_views=8):
    """创建 GT vs Pred 的对比拼图"""
    render_dir = Path(exp["path"]) / "test" / "iter_30000" / "render_test"
    if not render_dir.exists():
        return False

    # 找到所有 GT 和 Pred 文件
    gt_files = sorted(render_dir.glob("*_gt.png"))
    pred_files = sorted(render_dir.glob("*_pred.png"))
    if len(gt_files) < n_views:
        n_views = len(gt_files)

    # 均匀采样
    indices = np.linspace(0, len(gt_files) - 1, n_views, dtype=int)

    fig, axes = plt.subplots(n_views, 2, figsize=(10, 3 * n_views))

    for i, idx in enumerate(indices):
        gt_img = np.array(Image.open(gt_files[idx]).convert("L"))
        pred_img = np.array(Image.open(pred_files[idx]).convert("L"))

        vmin = min(gt_img.min(), pred_img.min())
        vmax = max(gt_img.max(), pred_img.max())

        axes[i, 0].imshow(gt_img, cmap="gray", vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f"GT View {idx}", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(pred_img, cmap="gray", vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f"Pred View {idx}", fontsize=9)
        axes[i, 1].axis("off")

    label = METHOD_LABELS.get(exp["method"], exp["method"])
    organ = ORGAN_LABELS.get(exp["organ"], exp["organ"])
    fig.suptitle(
        f"{organ} | {exp['views']} Views | {label}\n"
        f"PSNR: {exp['psnr_2d']:.2f} dB | SSIM: {exp['ssim_2d']:.4f}",
        fontsize=14, fontweight="bold", y=1.005
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return True


def create_novel_view_grid(exp, output_path, cols=4):
    """创建更美观的新视角合成对比网格：一行 GT，一行 Pred"""
    render_dir = Path(exp["path"]) / "test" / "iter_30000" / "render_test"
    if not render_dir.exists():
        return False

    gt_files = sorted(render_dir.glob("*_gt.png"))
    n_total = len(gt_files)
    n_show = min(cols * 2, n_total)
    indices = np.linspace(0, n_total - 1, n_show, dtype=int)

    fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 8))

    for i, idx in enumerate(indices):
        gt_img = np.array(Image.open(gt_files[idx]).convert("L"))
        pred_file = gt_files[idx].parent / f"{idx:05d}_pred.png"
        pred_img = np.array(Image.open(pred_file).convert("L"))

        vmin = min(gt_img.min(), pred_img.min())
        vmax = max(gt_img.max(), pred_img.max())

        axes[0, i].imshow(gt_img, cmap="gray", vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f"GT #{idx}", fontsize=8)
        axes[0, i].axis("off")

        axes[1, i].imshow(pred_img, cmap="gray", vmin=vmin, vmax=vmax)
        axes[1, i].set_title(f"Pred #{idx}", fontsize=8)
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Ground Truth", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Prediction", fontsize=12, fontweight="bold")

    label = METHOD_LABELS.get(exp["method"], exp["method"])
    organ = ORGAN_LABELS.get(exp["organ"], exp["organ"])
    fig.suptitle(
        f"{organ} | {exp['views']} Views | {label}\n"
        f"PSNR: {exp['psnr_2d']:.2f} dB | SSIM: {exp['ssim_2d']:.4f}",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return True


def create_dashboard(completed, output_path):
    """创建进度看板"""
    methods = ["r2_gaussian", "spags", "xgaussian", "fsgs", "corgs", "dngaussian"]

    # 统计完成度和 PSNR
    counts = {m: 0 for m in methods}
    psnrs = {m: [] for m in methods}
    best_psnr = {m: 0 for m in methods}
    best_info = {m: None for m in methods}

    for exp in completed:
        m = exp["method"]
        if m in counts:
            counts[m] += 1
            psnrs[m].append(exp["psnr_2d"])
            if exp["psnr_2d"] > best_psnr[m]:
                best_psnr[m] = exp["psnr_2d"]
                best_info[m] = f"{ORGAN_LABELS.get(exp['organ'], exp['organ'])} {exp['views']}v"

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 左：进度条
    labels = [METHOD_LABELS.get(m, m) for m in methods]
    values = [counts[m] for m in methods]
    colors = [METHOD_COLORS[m] for m in methods]

    bars = axes[0].barh(labels, values, color=colors, height=0.6)
    axes[0].set_xlim(0, 15)
    axes[0].set_xlabel("Completed (out of 15)", fontsize=12)
    axes[0].set_title("Experiment Progress Dashboard", fontsize=14, fontweight="bold")
    for bar, v, m in zip(bars, values, methods):
        axes[0].text(v + 0.1, bar.get_y() + bar.get_height() / 2,
                     f"{v}/15", va="center", fontsize=11, fontweight="bold")
    axes[0].invert_yaxis()

    # 中：PSNR 对比
    valid_methods = [m for m in methods if psnrs[m]]
    if valid_methods:
        lbls = [METHOD_LABELS.get(m, m) for m in valid_methods]
        avgs = [sum(psnrs[m]) / len(psnrs[m]) for m in valid_methods]
        clrs = [METHOD_COLORS[m] for m in valid_methods]
        bars2 = axes[1].barh(lbls, avgs, color=clrs, height=0.6)
        for bar, a in zip(bars2, avgs):
            axes[1].text(a + 0.1, bar.get_y() + bar.get_height() / 2,
                         f"{a:.2f}", va="center", fontsize=10, fontweight="bold")
        axes[1].set_xlabel("Average PSNR (dB)", fontsize=12)
        axes[1].set_title("Quality Comparison", fontsize=14, fontweight="bold")
    else:
        axes[1].text(0.5, 0.5, "Waiting for data...", ha="center", va="center",
                     transform=axes[1].transAxes, fontsize=14)

    # 右：各方法最佳成绩
    axes[2].axis("off")
    y_pos = 0.95
    axes[2].text(0.5, y_pos, "Best PSNR per Method", ha="center", fontsize=14, fontweight="bold",
                 transform=axes[2].transAxes)
    y_pos -= 0.08
    for m in methods:
        if best_info[m]:
            axes[2].text(0.1, y_pos,
                         f"● {METHOD_LABELS[m]:<16} {best_psnr[m]:.2f} dB",
                         fontsize=11, transform=axes[2].transAxes,
                         color=METHOD_COLORS[m])
            axes[2].text(0.6, y_pos, f"({best_info[m]})", fontsize=9,
                         transform=axes[2].transAxes, color="gray")
        else:
            axes[2].text(0.1, y_pos,
                         f"○ {METHOD_LABELS[m]:<16} ---",
                         fontsize=11, transform=axes[2].transAxes, color="lightgray")
        y_pos -= 0.06

    # 底部文字
    n_total = 90
    n_done = len(completed)
    fig.text(0.5, 0.01,
             f"Updated: {datetime.now().strftime('%H:%M:%S')}  |  "
             f"Progress: {n_done}/{n_total}  |  "
             f"Running: {n_total - n_done} remaining",
             ha="center", fontsize=10, color="gray", fontstyle="italic")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_report():
    """主函数：生成完整报告"""
    completed = find_completed()
    n_done = len(completed)
    n_total = 90

    # 确保 test.py 渲染已完成（最多新处理 2 个）
    print(f"实验进度: {n_done}/{n_total}")
    n_rendered = ensure_test_renders(completed, max_new=2)
    if n_rendered > 0:
        print(f"新渲染 {n_rendered} 个实验")
        # 重新读取以获取 has_test_render 状态
        completed = find_completed()

    # 1. 仪表盘
    dash_path = RESULTS / "dashboard.png"
    create_dashboard(completed, dash_path)
    print(f"仪表盘: {dash_path}")

    # 2. 最新实验的新视角合成对比
    # 找到最新完成的且有渲染的实验
    latest_with_render = None
    for exp in reversed(completed):
        if exp["has_test_render"]:
            latest_with_render = exp
            break

    latest_img_path = None
    latest_info = None
    if latest_with_render:
        novel_path = RESULTS / f"novel_view_{latest_with_render['dirname']}.png"
        ok = create_novel_view_grid(latest_with_render, novel_path)
        if ok:
            latest_img_path = str(novel_path)
            latest_info = latest_with_render
            print(f"最新对比图: {novel_path}")

        # 也创建详细蒙太奇
        montage_path = RESULTS / f"montage_{latest_with_render['dirname']}.png"
        create_comparison_montage(latest_with_render, montage_path)

    # 3. 生成按视角的指标摘要
    summary_lines = []
    for v in [3, 6, 9]:
        v_exp = [e for e in completed if e["views"] == v]
        if not v_exp:
            continue
        summary_lines.append(f"\n## {v} Views")
        for m in ["r2_gaussian", "spags", "xgaussian", "fsgs", "corgs", "dngaussian"]:
            m_exp = [e for e in v_exp if e["method"] == m]
            if m_exp:
                avg_psnr = sum(e["psnr_2d"] for e in m_exp) / len(m_exp)
                avg_ssim = sum(e["ssim_2d"] for e in m_exp) / len(m_exp)
                summary_lines.append(
                    f"  {METHOD_LABELS[m]:<18}: "
                    f"PSNR={avg_psnr:.2f}  SSIM={avg_ssim:.4f}  (n={len(m_exp)})"
                )

    summary_text = "\n".join(summary_lines)

    # 输出 JSON 供 cron agent 使用
    info = {
        "n_completed": n_done,
        "n_total": n_total,
        "n_rendered": n_rendered,
        "dash_img": str(dash_path),
        "latest_img": latest_img_path,
        "latest_info": {
            "organ": latest_info["organ"] if latest_info else None,
            "views": latest_info["views"] if latest_info else None,
            "method": latest_info["method"] if latest_info else None,
            "psnr": latest_info["psnr_2d"] if latest_info else None,
            "ssim": latest_info["ssim_2d"] if latest_info else None,
        } if latest_info else None,
        "summary": summary_text,
    }
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    generate_report()
