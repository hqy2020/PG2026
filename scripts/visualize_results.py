#!/usr/bin/env python3
"""
PG2026 完整实验报告 + 新视角合成可视化 + GitHub推送
"""
import os, sys, json, yaml, subprocess, glob
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
RESULTS.mkdir(parents=True, exist_ok=True)

METHOD_LABELS = {
    "r2_gaussian": "R²-Gaussian", "spags": "SPAGS",
    "xgaussian": "X-Gaussian", "fsgs": "FSGS",
    "corgs": "CoR-GS", "dngaussian": "DN-Gaussian",
}
ORGAN_LABELS = {"chest":"Chest","head":"Head","abdomen":"Abdomen","foot":"Foot","pancreas":"Pancreas"}
METHOD_COLORS = {"r2_gaussian":"#4CAF50","spags":"#2196F3","xgaussian":"#FF9800","fsgs":"#E91E63","corgs":"#9C27B0","dngaussian":"#607D8B"}
EVAL_PATTERNS = {
    "r2_gaussian": "eval2d_render_test.yml",
    "spags": "eval2d_render_test.yml",
    "xgaussian": "eval2d_xgaussian.yml",
    "fsgs": "eval2d_fsgs.yml",
    "corgs": "eval2d_corgs.yml",
    "dngaussian": "eval2d_dngaussian.yml",
}

def find_completed():
    completed = []
    for d in sorted(OUTPUT.glob("2026_04_30_*/")):
        dirname = d.name
        rest = dirname[len("2026_04_30_"):]
        try:
            organ, rest2 = rest.split("_", 1)
            views_str, method = rest2.split("views_", 1)
            views = int(views_str)
        except:
            continue
        # 找该方法的 eval 文件
        pattern = EVAL_PATTERNS.get(method, "eval2d_*.yml")
        eval_files = list(d.glob(f"eval/iter_030000/{pattern}"))
        if not eval_files:
            # fallback
            eval_files = list(d.glob("eval/iter_030000/eval2d_*.yml"))
        if not eval_files:
            continue
        ef = eval_files[0]
        with open(ef) as fh:
            data = yaml.safe_load(fh)
        completed.append({
            "dirname": dirname, "organ": organ, "views": views, "method": method,
            "psnr_2d": data.get("psnr_2d", 0), "ssim_2d": data.get("ssim_2d", 0),
            "has_test_render": (d / "test" / "iter_30000" / "render_test").exists(),
        })
    return completed


def create_dashboard(completed, output_path):
    methods = ["r2_gaussian","spags","xgaussian","fsgs","corgs","dngaussian"]
    counts = {m:0 for m in methods}
    psnrs = {m:[] for m in methods}
    best = {m:(0,"") for m in methods}
    for e in completed:
        m=e["method"]; counts[m]=counts.get(m,0)+1; psnrs.setdefault(m,[]).append(e["psnr_2d"])
        if e["psnr_2d"] > best[m][0]:
            best[m]=(e["psnr_2d"], f"{ORGAN_LABELS.get(e['organ'],e['organ'])} {e['views']}v")

    fig, axes = plt.subplots(1,3,figsize=(20,6))
    # 左：进度条
    labels=[METHOD_LABELS.get(m,m) for m in methods]
    vals=[counts[m] for m in methods]
    clrs=[METHOD_COLORS[m] for m in methods]
    bars=axes[0].barh(labels, vals, color=clrs, height=0.6)
    axes[0].set_xlim(0,15); axes[0].set_xlabel("Completed (out of 15)", fontsize=12)
    axes[0].set_title("Experiment Progress", fontsize=14, fontweight="bold")
    for bar,v,m in zip(bars,vals,methods):
        axes[0].text(v+0.1, bar.get_y()+bar.get_height()/2, f"{v}/15", va="center", fontsize=11, fontweight="bold")
    axes[0].invert_yaxis()

    # 中：PSNR
    valid=[m for m in methods if psnrs.get(m)]
    if valid:
        lbls=[METHOD_LABELS.get(m,m) for m in valid]
        avgs=[sum(psnrs[m])/len(psnrs[m]) for m in valid]
        clrs2=[METHOD_COLORS[m] for m in valid]
        bars2=axes[1].barh(lbls, avgs, color=clrs2, height=0.6)
        for bar,a in zip(bars2,avgs):
            axes[1].text(a+0.1, bar.get_y()+bar.get_height()/2, f"{a:.2f}", va="center", fontsize=10, fontweight="bold")
        axes[1].set_xlabel("Average PSNR (dB)", fontsize=12)
        axes[1].set_title("Quality Comparison", fontsize=14, fontweight="bold")
    else:
        axes[1].text(0.5,0.5,"Waiting...", ha="center", va="center", transform=axes[1].transAxes, fontsize=14)

    # 右：最佳
    axes[2].axis("off")
    y=0.95
    axes[2].text(0.5,y,"Best PSNR per Method", ha="center", fontsize=14, fontweight="bold", transform=axes[2].transAxes)
    y-=0.08
    for m in methods:
        if best[m][0]>0:
            axes[2].text(0.1, y, f"● {METHOD_LABELS[m]:<16} {best[m][0]:.2f} dB", fontsize=11,
                        transform=axes[2].transAxes, color=METHOD_COLORS[m])
            axes[2].text(0.6, y, f"({best[m][1]})", fontsize=9, transform=axes[2].transAxes, color="gray")
        else:
            axes[2].text(0.1, y, f"○ {METHOD_LABELS[m]:<16} ---", fontsize=11,
                        transform=axes[2].transAxes, color="lightgray")
        y-=0.06

    n_done=len(completed)
    fig.text(0.5, 0.01, f"Updated: {datetime.now().strftime('%H:%M:%S')}  |  Progress: {n_done}/90  |  {90-n_done} remaining",
             ha="center", fontsize=10, color="gray", fontstyle="italic")
    plt.tight_layout(rect=[0,0.03,1,0.97])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_novel_view_grid(exp, output_path, cols=6):
    render_dir = Path(exp["path"]) / "test" / "iter_30000" / "render_test"
    if not render_dir.exists():
        return False
    gt_files = sorted(render_dir.glob("*_gt.png"))
    n_total = len(gt_files)
    n_show = min(cols*2, n_total)
    indices = np.linspace(0, n_total-1, n_show, dtype=int)

    fig, axes = plt.subplots(2, n_show, figsize=(4*n_show, 8))
    for i, idx in enumerate(indices):
        gt_img = np.array(Image.open(gt_files[idx]).convert("L"))
        pred_file = gt_files[idx].parent / f"{idx:05d}_pred.png"
        pred_img = np.array(Image.open(pred_file).convert("L"))
        vmin=min(gt_img.min(),pred_img.min()); vmax=max(gt_img.max(),pred_img.max())
        axes[0,i].imshow(gt_img, cmap="gray", vmin=vmin, vmax=vmax)
        axes[0,i].set_title(f"GT #{idx}", fontsize=8); axes[0,i].axis("off")
        axes[1,i].imshow(pred_img, cmap="gray", vmin=vmin, vmax=vmax)
        axes[1,i].set_title(f"Pred #{idx}", fontsize=8); axes[1,i].axis("off")
    axes[0,0].set_ylabel("Ground Truth", fontsize=12, fontweight="bold")
    axes[1,0].set_ylabel("Prediction", fontsize=12, fontweight="bold")
    label=METHOD_LABELS.get(exp["method"],exp["method"]);
    organ=ORGAN_LABELS.get(exp["organ"],exp["organ"])
    fig.suptitle(f"{organ} | {exp['views']} Views | {label}\nPSNR: {exp['psnr_2d']:.2f} dB | SSIM: {exp['ssim_2d']:.4f}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return True


def get_best_by_method(completed):
    """找到各方法 PSNR 最高的实验"""
    best = {}
    for e in completed:
        m = e["method"]
        if m not in best or e["psnr_2d"] > best[m]["psnr_2d"]:
            best[m] = e
    return best


def main():
    completed = find_completed()
    n_done = len(completed)
    print(f"实验进度: {n_done}/90")

    # 1. 仪表盘
    dash_path = RESULTS / "dashboard.png"
    create_dashboard(completed, dash_path)
    print(f"仪表盘: {dash_path}")

    # 2. 各方法最佳结果的新视角合成图（只对 r2_gaussian/spags 可用）
    best = get_best_by_method(completed)
    latest_img_path = None
    latest_info = None
    for m in ["spags", "r2_gaussian"]:
        if m in best and best[m]["has_test_render"]:
            exp = best[m]
            exp["path"] = str(OUTPUT / exp["dirname"])
            novel_path = RESULTS / f"novel_view_{exp['dirname']}.png"
            ok = create_novel_view_grid(exp, novel_path)
            if ok:
                latest_img_path = str(novel_path)
                latest_info = exp
                break

    # 未渲染的实验用 test.py 处理最新的一个
    if latest_img_path is None:
        for e in reversed(completed):
            if e["method"] in ["r2_gaussian", "spags"]:
                exp_dir = OUTPUT / e["dirname"]
                cmd = f"source ~/anaconda3/etc/profile.d/conda.sh && conda activate r2_gaussian_new && CUDA_VISIBLE_DEVICES=0 python test.py -m {exp_dir} --iteration 30000 --skip_recon 2>/dev/null"
                subprocess.run(cmd, shell=True, timeout=120, capture_output=True)
                e["has_test_render"] = True
                e["path"] = str(exp_dir)
                novel_path = RESULTS / f"novel_view_{e['dirname']}.png"
                ok = create_novel_view_grid(e, novel_path)
                if ok:
                    latest_img_path = str(novel_path)
                    latest_info = e
                    break

    # 3. 按视角的摘要
    summary_lines = []
    for v in [3,6,9]:
        v_exp = [e for e in completed if e["views"]==v]
        if not v_exp: continue
        summary_lines.append(f"**{v} Views**")
        for m in ["r2_gaussian","spags","xgaussian","fsgs","corgs","dngaussian"]:
            m_exp=[e for e in v_exp if e["method"]==m]
            if m_exp:
                avg_p=sum(e["psnr_2d"] for e in m_exp)/len(m_exp)
                avg_s=sum(e["ssim_2d"] for e in m_exp)/len(m_exp)
                summary_lines.append(f"  {METHOD_LABELS[m]:<18}: PSNR={avg_p:.2f}  SSIM={avg_s:.4f}  (n={len(m_exp)})")
            else:
                summary_lines.append(f"  {METHOD_LABELS[m]:<18}: --- (0/5)")

    info = {
        "n_completed": n_done, "n_total": 90,
        "dash_img": str(dash_path),
        "latest_img": latest_img_path,
        "latest_info": latest_info,
        "summary": "\n".join(summary_lines),
    }
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
