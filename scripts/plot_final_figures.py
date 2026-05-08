#!/usr/bin/env python3
"""PG2026 最终消融图：8 configs × 2/3/4 views"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(str(PROJECT))
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# 数据
data = {
    "R²-Gaussian": {"2v": 21.27, "3v": 27.80, "4v": 29.10},
    "+SPS":        {"2v": 21.44, "3v": 28.01, "4v": 29.16},
    "+ADM":        {"2v": 21.31, "3v": 27.91, "4v": 29.18},
    "+GAP":        {"2v": 21.28, "3v": 28.22, "4v": 29.20},
    "SPS+ADM":     {"2v": 21.51, "3v": 28.09, "4v": 29.17},
    "SPS+GAP":     {"2v": 21.42, "3v": 28.22, "4v": 29.09},
    "ADM+GAP":     {"2v": 21.43, "3v": 28.22, "4v": 29.36},
    "SPS+ADM+GAP": {"2v": 21.44, "3v": 28.22, "4v": 29.20},
}

# 按 3v 排序
order = ["R²-Gaussian", "+SPS", "+ADM", "SPS+ADM", "+GAP", "SPS+GAP", "ADM+GAP", "SPS+ADM+GAP"]
colors = ['#95A5A6', '#E67E22', '#27AE60', '#2980B9', '#E74C3C', '#8E44AD', '#1ABC9C', '#C0392B']

views = [2, 3, 4]
view_labels = ['2 Views', '3 Views', '4 Views']

fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor='white')
fig.subplots_adjust(wspace=0.30, left=0.06, right=0.97, bottom=0.22, top=0.90)

x = np.arange(len(order))
bw = 0.55

for idx, (v, vlabel) in enumerate(zip(views, view_labels)):
    ax = axes[idx]
    vals = [data[c][f"{v}v"] for c in order]
    
    bars = ax.bar(x, vals, bw, color=colors, edgecolor='white', lw=0.5)
    
    # 数值标注
    for b, val in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=6.5, fontweight='bold')
    
    # 差值标注（vs 第一个 bar）
    baseline_val = vals[0]
    for i, (b, val) in enumerate(zip(bars, vals)):
        if i > 0 and abs(val - baseline_val) > 0.005:
            diff = val - baseline_val
            ax.annotate(f'+{diff:.2f}', (b.get_x()+b.get_width()/2, b.get_height()),
                       textcoords='offset points', xytext=(0, 22),
                       ha='center', fontsize=5.5, color='#333', alpha=0.7)
    
    ax.set_title(vlabel, fontsize=13, fontweight='bold', pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=6, rotation=35, ha='right')
    ax.set_ylabel('PSNR (dB)' if idx == 0 else '', fontsize=10)
    ax.set_ylim(min(vals)-0.15, max(vals)+0.55)
    ax.tick_params(axis='y', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.25, linestyle='--')

# 英雄标记
for idx in range(3):
    ax = axes[idx]
    v = views[idx]
    vals = [data[c][f"{v}v"] for c in order]
    best_idx = np.argmax(vals)
    ax.patches[best_idx].set_edgecolor('#FFD700')
    ax.patches[best_idx].set_linewidth(2.5)

fig.suptitle('Ablation Study: SPS + ADM + GAP (8 configs × 2/3/4 views, 5 organs avg)',
             fontsize=14, fontweight='bold', y=0.97)
fig.savefig(f'{FIG_DIR}/fig_ablation_final.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✅ {FIG_DIR}/fig_ablation_final.png")

# ════════════════════════════════════════
# FIG 2: 组件贡献累积 (3 views)
# ════════════════════════════════════════
stages = ['R²-Gaussian', '+SPS', '+ADM', '+GAP']
stage_vals = [27.80, 28.01, 27.91, 28.22]  # 累积
contribs = [27.80, 0.21, 0.11, 0.42]  # vs baseline
# SPS+ADM = 28.09, SPS+ADM+GAP = 28.22

# 用堆叠图展示累积贡献
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), facecolor='white')
fig2.subplots_adjust(wspace=0.30)

# 左: 堆叠bar
colors2 = ['#95A5A6', '#E67E22', '#27AE60', '#C0392B']
bars = ax1.bar(['R²-Gaussian', '+SPS', '+ADM', '+GAP'],
               [27.80, 0.21, 0.11, 0.42],
               color=colors2, edgecolor='white', lw=0.8, width=0.5)
cum = 0
for i, (b, v) in enumerate(zip(bars, [27.80, 0.21, 0.11, 0.42])):
    if i == 0:
        cum = v
        ax1.text(b.get_x()+b.get_width()/2, b.get_height()/2, f'{v:.2f}',
                ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    else:
        cum += v
        # 增量标注
        ax1.text(b.get_x()+b.get_width()/2, b.get_height()/2, f'+{v:.2f}',
                ha='center', va='center', fontsize=11, fontweight='bold', color='white')
        # 累积标注
        ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f'cum. {cum:.2f}',
                ha='center', va='bottom', fontsize=7.5, color='#555')

ax1.set_ylabel('PSNR (dB)', fontsize=10)
ax1.set_title('Cumulative Contribution\n(SPS → ADM → GAP)', fontsize=11, fontweight='bold')
ax1.set_ylim(0, 1.0)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(labelsize=9)
ax1.grid(axis='y', alpha=0.25, linestyle='--')

# 右: 最终对比柱状图
right_configs = ['R²-Gaussian', 'SPS+ADM', 'SPS+ADM+GAP', 'Full SPAGS\n(with GAR)']
right_vals = [27.80, 28.09, 28.22, 28.06]
right_colors = ['#95A5A6', '#2980B9', '#C0392B', '#F39C12']

bars2 = ax2.bar(right_configs, right_vals, color=right_colors, edgecolor='white', lw=0.8, width=0.5)
for b, v in zip(bars2, right_vals):
    diff = v - 27.80
    label = f'{v:.2f}' if v == 27.80 else f'{v:.2f}\n(+{diff:.2f})'
    ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, label,
            ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax2.set_ylabel('PSNR (dB)', fontsize=10)
ax2.set_title('Final Comparison\n(3 views average)', fontsize=11, fontweight='bold')
ax2.set_ylim(27.50, 28.50)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.tick_params(labelsize=8.5)
ax2.grid(axis='y', alpha=0.25, linestyle='--')

fig2.suptitle('SPAGS: Three-stage Framework (SPS → ADM → GAP)', fontsize=13, fontweight='bold', y=1.02)
fig2.tight_layout()
fig2.savefig(f'{FIG_DIR}/fig_three_stage.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✅ {FIG_DIR}/fig_three_stage.png")
