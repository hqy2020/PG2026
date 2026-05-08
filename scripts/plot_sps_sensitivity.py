#!/usr/bin/env python3
"""SPS 超参数敏感度分析图"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(str(PROJECT))
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════
# SPS 超参数据（所有配置均带 ADM + GAP 最优）
# ═══════════════════════════════════════════════════════
# 1. uniform_ratio sweep (gamma 同步调整: gamma = 1.2 - unif_ratio)
unif_data = {
    "ratio": [0.2, 0.3, 0.4],
    "gamma": [1.0, 0.8, 0.7],
    "avg":   [28.22, 28.17, 28.15],
    "chest": [26.88, 26.96, 26.97],
    "head":  [26.58, 26.66, 26.42],
    "abdo":  [29.64, 29.38, 29.53],
    "foot":  [28.57, 28.57, 28.30],
    "panc":  [29.40, 29.30, 29.52],
}

# 2. density_init_mode comparison
init_data = {
    "mode": ["raw", "match_valid_mean"],
    "avg":  [28.22, 28.20],
}

# 3. n_points comparison
npts_data = {
    "n":   [50000, 75000],
    "avg": [28.22, 28.18],
}

# ═══════════════════════════════════════════════════════
# 画图: 1 张大图，3 个子图
# ═══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), facecolor='white')
fig.subplots_adjust(wspace=0.40, left=0.06, right=0.97, bottom=0.18, top=0.85)

baseline_psnr = 28.09  # SPS+ADM without GAP

# ─── 子图A: uniform_ratio sweep ───
ax = axes[0]
x = unif_data["ratio"]
y = unif_data["avg"]

ax.plot(x, y, 'o-', color='#2980B9', lw=2.5, ms=9, zorder=3, label='SPS+ADM+GAP')
ax.axhline(y=baseline_psnr, color='gray', ls='--', lw=0.8, alpha=0.6, label='SPS+ADM (no GAP)')

# 标注具体数值
for xi, yi, gi in zip(x, y, unif_data["gamma"]):
    ax.annotate(f'{yi:.2f}\n(γ={gi:.1f})', (xi, yi),
                textcoords='offset points', xytext=(0, 18),
                ha='center', fontsize=8, fontweight='bold', color='#2980B9')

ax.set_xlabel('SPS Uniform Ratio (with adaptive γ)', fontsize=10)
ax.set_ylabel('Average PSNR (dB)', fontsize=10)
ax.set_title('(a) Uniform Ratio Sweep', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['0.20', '0.30', '0.40'], fontsize=9)
ax.set_ylim(28.00, 28.35)
ax.legend(fontsize=8, loc='lower left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.25, linestyle='--')

# 添加各器官的散点（透明）
for organ_key, color, marker, label in [("chest", '#E74C3C', 'v', 'Chest'), ("head", '#F39C12', 's', 'Head'),
                                         ("abdo", '#2ECC71', '^', 'Abdomen'), ("foot", '#9B59B6', 'D', 'Foot'),
                                         ("panc", '#1ABC9C', 'p', 'Pancreas')]:
    vals = [unif_data[organ_key][i] for i in range(len(x))]
    ax.scatter(x, vals, color=color, marker=marker, s=30, alpha=0.3, zorder=2)

# ─── 子图B: init_mode ───
ax = axes[1]
modes = init_data["mode"]
vals = init_data["avg"]
x_pos = [0, 1]

bars = ax.bar(x_pos, vals, 0.4, color=['#C0392B', '#E67E22'], edgecolor='white', lw=0.8)
ax.axhline(y=baseline_psnr, color='gray', ls='--', lw=0.8, alpha=0.6, label='SPS+ADM (no GAP)')

for b, v in zip(bars, vals):
    diff = v - baseline_psnr
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
            f'{v:.2f} dB\n(+{diff:.2f})', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(x_pos)
ax.set_xticklabels(['raw', 'match_valid_mean'], fontsize=9)
ax.set_ylabel('Average PSNR (dB)', fontsize=10)
ax.set_title('(b) Density Init Mode', fontsize=12, fontweight='bold')
ax.set_ylim(28.00, 28.35)
ax.legend(fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.25, linestyle='--')

# ─── 子图C: n_points ───
ax = axes[2]
npts = npts_data["n"]
vals_n = npts_data["avg"]

ax.plot(npts, vals_n, 's-', color='#8E44AD', lw=2.5, ms=9, zorder=3, label='SPS+ADM+GAP')
ax.axhline(y=baseline_psnr, color='gray', ls='--', lw=0.8, alpha=0.6, label='SPS+ADM (no GAP)')

for ni, vi in zip(npts, vals_n):
    ax.annotate(f'{vi:.2f}', (ni, vi), textcoords='offset points', xytext=(0, 15),
                ha='center', fontsize=9, fontweight='bold', color='#8E44AD')

ax.set_xlabel('Number of Initial Gaussians', fontsize=10)
ax.set_ylabel('Average PSNR (dB)', fontsize=10)
ax.set_title('(c) Initial Point Count', fontsize=12, fontweight='bold')
ax.set_xticks(npts)
ax.set_xticklabels(['50,000', '75,000'], fontsize=9)
ax.set_ylim(28.00, 28.35)
ax.legend(fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.25, linestyle='--')

fig.suptitle('SPS Hyperparameter Sensitivity (all with SPS+ADM+GAP, 3 views avg)',
             fontsize=13, fontweight='bold', y=0.97)

fig.savefig(f'{FIG_DIR}/fig_sps_sensitivity.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✅ {FIG_DIR}/fig_sps_sensitivity.png")
