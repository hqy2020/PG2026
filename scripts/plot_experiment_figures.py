#!/usr/bin/env python3
"""PG2026 论文实验图：消融 + GAP超参扫描"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(str(PROJECT))
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════
# 数据
# ═══════════════════════════════════════════════════════
organs = ["Chest", "Head", "Abdomen", "Foot", "Pancreas"]

data = {
    "R²-Gaussian":              (26.08, 26.53, 29.20, 28.57, 28.60),
    "+SPS":                     (26.65, 26.39, 29.45, 28.48, 29.07),
    "+ADM":                     (26.12, 26.58, 29.33, 28.70, 28.84),
    "SPS+ADM":                  (26.71, 26.59, 29.62, 28.35, 29.20),
    "SPS+ADM+GAP\n(t=0.015,r=2%)": (26.81, 26.63, 29.73, 28.49, 29.43),
    "GAP(t=0.01,r=3%)":         (26.68, 26.56, 29.69, 28.24, 29.39),
    "GAP(t=0.02,r=3%)":         (26.93, 26.44, 29.65, 28.31, 29.40),
    "GAP(t=0.015,r=3%)":        (26.88, 26.58, 29.64, 28.57, 29.40),
    "GAP(t=0.015,r=5%)":        (26.78, 26.56, 29.80, 28.38, 29.37),
}

# ═══════════════════════════════════════════════════════
# FIG 1: 消融柱状图 — 5个器官 + 平均
# ═══════════════════════════════════════════════════════
ablation_configs = ["R²-Gaussian", "+SPS", "+ADM", "SPS+ADM", "SPS+ADM+GAP\n(t=0.015,r=2%)"]
ablation_colors = ['#95A5A6', '#E67E22', '#27AE60', '#2980B9', '#C0392B']

fig, axes = plt.subplots(1, 6, figsize=(16, 4.5), facecolor='white',
                         gridspec_kw={'width_ratios': [1,1,1,1,1,1.3]})
fig.subplots_adjust(wspace=0.35, left=0.04, right=0.98, bottom=0.22, top=0.88)

x = np.arange(len(ablation_configs))
bw = 0.55

for idx, organ in enumerate(organs):
    ax = axes[idx]
    vals = [data[c][idx] for c in ablation_configs]
    bars = ax.bar(x, vals, bw, color=ablation_colors, edgecolor='white', linewidth=0.5)
    
    # 数值标注
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f'{v:.2f}',
                ha='center', va='bottom', fontsize=5.6, fontweight='bold')
    
    ax.set_title(organ, fontsize=11, fontweight='bold', pad=4)
    ax.set_xticks(x)
    ax.set_xticklabels(ablation_configs, fontsize=5.2, rotation=32, ha='right')
    ax.set_ylabel('PSNR (dB)' if idx == 0 else '', fontsize=9)
    ax.set_ylim(min(vals)-0.2, max(vals)+0.5)
    ax.tick_params(axis='y', labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.25, linestyle='--')

# 平均图
ax_avg = axes[5]
avg_vals = [np.mean([data[c][i] for i in range(5)]) for c in ablation_configs]
bars = ax_avg.bar(x, avg_vals, bw, color=ablation_colors, edgecolor='white', linewidth=0.5)

# baseline 线
ax_avg.axhline(y=avg_vals[0], color='gray', ls='--', lw=0.7, alpha=0.5)

# 差值标注
for i, (b, v) in enumerate(zip(bars, avg_vals)):
    diff = v - avg_vals[0]
    label = f'{v:.2f}' if i == 0 else f'{v:.2f}\n(+{diff:.2f})'
    ax_avg.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, label,
                ha='center', va='bottom', fontsize=6, fontweight='bold')

ax_avg.set_title('Average', fontsize=11, fontweight='bold', pad=4)
ax_avg.set_xticks(x)
ax_avg.set_xticklabels(ablation_configs, fontsize=5.2, rotation=32, ha='right')
ax_avg.set_ylabel('PSNR (dB)', fontsize=9)
ax_avg.set_ylim(min(avg_vals)-0.08, max(avg_vals)+0.30)
ax_avg.tick_params(axis='y', labelsize=7)
ax_avg.spines['top'].set_visible(False)
ax_avg.spines['right'].set_visible(False)
ax_avg.grid(axis='y', alpha=0.25, linestyle='--')

fig.suptitle('Ablation Study on SPAGS Components (3 views, 5 organs)', 
             fontsize=13, fontweight='bold', y=0.96)
fig.savefig(f'{FIG_DIR}/fig_ablation.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✅ {FIG_DIR}/fig_ablation.png")

# ═══════════════════════════════════════════════════════
# FIG 2: GAP 超参热力图 (threshold × ratio)
# ═══════════════════════════════════════════════════════
thresholds = [0.010, 0.015, 0.020]
ratios = [0.02, 0.03, 0.05]

# 构建热力图数据: ratios × thresholds
heat_data = np.full((len(ratios), len(thresholds)), np.nan)
# 已有数据点
heat_map = {
    (0.010, 0.03): 28.11,
    (0.015, 0.02): 28.22,
    (0.015, 0.03): 28.21,
    (0.015, 0.05): 28.18,
    (0.020, 0.03): 28.15,
}
for (t, r), v in heat_map.items():
    ti = thresholds.index(t)
    ri = ratios.index(r)
    heat_data[ri, ti] = v

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), facecolor='white')
fig2.subplots_adjust(wspace=0.35, left=0.08, right=0.92)

# --- 子图A: 折线图 (固定ratio=3%, 扫threshold) ---
ths_fixed3 = [0.010, 0.015, 0.020]
vals_fixed3 = [28.11, 28.21, 28.15]
ax1.plot(ths_fixed3, vals_fixed3, 'o-', color='#2980B9', lw=2, ms=8, zorder=3)
ax1.axhline(y=28.09, color='gray', ls='--', lw=0.8, alpha=0.6, label='SPS+ADM (no GAP)')
for t, v in zip(ths_fixed3, vals_fixed3):
    offset = 0.02 if v != 28.21 else 0.04
    ax1.annotate(f'{v:.2f}', (t, v), textcoords='offset points', xytext=(0, 12),
                 ha='center', fontsize=8, fontweight='bold')
ax1.set_xlabel('GAP Threshold', fontsize=10)
ax1.set_ylabel('Avg PSNR (dB)', fontsize=10)
ax1.set_title('GAP Threshold Sweep (ratio=3%)', fontsize=11, fontweight='bold')
ax1.set_xticks(ths_fixed3)
ax1.set_xticklabels(['0.010', '0.015', '0.020'], fontsize=9)
ax1.set_ylim(28.00, 28.30)
ax1.legend(fontsize=8)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(axis='y', alpha=0.25, linestyle='--')

# --- 子图B: 折线图 (固定threshold=0.015, 扫ratio) ---
rats_fixed015 = [0.02, 0.03, 0.05]
vals_fixed015 = [28.22, 28.21, 28.18]
ax2.plot(rats_fixed015, vals_fixed015, 's-', color='#C0392B', lw=2, ms=8, zorder=3)
ax2.axhline(y=28.09, color='gray', ls='--', lw=0.8, alpha=0.6, label='SPS+ADM (no GAP)')
for r, v in zip(rats_fixed015, vals_fixed015):
    ax2.annotate(f'{v:.2f}', (r, v), textcoords='offset points', xytext=(0, 12),
                 ha='center', fontsize=8, fontweight='bold')
ax2.set_xlabel('GAP Max Ratio', fontsize=10)
ax2.set_ylabel('Avg PSNR (dB)', fontsize=10)
ax2.set_title('GAP Ratio Sweep (threshold=0.015)', fontsize=11, fontweight='bold')
ax2.set_xticks(rats_fixed015)
ax2.set_xticklabels(['2%', '3%', '5%'], fontsize=9)
ax2.set_ylim(28.00, 28.30)
ax2.legend(fontsize=8)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(axis='y', alpha=0.25, linestyle='--')

fig2.suptitle('GAP Hyperparameter Sensitivity (SPS+ADM baseline = 28.09 dB)',
              fontsize=12, fontweight='bold', y=1.01)
fig2.savefig(f'{FIG_DIR}/fig_gap_sweep.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✅ {FIG_DIR}/fig_gap_sweep.png")

# ═══════════════════════════════════════════════════════
# FIG 3: 方法改进堆叠图 (性能累积)
# ═══════════════════════════════════════════════════════
stages = ['R²-Gaussian', '+SPS', '+ADM', '+GAP']
psnrs_stage = [27.80, 28.01, 28.09, 28.22]
deltas = [psnrs_stage[0]] + [psnrs_stage[i]-psnrs_stage[i-1] for i in range(1, len(psnrs_stage))]

fig3, ax = plt.subplots(1, 1, figsize=(7, 4), facecolor='white')

# 堆叠条形图
colors_stage = ['#95A5A6', '#E67E22', '#27AE60', '#C0392B']
bars = ax.bar(stages, deltas, color=colors_stage, edgecolor='white', lw=0.8,
              width=0.5)

# 累计值标注
cumulative = 0
for i, (b, d) in enumerate(zip(bars, deltas)):
    cumulative += d if i > 0 else d
    ax.text(b.get_x()+b.get_width()/2, b.get_height()/2,
            f'+{d:.2f}' if i > 0 else f'{d:.2f}',
            ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
            f'cum. {cumulative:.2f} dB' if i > 0 else f'',
            ha='center', va='bottom', fontsize=8, color='#555')

ax.set_ylabel('PSNR Contribution (dB)', fontsize=11)
ax.set_title('Component-wise Performance Contribution (3 views avg)', 
             fontsize=12, fontweight='bold')
ax.set_ylim(0, max(deltas)*1.6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=10)
ax.grid(axis='y', alpha=0.25, linestyle='--')

fig3.savefig(f'{FIG_DIR}/fig_contribution.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✅ {FIG_DIR}/fig_contribution.png")
