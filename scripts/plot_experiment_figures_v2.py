#!/usr/bin/env python3
"""PG2026 论文实验图 v2 — 含 GAP 超参 + SPS 超参完整结果"""
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

organs = ["Chest", "Head", "Abdomen", "Foot", "Pancreas"]

# ═══════════════════════════════════════════════════════
# FIG 1: 完整消融柱状图 — 8 个配置 × 5 器官 + 平均
# ═══════════════════════════════════════════════════════
ablation_data = {
    "R²-Gaussian":      (26.08, 26.53, 29.20, 28.57, 28.60),
    "+SPS":             (26.65, 26.39, 29.45, 28.48, 29.07),
    "+ADM":             (26.12, 26.58, 29.33, 28.70, 28.84),
    "SPS+ADM":          (26.71, 26.59, 29.62, 28.35, 29.20),
    "SPS+ADM+GAP":      (26.88, 26.58, 29.64, 28.57, 29.40),
}

# 排序: 按平均PSNR
configs_ordered = ["R²-Gaussian", "+SPS", "+ADM", "SPS+ADM", "SPS+ADM+GAP"]
colors_ablation = ['#95A5A6', '#E67E22', '#27AE60', '#2980B9', '#C0392B']

fig1, axes = plt.subplots(1, 6, figsize=(16, 4.5), facecolor='white',
                          gridspec_kw={'width_ratios': [1,1,1,1,1,1.3]})
fig1.subplots_adjust(wspace=0.35, left=0.04, right=0.98, bottom=0.22, top=0.88)

x = np.arange(len(configs_ordered))
bw = 0.55

for idx, organ in enumerate(organs):
    ax = axes[idx]
    vals = [ablation_data[c][idx] for c in configs_ordered]
    bars = ax.bar(x, vals, bw, color=colors_ablation, edgecolor='white', lw=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f'{v:.2f}',
                ha='center', va='bottom', fontsize=5.6, fontweight='bold')
    ax.set_title(organ, fontsize=11, fontweight='bold', pad=4)
    ax.set_xticks(x)
    ax.set_xticklabels(configs_ordered, fontsize=5.2, rotation=32, ha='right')
    ax.set_ylabel('PSNR (dB)' if idx == 0 else '', fontsize=9)
    ax.set_ylim(min(vals)-0.2, max(vals)+0.5)
    ax.tick_params(axis='y', labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.25, linestyle='--')

# 平均图
ax_avg = axes[5]
avg_vals = [np.mean([ablation_data[c][i] for i in range(5)]) for c in configs_ordered]
bars = ax_avg.bar(x, avg_vals, bw, color=colors_ablation, edgecolor='white', lw=0.5)
ax_avg.axhline(y=avg_vals[0], color='gray', ls='--', lw=0.7, alpha=0.5)
for i, (b, v) in enumerate(zip(bars, avg_vals)):
    diff = v - avg_vals[0]
    label = f'{v:.2f}' if i == 0 else f'{v:.2f}\n(+{diff:.2f})'
    ax_avg.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, label,
                ha='center', va='bottom', fontsize=6, fontweight='bold')
ax_avg.set_title('Average', fontsize=11, fontweight='bold', pad=4)
ax_avg.set_xticks(x)
ax_avg.set_xticklabels(configs_ordered, fontsize=5.2, rotation=32, ha='right')
ax_avg.set_ylabel('PSNR (dB)', fontsize=9)
ax_avg.set_ylim(min(avg_vals)-0.08, max(avg_vals)+0.30)
ax_avg.tick_params(axis='y', labelsize=7)
ax_avg.spines['top'].set_visible(False)
ax_avg.spines['right'].set_visible(False)
ax_avg.grid(axis='y', alpha=0.25, linestyle='--')

fig1.suptitle('Ablation Study on SPAGS Components (3 views, 5 organs average)',
              fontsize=13, fontweight='bold', y=0.96)
fig1.savefig(f'{FIG_DIR}/fig1_ablation.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✅ {FIG_DIR}/fig1_ablation.png")

# ═══════════════════════════════════════════════════════
# FIG 2: GAP 超参折线图
# ═══════════════════════════════════════════════════════
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), facecolor='white')
fig2.subplots_adjust(wspace=0.35, left=0.08, right=0.92)

# 左: 固定 ratio=3%, 扫 threshold
ths = [0.010, 0.015, 0.020]
vals_th = [28.11, 28.21, 28.15]
ax1.plot(ths, vals_th, 'o-', color='#2980B9', lw=2, ms=8, zorder=3)
ax1.axhline(y=28.09, color='gray', ls='--', lw=0.8, alpha=0.6, label='SPS+ADM (no GAP)')
for t, v in zip(ths, vals_th):
    offset = 0.02 if v != 28.21 else 0.04
    ax1.annotate(f'{v:.2f}', (t, v), textcoords='offset points', xytext=(0, 12),
                 ha='center', fontsize=9, fontweight='bold')
ax1.set_xlabel('GAP Threshold', fontsize=10)
ax1.set_ylabel('Avg PSNR (dB)', fontsize=10)
ax1.set_title('(a) Threshold Sweep (ratio=3%)', fontsize=11, fontweight='bold')
ax1.set_xticks(ths)
ax1.set_xticklabels(['0.010', '0.015', '0.020'], fontsize=9)
ax1.set_ylim(28.00, 28.32)
ax1.legend(fontsize=8)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(axis='y', alpha=0.25, linestyle='--')

# 右: 固定 threshold=0.015, 扫 ratio
rats = [0.02, 0.03, 0.05]
vals_r = [28.22, 28.21, 28.18]
ax2.plot(rats, vals_r, 's-', color='#C0392B', lw=2, ms=8, zorder=3)
ax2.axhline(y=28.09, color='gray', ls='--', lw=0.8, alpha=0.6, label='SPS+ADM (no GAP)')
for r, v in zip(rats, vals_r):
    ax2.annotate(f'{v:.2f}', (r, v), textcoords='offset points', xytext=(0, 12),
                 ha='center', fontsize=9, fontweight='bold')
ax2.set_xlabel('GAP Max Ratio', fontsize=10)
ax2.set_ylabel('Avg PSNR (dB)', fontsize=10)
ax2.set_title('(b) Ratio Sweep (threshold=0.015)', fontsize=11, fontweight='bold')
ax2.set_xticks(rats)
ax2.set_xticklabels(['2%', '3%', '5%'], fontsize=9)
ax2.set_ylim(28.00, 28.32)
ax2.legend(fontsize=8)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(axis='y', alpha=0.25, linestyle='--')

fig2.suptitle('GAP Hyperparameter Sensitivity Analysis (SPS+ADM baseline = 28.09 dB)',
              fontsize=12, fontweight='bold', y=1.01)
fig2.savefig(f'{FIG_DIR}/fig2_gap_sweep.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✅ {FIG_DIR}/fig2_gap_sweep.png")

# ═══════════════════════════════════════════════════════
# FIG 3: SPS 超参柱状图（带 GAP）
# ═══════════════════════════════════════════════════════
sps_sweep_data = {
    "SPS-v1\n(orig)":       (26.88, 26.58, 29.64, 28.57, 29.40),
    "SPS-v5\n(mean_init)":  (26.95, 26.59, 29.57, 28.56, 29.33),
    "SPS-v6\n(75Kpts)":     (26.93, 26.52, 29.48, 28.53, 29.46),
    "SPS-v4\n(unif0.3)":    (26.96, 26.66, 29.38, 28.57, 29.30),
    "SPS-v2\n(unif0.4)":    (26.97, 26.42, 29.53, 28.30, 29.52),
}
sps_order = ["SPS-v1\n(orig)", "SPS-v5\n(mean_init)", "SPS-v6\n(75Kpts)", "SPS-v4\n(unif0.3)", "SPS-v2\n(unif0.4)"]
sps_colors = ['#C0392B', '#E67E22', '#F39C12', '#3498DB', '#95A5A6']

fig3, axes = plt.subplots(1, 6, figsize=(16, 4.5), facecolor='white',
                          gridspec_kw={'width_ratios': [1,1,1,1,1,1.3]})
fig3.subplots_adjust(wspace=0.35, left=0.04, right=0.98, bottom=0.24, top=0.88)

x = np.arange(len(sps_order))
for idx, organ in enumerate(organs):
    ax = axes[idx]
    vals = [sps_sweep_data[c][idx] for c in sps_order]
    bars = ax.bar(x, vals, 0.6, color=sps_colors, edgecolor='white', lw=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f'{v:.2f}',
                ha='center', va='bottom', fontsize=5.8, fontweight='bold')
    ax.set_title(organ, fontsize=11, fontweight='bold', pad=4)
    ax.set_xticks(x)
    ax.set_xticklabels(sps_order, fontsize=5.5, rotation=32, ha='right')
    ax.set_ylabel('PSNR (dB)' if idx == 0 else '', fontsize=9)
    ax.set_ylim(min(vals)-0.15, max(vals)+0.35)
    ax.tick_params(axis='y', labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.25, linestyle='--')

# 平均图
ax_avg = axes[5]
avg_vals = [np.mean([sps_sweep_data[c][i] for i in range(5)]) for c in sps_order]
bars = ax_avg.bar(x, avg_vals, 0.6, color=sps_colors, edgecolor='white', lw=0.5)
baseline_avg = avg_vals[0]
for i, (b, v) in enumerate(zip(bars, avg_vals)):
    diff = v - baseline_avg
    label = f'{v:.2f}' if i == 0 else f'{v:.2f}\n({diff:+.2f})'
    ax_avg.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, label,
                ha='center', va='bottom', fontsize=6, fontweight='bold')
ax_avg.set_title('Average', fontsize=11, fontweight='bold', pad=4)
ax_avg.set_xticks(x)
ax_avg.set_xticklabels(sps_order, fontsize=5.5, rotation=32, ha='right')
ax_avg.set_ylabel('PSNR (dB)', fontsize=9)
ax_avg.set_ylim(min(avg_vals)-0.05, max(avg_vals)+0.18)
ax_avg.tick_params(axis='y', labelsize=7)
ax_avg.spines['top'].set_visible(False)
ax_avg.spines['right'].set_visible(False)
ax_avg.grid(axis='y', alpha=0.25, linestyle='--')

fig3.suptitle('SPS Variants with GAP (all with ADM + GAP, 3 views average)',
              fontsize=13, fontweight='bold', y=0.96)
fig3.savefig(f'{FIG_DIR}/fig3_sps_sweep.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✅ {FIG_DIR}/fig3_sps_sweep.png")

# ═══════════════════════════════════════════════════════
# FIG 4: 组件贡献堆叠图
# ═══════════════════════════════════════════════════════
stages = ['R²-Gaussian', '+SPS', '+ADM', '+GAP']
psnrs_stage = [27.80, 28.01, 28.09, 28.22]
deltas = [psnrs_stage[0]] + [psnrs_stage[i]-psnrs_stage[i-1] for i in range(1, len(psnrs_stage))]
colors_stage = ['#95A5A6', '#E67E22', '#27AE60', '#C0392B']

fig4, ax = plt.subplots(1, 1, figsize=(6, 4), facecolor='white')
bars = ax.bar(stages, deltas, color=colors_stage, edgecolor='white', lw=0.8, width=0.5)
cumulative = 0
for i, (b, d) in enumerate(zip(bars, deltas)):
    cumulative += d if i > 0 else d
    txt = f'+{d:.2f}' if i > 0 else f'{d:.2f}'
    ax.text(b.get_x()+b.get_width()/2, b.get_height()/2, txt,
            ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    if i > 0:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                f'cum. {cumulative:.2f} dB', ha='center', va='bottom', fontsize=8, color='#555')
ax.set_ylabel('PSNR Contribution (dB)', fontsize=11)
ax.set_title('Component-wise Contribution (3 views, 5 organs avg)',
             fontsize=12, fontweight='bold')
ax.set_ylim(0, max(deltas)*1.6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=10)
ax.grid(axis='y', alpha=0.25, linestyle='--')
fig4.tight_layout()
fig4.savefig(f'{FIG_DIR}/fig4_contribution.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✅ {FIG_DIR}/fig4_contribution.png")

print("\n✅ 全部完成！4 张图已保存至 figures/")
