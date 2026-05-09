#!/usr/bin/env python3
"""PG2026 消融实验柱状图：8 configs × 5 organs"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT)
FIGURE_DIR = "figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

# ── 数据 ──
data = {
    "R²-Gaussian (baseline)": {"Chest": 26.08, "Head": 26.53, "Abdomen": 29.20, "Foot": 28.57, "Pancreas": 28.60},
    "+SPS":                   {"Chest": 26.65, "Head": 26.39, "Abdomen": 29.45, "Foot": 28.48, "Pancreas": 29.07},
    "+GAP":                   {"Chest": 25.95, "Head": 26.72, "Abdomen": 29.23, "Foot": 28.60, "Pancreas": 28.83},
    "+ADM":                   {"Chest": 26.12, "Head": 26.58, "Abdomen": 29.33, "Foot": 28.70, "Pancreas": 28.84},
    "SPS+GAP":                {"Chest": 26.57, "Head": 26.54, "Abdomen": 29.47, "Foot": 28.18, "Pancreas": 29.20},
    "SPS+ADM":                {"Chest": 26.71, "Head": 26.59, "Abdomen": 29.62, "Foot": 28.35, "Pancreas": 29.20},
    "GAP+ADM":                {"Chest": 25.99, "Head": 26.78, "Abdomen": 29.34, "Foot": 28.63, "Pancreas": 29.14},
    "SPAGS (full)":           {"Chest": 26.63, "Head": 26.55, "Abdomen": 29.55, "Foot": 28.34, "Pancreas": 29.24},
}

config_names = list(data.keys())
organs = ["Chest", "Head", "Abdomen", "Foot", "Pancreas"]

# 每个 config 含哪些组件 (用于着色)
components = {
    "R²-Gaussian (baseline)": (0, 0, 0),  # SPS, GAP, ADM
    "+SPS":         (1, 0, 0),
    "+GAP":         (0, 1, 0),
    "+ADM":         (0, 0, 1),
    "SPS+GAP":      (1, 1, 0),
    "SPS+ADM":      (1, 0, 1),
    "GAP+ADM":      (0, 1, 1),
    "SPAGS (full)": (1, 1, 1),
}

# ── 颜色方案 ──
# SPS_init=红/橙, GAP=蓝/青, ADM=绿
base_color = np.array([0.55, 0.55, 0.55])  # baseline 灰色
sps_color = np.array([0.90, 0.30, 0.20])    # 红橙
gap_color = np.array([0.20, 0.50, 0.85])    # 蓝
adm_color = np.array([0.20, 0.70, 0.30])    # 绿

def blend_color(has_sps, has_gap, has_adm):
    """混合颜色表示组件组合"""
    if not (has_sps or has_gap or has_adm):
        return base_color
    colors = []
    weights = []
    if has_sps:
        colors.append(sps_color)
        weights.append(1.0)
    if has_gap:
        colors.append(gap_color)
        weights.append(0.7)
    if has_adm:
        colors.append(adm_color)
        weights.append(0.8)
    total = sum(weights)
    blended = sum(c * w for c, w in zip(colors, weights)) / total
    # 增加亮度
    blended = np.minimum(blended + 0.08, 1.0)
    return blended

bar_colors = {}
for name, (s, g, a) in components.items():
    bar_colors[name] = blend_color(s, g, a)

# ── 创建画布: 2行 (5子图 + 1平均) ──
fig = plt.figure(figsize=(18, 8), facecolor='white')

# 主网格: 2 行, 第一行 5 列(器官), 第二行 1 列(平均, span 5)
gs = fig.add_gridspec(2, 10, hspace=0.35, wspace=0.28,
                       left=0.05, right=0.98, top=0.92, bottom=0.12)

axes = {}
for i, organ in enumerate(organs):
    ax = fig.add_subplot(gs[0, i*2:(i+1)*2])
    axes[organ] = ax

ax_avg = fig.add_subplot(gs[1, 1:9])

# ── 绘制各器官 ──
x = np.arange(len(config_names))
bar_width = 0.60

for organ in organs:
    ax = axes[organ]
    values = [data[name][organ] for name in config_names]
    colors = [bar_colors[name] for name in config_names]
    
    bars = ax.bar(x, values, bar_width, color=colors, edgecolor='white', linewidth=0.5)
    
    # 在每个 bar 上标注 PSNR 值
    for bar_obj, val in zip(bars, values):
        ax.text(bar_obj.get_x() + bar_obj.get_width()/2, bar_obj.get_height() + 0.03,
                f'{val:.2f}', ha='center', va='bottom', fontsize=5.5, fontweight='bold')
    
    ax.set_title(organ, fontsize=13, fontweight='bold', pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels(config_names, fontsize=6.5, rotation=40, ha='right')
    ax.set_ylabel('PSNR (dB)', fontsize=9)
    ax.set_ylim(min(values) - 0.3, max(values) + 0.6)
    ax.tick_params(axis='y', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

# ── 绘制平均 ──
values_avg = [np.mean([data[name][o] for o in organs]) for name in config_names]
colors_avg = [bar_colors[name] for name in config_names]

bars_avg = ax_avg.bar(x, values_avg, bar_width, color=colors_avg, edgecolor='white', linewidth=0.5)

for bar_obj, val in zip(bars_avg, values_avg):
    ax_avg.text(bar_obj.get_x() + bar_obj.get_width()/2, bar_obj.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

ax_avg.set_title('Average PSNR (5 organs)', fontsize=13, fontweight='bold', pad=6)
ax_avg.set_xticks(x)
ax_avg.set_xticklabels(config_names, fontsize=7, rotation=40, ha='right')
ax_avg.set_ylabel('PSNR (dB)', fontsize=10)
ax_avg.set_ylim(min(values_avg) - 0.1, max(values_avg) + 0.35)
ax_avg.tick_params(axis='y', labelsize=9)
ax_avg.spines['top'].set_visible(False)
ax_avg.spines['right'].set_visible(False)
ax_avg.axhline(y=values_avg[0], color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
ax_avg.grid(axis='y', alpha=0.3, linestyle='--')

# ── 图例 (组件指示) ──
legend_patches = [
    mpatches.Patch(color=sps_color, label='SPS init'),
    mpatches.Patch(color=gap_color, label='GAP'),
    mpatches.Patch(color=adm_color, label='ADM'),
    mpatches.Patch(color=base_color, label='(none)'),
]
fig.legend(handles=legend_patches, loc='upper center', ncol=4,
           fontsize=10, frameon=True, fancybox=False,
           borderaxespad=0.2, handlelength=1.0, columnspacing=2.0)

# ── 标注 baseline 均值线 ──
baseline_avg = values_avg[0]
ax_avg.text(0, baseline_avg + 0.02, f'baseline {baseline_avg:.2f}',
            color='gray', fontsize=7, ha='center', va='bottom', alpha=0.7)

plt.savefig(f'{FIGURE_DIR}/ablation_bars.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"✅ 已保存: {FIGURE_DIR}/ablation_bars.png")

# ============================================================
# 第二张图: 效应分解图 (主效应 + 交互) — 类似 Pareto 图
# ============================================================
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor='white')

effects = {
    'SPS init': 0.0755,
    'ADM': 0.0480,
    'SPS×GAP': -0.0225,
    'GAP': 0.0105,
    'SPS×ADM': -0.0090,
    'GAP×ADM': -0.0030,
    'SPS×GAP×ADM': -0.0010,
}
contribs = {
    'SPS init': 65.43,
    'ADM': 26.45,
    'SPS×GAP': 5.81,
    'GAP': 1.27,
    'SPS×ADM': 0.93,
    'GAP×ADM': 0.10,
    'SPS×GAP×ADM': 0.01,
}

# 按效应绝对值排序
sorted_items = sorted(effects.items(), key=lambda kv: abs(kv[1]), reverse=True)
names_sorted = [k for k, v in sorted_items]
vals_sorted = [v for k, v in sorted_items]
contrib_sorted = [contribs[k] for k in names_sorted]

# 颜色: 正效应/负效应
colors_eff = []
for v in vals_sorted:
    if v > 0:
        colors_eff.append('#2E86AB')  # 蓝
    else:
        colors_eff.append('#D64933')  # 红

bars1 = ax1.barh(range(len(names_sorted)), vals_sorted, color=colors_eff,
                 edgecolor='white', linewidth=0.5, height=0.6)
ax1.set_yticks(range(len(names_sorted)))
ax1.set_yticklabels(names_sorted, fontsize=11)
ax1.set_xlabel('Effect on PSNR (dB)', fontsize=12)
ax1.axvline(x=0, color='gray', linewidth=0.8)
ax1.set_title('Main & Interaction Effects', fontsize=14, fontweight='bold')
ax1.invert_yaxis()
ax1.tick_params(labelsize=10)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

for bar_obj, val in zip(bars1, vals_sorted):
    label_x = bar_obj.get_width() + 0.002 if val >= 0 else bar_obj.get_width() - 0.012
    ha = 'left' if val >= 0 else 'right'
    ax1.text(label_x, bar_obj.get_y() + bar_obj.get_height()/2,
             f'{val:+.4f}', va='center', ha=ha, fontsize=9, fontweight='bold')

# 饼图或柱状: 贡献占比
explode = [0.05 if c > 10 else 0 for c in contrib_sorted]
wedges, texts, autotexts = ax2.pie(
    contrib_sorted, labels=None, autopct='%1.1f%%',
    startangle=90, explode=explode,
    colors=['#2E86AB', '#27AE60', '#F39C12', '#95A5A6', '#E74C3C', '#9B59B6', '#BDC3C7'],
    pctdistance=0.85, textprops={'fontsize': 9, 'fontweight': 'bold'}
)
# 手动加图例
legend_labels = [f'{n}  ({c:.1f}%)' for n, c in zip(names_sorted, contrib_sorted)]
ax2.legend(wedges, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5),
           fontsize=9, frameon=False)
ax2.set_title('Contribution Breakdown', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{FIGURE_DIR}/ablation_effects.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"✅ 已保存: {FIGURE_DIR}/ablation_effects.png")
