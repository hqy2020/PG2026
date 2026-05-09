import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np, os

FIGS_DIR = os.path.dirname(os.path.abspath(__file__))

def setup_fig(w=12, h=5):
    fig, ax = plt.subplots(1, 1, figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis('off')
    return fig, ax

def round_box(ax, x, y, w, h, text, color='#dae8fc', edge='#6c8ebf', fs=10, fw='normal', ha='center', va='center'):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                          facecolor=color, edgecolor=edge, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha=ha, va=va, fontsize=fs, fontweight=fw, zorder=5)

def arrow(ax, x1, y1, x2, y2, color='#333333', lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))

def big_box(ax, x, y, w, h, label='', edge='#999', ls='--'):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                          facecolor='none', edgecolor=edge, linewidth=1.0, linestyle=ls)
    ax.add_patch(box)
    if label:
        ax.text(x + 0.3, y + h - 0.3, label, ha='left', va='top', fontsize=10, fontweight='bold', color=edge)

# FIGURE 1
def draw_fig1():
    fig, ax = setup_fig(14, 5.5)
    ax.text(7, 5.2, 'SPAGS: Spatial-aware Progressive Adaptive Gaussian Splatting',
            ha='center', va='center', fontsize=16, fontweight='bold')
    round_box(ax, 0.5, 3.8, 2.0, 1.0, 'Sparse CT\nProjections\n(2/3/4 views)', '#f5f5f5', '#999')
    big_box(ax, 3.0, 3.3, 10.5, 2.0, 'SPAGS Pipeline', '#666')
    y_s = 4.5
    round_box(ax, 3.3, y_s, 3.0, 0.4, 'Stage 1: SPS (Spatial Prior Seeding)', '#dae8fc', '#6c8ebf', 9)
    round_box(ax, 6.5, y_s, 3.0, 0.4, 'Stage 2: GAP (Geometry-aware Pruning)', '#d5e8d4', '#82b366', 9)
    round_box(ax, 9.7, y_s, 3.0, 0.4, 'Stage 3: ADM (Adaptive Density Modulation)', '#fff2cc', '#d6b656', 9)
    round_box(ax, 3.3, 3.5, 3.0, 0.9, 'FDK → Density-weighted\nsampling → 50K Gaussians\n\u03b3=1.0, \u03b1=0.2, raw init',
              '#dae8fc', '#6c8ebf', 8)
    round_box(ax, 6.5, 3.5, 3.0, 0.9, 'KNN proximity (K=5)\n\u03c4=0.015 pruning\n\u03b2=2%/iter, \u03b4=0.0002',
              '#d5e8d4', '#82b366', 8)
    round_box(ax, 9.7, 3.5, 3.0, 0.9, 'K-Planes (64\u00d764\u00d732)\nMLP decoder (3\u00d7128)\nr=0.3, warmup=15K',
              '#fff2cc', '#d6b656', 8)
    arrow(ax, 6.3, 4.7, 6.5, 4.7); arrow(ax, 9.5, 4.7, 9.7, 4.7); arrow(ax, 2.5, 4.3, 3.3, 4.3)
    round_box(ax, 11.8, 4.8, 1.3, 0.6, 'High-quality\nNovel Views', '#f8cecc', '#b85450', 9, 'bold')
    arrow(ax, 11.0, 4.7, 11.8, 4.7)
    ax.annotate('', xy=(10.5, 3.0), xytext=(4.0, 3.0),
                arrowprops=dict(arrowstyle='->', color='#888', lw=2))
    ax.text(7, 2.8, 'Training Iterations (0 \u2192 30,000)', ha='center', va='center', fontsize=9, color='#888')
    y_m = 1.5
    round_box(ax, 1.0, y_m, 2.0, 0.7, 'R\u00b2-Gaussian\n27.80 dB', '#f0f0f0', '#999', 10)
    arrow(ax, 3.0, 1.85, 3.8, 1.85)
    ax.text(3.4, 1.65, '+0.39', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#b85450')
    round_box(ax, 3.8, y_m, 2.0, 0.7, 'SPAGS (Ours)\n28.22 dB', '#f8cecc', '#b85450', 10, 'bold')
    for i, c in enumerate([
        '\u2713 Anatomically-aware initialization via FDK priors',
        '\u2713 Redundancy removal through proximity-guided pruning',
        '\u2713 Spatial density modulation with K-planes encoding',
    ]):
        ax.text(6.5, 2.0 - i*0.3, c, ha='left', va='center', fontsize=9, color='#333')
    ax.text(0.5, 0.3, 'Figure 1: Overview of the SPAGS framework.',
            ha='left', va='center', fontsize=8, color='#666', style='italic')
    fig.tight_layout()
    fig.savefig(f'{FIGS_DIR}/fig_intro_overview.pdf', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("OK fig1")

# FIGURE 2
def draw_fig2():
    fig, ax = setup_fig(16, 7.5)
    ax.text(8, 7.3, 'SPAGS Three-Stage Pipeline', ha='center', fontsize=14, fontweight='bold')
    big_box(ax, 0.3, 0.2, 15.4, 6.8, 'SPAGS Training Framework', '#666')
    round_box(ax, 0.5, 6.0, 4.5, 0.4, 'Stage 1: SPS \u2014 Spatial Prior Seeding (Before Training)', '#dae8fc', '#6c8ebf', 9)
    round_box(ax, 0.5, 5.0, 1.5, 0.7, 'CT Projections\n(2/3/4 views)', 'white', '#999', 8)
    round_box(ax, 2.3, 5.0, 1.5, 0.7, 'FDK\nReconstruction', '#dae8fc', '#6c8ebf', 8)
    round_box(ax, 4.1, 5.0, 1.8, 0.7, 'Density-weighted\nSampling (\u03b3=1.0)', '#dae8fc', '#6c8ebf', 8)
    round_box(ax, 6.2, 5.0, 1.5, 0.7, '+Uniform\n(\u03b1=0.2)', '#dae8fc', '#6c8ebf', 8)
    round_box(ax, 8.0, 5.0, 1.8, 0.7, '50K Initial\nGaussians', '#e1d5e7', '#9673a6', 8)
    arrow(ax, 2.0, 5.35, 2.3, 5.35); arrow(ax, 3.8, 5.35, 4.1, 5.35)
    arrow(ax, 5.9, 5.35, 6.2, 5.35); arrow(ax, 7.7, 5.35, 8.0, 5.35)
    ax.text(0.5, 4.5, 'p(x) \u221d V_FDK(x)^\u03b3,  X = X_uniform \u222a X_weighted', ha='left', fontsize=8, style='italic', color='#6c8ebf')
    round_box(ax, 0.5, 3.8, 4.5, 0.4, 'Stage 2: GAP \u2014 Geometry-aware Pruning [Iters 2K-20K, every 500 iter]', '#d5e8d4', '#82b366', 9)
    round_box(ax, 0.5, 2.8, 1.3, 0.7, 'Current\nGaussians', 'white', '#999', 8)
    round_box(ax, 2.0, 2.8, 1.8, 0.7, 'KNN Proximity\nK=5, s(p)=mean(d)', '#d5e8d4', '#82b366', 8)
    round_box(ax, 4.0, 2.8, 1.5, 0.7, 's(p) < 0.015?\n\u2192 Candidate', '#d5e8d4', '#82b366', 8)
    round_box(ax, 5.7, 2.8, 1.5, 0.7, 'Gradient\n< 0.0002?', '#d5e8d4', '#82b366', 8)
    round_box(ax, 7.4, 2.8, 1.5, 0.7, 'Prune\n(max 2%/iter)', '#f8cecc', '#b85450', 8)
    round_box(ax, 9.2, 2.8, 1.5, 0.7, 'Refined\nGaussians', '#e1d5e7', '#9673a6', 8)
    arrow(ax, 1.8, 3.15, 2.0, 3.15); arrow(ax, 3.8, 3.15, 4.0, 3.15)
    arrow(ax, 5.5, 3.15, 5.7, 3.15); arrow(ax, 7.2, 3.15, 7.4, 3.15); arrow(ax, 8.9, 3.15, 9.2, 3.15)
    ax.text(0.5, 2.25, 's(p_i) = \u00b9/\u2096 \u00d7 \u03a3||p_i-p_j||\u2082,  N_prune = min(N_cand, \u230a\u03b2\u00b7N_total\u230b)',
            ha='left', fontsize=8, style='italic', color='#82b366')
    round_box(ax, 0.5, 1.5, 4.5, 0.4, 'Stage 3: ADM \u2014 Adaptive Density Modulation [After 15K warmup]', '#fff2cc', '#d6b656', 9)
    round_box(ax, 0.5, 0.5, 1.5, 0.7, 'Position x\n(x,y,z)', 'white', '#999', 8)
    round_box(ax, 2.2, 0.5, 1.8, 0.7, 'K-Planes\nF_xy+F_xz+F_yz\n(64\u00d764\u00d732)', '#fff2cc', '#d6b656', 7)
    round_box(ax, 4.2, 0.5, 1.5, 0.7, 'MLP Decoder\n3\u00d7128 units', '#fff2cc', '#d6b656', 8)
    round_box(ax, 5.9, 0.5, 1.8, 0.7, '\u0394\u03c1 = tanh(\u00b7)\u00d7r\nc = \u03c3(\u00b7)', '#fff2cc', '#d6b656', 8)
    round_box(ax, 7.9, 0.5, 1.5, 0.7, 'Zero-mean\nModulation\n\u03c1_final=\u03c1_base+c\u00b7\u0394\u03c1', '#ffd966', '#bf9000', 7)
    round_box(ax, 9.6, 0.5, 1.5, 0.7, 'Final\nAdjusted \u03c1', '#e1d5e7', '#9673a6', 8)
    arrow(ax, 2.0, 0.85, 2.2, 0.85); arrow(ax, 4.0, 0.85, 4.2, 0.85)
    arrow(ax, 5.7, 0.85, 5.9, 0.85); arrow(ax, 7.7, 0.85, 7.9, 0.85); arrow(ax, 9.4, 0.85, 9.6, 0.85)
    ax.annotate('', xy=(1.5, 5.0), xytext=(1.5, 3.5),
                arrowprops=dict(arrowstyle='->', color='#555', lw=2, ls='dashed'))
    ax.annotate('', xy=(1.5, 2.8), xytext=(1.5, 1.9),
                arrowprops=dict(arrowstyle='->', color='#555', lw=2, ls='dashed'))
    ax.text(0.2, 3.0, 'Training Iterations \u2192', ha='left', fontsize=8, color='#555', rotation=90, va='center')
    fig.tight_layout()
    fig.savefig(f'{FIGS_DIR}/fig_pipeline.pdf', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("OK fig2")

# FIGURE 3
def draw_fig3():
    fig, ax = setup_fig(12, 4.5)
    ax.text(6, 4.3, 'SPS: Spatial Prior Seeding', ha='center', fontsize=14, fontweight='bold')
    round_box(ax, 0.3, 3.0, 1.8, 0.7, 'Sparse CT\nProjections\n{\u03b8\u2081,\u03b8\u2082,...\u03b8\u2099}', '#f0f0f0', '#999', 8)
    round_box(ax, 2.4, 3.0, 1.8, 0.7, 'FDK Filtered\nBack-Projection', '#dae8fc', '#6c8ebf', 8)
    round_box(ax, 4.5, 3.0, 1.5, 0.7, 'FDK Volume\nV(x,y,z)', '#dae8fc', '#6c8ebf', 8)
    round_box(ax, 6.3, 2.6, 1.8, 0.5, 'Density-Weighted\np(x) \u221d V(x)^\u03b3', '#dae8fc', '#6c8ebf', 7)
    round_box(ax, 6.3, 3.3, 1.8, 0.5, 'Uniform Sampling\n\u03b1 = 0.2', '#e8f0fe', '#6c8ebf', 7)
    round_box(ax, 8.4, 2.9, 1.8, 0.7, 'Mixed Strategy\nX = X\u1d56 \u222a X_w', '#dae8fc', '#6c8ebf', 8)
    round_box(ax, 10.5, 2.9, 1.3, 0.7, '50K Initial\nGaussians', '#e1d5e7', '#9673a6', 8, 'bold')
    arrow(ax, 2.1, 3.35, 2.4, 3.35); arrow(ax, 4.2, 3.35, 4.5, 3.35); arrow(ax, 6.0, 3.35, 6.3, 3.35)
    arrow(ax, 6.3, 2.85, 8.4, 3.0); arrow(ax, 6.3, 3.55, 8.4, 3.25); arrow(ax, 10.2, 3.25, 10.5, 3.25)
    ax.text(0.5, 1.8, 'Key Formulas:', fontsize=10, fontweight='bold')
    ax.text(0.5, 1.4, 'p(x) \u221d V_FDK(x)^\u03b3    (density-weighted sampling probability)', fontsize=9)
    ax.text(0.5, 1.0, 'X = X_uniform \u222a X_weighted    (mixed sampling strategy)', fontsize=9)
    ax.text(0.5, 0.4, 'Parameters: \u03b1=0.2, \u03b3=1.0, K=50,000, init_mode=raw', fontsize=8, style='italic', color='#666')
    fig.tight_layout()
    fig.savefig(f'{FIGS_DIR}/fig_sps.pdf', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("OK fig3")

# FIGURE 4
def draw_fig4():
    fig, ax = setup_fig(12, 4.5)
    ax.text(6, 4.3, 'GAP: Geometry-aware Pruning', ha='center', fontsize=14, fontweight='bold')
    round_box(ax, 0.3, 3.0, 1.8, 0.7, 'All Gaussians\nat iteration t\n(n \u2248 80-120K)', '#f0f0f0', '#999', 8)
    round_box(ax, 2.4, 2.8, 2.0, 0.8, 'For each Gaussian:\ns(p\u1d62) = \u00b9/\u2096 \u03a3||p\u1d62-p\u2c7c||\u2082\nK = 5 nearest neighbors', '#d5e8d4', '#82b366', 7)
    round_box(ax, 4.7, 2.8, 1.5, 0.8, 's(p\u1d62) < 0.015?\n\u2192 Candidate', '#d5e8d4', '#82b366', 8)
    round_box(ax, 6.5, 2.8, 1.5, 0.8, 'Gradient\n< 0.0002?\n\u2192 Not active', '#d5e8d4', '#82b366', 7)
    round_box(ax, 8.3, 2.8, 1.5, 0.8, 'Prune\nN=min(N_c,\u230a\u03b2N\u230b)\n\u03b2=0.02', '#f8cecc', '#b85450', 7)
    round_box(ax, 10.1, 2.8, 1.3, 0.8, 'Refined\nGaussians', '#e1d5e7', '#9673a6', 8, 'bold')
    arrow(ax, 2.1, 3.35, 2.4, 3.35); arrow(ax, 4.4, 3.35, 4.7, 3.35)
    arrow(ax, 6.2, 3.35, 6.5, 3.35); arrow(ax, 8.0, 3.35, 8.3, 3.35); arrow(ax, 9.8, 3.35, 10.1, 3.35)
    ax.text(0.5, 1.8, 'Schedule:', fontsize=10, fontweight='bold')
    ax.text(0.5, 1.4, '\u2022 Start: iteration 2,000', fontsize=9)
    ax.text(0.5, 1.0, '\u2022 Until: iteration 20,000', fontsize=9)
    ax.text(0.5, 0.6, '\u2022 Frequency: every 500 iterations', fontsize=9)
    ax.text(5.5, 1.8, 'Key Parameters:', fontsize=10, fontweight='bold')
    ax.text(5.5, 1.4, '\u03c4 (threshold) = 0.015', fontsize=9)
    ax.text(5.5, 1.0, '\u03b2 (max prune ratio) = 2%', fontsize=9)
    ax.text(5.5, 0.6, '\u03b4 (gradient threshold) = 0.0002', fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{FIGS_DIR}/fig_gap.pdf', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("OK fig4")

# FIGURE 5
def draw_fig5():
    fig, ax = setup_fig(14, 4.5)
    ax.text(7, 4.3, 'ADM: Adaptive Density Modulation', ha='center', fontsize=14, fontweight='bold')
    round_box(ax, 0.5, 2.8, 1.5, 0.7, 'Position\nx = (x,y,z)', '#f0f0f0', '#999', 8)
    round_box(ax, 2.3, 2.5, 2.5, 1.0, 'K-Planes Encoder\n(Resolution=64\u00d764)\n\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\nF_xy(x,y) \u2208 R\u00b3\u00b2\nF_xz(x,z) \u2208 R\u00b3\u00b2\nF_yz(y,z) \u2208 R\u00b3\u00b2', '#fff2cc', '#d6b656', 7)
    round_box(ax, 5.1, 2.8, 1.5, 0.7, 'Concatenate\nF(x) \u2208 R\u2079\u2076', '#fff2cc', '#d6b656', 8)
    round_box(ax, 6.9, 2.3, 2.5, 1.2, 'MLP Decoder\n(3 layers \u00d7 128 units, ReLU)\n\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\nMLP_\u0394 \u2192 \u0394\u03c1 \u2208 [-r, r]\nMLP_c \u2192 c \u2208 [0, 1]', '#fff2cc', '#d6b656', 7)
    round_box(ax, 9.7, 2.5, 2.2, 1.0, '\u03c1_final = \u03c1_base\n+ c(x)\u00b7\u0394\u03c1(x)\n(Zero-mean norm)', '#ffd966', '#bf9000', 7)
    round_box(ax, 12.2, 2.8, 1.3, 0.7, 'Modulated\nDensity \u03c1', '#e1d5e7', '#9673a6', 8, 'bold')
    arrow(ax, 2.0, 3.15, 2.3, 3.15); arrow(ax, 4.8, 3.15, 5.1, 3.15)
    arrow(ax, 6.6, 3.15, 6.9, 3.15); arrow(ax, 9.4, 3.15, 9.7, 3.15); arrow(ax, 11.9, 3.15, 12.2, 3.15)
    ax.text(0.5, 1.5, 'Key Formulas:', fontsize=10, fontweight='bold')
    ax.text(0.5, 1.1, 'F(x) = [f_xy, f_xz, f_yz] \u2208 R^(3d)    (feature encoding)', fontsize=9)
    ax.text(0.5, 0.7, '\u0394\u03c1(x) = tanh(MLP_\u0394(F(x))) \u00b7 r,   c(x) = \u03c3(MLP_c(F(x)))    (offset + confidence)', fontsize=9)
    ax.text(0.5, 0.3, '\u03c1_final(x) = \u03c1_base(x) + c(x) \u00b7 \u0394\u03c1(x)    (confidence-weighted modulation)', fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{FIGS_DIR}/fig_adm.pdf', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("OK fig5")

draw_fig1()
draw_fig2()
draw_fig3()
draw_fig4()
draw_fig5()
print("All figures done!")
