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

def box(ax, x, y, w, h, text, color='#dae8fc', edge='#6c8ebf', fs=10, fw='normal', ha='center', va='center'):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                          facecolor=color, edgecolor=edge, linewidth=1.2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha=ha, va=va, fontsize=fs, fontweight=fw, zorder=5)

def arr(ax, x1, y1, x2, y2, color='#333', lw=1.2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))

def bigbox(ax, x, y, w, h, label='', edge='#999', ls='--'):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                          facecolor='none', edgecolor=edge, linewidth=0.8, linestyle=ls)
    ax.add_patch(box)
    if label:
        ax.text(x + 0.2, y + h - 0.2, label, ha='left', va='top', fontsize=9, fontweight='bold', color=edge)

# ============================================
# FIG 1: Overview — three-stage comparison
# ============================================
def draw_fig1():
    fig, ax = setup_fig(13, 4.5)
    ax.text(6.5, 4.2, 'SPAGS: Three-Stage Pipeline', ha='center', fontsize=14, fontweight='bold')

    # Input
    box(ax, 0.5, 3.0, 1.5, 0.7, 'Input\n2/3/4 views', '#f5f5f5', '#999', 9)
    bigbox(ax, 2.5, 2.5, 10.0, 1.5, 'SPAGS Framework', '#666')

    y_s = 3.4
    # Stage 1
    box(ax, 2.8, y_s, 3.0, 0.35, 'SPS: Spatial Prior', '#dae8fc', '#6c8ebf', 8)
    box(ax, 2.8, 2.65, 3.0, 0.65,
        'FDK → Density Samp.\nUniform → 50K GS',
        '#dae8fc', '#6c8ebf', 8)
    arr(ax, 2.0, 3.35, 2.8, 3.35)

    # Stage 2
    box(ax, 6.0, y_s, 3.0, 0.35, 'GAP: Geo. Pruning', '#d5e8d4', '#82b366', 8)
    box(ax, 6.0, 2.65, 3.0, 0.65,
        'KNN+τ Pruning\nGrad Filtering',
        '#d5e8d4', '#82b366', 8)
    arr(ax, 5.8, 3.35, 6.0, 3.35)

    # Stage 3
    box(ax, 9.2, y_s, 3.0, 0.35, 'ADM: Density Mod.', '#fff2cc', '#d6b656', 8)
    box(ax, 9.2, 2.65, 3.0, 0.65,
        'K-Planes+MLP\nZero-mean Adjust',
        '#fff2cc', '#d6b656', 8)
    arr(ax, 9.0, 3.35, 9.2, 3.35)

    # Output
    box(ax, 11.0, 3.7, 1.5, 0.5, 'Novel Views\n+0.39 dB ↑', '#f8cecc', '#b85450', 9, 'bold')
    arr(ax, 11.0, 3.35, 12.2, 3.7)

    # Baseline comparison
    box(ax, 2.0, 1.0, 1.8, 0.5, 'R²-Gaussian\n27.80 dB', '#f0f0f0', '#999', 8)
    ax.text(3.8, 1.05, '+0.39', ha='center', fontsize=9, fontweight='bold', color='#c44')
    box(ax, 4.5, 1.0, 1.8, 0.5, 'SPAGS (Ours)\n28.22 dB', '#f8cecc', '#b85450', 9, 'bold')
    arr(ax, 3.8, 1.25, 4.5, 1.25)

    # Key contributions — short
    c = ['✓ FDK-guided seeding', '✓ Proximity pruning', '✓ Density modulation']
    for i, t in enumerate(c):
        ax.text(7.0, 1.5 - i*0.25, t, ha='left', fontsize=9, color='#333')

    fig.tight_layout()
    fig.savefig(f'{FIGS_DIR}/fig_intro_overview.pdf', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("OK fig1")


# ============================================
# FIG 2: Full Pipeline — three stages
# ============================================
def draw_fig2():
    fig, ax = setup_fig(14, 6.5)
    ax.text(7, 6.2, 'SPAGS Training Pipeline', ha='center', fontsize=13, fontweight='bold')

    bigbox(ax, 0.3, 0.2, 13.4, 5.5, 'Training Process', '#666')

    # Stage 1: SPS
    y1 = 4.8
    box(ax, 0.5, y1, 4.0, 0.35, 'Stage 1: SPS — Spatial Prior Seeding', '#dae8fc', '#6c8ebf', 8)
    box(ax, 0.5, 3.9, 1.2, 0.6, 'CT Proj.\n2–4 views', 'white', '#999', 8)
    box(ax, 2.0, 3.9, 1.2, 0.6, 'FDK', '#dae8fc', '#6c8ebf', 9)
    box(ax, 3.5, 3.9, 1.5, 0.6, 'Density\nSampling', '#dae8fc', '#6c8ebf', 8)
    box(ax, 5.3, 3.9, 1.2, 0.6, 'Uniform\n(α=0.2)', '#dae8fc', '#6c8ebf', 8)
    box(ax, 6.8, 3.9, 1.3, 0.6, '50K Init\nGaussians', '#e1d5e7', '#9673a6', 8)
    arr(ax, 1.7, 4.2, 2.0, 4.2); arr(ax, 3.2, 4.2, 3.5, 4.2)
    arr(ax, 5.0, 4.2, 5.3, 4.2); arr(ax, 6.5, 4.2, 6.8, 4.2)

    # Stage 2: GAP
    y2 = 3.1
    box(ax, 0.5, y2, 4.0, 0.35, 'Stage 2: GAP — Geometry-aware Pruning', '#d5e8d4', '#82b366', 8)
    box(ax, 0.5, 2.2, 1.2, 0.6, 'Current\nGaussians', 'white', '#999', 8)
    box(ax, 2.0, 2.2, 1.4, 0.6, 'KNN\nK=5', '#d5e8d4', '#82b366', 9)
    box(ax, 3.7, 2.2, 1.0, 0.6, 's<τ?', '#d5e8d4', '#82b366', 10)
    box(ax, 5.0, 2.2, 1.2, 0.6, 'Grad\n<δ?', '#f8cecc', '#b85450', 9)
    box(ax, 6.5, 2.2, 1.2, 0.6, 'Prune\n2%/iter', '#f8cecc', '#b85450', 8)
    box(ax, 8.0, 2.2, 1.3, 0.6, 'Refined\nGaussians', '#e1d5e7', '#9673a6', 8)
    arr(ax, 1.7, 2.5, 2.0, 2.5); arr(ax, 3.4, 2.5, 3.7, 2.5)
    arr(ax, 4.7, 2.5, 5.0, 2.5); arr(ax, 6.2, 2.5, 6.5, 2.5); arr(ax, 7.7, 2.5, 8.0, 2.5)

    # Stage 3: ADM
    y3 = 1.3
    box(ax, 0.5, y3, 4.0, 0.35, 'Stage 3: ADM — Adaptive Density Modulation', '#fff2cc', '#d6b656', 8)
    box(ax, 0.5, 0.4, 1.2, 0.6, 'Position\nx,y,z', 'white', '#999', 8)
    box(ax, 2.0, 0.4, 1.5, 0.6, 'K-Planes\n64×64×32', '#fff2cc', '#d6b656', 8)
    box(ax, 3.8, 0.4, 1.2, 0.6, 'MLP\n3×128', '#fff2cc', '#d6b656', 9)
    box(ax, 5.3, 0.4, 1.5, 0.6, 'Δρ, c\nDual Head', '#ffd966', '#bf9000', 8)
    box(ax, 7.1, 0.4, 1.3, 0.6, 'ρ_final\nAdjust', '#e1d5e7', '#9673a6', 8)
    arr(ax, 1.7, 0.7, 2.0, 0.7); arr(ax, 3.5, 0.7, 3.8, 0.7)
    arr(ax, 5.0, 0.7, 5.3, 0.7); arr(ax, 6.8, 0.7, 7.1, 0.7)

    # Timeline arrows between stages
    ax.annotate('', xy=(1.3, 3.9), xytext=(1.3, 2.8),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.5, ls='dashed'))
    ax.annotate('', xy=(1.3, 2.2), xytext=(1.3, 1.0),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.5, ls='dashed'))
    ax.text(0.1, 2.5, 'Iter →', ha='left', fontsize=8, color='#555', rotation=90, va='center')

    fig.tight_layout()
    fig.savefig(f'{FIGS_DIR}/fig_pipeline.pdf', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("OK fig2")


# ============================================
# FIG 3: SPS Detail
# ============================================
def draw_fig3():
    fig, ax = setup_fig(11, 3.5)
    ax.text(5.5, 3.3, 'SPS: Spatial Prior Seeding', ha='center', fontsize=13, fontweight='bold')

    box(ax, 0.3, 2.2, 1.3, 0.6, 'Projections\n{θ₁..θₙ}', '#f0f0f0', '#999', 8)
    box(ax, 1.9, 2.2, 1.3, 0.6, 'FDK\nBackproj.', '#dae8fc', '#6c8ebf', 8)
    box(ax, 3.5, 2.2, 1.3, 0.6, 'Volume\nV(x)', '#dae8fc', '#6c8ebf', 8)
    box(ax, 5.1, 1.9, 1.5, 0.4, 'Weighted\np∝V^γ', '#dae8fc', '#6c8ebf', 7)
    box(ax, 5.1, 2.4, 1.5, 0.4, 'Uniform\nα=0.2', '#e8f0fe', '#6c8ebf', 7)
    box(ax, 6.9, 2.0, 1.5, 0.6, 'Mixed\nX=Xᵤ∪X_w', '#dae8fc', '#6c8ebf', 8)
    box(ax, 8.7, 2.0, 1.5, 0.6, '50K Init\nGaussians', '#e1d5e7', '#9673a6', 8)

    arr(ax, 1.6, 2.5, 1.9, 2.5); arr(ax, 3.2, 2.5, 3.5, 2.5)
    arr(ax, 4.8, 1.95, 5.1, 2.05); arr(ax, 4.8, 2.5, 5.1, 2.5)
    arr(ax, 6.6, 2.3, 6.9, 2.3); arr(ax, 8.4, 2.3, 8.7, 2.3)

    # Formula — single line
    ax.text(0.5, 0.8, 'p(x) ∝ V(x)^γ,   X = X_u ∪ X_w', ha='left', fontsize=9, style='italic', color='#555')
    ax.text(0.5, 0.3, 'Parameters: γ=1.0, α=0.2, 50K points, raw init', ha='left', fontsize=8, style='italic', color='#888')

    fig.tight_layout()
    fig.savefig(f'{FIGS_DIR}/fig_sps.pdf', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("OK fig3")


# ============================================
# FIG 4: GAP Detail
# ============================================
def draw_fig4():
    fig, ax = setup_fig(11, 3.5)
    ax.text(5.5, 3.3, 'GAP: Geometry-aware Pruning', ha='center', fontsize=13, fontweight='bold')

    box(ax, 0.3, 2.2, 1.3, 0.6, 'All GS\n80–120K', '#f0f0f0', '#999', 8)
    box(ax, 1.9, 2.0, 1.5, 0.8, 'KNN Score\ns(p)=¹/ₖΣ||·||\nK=5', '#d5e8d4', '#82b366', 7)
    box(ax, 3.7, 2.0, 1.2, 0.4, 's<τ?', '#d5e8d4', '#82b366', 10)
    box(ax, 3.7, 2.5, 1.2, 0.4, 'Grad<δ?', '#d5e8d4', '#82b366', 10)
    box(ax, 5.2, 2.2, 1.3, 0.6, 'Prune\nmax 2%/iter', '#f8cecc', '#b85450', 8)
    box(ax, 6.8, 2.2, 1.3, 0.6, 'Refined\nGS', '#e1d5e7', '#9673a6', 9)

    arr(ax, 1.6, 2.5, 1.9, 2.5); arr(ax, 3.4, 2.5, 3.7, 2.5)
    arr(ax, 4.9, 2.5, 5.2, 2.5); arr(ax, 6.5, 2.5, 6.8, 2.5)

    # Schedule — minimal
    ax.text(0.5, 0.8, 'Schedule: iters 2K–20K, every 500 iter', ha='left', fontsize=9, style='italic', color='#555')
    ax.text(0.5, 0.3, 'τ=0.015, β=2%, δ=0.0002, K=5', ha='left', fontsize=8, style='italic', color='#888')

    fig.tight_layout()
    fig.savefig(f'{FIGS_DIR}/fig_gap.pdf', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("OK fig4")


# ============================================
# FIG 5: ADM Detail
# ============================================
def draw_fig5():
    fig, ax = setup_fig(11, 3.5)
    ax.text(5.5, 3.3, 'ADM: Adaptive Density Modulation', ha='center', fontsize=13, fontweight='bold')

    box(ax, 0.5, 2.2, 1.2, 0.6, 'Position\n(x,y,z)', '#f0f0f0', '#999', 8)
    box(ax, 2.0, 2.0, 1.8, 0.8, 'K-Planes\nF_xy+F_xz+F_yz\n64×64×32', '#fff2cc', '#d6b656', 7)
    box(ax, 4.1, 2.2, 1.2, 0.6, 'MLP\n3×128', '#fff2cc', '#d6b656', 9)
    box(ax, 5.6, 2.0, 1.5, 0.8, 'Δρ=tanh·r\nc=σ(·)', '#ffd966', '#bf9000', 8)
    box(ax, 7.4, 2.0, 1.5, 0.8, 'ρ_final=ρ_base\n+c·Δρ\nZero-mean', '#e1d5e7', '#9673a6', 7)
    box(ax, 9.2, 2.2, 1.2, 0.6, 'Final ρ', '#e1d5e7', '#9673a6', 9)

    arr(ax, 1.7, 2.5, 2.0, 2.5); arr(ax, 3.8, 2.5, 4.1, 2.5)
    arr(ax, 5.3, 2.5, 5.6, 2.5); arr(ax, 7.1, 2.5, 7.4, 2.5); arr(ax, 8.9, 2.5, 9.2, 2.5)

    # Formulas — single line each
    ax.text(0.5, 0.8, 'F(x) = [f_xy, f_xz, f_yz]', ha='left', fontsize=9, style='italic', color='#555')
    ax.text(0.5, 0.3, 'Parameters: d=32, r=0.3, warmup=15K', ha='left', fontsize=8, style='italic', color='#888')

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
