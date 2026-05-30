#!/usr/bin/env python3
"""
Reproduce key paper figures from precomputed statistics.

Figure 2a:  Dead features by model — TopK vs TopK + AuxK (7 models).
Figure 4b:  Gamma vs init death scatter (454 model-layers) with theory curves.
Figure S11: 4-condition comparison — At Init / TopK / +AuxK / +Centering (14 models).

Usage:
    python reproduction/plot_figures.py

Outputs:
    reproduction/figure2a.pdf
    reproduction/figure4b.pdf
    reproduction/figure_s11.pdf
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from pathlib import Path

# Add parent dir so we can import metrics
sys.path.insert(0, str(Path(__file__).parent.parent))
from metrics import predicted_dead_by_relu, predicted_dead_by_topk

HERE = Path(__file__).parent

# =============================================================================
# Style + Colors (matching paper)
# =============================================================================
COLORS = {
    'sage_green': '#6b9a6b',  # At Init / Alive
    'purple': '#7b4bbf',      # TopK baseline (trained, no AuxK)
    'teal': '#4ba8bf',         # + AuxK
    'magenta': '#bf4b9a',      # + Mean Centering
}


def setup_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 13,
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'xtick.labelsize': 11.5,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        'axes.edgecolor': '#aaaaaa',
        'axes.linewidth': 0.5,
    })


SHORT_NAMES = {
    'GPT-2': 'GPT-2',
    'ModernBERT-Large': 'ModernBERT',
    'Pythia 410m': 'Pythia',
    'DINOv2-Large': 'DINOv2',
    'DINOv3-1B': 'DINOv3-1B',
    'DINOv3-7B': 'DINOv3',
    'Stable Diffusion 3.5 Large': 'SD 3.5',
    'ProGen2-large': 'ProGen2',
    'gLM2': 'gLM2',
    'ESM2-650M': 'ESM2-650M',
    'ESM2-3B': 'ESM2-3B',
    'ESM3': 'ESM3',
    'Evo1': 'Evo1',
    'AlphaFold3': 'AF3',
}


def domain_x_positions(n, domain_boundaries, gap=0.6, spacing=1.0):
    """Compute x positions with gaps between domain groups."""
    x = np.zeros(n)
    pos = 0
    for domain, (start, end) in domain_boundaries.items():
        if start > 0:
            pos += gap
        for i in range(start, end):
            x[i] = pos
            pos += spacing
    return x


def add_domain_labels(ax, x, domain_boundaries, y_pos, fontsize=14):
    """Add domain group labels with a thin line underneath."""
    for domain, (start, end) in domain_boundaries.items():
        positions = x[start:end]
        mid = np.mean(positions)
        left = min(positions) - 0.4
        right = max(positions) + 0.4
        ax.text(mid, y_pos, domain, ha='center', va='bottom',
                fontsize=fontsize, color='#222222')
        ax.plot([left, right], [y_pos - 0.5, y_pos - 0.5],
                color='#bbbbbb', linewidth=0.8, clip_on=False, zorder=5)


def add_value_labels(ax, bars, vals, fontsize=9.5, min_val=0.3):
    """Add value labels above bars."""
    for bar, val in zip(bars, vals):
        if val > min_val:
            y_offset = max(1.5, val * 0.03)
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + y_offset,
                    f'{val:.1f}', ha='center', va='bottom',
                    fontsize=fontsize, color='#333333')


# =============================================================================
# Figure 2a: TopK vs TopK + AuxK (7 models)
# =============================================================================
def plot_figure2a(df):
    """Reproduce Figure 2a: bar chart comparing TopK baseline vs +AuxK."""
    fig2_models = ['GPT-2', 'ModernBERT-Large', 'DINOv3-7B',
                   'Stable Diffusion 3.5 Large', 'ProGen2-large', 'ESM3', 'AlphaFold3']
    fig2_domains = {'Language': (0, 2), 'Vision': (2, 4), 'Biology': (4, 7)}
    # Override short name: only one DINOv3, drop the size
    names = {**SHORT_NAMES, 'DINOv3-7B': 'DINOv3'}

    sub = df[df.model.isin(fig2_models)].set_index('model').loc[fig2_models]

    baseline_vals = sub.dead_pct_final_baseline.values
    auxk_vals = sub.dead_pct_final_auxk.values

    n = len(fig2_models)
    x = domain_x_positions(n, fig2_domains, gap=1.2, spacing=1.4)
    bw = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.5))

    bars1 = ax.bar(x - bw / 2, baseline_vals, bw, color=COLORS['purple'],
                   label='TopK', zorder=3, edgecolor='white', linewidth=0.3)
    bars2 = ax.bar(x + bw / 2, auxk_vals, bw, color=COLORS['teal'],
                   label='TopK + AuxK', zorder=3, edgecolor='white', linewidth=0.3)

    add_value_labels(ax, bars1, baseline_vals)
    add_value_labels(ax, bars2, auxk_vals)

    ax.yaxis.grid(True, alpha=0.2, linewidth=0.4, color='#cccccc', zorder=0)
    ax.set_axisbelow(True)

    max_val = max(max(baseline_vals), max(auxk_vals))
    ax.set_ylim(0, min(118, max_val + 18))

    add_domain_labels(ax, x, fig2_domains,
                      y_pos=min(112, max_val + 13), fontsize=14)

    ax.set_ylabel('Dead Features (%)')
    ax.set_xticks(x)
    ax.set_xticklabels([names.get(m, m) for m in fig2_models], rotation=0, ha='center')
    ax.set_xlim(x[0] - 0.8, x[-1] + 0.8)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
              ncol=2, frameon=False, fontsize=11, columnspacing=2.0)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)

    out = HERE / 'figure2a.pdf'
    fig.savefig(out, bbox_inches='tight', dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


# =============================================================================
# Figure S11: 4-condition bar chart (14 models)
# =============================================================================
def plot_figure_s11(df):
    """Reproduce Figure S11: At Init / TopK / +AuxK / +Centering for all models."""
    s11_models = [
        'GPT-2', 'Pythia 410m', 'ModernBERT-Large',
        'DINOv2-Large', 'DINOv3-1B', 'DINOv3-7B', 'Stable Diffusion 3.5 Large',
        'ProGen2-large', 'gLM2', 'ESM2-650M', 'ESM2-3B', 'ESM3', 'Evo1', 'AlphaFold3',
    ]
    s11_domains = {'Language': (0, 3), 'Vision': (3, 7), 'Biology': (7, 14)}

    sub = df[df.model.isin(s11_models)].set_index('model').loc[s11_models]

    init_vals = sub.dead_pct_init_baseline.values
    baseline_vals = sub.dead_pct_final_baseline.values
    auxk_vals = sub.dead_pct_final_auxk.values
    mc_vals = sub.dead_pct_final_centered.values

    n = len(s11_models)
    x = domain_x_positions(n, s11_domains, gap=0.8)
    bw = 0.19

    fig, ax = plt.subplots(figsize=(15, 5.5))

    ax.bar(x - 1.5 * bw, init_vals, bw, color=COLORS['sage_green'],
           label='At Init', zorder=3, edgecolor='white', linewidth=0.3)
    ax.bar(x - 0.5 * bw, baseline_vals, bw, color=COLORS['purple'],
           label='TopK', zorder=3, edgecolor='white', linewidth=0.3)
    ax.bar(x + 0.5 * bw, auxk_vals, bw, color=COLORS['teal'],
           label='+ AuxK', zorder=3, edgecolor='white', linewidth=0.3)
    ax.bar(x + 1.5 * bw, mc_vals, bw, color=COLORS['magenta'],
           label='+ Mean Centering', zorder=3, edgecolor='white', linewidth=0.3)

    ax.yaxis.grid(True, alpha=0.2, linewidth=0.4, color='#cccccc', zorder=0)
    ax.set_axisbelow(True)

    add_domain_labels(ax, x, s11_domains, y_pos=105, fontsize=14)

    ax.set_ylabel('Dead Features (%)')
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_NAMES.get(m, m) for m in s11_models],
                       rotation=25, ha='right', fontsize=11)
    ax.set_ylim(0, 112)
    ax.set_xlim(x[0] - 0.7, x[-1] + 0.7)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=4, frameon=False, fontsize=11, columnspacing=1.5)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)

    out = HERE / 'figure_s11.pdf'
    fig.savefig(out, bbox_inches='tight', dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


# =============================================================================
# Figure 4b: Gamma vs init death scatter with theory curves (454 model-layers)
# =============================================================================
DOMAIN_COLORS = {
    'biology': '#4ba8bf',
    'vision': '#7b4bbf',
    'language': '#d4874b',
}


def plot_figure4b(df_init):
    """Reproduce Figure 4b: gamma vs init death with theory curves."""
    # SAE parameters used in the paper
    n_eval = 100_000
    k = 64
    n_features = 8192

    gamma_range = np.linspace(0.3, df_init.gamma.max() * 1.02, 500)
    theory_relu = predicted_dead_by_relu(gamma_range, n_eval=n_eval)
    theory_topk = predicted_dead_by_topk(gamma_range, n_eval=n_eval,
                                          k=k, n_features=n_features)

    fig, ax = plt.subplots(figsize=(6, 4))

    # Theory curves
    ax.plot(gamma_range, theory_relu, 'k--', linewidth=1.2, alpha=0.7,
            label='ReLU theory', zorder=1)
    ax.plot(gamma_range, theory_topk, color='#555555',
            linestyle=(0, (5, 2, 1, 2)), linewidth=1.2, alpha=0.7,
            label='TopK theory', zorder=1)

    # Data points by domain, TopK (triangles) and ReLU (circles)
    for domain in ['language', 'vision', 'biology']:
        fd = df_init[df_init.domain == domain]
        color = DOMAIN_COLORS[domain]
        ax.scatter(fd.gamma, fd.dead_pct_init_topk,
                   s=16, c=color, alpha=0.6, marker='^', zorder=2, linewidths=0)
        ax.scatter(fd.gamma, fd.dead_pct_init_relu,
                   s=16, c=color, alpha=0.6, marker='o', zorder=2, linewidths=0)

    ax.set_xlabel(r'Outlier severity $\gamma$', fontsize=12)
    ax.set_ylabel('Dead features (%)', fontsize=12)
    ax.set_xlim(-2, gamma_range[-1])
    ax.set_ylim(-2, 105)
    ax.grid(True, alpha=0.25, linewidth=0.4)

    # Legend
    shape_handles = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
               markersize=6, label='TopK', linestyle='None', alpha=0.7),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=6, label='ReLU', linestyle='None', alpha=0.7),
    ]
    theory_handles = [
        Line2D([0], [0], color='black', linestyle='--', linewidth=1.2,
               label='ReLU theory', alpha=0.7),
        Line2D([0], [0], color='#555555', linestyle=(0, (5, 2, 1, 2)),
               linewidth=1.2, label='TopK theory', alpha=0.7),
    ]
    domain_handles = [
        Line2D([0], [0], marker='s', color='w',
               markerfacecolor=DOMAIN_COLORS[d],
               markersize=6, label=d.capitalize(), linestyle='None', alpha=0.7)
        for d in ['biology', 'language', 'vision']
    ]

    all_handles = shape_handles + theory_handles + domain_handles
    ax.legend(handles=all_handles, fontsize=8, loc='lower right',
              ncol=2, frameon=True, framealpha=0.9, columnspacing=0.8)

    plt.tight_layout()

    out = HERE / 'figure4b.pdf'
    fig.savefig(out, bbox_inches='tight', dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    setup_style()

    # Load trained results (14 models, 1 layer each)
    df_trained = pd.read_csv(HERE / 'trained_results.csv')
    print(f"Loaded {len(df_trained)} models from trained_results.csv\n")

    print(f"{'Model':<28} {'Init%':>7} {'Baseline%':>10} {'+AuxK%':>8} {'+MC%':>7}")
    print("-" * 65)
    for _, row in df_trained.iterrows():
        print(f"{row.model:<28} {row.dead_pct_init_baseline:>7.1f} "
              f"{row.dead_pct_final_baseline:>10.1f} "
              f"{row.dead_pct_final_auxk:>8.1f} "
              f"{row.dead_pct_final_centered:>7.1f}")

    # Load init stats (454 model-layers)
    df_init = pd.read_csv(HERE / 'init_stats.csv')
    print(f"\nLoaded {len(df_init)} model-layers from init_stats.csv\n")

    print("Generating figures...")
    plot_figure2a(df_trained)
    plot_figure4b(df_init)
    plot_figure_s11(df_trained)
    print("\nDone.")
