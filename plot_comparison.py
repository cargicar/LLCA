#!/usr/bin/env python3
"""
Compare LCA vs SVD compression performance on the same canvas.

Usage:
  python plot_comparison.py \
      --lca  experiments/lca_ddp_2026-06-25_21-25-28/lca_results.csv \
      --svd  experiments/svd_v2_2026-06-25_16-51-53/svd_results.csv \
      --out  comparison.png
"""
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='LCA vs SVD compression comparison')
parser.add_argument('--lca', required=True, metavar='CSV',
                    help='lca_results.csv from an lca_ddp / LCA.py experiment')
parser.add_argument('--svd', required=True, metavar='CSV',
                    help='svd_results.csv from svd_compression.py experiment')
parser.add_argument('--out-dir', default=None, metavar='DIR',
                    help='output directory for PNGs (default: same dir as --lca)')
parser.add_argument('--lca-label', default='LCA (SVD init)', metavar='STR')
parser.add_argument('--svd-label', default='SVD', metavar='STR')
parser.add_argument('--rel-err-target', type=float, default=0.01,
                    help='horizontal guide line (default 0.01 = 1%%)')
args = parser.parse_args()

out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.lca))
os.makedirs(out_dir, exist_ok=True)

# ── Load ─────────────────────────────────────────────────────────────────────
lca = pd.read_csv(args.lca)
svd = pd.read_csv(args.svd)

# Normalise column names to lower-case
lca.columns = [c.lower() for c in lca.columns]
svd.columns = [c.lower() for c in svd.columns]

# Cap at 10% relative error
lca = lca[lca['rel_err'] <= 0.10]
svd = svd[svd['rel_err'] <= 0.10]

TARGET    = args.rel_err_target
LCA_COLOR = '#1f77b4'   # blue
SVD_COLOR = '#d62728'   # red

def _mark_target(ax, df, color, marker, x_col):
    below = df[df['rel_err'] <= TARGET]
    if not below.empty:
        best = below.iloc[0]
        ax.plot(best[x_col], best['rel_err'],
                marker, color=color, markersize=9, markeredgecolor='k',
                markeredgewidth=0.8, zorder=5)
        label = (f"{best[x_col]:.2f}×\n{best['rel_err']*100:.2f}%"
                 if x_col == 'comp_coeff' else
                 f"{best[x_col]:.1f} BPV\n{best['rel_err']*100:.2f}%")
        ax.annotate(label, xy=(best[x_col], best['rel_err']),
                    xytext=(6, 4), textcoords='offset points',
                    fontsize=7.5, color=color)

def _style(ax, xlabel, title):
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, _: f'{y*100:.3g}%' if y < 0.1 else f'{y*100:.0f}%'))
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel('Relative error (log scale)', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

# ── Plot 1: compression ratio vs rel_err ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.plot(lca['comp_coeff'], lca['rel_err'],
        'o-', color=LCA_COLOR, label=args.lca_label, markersize=4, linewidth=1.5)
ax.plot(svd['comp_coeff'], svd['rel_err'],
        's-', color=SVD_COLOR, label=args.svd_label, markersize=4, linewidth=1.5)
ax.axhline(TARGET, color='grey', linestyle='--', linewidth=0.9,
           label=f'{TARGET*100:.0f}% rel-err target')
for df, color, marker in [(lca, LCA_COLOR, 'o'), (svd, SVD_COLOR, 's')]:
    _mark_target(ax, df, color, marker, 'comp_coeff')
_style(ax, 'Compression ratio (P³ / avg_active)', 'Compression vs Reconstruction Error')
plt.tight_layout()
out1 = os.path.join(out_dir, 'comparison_compression.png')
plt.savefig(out1, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved {out1}')

# ── Plot 2: BPV vs rel_err (rate–distortion) ─────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.plot(lca['bpv_coeff'], lca['rel_err'],
        'o-', color=LCA_COLOR, label=args.lca_label, markersize=4, linewidth=1.5)
ax.plot(svd['bpv_coeff'], svd['rel_err'],
        's-', color=SVD_COLOR, label=args.svd_label, markersize=4, linewidth=1.5)
ax.axhline(TARGET, color='grey', linestyle='--', linewidth=0.9,
           label=f'{TARGET*100:.0f}% rel-err target')
for df, color, marker in [(lca, LCA_COLOR, 'o'), (svd, SVD_COLOR, 's')]:
    _mark_target(ax, df, color, marker, 'bpv_coeff')
_style(ax, 'Bits per voxel (coeff only)', 'Rate–Distortion')
plt.tight_layout()
out2 = os.path.join(out_dir, 'comparison_rate_distortion.png')
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved {out2}')
