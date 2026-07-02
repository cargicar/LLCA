"""
Plot relative-error and compression-ratio curves vs epoch, parsed directly
from a run.log produced by lca_sim_mldc_SingleSnaptshot.py or
svd_lca_hybrid.py.

Auto-detects which script produced the log from its epoch-line format:
  - lca_sim_mldc_SingleSnaptshot.py :  "... Rel.err: 0.0491 ... CompRatio: 12.3x ..."
  - svd_lca_hybrid.py               :  "... LCA_res_err=0.0491 ... comp_ratio=12.3x ..."
                                        (plots LCA_res_err, not Hybrid_err)

Produces two separate figures: error vs epoch, and compression ratio vs epoch.

Usage
-----
    python plot_run_log.py experiments/simmldc_2026-06-11_13-28-15/run.log
    python plot_run_log.py experiments/svd_lca_2026-07-02_17-20-47          # dir → run.log inside it
    python plot_run_log.py path/to/run.log --out-dir plots/
"""

import argparse
import os
import re
import sys

import matplotlib.pyplot as plt

_NUM = r'[\d.eE+-]+'

# lca_sim_mldc_SingleSnaptshot.py:
#   Epoch 02 | 1.0s (0.02s/batch) | Sparsity: ... Rel.err: 0.049123 ... CompRatio: 12.34x ...
_PURE_LCA_RE = re.compile(
    rf'^Epoch\s+(\d+).*?Rel\.err:\s*({_NUM}).*?CompRatio:\s*({_NUM})x'
)
# svd_lca_hybrid.py:
#   Epoch 003 | 40.5s (30p x 6snap) | Sparsity=... LCA_res_err=0.049123 Hybrid_err=... comp_ratio=9.74x ...
_HYBRID_RE = re.compile(
    rf'^Epoch\s+(\d+).*?LCA_res_err=({_NUM}).*?comp_ratio=({_NUM})x'
)


def parse_run_log(path):
    """Returns (mode, epochs, err_values, comp_ratios). mode is 'hybrid' or 'pure_lca'."""
    epochs, errs, comps = [], [], []
    mode = None
    with open(path) as f:
        for line in f:
            m = _HYBRID_RE.search(line)
            if m:
                mode = 'hybrid'
            else:
                m = _PURE_LCA_RE.search(line)
                if m:
                    mode = mode or 'pure_lca'
            if m:
                epochs.append(int(m.group(1)))
                errs.append(float(m.group(2)))
                comps.append(float(m.group(3)))

    if not epochs:
        raise ValueError(
            f"No per-epoch training lines found in {path} — is this an SVD-only "
            f"sweep log (no `lca:` section), rather than a training run.log?"
        )
    return mode, epochs, errs, comps


def main():
    parser = argparse.ArgumentParser(
        description='Plot rel_err (or LCA_res_err for hybrid logs) and compression '
                     'ratio vs epoch from a run.log'
    )
    parser.add_argument('run_log', help='path to run.log, or a directory containing one')
    parser.add_argument('--out-dir', default=None,
                        help='directory to save PNGs into (default: alongside run.log)')
    args = parser.parse_args()

    run_log_path = args.run_log
    if os.path.isdir(run_log_path):
        run_log_path = os.path.join(run_log_path, 'run.log')

    mode, epochs, errs, comps = parse_run_log(run_log_path)
    err_label = 'LCA residual rel_err (LCA_res_err)' if mode == 'hybrid' else 'Relative error (Rel.err)'

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(run_log_path))
    os.makedirs(out_dir, exist_ok=True)

    # Canvas 1 — error vs epoch
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(epochs, errs, marker='o', markersize=3, color='steelblue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel(err_label)
    ax1.set_yscale('log')
    ax1.set_title(f'{err_label} vs epoch  ({os.path.basename(run_log_path)})')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    err_out = os.path.join(out_dir, 'rel_err_vs_epoch.png')
    plt.savefig(err_out, dpi=150)
    plt.close(fig1)
    print(f"Saved {err_out}")

    # Canvas 2 — compression ratio vs epoch
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.plot(epochs, comps, marker='o', markersize=3, color='darkorange')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Compression ratio (x)')
    ax2.set_title(f'Compression ratio vs epoch  ({os.path.basename(run_log_path)})')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    comp_out = os.path.join(out_dir, 'compression_ratio_vs_epoch.png')
    plt.savefig(comp_out, dpi=150)
    plt.close(fig2)
    print(f"Saved {comp_out}")


if __name__ == '__main__':
    main()
