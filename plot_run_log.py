"""
Plot relative-error and compression-ratio curves vs epoch, parsed directly
from a run.log produced by lca_sim_mldc_SingleSnaptshot.py or
svd_lca_hybrid.py.

Auto-detects which script produced the log from its epoch-line format:
  - lca_sim_mldc_SingleSnaptshot.py :  "... Rel.err: 0.0491 ... CompRatio: 12.3x ..."
  - svd_lca_hybrid.py               :  "... LCA_res_err=... Hybrid_err=0.0491 comp_ratio=12.3x ..."
                                        (plots Hybrid_err, not LCA_res_err)

Produces three figures: error vs epoch, compression ratio vs epoch, and — for
hybrid logs that include the SVD sweep table — a Pareto comparison of the
hybrid's (rel_err, comp_ratio) trajectory against the pure-SVD sweep curve,
since hybrid's comp_ratio alone is meaningless without that reference: it can
only ever fall *below* the SVD-only floor at the same k (LCA is a strictly
additive cost), so the real question is whether it beats pure SVD at *matched
accuracy* — i.e. whether its point sits above the SVD curve, not just where
it falls on its own epoch axis.

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
import numpy as np

_NUM = r'[\d.eE+-]+'

# lca_sim_mldc_SingleSnaptshot.py:
#   Epoch 02 | 1.0s (0.02s/batch) | Sparsity: ... Rel.err: 0.049123 ... CompRatio: 12.34x ...
_PURE_LCA_RE = re.compile(
    rf'^Epoch\s+(\d+).*?Rel\.err:\s*({_NUM}).*?CompRatio:\s*({_NUM})x'
)
# svd_lca_hybrid.py:
#   Epoch 003 | 40.5s (30p x 6snap) | Sparsity=... LCA_res_err=... Hybrid_err=0.049123 comp_ratio=9.74x ...
_HYBRID_RE = re.compile(
    rf'^Epoch\s+(\d+).*?Hybrid_err=({_NUM}).*?comp_ratio=({_NUM})x'
)
# svd_lca_hybrid.py, printed once right before the epoch loop starts — the SVD-only
# compression ratio never changes during LCA training (bytes_svd is fixed once k_lca
# is chosen), so it's a constant baseline, not a per-epoch curve:
#   SVD-only comp=10.4x  BPV=3.0727
_SVD_COMP_RE = re.compile(rf'^SVD-only comp=({_NUM})x')
# svd_lca_hybrid.py's SVD sweep table row, e.g.:
#   k     rel_err    PSNR(dB)    Comp(coeff)    BPV(coeff)    Comp(+basis)    BPV(+basis)
#   24    0.117928       44.14          30.38x         1.053           30.21x          1.059
_SVD_SWEEP_ROW_RE = re.compile(
    rf'^\s*(\d+)\s+({_NUM})\s+({_NUM})\s+({_NUM})x\s+({_NUM})\s+({_NUM})x\s+({_NUM})\s*$'
)
# svd_lca_hybrid.py's best-checkpoint marker, e.g.:
#   [best] comp_ratio=6.79x  hybrid_rel_err=0.099650
_BEST_RE = re.compile(rf'\[best\]\s+comp_ratio=({_NUM})x\s+hybrid_rel_err=({_NUM})')


def parse_run_log(path):
    """Returns (mode, epochs, err_values, comp_ratios, svd_comp_ratio).
    mode is 'hybrid' or 'pure_lca'. svd_comp_ratio is the constant SVD-only
    compression baseline (hybrid logs only; None otherwise)."""
    epochs, errs, comps = [], [], []
    mode = None
    svd_comp_ratio = None
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
                continue
            svd_m = _SVD_COMP_RE.search(line)
            if svd_m:
                svd_comp_ratio = float(svd_m.group(1))

    if not epochs:
        raise ValueError(
            f"No per-epoch training lines found in {path} — is this an SVD-only "
            f"sweep log (no `lca:` section), rather than a training run.log?"
        )
    return mode, epochs, errs, comps, svd_comp_ratio


def parse_svd_sweep(path):
    """Parse the SVD-only sweep table (k, rel_err, comp_coeff) that svd_lca_hybrid.py
    prints before LCA training starts. Returns (ks, rel_errs, comps), all empty if
    the table isn't present (e.g. pure-LCA logs never have it)."""
    ks, rel_errs, comps = [], [], []
    with open(path) as f:
        for line in f:
            m = _SVD_SWEEP_ROW_RE.match(line)
            if m:
                ks.append(int(m.group(1)))
                rel_errs.append(float(m.group(2)))
                comps.append(float(m.group(4)))
    return ks, rel_errs, comps


def parse_best_checkpoint(path):
    """Returns (comp_ratio, rel_err) of the last `[best]` checkpoint line logged
    by svd_lca_hybrid.py (the actual saved lca_hybrid_best_compression.pth), or
    (None, None) if no such line exists."""
    best_comp, best_err = None, None
    with open(path) as f:
        for line in f:
            m = _BEST_RE.search(line)
            if m:
                best_comp, best_err = float(m.group(1)), float(m.group(2))
    return best_comp, best_err


def _interp_comp_at_rel_err(svd_rel_errs, svd_comps, target_rel_err):
    """Linearly interpolate the pure-SVD sweep's comp_ratio at a given rel_err.
    svd_rel_errs decreases as k increases, so reverse to ascending x for np.interp.
    Returns None if target_rel_err falls outside the swept range."""
    xs = list(reversed(svd_rel_errs))
    ys = list(reversed(svd_comps))
    if not xs or target_rel_err < xs[0] or target_rel_err > xs[-1]:
        return None
    return float(np.interp(target_rel_err, xs, ys))


def main():
    parser = argparse.ArgumentParser(
        description='Plot rel_err (or Hybrid_err for hybrid logs) and compression '
                     'ratio vs epoch from a run.log'
    )
    parser.add_argument('run_log', help='path to run.log, or a directory containing one')
    parser.add_argument('--out-dir', default=None,
                        help='directory to save PNGs into (default: alongside run.log)')
    args = parser.parse_args()

    run_log_path = args.run_log
    if os.path.isdir(run_log_path):
        run_log_path = os.path.join(run_log_path, 'run.log')

    mode, epochs, errs, comps, svd_comp_ratio = parse_run_log(run_log_path)
    err_label  = 'Hybrid (SVD+LCA) rel_err' if mode == 'hybrid' else 'Relative error (Rel.err)'
    comp_label = 'Hybrid (SVD+LCA) comp_ratio' if mode == 'hybrid' else 'comp_ratio'

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

    # Canvas 2 — compression ratio vs epoch (total/hybrid, + SVD-only baseline if available)
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.plot(epochs, comps, marker='o', markersize=3, color='darkorange', label=comp_label)
    if svd_comp_ratio is not None:
        ax2.axhline(svd_comp_ratio, color='green', linestyle='--', linewidth=1.2,
                    label=f'SVD-only comp_ratio ({svd_comp_ratio:.2f}x, constant)')
        ax2.legend(fontsize=9)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Compression ratio (x)')
    ax2.set_title(f'Compression ratio vs epoch  ({os.path.basename(run_log_path)})')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    comp_out = os.path.join(out_dir, 'compression_ratio_vs_epoch.png')
    plt.savefig(comp_out, dpi=150)
    plt.close(fig2)
    print(f"Saved {comp_out}")

    # Canvas 3 — hybrid vs. pure-SVD Pareto frontier (hybrid logs only)
    if mode == 'hybrid':
        svd_ks, svd_rel_errs, svd_comps = parse_svd_sweep(run_log_path)
        if not svd_rel_errs:
            print("Note: no SVD sweep table found in this log — skipping Pareto comparison plot.")
        else:
            best_comp, best_err = parse_best_checkpoint(run_log_path)
            final_err, final_comp = errs[-1], comps[-1]

            fig3, ax3 = plt.subplots(figsize=(9, 6))
            ax3.loglog(svd_rel_errs, svd_comps, 'o-', color='steelblue', markersize=4,
                       label='Pure SVD sweep (varying k)')
            ax3.loglog(errs, comps, '-', color='gray', linewidth=1, alpha=0.6,
                       label='Hybrid training trajectory (epoch-by-epoch)')
            ax3.scatter([final_err], [final_comp], marker='*', s=250, color='red', zorder=5,
                        label=f'Hybrid final epoch (rel_err={final_err:.4f}, comp={final_comp:.2f}x)')
            if best_comp is not None:
                ax3.scatter([best_err], [best_comp], marker='D', s=80, color='purple', zorder=5,
                            label=f'Hybrid best checkpoint (rel_err={best_err:.4f}, comp={best_comp:.2f}x)')
            ax3.set_xlabel('Relative error (log scale)')
            ax3.set_ylabel('Compression ratio (x, log scale)')
            ax3.set_title(f'Hybrid vs. pure-SVD Pareto frontier  ({os.path.basename(run_log_path)})')
            ax3.grid(True, which='both', alpha=0.3)
            ax3.legend(fontsize=8)
            plt.tight_layout()
            pareto_out = os.path.join(out_dir, 'svd_pareto_vs_hybrid.png')
            plt.savefig(pareto_out, dpi=150)
            plt.close(fig3)
            print(f"Saved {pareto_out}")

            # Automated verdict: at the SAME rel_err the hybrid reached, would pure
            # SVD alone (interpolated from the sweep) have given more compression?
            for label, err, comp in [('final epoch', final_err, final_comp),
                                      ('best checkpoint', best_err, best_comp)]:
                if err is None:
                    continue
                svd_comp_at_err = _interp_comp_at_rel_err(svd_rel_errs, svd_comps, err)
                if svd_comp_at_err is None:
                    print(f"Verdict ({label}): rel_err={err:.4f} is outside the swept SVD "
                          f"k-range — can't compare.")
                elif comp > svd_comp_at_err:
                    print(f"Verdict ({label}): hybrid WINS — {comp:.2f}x vs. pure SVD's "
                          f"~{svd_comp_at_err:.2f}x at matched rel_err={err:.4f}.")
                else:
                    print(f"Verdict ({label}): hybrid LOSES — {comp:.2f}x vs. pure SVD's "
                          f"~{svd_comp_at_err:.2f}x at matched rel_err={err:.4f} "
                          f"(pure SVD alone would compress {svd_comp_at_err/comp:.2f}x better).")


if __name__ == '__main__':
    main()
