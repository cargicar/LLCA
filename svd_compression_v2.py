"""
SVD compression baseline for 3D simulation data — v2.

Identical to svd_compression.py except the k sweep stops as soon as a
user-defined relative-error bound is met, rather than sweeping a fixed list.

Sweeps k = 1, 2, 3, … in ascending order, evaluating reconstruction quality
at every step, and halts the moment rel_err ≤ --rel-err-target.  k_max is
a safety cap (default: n_patches).

The full trajectory k = 1 … k_stop is recorded and plotted so you can see
both the convergence curve and the exact stopping point.

Usage
-----
    python svd_compression_v2.py config_simmldc.yaml
    python svd_compression_v2.py config_simmldc.yaml --rel-err-target 0.005
    python svd_compression_v2.py config_simmldc.yaml --rel-err-target 0.01 --k-max 200
    python svd_compression_v2.py config_simmldc.yaml --lca-bpv 3.5 --lca-rel-err 0.0098
    python svd_compression_v2.py config_simmldc.yaml --output-dir results/svd_v2_run1
"""

import argparse
import csv
import os
import shutil
import sys
from datetime import datetime

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml

try:
    from sklearn.utils.extmath import randomized_svd
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Patch utilities  (unchanged from svd_compression.py)
# ---------------------------------------------------------------------------

def extract_tiled_patches(vol: np.ndarray, patch_size: int):
    D, H, W = vol.shape
    P = patch_size
    nD, nH, nW = D // P, H // P, W // P
    positions = [(i*P, j*P, k*P)
                 for i in range(nD)
                 for j in range(nH)
                 for k in range(nW)]

    n = len(positions)
    X     = np.empty((n, P**3), dtype=np.float32)
    means = np.empty(n, dtype=np.float32)
    stds  = np.empty(n, dtype=np.float32)

    for idx, (x, y, z) in enumerate(positions):
        patch      = vol[x:x+P, y:y+P, z:z+P].ravel().astype(np.float32)
        m          = patch.mean()
        s          = patch.std() + 1e-8
        means[idx] = m
        stds[idx]  = s
        X[idx]     = (patch - m) / s

    return X, means, stds, positions, (nD, nH, nW)


def reconstruct_volume(coeffs, Vt_k, means, stds, positions, vol_shape, patch_size):
    D, H, W = vol_shape
    P = patch_size
    recon_vol = np.zeros((D, H, W), dtype=np.float32)
    for idx, (x, y, z) in enumerate(positions):
        patch_norm = coeffs[idx] @ Vt_k
        patch      = patch_norm * stds[idx] + means[idx]
        recon_vol[x:x+P, y:y+P, z:z+P] = patch.reshape(P, P, P)
    return recon_vol


def compute_metrics(input_vol, recon_vol, k, patch_size, n_patches):
    error     = input_vol - recon_vol
    mse       = float((error**2).mean())
    rmse      = float(np.sqrt(mse))
    rel_err   = float(np.sqrt(mse) / (np.sqrt((input_vol**2).mean()) + 1e-8))
    sig_range = float(input_vol.max() - input_vol.min())
    psnr      = float(20 * np.log10(sig_range / (rmse + 1e-12)))

    P3            = patch_size**3
    bytes_coeff   = k * 4
    bytes_in      = P3 * 4
    comp_coeff    = P3 / k
    bpv_coeff     = (bytes_coeff * 8) / P3

    bytes_basis_pp = (k * P3 * 4) / n_patches
    bytes_total    = bytes_coeff + bytes_basis_pp
    comp_total     = bytes_in / bytes_total if bytes_total > 0 else float('inf')
    bpv_total      = (bytes_total * 8) / P3

    _index_bits_svd = int(np.ceil(np.log2(k + 1))) if k > 1 else 1
    _bytes_per_coo  = 4 + (_index_bits_svd + 7) // 8
    _bytes_code_coo = k * _bytes_per_coo
    comp_lca_equiv  = bytes_in / _bytes_code_coo if _bytes_code_coo > 0 else float('inf')
    bpv_lca_equiv   = (_bytes_code_coo * 8) / P3

    return dict(
        k=k,
        rel_err=rel_err,
        rmse=rmse,
        psnr=psnr,
        comp_coeff=comp_coeff,
        bpv_coeff=bpv_coeff,
        comp_total=comp_total,
        bpv_total=bpv_total,
        comp_lca_equiv=comp_lca_equiv,
        bpv_lca_equiv=bpv_lca_equiv,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='SVD compression baseline v2')
    parser.add_argument('config', help='path to config_simmldc.yaml')
    parser.add_argument('--rel-err-target', type=float, default=0.01,
                        help='stop as soon as rel_err ≤ this value (default: 0.01)')
    parser.add_argument('--k-max', type=int, default=None,
                        help='safety cap on k (default: n_patches)')
    parser.add_argument('--lca-bpv', type=float, default=None,
                        help='LCA BPV result for comparison line on plots')
    parser.add_argument('--lca-rel-err', type=float, default=None,
                        help='LCA relative error result for comparison line on plots')
    parser.add_argument('--output-dir', default=None,
                        help='output directory (default: experiments/svd_v2_TIMESTAMP)')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dcfg = cfg['data']
    P    = dcfg['patch_size']

    _svd_backend = cfg.get('svd', {}).get('backend', 'sklearn')
    if _svd_backend == 'sklearn' and not _SKLEARN_AVAILABLE:
        print("Warning: sklearn not installed — falling back to numpy SVD")
        _svd_backend = 'numpy'
    _USE_SKLEARN = (_svd_backend == 'sklearn')

    out_dir   = args.output_dir or os.path.join(
        'experiments', 'svd_v2_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    )
    plots_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(out_dir, 'config_simmldc.yaml'))

    log_path = os.path.join(out_dir, 'run.log')

    class _Tee:
        def __init__(self, *files): self.files = files
        def write(self, obj):
            for f in self.files: f.write(obj); f.flush()
        def flush(self):
            for f in self.files: f.flush()

    _log = open(log_path, 'w')
    sys.stdout = _Tee(sys.__stdout__, _log)
    sys.stderr = _Tee(sys.__stderr__, _log)

    print(f"Output dir   : {out_dir}")
    print(f"Config       : {args.config}")
    print(f"Patch size   : {P}³ = {P**3:,} voxels")
    print(f"rel_err target: {args.rel_err_target}")
    print(f"SVD backend  : {'sklearn randomized_svd' if _USE_SKLEARN else 'numpy linalg.svd'}\n")

    # ------------------------------------------------------------------ #
    # Load volume
    # ------------------------------------------------------------------ #
    print(f"Loading {dcfg['h5_path']} — field={dcfg['field_key']} t={dcfg['timestep']} ...")
    with h5py.File(dcfg['h5_path'], 'r') as f:
        vol = f[dcfg['field_key']][dcfg['timestep']].astype(np.float32)

    vol = (vol - vol.mean()) / (vol.std() + 1e-8)
    D, H, W = vol.shape
    print(f"Volume       : {D}×{H}×{W}\n")

    # ------------------------------------------------------------------ #
    # Extract non-overlapping patches
    # ------------------------------------------------------------------ #
    X, means, stds, positions, (nD, nH, nW) = extract_tiled_patches(vol, P)
    n_patches = len(positions)
    D_out, H_out, W_out = nD * P, nH * P, nW * P
    input_vol = vol[:D_out, :H_out, :W_out].copy()

    print(f"Tiles        : {nD}×{nH}×{nW} = {n_patches} patches "
          f"(covering {D_out}×{H_out}×{W_out} of {D}×{H}×{W})")
    print(f"Patch matrix X : {X.shape}  ({X.nbytes/1024/1024:.1f} MB)\n")

    # ------------------------------------------------------------------ #
    # Compute SVD  (truncated to k_max)
    # ------------------------------------------------------------------ #
    k_max = min(args.k_max or n_patches, n_patches, P**3)
    print(f"Computing truncated SVD (k_max={k_max}) ...")

    if _USE_SKLEARN:
        U, s, Vt = randomized_svd(X, n_components=k_max, random_state=42)
    else:
        U_full, s_full, Vt_full = np.linalg.svd(X, full_matrices=False)
        U, s, Vt = U_full[:, :k_max], s_full[:k_max], Vt_full[:k_max]

    coeffs_full = U * s[np.newaxis, :]   # (n_patches, k_max)
    print(f"SVD done.  Singular values: max={s[0]:.3f}  min={s[-1]:.4f}\n")

    # ------------------------------------------------------------------ #
    # Ascending sweep — stop when rel_err ≤ target
    # ------------------------------------------------------------------ #
    print(f"{'k':>6}  {'rel_err':>10}  {'PSNR(dB)':>10}  "
          f"{'Comp(coeff)':>13}  {'BPV(coeff)':>12}  "
          f"{'Comp(+basis)':>14}  {'BPV(+basis)':>13}  "
          f"{'Comp(LCA-eq)':>14}  {'BPV(LCA-eq)':>13}")
    print('-' * 118)

    results  = []
    k_stop   = None

    for k in range(1, k_max + 1):
        recon_vol = reconstruct_volume(
            coeffs_full[:, :k], Vt[:k], means, stds,
            positions, (D_out, H_out, W_out), P
        )
        m = compute_metrics(input_vol, recon_vol, k, P, n_patches)
        results.append(m)

        print(f"{k:>6}  {m['rel_err']:>10.6f}  {m['psnr']:>10.2f}  "
              f"{m['comp_coeff']:>13.2f}x  {m['bpv_coeff']:>12.3f}  "
              f"{m['comp_total']:>14.2f}x  {m['bpv_total']:>13.3f}  "
              f"{m['comp_lca_equiv']:>14.2f}x  {m['bpv_lca_equiv']:>13.3f}")

        if m['rel_err'] <= args.rel_err_target:
            k_stop = k
            print(f"\n  ✓  rel_err={m['rel_err']:.6f} ≤ target={args.rel_err_target} "
                  f"reached at k={k_stop}\n")
            break
    else:
        print(f"\n  ✗  target {args.rel_err_target} not reached within k_max={k_max}\n")

    print()
    best = results[-1]   # last evaluated k (either k_stop or k_max)

    # ------------------------------------------------------------------ #
    # Plots
    # ------------------------------------------------------------------ #
    ks       = [r['k']           for r in results]
    rel_errs = [r['rel_err']     for r in results]
    bpvs     = [r['bpv_coeff']   for r in results]
    cumvar   = np.cumsum(s**2) / np.sum(s**2) * 100

    # Plot 1 — singular value spectrum
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.semilogy(np.arange(1, len(s)+1), s, color='steelblue')
    if k_stop is not None:
        ax1.axvline(k_stop, color='red', linestyle='--', linewidth=1,
                    label=f'k_stop={k_stop}')
        ax1.legend(fontsize=8)
    ax1.set_xlabel('Component index')
    ax1.set_ylabel('Singular value (log scale)')
    ax1.set_title(f'Singular value spectrum  (patch {P}³, {n_patches} patches)')
    ax1.grid(True, alpha=0.3)

    ax2.plot(np.arange(1, len(s)+1), cumvar, color='darkorange')
    ax2.axhline(99, color='red',  linestyle='--', linewidth=0.8, label='99%')
    ax2.axhline(95, color='gray', linestyle=':',  linewidth=0.8, label='95%')
    if k_stop is not None:
        ax2.axvline(k_stop, color='red', linestyle='--', linewidth=1,
                    label=f'k_stop={k_stop}')
    ax2.set_xlabel('k (number of components)')
    ax2.set_ylabel('Cumulative explained variance (%)')
    ax2.set_title('Cumulative explained variance')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    for pct, color in [(99, 'red'), (95, 'gray')]:
        idx = np.searchsorted(cumvar, pct)
        if idx < len(s):
            ax2.axvline(idx + 1, color=color, linestyle='--', linewidth=0.8)
            ax2.text(idx + 1.5, pct - 3, f'k={idx+1}', fontsize=8, color=color)

    plt.tight_layout()
    out = os.path.join(plots_dir, 'singular_values.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")

    # Plot 2 — rel_err vs k (convergence curve)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.semilogy(ks, rel_errs, 'o-', color='steelblue', markersize=4, label='rel_err')
    ax2.plot(ks, [r['comp_coeff'] for r in results], 's--',
             color='darkorange', markersize=4, label='comp_ratio (coeff)')

    ax1.axhline(args.rel_err_target, color='red', linestyle=':', linewidth=1,
                label=f'target={args.rel_err_target}')
    if k_stop is not None:
        ax1.axvline(k_stop, color='red', linestyle='--', linewidth=1,
                    label=f'k_stop={k_stop}')
    if args.lca_rel_err is not None:
        ax1.axhline(args.lca_rel_err, color='purple', linestyle='--', linewidth=1,
                    label=f'LCA rel_err={args.lca_rel_err:.4f}')

    ax1.set_xlabel('k (SVD components)')
    ax1.set_ylabel('Relative reconstruction error (log)', color='steelblue')
    ax2.set_ylabel('Compression ratio (coeff only)', color='darkorange')
    ax1.set_title(f'SVD v2: convergence to target  (patch {P}³, {n_patches} patches)')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(plots_dir, 'rel_err_vs_k.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")

    # Plot 3 — rate-distortion
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(bpvs, rel_errs, 'o-', color='steelblue', markersize=5,
                label='SVD (coeff only, 4 B/coeff)')
    ax.semilogy([r['bpv_total']      for r in results], rel_errs, 's--',
                color='teal', markersize=4, alpha=0.7, label='SVD (+ amortised basis)')
    ax.semilogy([r['bpv_lca_equiv']  for r in results], rel_errs, '^:',
                color='darkorange', markersize=4, alpha=0.9, label='SVD (LCA-equiv COO)')

    ax.axhline(args.rel_err_target, color='red', linestyle=':', linewidth=1,
               label=f'target={args.rel_err_target}')

    if args.lca_bpv is not None and args.lca_rel_err is not None:
        ax.scatter([args.lca_bpv], [args.lca_rel_err], marker='*', s=200,
                   color='red', zorder=5,
                   label=f'LCA ({args.lca_bpv:.2f} BPV, {args.lca_rel_err:.4f} err)')

    if k_stop is not None:
        ax.scatter([best['bpv_coeff']], [best['rel_err']], marker='D', s=80,
                   color='red', zorder=6, label=f'k_stop={k_stop}')

    for r in results[::max(1, len(results)//8)]:
        ax.annotate(f"k={r['k']}", (r['bpv_coeff'], r['rel_err']),
                    textcoords='offset points', xytext=(4, 4), fontsize=7)

    ax.set_xlabel('Bits per voxel (BPV)')
    ax.set_ylabel('Relative reconstruction error (log scale)')
    ax.set_title(f'SVD Rate–Distortion  |  patch {P}³  |  {n_patches} patches')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(plots_dir, 'rate_distortion.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")

    # Plot 4 — full-volume reconstruction at k_stop (or k_max)
    best_k      = best['k']
    recon_vol   = reconstruct_volume(
        coeffs_full[:, :best_k], Vt[:best_k],
        means, stds, positions, (D_out, H_out, W_out), P
    )

    mD, mH, mW = D_out // 2, H_out // 2, W_out // 2
    plane_defs = [
        ('XY (z=mid)', input_vol[:, :, mW], recon_vol[:, :, mW]),
        ('XZ (y=mid)', input_vol[:, mH, :], recon_vol[:, mH, :]),
        ('YZ (x=mid)', input_vol[mD, :, :], recon_vol[mD, :, :]),
    ]

    stop_label = f'k_stop={best_k}' if k_stop is not None else f'k_max={best_k} (target not reached)'
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(
        f'SVD Full-volume reconstruction  |  {stop_label}  '
        f'rel_err={best["rel_err"]:.4f}  '
        f'comp_coeff={best["comp_coeff"]:.1f}x  BPV={best["bpv_coeff"]:.3f}',
        fontsize=10
    )
    for col, (lbl, inp_p, rec_p) in enumerate(plane_defs):
        vmax = np.percentile(np.abs(inp_p), 99)
        for row, (data, row_lbl) in enumerate([(inp_p, 'Input'), (rec_p, 'Reconstruction')]):
            ax = axes[row, col]
            im = ax.imshow(data, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                           origin='lower', aspect='equal')
            if row == 0:
                ax.set_title(lbl, fontsize=9)
            if col == 0:
                ax.set_ylabel(row_lbl, fontsize=9)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            plt.colorbar(im, ax=ax, shrink=0.85)

    plt.tight_layout()
    out = os.path.join(plots_dir, f'full_volume_k{best_k}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")

    # Plot 5 — error maps
    error_vol  = input_vol - recon_vol
    fig, axes  = plt.subplots(1, 3, figsize=(14, 4))
    plane_data = [
        ('XY', input_vol[:, :, mW], error_vol[:, :, mW]),
        ('XZ', input_vol[:, mH, :], error_vol[:, mH, :]),
        ('YZ', input_vol[mD, :, :], error_vol[mD, :, :]),
    ]
    for ax, (lbl, inp_p, err_p) in zip(axes, plane_data):
        vmax = np.percentile(np.abs(inp_p), 99)
        ax.imshow(err_p, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                  origin='lower', aspect='equal')
        ax.set_title(f'SVD error  {lbl}', fontsize=9)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    lca_ref = ('' if args.lca_rel_err is None
               else f'  vs LCA rel_err={args.lca_rel_err:.4f}')
    fig.suptitle(
        f'SVD error maps  |  {stop_label}  rel_err={best["rel_err"]:.4f}{lca_ref}',
        fontsize=9
    )
    plt.tight_layout()
    out = os.path.join(plots_dir, f'error_maps_k{best_k}.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")

    # ------------------------------------------------------------------ #
    # Save results table
    # ------------------------------------------------------------------ #
    csv_path = os.path.join(out_dir, 'svd_results.csv')
    with open(csv_path, 'w', newline='') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults table saved to {csv_path}")

    print(f"\n{'='*60}")
    print(f"SVD v2 SUMMARY  (patch {P}³, {n_patches} tiles, t={dcfg['timestep']})")
    print(f"{'='*60}")
    print(f"rel_err target : {args.rel_err_target}")
    if k_stop is not None:
        print(f"k_stop         : {k_stop}  ← first k meeting the target")
        print(f"rel_err        : {best['rel_err']:.6f}")
        print(f"comp(coeff)    : {best['comp_coeff']:.2f}x  BPV(coeff)={best['bpv_coeff']:.3f}")
        print(f"comp(LCA-eq)   : {best['comp_lca_equiv']:.2f}x  BPV(LCA-eq)={best['bpv_lca_equiv']:.3f}")
        print(f"comp(+basis)   : {best['comp_total']:.2f}x  BPV(+basis)={best['bpv_total']:.3f}")
    else:
        print(f"Target NOT reached within k_max={k_max}")
        print(f"Best rel_err   : {best['rel_err']:.6f}  at k={best['k']}")
    k99 = int(np.searchsorted(cumvar, 99)) + 1
    k95 = int(np.searchsorted(cumvar, 95)) + 1
    print(f"k for 95% var  : {k95}  →  comp={P**3/k95:.1f}x  BPV={k95*32/P**3:.3f}")
    print(f"k for 99% var  : {k99}  →  comp={P**3/k99:.1f}x  BPV={k99*32/P**3:.3f}")
    if args.lca_bpv:
        print(f"LCA reference  : BPV={args.lca_bpv:.3f}  rel_err={args.lca_rel_err}")

    print("\nDone.")
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    _log.close()


if __name__ == '__main__':
    main()
