"""
SVD compression baseline for 3D simulation data.

Computes a truncated SVD on non-overlapping 3D patches extracted from a single
HDF5 snapshot.  Sweeps over a range of k (number of SVD components) and
reports the same metrics as the LCA pipeline (rel_err, comp_ratio, BPV, PSNR),
enabling direct comparison.

No gradient descent — SVD is computed analytically from the data in one pass.

Compression model (per patch)
------------------------------
  - Encoder  : project normalised patch onto top-k right singular vectors
               coefficients = x_norm @ Vt.T   shape: (k,)
  - Decoder  : reconstruct from coefficients + undo normalisation
               x_norm_hat = coefficients @ Vt  → x_hat = x_norm_hat * std + mean
  - Storage  : k × float32 = k × 4 bytes  (no index overhead, unlike COO LCA)
  - Basis Vt : k × P³ × 4 bytes total, amortised over all patches in the volume

Compression ratio (coefficients only, basis overhead excluded):
  comp_ratio = (P³ × 4) / (k × 4) = P³ / k

Usage
-----
    python svd_compression.py config_simmldc.yaml
    python svd_compression.py config_simmldc.yaml --k-max 60
    python svd_compression.py config_simmldc.yaml --k-values 5 10 20 50
    python svd_compression.py config_simmldc.yaml --lca-bpv 3.5 --lca-rel-err 0.0098
    python svd_compression.py config_simmldc.yaml --output-dir results/svd_run1
"""

import argparse
import os
import sys
from datetime import datetime

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml

try:
    from sklearn.utils.extmath import randomized_svd
    _HAVE_SKLEARN = True
except ImportError:
    _HAVE_SKLEARN = False


# ---------------------------------------------------------------------------
# Patch utilities
# ---------------------------------------------------------------------------

def extract_tiled_patches(vol: np.ndarray, patch_size: int):
    """
    Extract all non-overlapping patch_size³ tiles from a 3D volume.

    Tiles that do not fit are discarded (same behaviour as the LCA inference
    script).  Each patch is normalised to zero mean, unit variance before
    being returned — matching LCA's internal normalisation.

    Returns
    -------
    X        : (n_patches, P³) float32 — normalised, flattened patches
    means    : (n_patches,) float32 — per-patch mean (to undo normalisation)
    stds     : (n_patches,) float32 — per-patch std  (to undo normalisation)
    positions: list of (x,y,z) top-left corners
    grid     : (nD, nH, nW) tile counts per dimension
    """
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
        patch = vol[x:x+P, y:y+P, z:z+P].ravel().astype(np.float32)
        m = patch.mean()
        s = patch.std() + 1e-8
        means[idx] = m
        stds[idx]  = s
        X[idx]     = (patch - m) / s

    return X, means, stds, positions, (nD, nH, nW)


def reconstruct_volume(
    coeffs: np.ndarray,      # (n_patches, k) — projection of each patch onto Vt[:k]
    Vt_k: np.ndarray,        # (k, P³) — top-k right singular vectors
    means: np.ndarray,       # (n_patches,)
    stds: np.ndarray,        # (n_patches,)
    positions: list,
    vol_shape: tuple,        # (D_out, H_out, W_out) — tiled region only
    patch_size: int,
) -> np.ndarray:
    """Reconstruct the tiled region from SVD coefficients."""
    D, H, W = vol_shape
    P = patch_size
    recon_vol = np.zeros((D, H, W), dtype=np.float32)

    for idx, (x, y, z) in enumerate(positions):
        patch_norm = coeffs[idx] @ Vt_k          # (P³,)
        patch = patch_norm * stds[idx] + means[idx]
        recon_vol[x:x+P, y:y+P, z:z+P] = patch.reshape(P, P, P)

    return recon_vol


def compute_metrics(
    input_vol: np.ndarray,
    recon_vol: np.ndarray,
    k: int,
    patch_size: int,
    n_patches: int,
) -> dict:
    """Compute reconstruction quality and compression metrics."""
    error     = input_vol - recon_vol
    mse       = float((error**2).mean())
    rmse      = float(np.sqrt(mse))
    rel_err   = float(np.sqrt(mse) / (np.sqrt((input_vol**2).mean()) + 1e-8))
    sig_range = float(input_vol.max() - input_vol.min())
    psnr      = float(20 * np.log10(sig_range / (rmse + 1e-12)))

    P3 = patch_size**3
    # Coefficients only (basis amortised / excluded):
    bytes_coeff   = k * 4                         # k × float32
    bytes_in      = P3 * 4                        # P³ × float32
    comp_coeff    = P3 / k                        # = bytes_in / bytes_coeff
    bpv_coeff     = (bytes_coeff * 8) / P3        # bits per voxel, coeff only

    # Amortised basis cost per patch:
    bytes_basis_pp = (k * P3 * 4) / n_patches
    bytes_total    = bytes_coeff + bytes_basis_pp
    comp_total     = bytes_in / bytes_total if bytes_total > 0 else float('inf')
    bpv_total      = (bytes_total * 8) / P3

    return dict(
        k=k,
        rel_err=rel_err,
        rmse=rmse,
        psnr=psnr,
        comp_coeff=comp_coeff,
        bpv_coeff=bpv_coeff,
        comp_total=comp_total,
        bpv_total=bpv_total,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='SVD compression baseline')
    parser.add_argument('config', help='path to config_simmldc.yaml')
    parser.add_argument('--k-values', type=int, nargs='+', default=None,
                        help='specific k values to evaluate (default: auto)')
    parser.add_argument('--k-max', type=int, default=None,
                        help='maximum k to sweep (default: n_patches)')
    parser.add_argument('--lca-bpv', type=float, default=None,
                        help='LCA BPV result for comparison line on plots')
    parser.add_argument('--lca-rel-err', type=float, default=None,
                        help='LCA relative error result for comparison line on plots')
    parser.add_argument('--output-dir', default=None,
                        help='output directory (default: experiments/svd_TIMESTAMP)')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dcfg = cfg['data']
    P    = dcfg['patch_size']

    # Output directory
    out_dir = args.output_dir or os.path.join(
        'experiments', 'svd_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    )
    plots_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    import shutil
    shutil.copy(args.config, os.path.join(out_dir, 'config_simmldc.yaml'))

    log_path = os.path.join(out_dir, 'run.log')

    class _Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files: f.write(obj); f.flush()
        def flush(self):
            for f in self.files: f.flush()

    _log = open(log_path, 'w')
    sys.stdout = _Tee(sys.__stdout__, _log)
    sys.stderr = _Tee(sys.__stderr__, _log)

    print(f"Output dir : {out_dir}")
    print(f"Config     : {args.config}")
    print(f"Patch size : {P}³ = {P**3:,} voxels")
    print(f"SVD backend: {'sklearn randomized_svd' if _HAVE_SKLEARN else 'numpy linalg.svd'}\n")

    # ------------------------------------------------------------------ #
    # Load volume
    # ------------------------------------------------------------------ #
    print(f"Loading {dcfg['h5_path']} — field={dcfg['field_key']} t={dcfg['timestep']} ...")
    with h5py.File(dcfg['h5_path'], 'r') as f:
        vol = f[dcfg['field_key']][dcfg['timestep']].astype(np.float32)

    vol = (vol - vol.mean()) / (vol.std() + 1e-8)   # global normalisation
    D, H, W = vol.shape
    print(f"Volume     : {D}×{H}×{W}\n")

    # ------------------------------------------------------------------ #
    # Extract non-overlapping patches
    # ------------------------------------------------------------------ #
    X, means, stds, positions, (nD, nH, nW) = extract_tiled_patches(vol, P)
    n_patches = len(positions)
    D_out, H_out, W_out = nD * P, nH * P, nW * P
    input_vol = vol[:D_out, :H_out, :W_out].copy()

    print(f"Tiles      : {nD}×{nH}×{nW} = {n_patches} patches "
          f"(covering {D_out}×{H_out}×{W_out} of {D}×{H}×{W})")
    print(f"Patch matrix X : {X.shape}  ({X.nbytes/1024/1024:.1f} MB)\n")

    # ------------------------------------------------------------------ #
    # Compute SVD
    # ------------------------------------------------------------------ #
    k_max = min(args.k_max or n_patches, n_patches, P**3)
    print(f"Computing truncated SVD (k_max={k_max}) ...")

    if _HAVE_SKLEARN:
        U, s, Vt = randomized_svd(X, n_components=k_max, random_state=42)
    else:
        # Full SVD then truncate — works since n_patches << P³
        U_full, s_full, Vt_full = np.linalg.svd(X, full_matrices=False)
        U, s, Vt = U_full[:, :k_max], s_full[:k_max], Vt_full[:k_max]

    # Pre-compute all projections (n_patches, k_max) — cheap since k_max is small
    # coefficients[i, :k] = U[i, :k] * s[:k]  for any k
    coeffs_full = U * s[np.newaxis, :]   # (n_patches, k_max)

    print(f"SVD done.  Singular values: max={s[0]:.3f}  min={s[-1]:.4f}\n")

    # ------------------------------------------------------------------ #
    # Sweep k values
    # ------------------------------------------------------------------ #
    if args.k_values:
        k_values = sorted(v for v in args.k_values if 1 <= v <= k_max)
    else:
        # Auto: logarithmically spaced + exact k_max
        k_log = np.unique(np.round(np.geomspace(1, k_max, 30)).astype(int))
        k_values = sorted(set(k_log.tolist() + [k_max]))

    results = []
    print(f"{'k':>6}  {'rel_err':>10}  {'PSNR(dB)':>10}  "
          f"{'CompRatio(coeff)':>18}  {'BPV(coeff)':>12}  "
          f"{'CompRatio(+basis)':>19}  {'BPV(+basis)':>13}")
    print('-' * 100)

    for k in k_values:
        coeffs_k = coeffs_full[:, :k]           # (n_patches, k)
        Vt_k     = Vt[:k]                       # (k, P³)
        recon_vol = reconstruct_volume(
            coeffs_k, Vt_k, means, stds, positions,
            (D_out, H_out, W_out), P
        )
        m = compute_metrics(input_vol, recon_vol, k, P, n_patches)
        results.append(m)
        print(f"{k:>6}  {m['rel_err']:>10.6f}  {m['psnr']:>10.2f}  "
              f"{m['comp_coeff']:>18.2f}x  {m['bpv_coeff']:>12.3f}  "
              f"{m['comp_total']:>19.2f}x  {m['bpv_total']:>13.3f}")

    print()

    # Best k at rel_err <= 1%
    under_1pct = [r for r in results if r['rel_err'] <= 0.01]
    if under_1pct:
        best = max(under_1pct, key=lambda r: r['comp_coeff'])
        print(f"Best (rel_err ≤ 1%): k={best['k']}  "
              f"rel_err={best['rel_err']:.4f}  "
              f"comp_ratio={best['comp_coeff']:.2f}x  BPV={best['bpv_coeff']:.3f}")
    else:
        best = min(results, key=lambda r: r['rel_err'])
        print(f"Note: rel_err never reaches 1% — "
              f"best is k={best['k']}  rel_err={best['rel_err']:.4f}")

    # ------------------------------------------------------------------ #
    # Plot 1 — Singular value spectrum
    # ------------------------------------------------------------------ #
    cumvar = np.cumsum(s**2) / np.sum(s**2) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.semilogy(np.arange(1, len(s)+1), s, color='steelblue')
    ax1.set_xlabel('Component index')
    ax1.set_ylabel('Singular value (log scale)')
    ax1.set_title(f'Singular value spectrum  (patch {P}³, {n_patches} patches)')
    ax1.grid(True, alpha=0.3)

    ax2.plot(np.arange(1, len(s)+1), cumvar, color='darkorange')
    ax2.axhline(99, color='red', linestyle='--', linewidth=0.8, label='99%')
    ax2.axhline(95, color='gray', linestyle=':', linewidth=0.8, label='95%')
    ax2.set_xlabel('k (number of components)')
    ax2.set_ylabel('Cumulative explained variance (%)')
    ax2.set_title('Cumulative explained variance')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Mark where 99% and 95% variance is reached
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

    # ------------------------------------------------------------------ #
    # Plot 2 — rel_err vs k
    # ------------------------------------------------------------------ #
    ks       = [r['k']        for r in results]
    rel_errs = [r['rel_err']  for r in results]
    comps    = [r['comp_coeff'] for r in results]
    bpvs     = [r['bpv_coeff']  for r in results]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.semilogy(ks, rel_errs, 'o-', color='steelblue', markersize=4, label='rel_err')
    ax2.plot(ks, comps, 's--', color='darkorange', markersize=4, label='comp_ratio (coeff)')

    ax1.axhline(0.01, color='red', linestyle=':', linewidth=1, label='1% target')
    if args.lca_rel_err is not None:
        ax1.axhline(args.lca_rel_err, color='purple', linestyle='--', linewidth=1,
                    label=f'LCA rel_err={args.lca_rel_err:.4f}')

    ax1.set_xlabel('k (SVD components)')
    ax1.set_ylabel('Relative reconstruction error (log)', color='steelblue')
    ax2.set_ylabel('Compression ratio (coeff only)', color='darkorange')
    ax1.set_title(f'SVD: reconstruction quality vs k  (patch {P}³, {n_patches} patches)')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(plots_dir, 'rel_err_vs_k.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")

    # ------------------------------------------------------------------ #
    # Plot 3 — rel_err vs BPV (rate-distortion curve)
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(bpvs, rel_errs, 'o-', color='steelblue', markersize=5,
                label='SVD (coeff only)')
    ax.semilogy([r['bpv_total'] for r in results], rel_errs, 's--',
                color='teal', markersize=4, alpha=0.7, label='SVD (+ amortised basis)')

    ax.axhline(0.01, color='red', linestyle=':', linewidth=1, label='1% error target')

    if args.lca_bpv is not None and args.lca_rel_err is not None:
        ax.scatter([args.lca_bpv], [args.lca_rel_err], marker='*', s=200,
                   color='red', zorder=5, label=f'LCA ({args.lca_bpv:.2f} BPV, '
                                                  f'{args.lca_rel_err:.4f} err)')

    # Annotate some k values
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

    # ------------------------------------------------------------------ #
    # Plot 4 — Full-volume reconstruction at best k
    # ------------------------------------------------------------------ #
    best_k = best['k']
    coeffs_best = coeffs_full[:, :best_k]
    Vt_best     = Vt[:best_k]
    recon_vol   = reconstruct_volume(
        coeffs_best, Vt_best, means, stds, positions,
        (D_out, H_out, W_out), P
    )

    mD, mH, mW = D_out // 2, H_out // 2, W_out // 2
    plane_defs = [
        ('XY (z=mid)', input_vol[:, :, mW], recon_vol[:, :, mW]),
        ('XZ (y=mid)', input_vol[:, mH, :], recon_vol[:, mH, :]),
        ('YZ (x=mid)', input_vol[mD, :, :], recon_vol[mD, :, :]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(
        f'SVD Full-volume reconstruction  |  k={best_k}  '
        f'rel_err={best["rel_err"]:.4f}  '
        f'CompRatio={best["comp_coeff"]:.1f}x  BPV={best["bpv_coeff"]:.3f}',
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

    # ------------------------------------------------------------------ #
    # Plot 5 — Comparison: SVD error map vs LCA (if lca_rel_err provided)
    # ------------------------------------------------------------------ #
    if args.lca_rel_err is not None:
        error_vol = input_vol - recon_vol
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
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
        fig.suptitle(
            f'SVD error maps  |  k={best_k}  rel_err={best["rel_err"]:.4f}  '
            f'vs LCA rel_err={args.lca_rel_err:.4f}',
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
    import csv
    csv_path = os.path.join(out_dir, 'svd_results.csv')
    with open(csv_path, 'w', newline='') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults table saved to {csv_path}")

    print(f"\n{'='*60}")
    print(f"SVD SUMMARY  (patch {P}³, {n_patches} tiles, t={dcfg['timestep']})")
    print(f"{'='*60}")
    print(f"Max components (n_patches): {n_patches}")
    print(f"Patch volume:               {P**3:,} voxels")
    print(f"Covered volume:             {D_out}×{H_out}×{W_out}")
    k99 = int(np.searchsorted(cumvar, 99)) + 1
    k95 = int(np.searchsorted(cumvar, 95)) + 1
    print(f"k for 95% variance:         {k95}  →  comp={P**3/k95:.1f}x  BPV={k95*32/P**3:.3f}")
    print(f"k for 99% variance:         {k99}  →  comp={P**3/k99:.1f}x  BPV={k99*32/P**3:.3f}")
    if under_1pct:
        print(f"Best (rel_err ≤ 1%):        k={best['k']}  "
              f"comp={best['comp_coeff']:.2f}x  BPV={best['bpv_coeff']:.3f}")
    if args.lca_bpv:
        print(f"LCA reference:              BPV={args.lca_bpv:.3f}  rel_err={args.lca_rel_err}")

    print("\nDone.")
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    _log.close()


if __name__ == '__main__':
    main()
