"""
SVD + LCA hybrid compression for 3D simulation data.

Extends svd_compression.py by training an LCA network on the SVD residual.
The full SVD sweep is preserved unchanged — output is directly comparable to
svd_compression.py.  After the sweep, if an `lca:` section is present in the
config, an LCAConv3D is trained on:

    residual = input_vol − svd_recon(svd_k)

Hybrid reconstruction:
    hybrid = svd_recon(svd_k) + lca_recon(residual)

Compression metric (both components exclude dictionary / basis overhead):
    bytes_svd   = n_patches × svd_k × 4        (dense float32 SVD coefficients)
    bytes_lca   = active_nz × bytes_per_nz      (COO: float32 value + flat index)
    comp_hybrid = bytes_in / (bytes_svd + bytes_lca)

If the `lca:` section is absent the script is identical to svd_compression.py.

Multi-snapshot training + held-out evaluation
----------------------------------------------
`data.train_timesteps` (list) pools patches from multiple HDF5 timesteps to
fit both the SVD basis and the LCA dictionary — this is closer to a fair
comparison than fitting on the exact snapshot being reconstructed, and gives
the Hebbian dictionary more diverse examples per atom. `data.eval_timestep`
holds out a snapshot never used in fitting: SVD coefficients for it are
obtained by *projecting* onto the shared basis (no re-fitting), and the
frozen LCA dictionary encodes its residual — this is the regime where
"beating SVD" is a meaningful generalization claim rather than an artifact
of SVD being optimal for the exact data it was fit on. Both keys are
optional; omitting them reproduces the original single-snapshot fit==eval
behaviour exactly.

Usage
-----
    python svd_lca_hybrid.py config_simmldc.yaml            # single GPU / SVD only
    python svd_lca_hybrid.py config_simmldc.yaml            # SVD + LCA (lca: in config)
    torchrun --nproc_per_node=4 svd_lca_hybrid.py config_simmldc.yaml

    # multi-snapshot training + held-out eval, overriding the config
    python svd_lca_hybrid.py config_simmldc.yaml \
        --train-timesteps 5 10 15 20 25 30 --eval-timestep 35
"""

import argparse
import csv
import os
import random
import shutil
import sys
import time
from datetime import datetime
from math import ceil

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import yaml

from lcapt.lca import LCAConv3D
from lcapt.metric import compute_l1_sparsity, compute_l2_error

try:
    from sklearn.utils.extmath import randomized_svd
    _HAVE_SKLEARN = True
except ImportError:
    _HAVE_SKLEARN = False


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def setup_ddp():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), local_rank, dist.get_world_size()


def cleanup_ddp():
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class _Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


# ---------------------------------------------------------------------------
# SVD utilities
# ---------------------------------------------------------------------------

def load_normalized_volume(h5_path: str, field_key: str, timestep: int):
    """Load one HDF5 timestep, z-scored to zero mean / unit variance."""
    with h5py.File(h5_path, 'r') as f:
        vol = f[field_key][timestep].astype(np.float32)
    mean = float(vol.mean())
    std  = float(vol.std()) + 1e-8
    return (vol - mean) / std, mean, std


def extract_tiled_patches(vol: np.ndarray, patch_size: int):
    """
    Extract all non-overlapping patch_size³ tiles from a 3D volume.

    Tiles that do not fit are discarded.  Each patch is normalised to zero
    mean, unit variance before being returned — matching LCA's internal
    normalisation.

    Returns
    -------
    X        : (n_patches, P³) float32 — normalised, flattened patches
    means    : (n_patches,) float32
    stds     : (n_patches,) float32
    positions: list of (x,y,z) top-left corners
    grid     : (nD, nH, nW)
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
    coeffs: np.ndarray,
    Vt_k: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    positions: list,
    vol_shape: tuple,
    patch_size: int,
) -> np.ndarray:
    """Reconstruct the tiled region from SVD coefficients."""
    D, H, W = vol_shape
    P = patch_size
    recon_vol = np.zeros((D, H, W), dtype=np.float32)
    for idx, (x, y, z) in enumerate(positions):
        patch_norm = coeffs[idx] @ Vt_k
        patch = patch_norm * stds[idx] + means[idx]
        recon_vol[x:x+P, y:y+P, z:z+P] = patch.reshape(P, P, P)
    return recon_vol


def compute_metrics_from_patches(
    X_norm: np.ndarray,
    coeffs_full: np.ndarray,
    Vt_full: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    patch_size: int,
    k: int,
) -> dict:
    """Reconstruction quality + compression metrics computed directly from a
    (possibly multi-snapshot) patch set, denormalising each patch with its
    own mean/std. Numerically identical to reconstructing full volumes and
    comparing spatially, but works uniformly whether the patches came from
    one snapshot or several pooled together.

    comp_coeff = P³/k  (dense coefficients, no index overhead)
    comp_total = same, amortising the shared Vt basis across all patches
    """
    P = patch_size
    n_patches = X_norm.shape[0]
    Xhat_norm = coeffs_full[:, :k] @ Vt_full[:k]
    X_real    = X_norm * stds[:, None] + means[:, None]
    Xhat_real = Xhat_norm * stds[:, None] + means[:, None]

    error     = X_real - Xhat_real
    mse       = float((error ** 2).mean())
    rmse      = float(np.sqrt(mse))
    rel_err   = rmse / (float(np.sqrt((X_real ** 2).mean())) + 1e-8)
    sig_range = float(X_real.max() - X_real.min())
    psnr      = float(20 * np.log10(sig_range / (rmse + 1e-12)))

    P3 = P ** 3
    bytes_in    = n_patches * P3 * 4
    bytes_coeff = n_patches * k * 4           # dense coefficients, no index overhead
    comp_coeff  = bytes_in / bytes_coeff      # = P³/k
    bpv_coeff   = (bytes_coeff * 8) / (n_patches * P3)

    bytes_basis = k * P3 * 4                  # Vt basis, paid once, amortised over all patches
    bytes_total = bytes_coeff + bytes_basis
    comp_total  = bytes_in / bytes_total if bytes_total > 0 else float('inf')
    bpv_total   = (bytes_total * 8) / (n_patches * P3)

    return dict(
        k=k, rel_err=rel_err, rmse=rmse, psnr=psnr,
        bytes_in=bytes_in, bytes_coeff=bytes_coeff,
        comp_coeff=comp_coeff, bpv_coeff=bpv_coeff,
        comp_total=comp_total, bpv_total=bpv_total,
    )


def _plot_reconstruction_grid(plots_dir, filename, input_vol, svd_recon, hybrid_recon, title):
    """Input / SVD recon / Hybrid recon / Error, three orthogonal mid-planes."""
    D, H, W = input_vol.shape
    mD, mH, mW = D // 2, H // 2, W // 2
    slicers = [
        ('XY  (z=mid)', lambda a: a[mD]),
        ('XZ  (y=mid)', lambda a: a[:, mH, :]),
        ('YZ  (x=mid)', lambda a: a[:, :, mW]),
    ]
    error_np = input_vol - hybrid_recon
    vmax_sig = float(np.percentile(np.abs(input_vol), 99))
    vmax_err = float(np.percentile(np.abs(error_np), 99))

    rows_data = [
        ('Input',                input_vol,    'RdBu_r', vmax_sig),
        ('SVD recon',            svd_recon,    'RdBu_r', vmax_sig),
        ('Hybrid recon',         hybrid_recon, 'RdBu_r', vmax_sig),
        ('Error (Input−Hybrid)', error_np,     'bwr',    vmax_err),
    ]
    fig, ax_grid = plt.subplots(4, 3, figsize=(11, 13))
    for ri, (row_label, arr, cmap, vmax_r) in enumerate(rows_data):
        ims = []
        for ci, (plane_label, slicer) in enumerate(slicers):
            ax = ax_grid[ri, ci]
            im = ax.imshow(slicer(arr), cmap=cmap, vmin=-vmax_r, vmax=vmax_r)
            ims.append(im)
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0: ax.set_title(plane_label, fontsize=9)
        ax_grid[ri, 0].set_ylabel(row_label, fontsize=8, labelpad=4)
        cbar = fig.colorbar(ims[1], ax=list(ax_grid[ri, :]), shrink=0.82, pad=0.02)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle(title, fontsize=9, y=1.002)
    plt.tight_layout()
    out = os.path.join(plots_dir, filename)
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='SVD + LCA hybrid compression')
    parser.add_argument('config', help='path to config YAML')
    parser.add_argument('--k-values', type=int, nargs='+', default=None,
                        help='specific k values to evaluate (default: auto)')
    parser.add_argument('--k-max', type=int, default=None,
                        help='maximum k for SVD sweep (default: n_patches)')
    parser.add_argument('--lca-bpv', type=float, default=None,
                        help='reference LCA BPV for rate-distortion plot')
    parser.add_argument('--lca-rel-err', type=float, default=None,
                        help='reference LCA rel_err for plots')
    parser.add_argument('--train-timesteps', type=int, nargs='+', default=None,
                        help='override data.train_timesteps — HDF5 timestep indices '
                             'pooled to fit the SVD basis + LCA dictionary')
    parser.add_argument('--eval-timestep', type=int, default=None,
                        help='override data.eval_timestep — held-out timestep for '
                             'generalization evaluation (never used in fitting)')
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ------------------------------------------------------------------ #
    # DDP setup
    # ------------------------------------------------------------------ #
    using_ddp = dist.is_available() and 'LOCAL_RANK' in os.environ
    if using_ddp:
        rank, local_rank, world_size = setup_ddp()
        device = torch.device(f'cuda:{local_rank}')
    else:
        rank = local_rank = 0
        world_size = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_main = (rank == 0)

    dcfg = cfg['data']
    P    = dcfg['patch_size']
    lca_cfg = cfg.get('lca')  # None → SVD-only run

    train_timesteps = args.train_timesteps or dcfg.get('train_timesteps') or [dcfg['timestep']]
    eval_timestep = (args.eval_timestep if args.eval_timestep is not None
                     else dcfg.get('eval_timestep'))

    # Output directory (rank 0 creates, all ranks share the same path)
    if is_main:
        ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        tag = 'svd_lca_' if lca_cfg is not None else 'svd_'
        out_dir = args.output_dir or os.path.join('experiments', tag + ts)
    else:
        out_dir = None
    if using_ddp:
        container = [out_dir]
        dist.broadcast_object_list(container, src=0)
        out_dir = container[0]

    plots_dir  = os.path.join(out_dir, 'plots')
    models_dir = os.path.join(out_dir, 'models')

    if is_main:
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        shutil.copy(args.config, os.path.join(out_dir, os.path.basename(args.config)))

    if using_ddp:
        dist.barrier()

    # ------------------------------------------------------------------ #
    # Logging (rank 0 only)
    # ------------------------------------------------------------------ #
    if is_main:
        _log = open(os.path.join(out_dir, 'run.log'), 'w')
        sys.stdout = _Tee(sys.__stdout__, _log)
        sys.stderr = _Tee(sys.__stderr__, _log)
        print(f"Output dir : {out_dir}")
        print(f"Config     : {args.config}")
        print(f"Patch size : {P}³ = {P**3:,} voxels")
        print(f"SVD backend: {'sklearn randomized_svd' if _HAVE_SKLEARN else 'numpy linalg.svd'}")
        print(f"Train timesteps : {train_timesteps}")
        print(f"Eval timestep   : "
              f"{eval_timestep if eval_timestep is not None else '(none — no held-out generalization test)'}\n")

    # ------------------------------------------------------------------ #
    # Load training snapshot(s) — each z-scored independently, tiled into
    # non-overlapping P³ patches, then pooled into one patch matrix so the
    # SVD basis and LCA dictionary are fit jointly across all of them.
    # ------------------------------------------------------------------ #
    train_vols = []
    for t in train_timesteps:
        vol_t, vmean_t, vstd_t = load_normalized_volume(dcfg['h5_path'], dcfg['field_key'], t)
        Xt, means_t, stds_t, positions_t, (nD, nH, nW) = extract_tiled_patches(vol_t, P)
        D_out, H_out, W_out = nD * P, nH * P, nW * P
        train_vols.append(dict(
            t=t, mean=vmean_t, std=vstd_t,
            X=Xt, means=means_t, stds=stds_t, positions=positions_t,
            shape=(D_out, H_out, W_out),
            input_vol=vol_t[:D_out, :H_out, :W_out].copy(),
            n_patches=len(positions_t),
        ))

    ref_shape = train_vols[0]['shape']
    for tv in train_vols[1:]:
        if tv['shape'] != ref_shape:
            raise ValueError(
                f"Training timestep {tv['t']} has tiled shape {tv['shape']} != "
                f"{ref_shape} from timestep {train_vols[0]['t']} — all training "
                f"snapshots must share the same spatial resolution."
            )

    X_train      = np.concatenate([tv['X'] for tv in train_vols], axis=0)
    means_train  = np.concatenate([tv['means'] for tv in train_vols])
    stds_train   = np.concatenate([tv['stds'] for tv in train_vols])
    patch_counts = [tv['n_patches'] for tv in train_vols]
    offsets      = np.cumsum([0] + patch_counts)   # offsets[i]:offsets[i+1] → patches of train_vols[i]
    n_patches    = X_train.shape[0]

    if is_main:
        print(f"Tiles/snapshot : {ref_shape[0]//P}×{ref_shape[1]//P}×{ref_shape[2]//P} "
              f"= {patch_counts[0]} patches  ×  {len(train_vols)} snapshot(s)")
        print(f"Patch matrix X : {X_train.shape}  ({X_train.nbytes/1024/1024:.1f} MB)\n")

    # ------------------------------------------------------------------ #
    # === SVD SWEEP — fit on the pooled multi-snapshot patch matrix ===
    # ------------------------------------------------------------------ #
    k_max = min(args.k_max or n_patches, n_patches, P**3)
    if is_main:
        print(f"Computing truncated SVD (k_max={k_max}) ...")

    if _HAVE_SKLEARN:
        U, s, Vt = randomized_svd(X_train, n_components=k_max, random_state=42)
    else:
        U_full, s_full, Vt_full = np.linalg.svd(X_train, full_matrices=False)
        U, s, Vt = U_full[:, :k_max], s_full[:k_max], Vt_full[:k_max]

    coeffs_full = U * s[np.newaxis, :]   # (n_patches, k_max)
    cumvar = np.cumsum(s**2) / np.sum(s**2) * 100

    if is_main:
        print(f"SVD done.  Singular values: max={s[0]:.3f}  min={s[-1]:.4f}\n")

    if args.k_values:
        k_values = sorted(v for v in args.k_values if 1 <= v <= k_max)
    else:
        k_log = np.unique(np.round(np.geomspace(1, k_max, 30)).astype(int))
        k_values = sorted(set(k_log.tolist() + [k_max]))

    results = []
    if is_main:
        print(f"{'k':>6}  {'rel_err':>10}  {'PSNR(dB)':>10}  "
              f"{'Comp(coeff)':>13}  {'BPV(coeff)':>12}  "
              f"{'Comp(+basis)':>14}  {'BPV(+basis)':>13}")
        print(f"       (pooled over {len(train_vols)} training snapshot(s); "
              f"Comp(coeff) directly comparable to hybrid comp_ratio)")
        print('-' * 90)

    for k in k_values:
        m = compute_metrics_from_patches(X_train, coeffs_full, Vt, means_train, stds_train, P, k)
        results.append(m)
        if is_main:
            print(f"{k:>6}  {m['rel_err']:>10.6f}  {m['psnr']:>10.2f}  "
                  f"{m['comp_coeff']:>13.2f}x  {m['bpv_coeff']:>12.3f}  "
                  f"{m['comp_total']:>14.2f}x  {m['bpv_total']:>13.3f}")

    if is_main:
        print()

    under_1pct = [r for r in results if r['rel_err'] <= 0.01]
    best = (max(under_1pct, key=lambda r: r['comp_coeff']) if under_1pct
            else min(results, key=lambda r: r['rel_err']))

    if is_main:
        if under_1pct:
            print(f"Best (rel_err ≤ 1%): k={best['k']}  rel_err={best['rel_err']:.4f}  "
                  f"comp(coeff)={best['comp_coeff']:.2f}x  BPV(coeff)={best['bpv_coeff']:.3f}  "
                  f"comp(+basis)={best['comp_total']:.2f}x  BPV(+basis)={best['bpv_total']:.3f}")
        else:
            print(f"Note: rel_err never reaches 1% — "
                  f"best is k={best['k']}  rel_err={best['rel_err']:.4f}  "
                  f"comp(coeff)={best['comp_coeff']:.2f}x  BPV(coeff)={best['bpv_coeff']:.3f}")

    # SVD plots ----------------------------------------------------------
    if is_main:
        ks       = [r['k']            for r in results]
        rel_errs = [r['rel_err']      for r in results]
        comps    = [r['comp_coeff']   for r in results]
        bpvs     = [r['bpv_coeff']    for r in results]

        # Plot 1 — singular value spectrum
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.semilogy(np.arange(1, len(s)+1), s, color='steelblue')
        ax1.set_xlabel('Component index')
        ax1.set_ylabel('Singular value (log scale)')
        ax1.set_title(f'Singular value spectrum  (patch {P}³, {n_patches} patches, '
                       f'{len(train_vols)} snapshot(s))')
        ax1.grid(True, alpha=0.3)
        ax2.plot(np.arange(1, len(s)+1), cumvar, color='darkorange')
        ax2.axhline(99, color='red',  linestyle='--', linewidth=0.8, label='99%')
        ax2.axhline(95, color='gray', linestyle=':',  linewidth=0.8, label='95%')
        ax2.set_xlabel('k'); ax2.set_ylabel('Cumulative explained variance (%)')
        ax2.set_title('Cumulative explained variance')
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
        for pct, color in [(99, 'red'), (95, 'gray')]:
            idx = np.searchsorted(cumvar, pct)
            if idx < len(s):
                ax2.axvline(idx + 1, color=color, linestyle='--', linewidth=0.8)
                ax2.text(idx + 1.5, pct - 3, f'k={idx+1}', fontsize=8, color=color)
        plt.tight_layout()
        out = os.path.join(plots_dir, 'singular_values.png')
        plt.savefig(out, dpi=150); plt.close(); print(f"Saved {out}")

        # Plot 2 — rel_err vs k
        fig, ax1_p = plt.subplots(figsize=(10, 5))
        ax2_p = ax1_p.twinx()
        ax1_p.semilogy(ks, rel_errs, 'o-', color='steelblue', markersize=4, label='rel_err')
        ax2_p.plot(ks, comps, 's--', color='darkorange', markersize=4,
                   label='comp_ratio (coeff)')
        ax1_p.axhline(0.01, color='red', linestyle=':', linewidth=1, label='1% target')
        if args.lca_rel_err is not None:
            ax1_p.axhline(args.lca_rel_err, color='purple', linestyle='--', linewidth=1,
                          label=f'LCA rel_err={args.lca_rel_err:.4f}')
        ax1_p.set_xlabel('k (SVD components)')
        ax1_p.set_ylabel('Relative reconstruction error (log)', color='steelblue')
        ax2_p.set_ylabel('Compression ratio (coeff only)', color='darkorange')
        ax1_p.set_title(f'SVD: reconstruction quality vs k  (patch {P}³, pooled {n_patches} patches)')
        lines1, labels1 = ax1_p.get_legend_handles_labels()
        lines2, labels2 = ax2_p.get_legend_handles_labels()
        ax1_p.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')
        ax1_p.grid(True, alpha=0.3)
        plt.tight_layout()
        out = os.path.join(plots_dir, 'rel_err_vs_k.png')
        plt.savefig(out, dpi=150); plt.close(); print(f"Saved {out}")

        # Plot 3 — rate-distortion
        fig, ax_p = plt.subplots(figsize=(9, 5))
        ax_p.semilogy(bpvs, rel_errs, 'o-', color='steelblue', markersize=5,
                      label='SVD (coeff only, 4 B/coeff)')
        ax_p.semilogy([r['bpv_total']      for r in results], rel_errs, 's--',
                      color='teal', markersize=4, alpha=0.7,
                      label='SVD (+ amortised basis)')
        ax_p.axhline(0.01, color='red', linestyle=':', linewidth=1, label='1% error target')
        if args.lca_bpv is not None and args.lca_rel_err is not None:
            ax_p.scatter([args.lca_bpv], [args.lca_rel_err], marker='*', s=200,
                         color='red', zorder=5,
                         label=f'LCA ({args.lca_bpv:.2f} BPV, {args.lca_rel_err:.4f} err)')
        for r in results[::max(1, len(results)//8)]:
            ax_p.annotate(f"k={r['k']}", (r['bpv_coeff'], r['rel_err']),
                          textcoords='offset points', xytext=(4, 4), fontsize=7)
        ax_p.set_xlabel('Bits per voxel (BPV)')
        ax_p.set_ylabel('Relative reconstruction error (log scale)')
        ax_p.set_title(f'SVD Rate–Distortion  |  patch {P}³  |  pooled {n_patches} patches')
        ax_p.legend(fontsize=9); ax_p.grid(True, alpha=0.3)
        plt.tight_layout()
        out = os.path.join(plots_dir, 'rate_distortion.png')
        plt.savefig(out, dpi=150); plt.close(); print(f"Saved {out}")

        # Plot 4 — full-volume reconstruction of the FIRST training snapshot at best SVD k
        best_k = best['k']
        tv0 = train_vols[0]
        sl0 = slice(offsets[0], offsets[1])
        recon_best = reconstruct_volume(
            coeffs_full[sl0, :best_k], Vt[:best_k],
            tv0['means'], tv0['stds'], tv0['positions'], tv0['shape'], P
        )
        input_vol0 = tv0['input_vol']
        D_out0, H_out0, W_out0 = tv0['shape']
        mD, mH, mW = D_out0 // 2, H_out0 // 2, W_out0 // 2
        plane_defs = [
            ('XY (z=mid)', input_vol0[:, :, mW], recon_best[:, :, mW]),
            ('XZ (y=mid)', input_vol0[:, mH, :], recon_best[:, mH, :]),
            ('YZ (x=mid)', input_vol0[mD, :, :], recon_best[mD, :, :]),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        fig.suptitle(
            f'SVD Full-volume reconstruction  (train t={tv0["t"]})  |  k={best_k}  '
            f'rel_err={best["rel_err"]:.4f}  comp={best["comp_coeff"]:.1f}x  '
            f'BPV={best["bpv_coeff"]:.3f}', fontsize=10
        )
        for col, (lbl, inp_p, rec_p) in enumerate(plane_defs):
            vmax = np.percentile(np.abs(inp_p), 99)
            for row, (data, row_lbl) in enumerate([(inp_p, 'Input'), (rec_p, 'Reconstruction')]):
                ax = axes[row, col]
                im = ax.imshow(data, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                               origin='lower', aspect='equal')
                if row == 0: ax.set_title(lbl, fontsize=9)
                if col == 0: ax.set_ylabel(row_lbl, fontsize=9)
                ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                plt.colorbar(im, ax=ax, shrink=0.85)
        plt.tight_layout()
        out = os.path.join(plots_dir, f'full_volume_svd_k{best_k}.png')
        plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close(); print(f"Saved {out}")

    # CSV + summary
    if is_main:
        csv_path = os.path.join(out_dir, 'svd_results.csv')
        with open(csv_path, 'w', newline='') as csvf:
            writer = csv.DictWriter(csvf, fieldnames=results[0].keys())
            writer.writeheader(); writer.writerows(results)
        print(f"\nResults table saved to {csv_path}")

        k99 = int(np.searchsorted(cumvar, 99)) + 1
        k95 = int(np.searchsorted(cumvar, 95)) + 1
        print(f"\n{'='*60}")
        print(f"SVD SUMMARY  (patch {P}³, {n_patches} pooled tiles, "
              f"train_timesteps={train_timesteps})")
        print(f"{'='*60}")
        print(f"Max components (n_patches): {n_patches}")
        print(f"Patch volume:               {P**3:,} voxels")
        print(f"Covered volume/snapshot:    {'×'.join(map(str, ref_shape))}")
        print(f"k for 95% variance:         {k95}  →  comp={P**3/k95:.1f}x  "
              f"BPV={k95*32/P**3:.3f}")
        print(f"k for 99% variance:         {k99}  →  comp={P**3/k99:.1f}x  "
              f"BPV={k99*32/P**3:.3f}")
        if under_1pct:
            print(f"Best (rel_err ≤ 1%):        k={best['k']}  "
                  f"comp(coeff)={best['comp_coeff']:.2f}x  BPV={best['bpv_coeff']:.3f}  "
                  f"comp(+basis)={best['comp_total']:.2f}x  BPV(+basis)={best['bpv_total']:.3f}")
        if args.lca_bpv:
            print(f"LCA reference:              BPV={args.lca_bpv:.3f}  "
                  f"rel_err={args.lca_rel_err}")

    # ------------------------------------------------------------------ #
    # Exit if no LCA config
    # ------------------------------------------------------------------ #
    if lca_cfg is None:
        if is_main:
            print("\nNo `lca:` section in config — SVD-only run complete.\nDone.")
            sys.stdout = sys.__stdout__; sys.stderr = sys.__stderr__
        if using_ddp:
            cleanup_ddp()
        return

    # ------------------------------------------------------------------ #
    # === LCA TRAINING — Hebbian dictionary on the pooled SVD residuals ===
    # ------------------------------------------------------------------ #
    if using_ddp:
        dist.barrier()

    k_lca = min(lca_cfg.get('svd_k') or best['k'], k_max)
    D_out, H_out, W_out = ref_shape

    # LCAConv3D with pad='same' requires stride to evenly divide the volume
    # it runs on. Unlike the patch-level scripts, this script runs LCA on
    # the FULL tiled residual volume (D_out³), not on individual P³ patches
    # — so it's D_out % stride that must be zero, not patch_size % stride.
    stride_lca = lca_cfg['stride']
    if D_out % stride_lca != 0:
        divisors = [d for d in range(1, D_out + 1) if D_out % d == 0]
        raise ValueError(
            f"lca.stride={stride_lca} does not evenly divide the tiled volume "
            f"size D_out={D_out} (from patch_size={P} × {D_out // P} tiles). "
            f"LCAConv3D's pad='same' forward pass requires D_out % stride == 0 "
            f"(it runs on the full residual volume, not per-patch). "
            f"Valid strides for D_out={D_out}: {divisors}"
        )

    train_residuals = []
    for idx, tv in enumerate(train_vols):
        sl = slice(offsets[idx], offsets[idx + 1])
        svd_recon_tv = reconstruct_volume(
            coeffs_full[sl, :k_lca], Vt[:k_lca],
            tv['means'], tv['stds'], tv['positions'], tv['shape'], P
        )
        residual_tv = (tv['input_vol'] - svd_recon_tv).astype(np.float32)
        res_mean_tv = float(residual_tv.mean())
        res_std_tv  = float(residual_tv.std()) + 1e-8
        svd_rel_err_tv = float(
            np.linalg.norm(tv['input_vol'] - svd_recon_tv) /
            (np.linalg.norm(tv['input_vol']) + 1e-8)
        )
        train_residuals.append(dict(
            t=tv['t'], input_vol=tv['input_vol'], svd_recon=svd_recon_tv,
            residual=residual_tv, res_mean=res_mean_tv, res_std=res_std_tv,
            svd_rel_err=svd_rel_err_tv,
        ))

    svd_rel_err = float(np.mean([r['svd_rel_err'] for r in train_residuals]))

    bytes_in  = D_out * H_out * W_out * 4
    bytes_svd = patch_counts[0] * k_lca * 4   # one snapshot's worth — comp_ratio is per-snapshot

    if is_main:
        print(f"\n{'='*60}")
        print(f"LCA TRAINING  (SVD k_lca={k_lca}  "
              f"mean train SVD_rel_err={svd_rel_err:.4f}  over {len(train_vols)} snapshot(s))")
        print(f"{'='*60}")
        print(f"bytes_in={bytes_in:,}  bytes_svd={bytes_svd:,}")
        print(f"SVD-only comp={bytes_in/bytes_svd:.1f}x  BPV={bytes_svd*8/(D_out*H_out*W_out):.4f}")
        print(f"  (same formula as hybrid comp_ratio below — directly comparable)\n")

    dtype_str = lca_cfg.get('dtype', 'float32')
    dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16}[dtype_str]

    residual_tensors = [
        torch.tensor((r['residual'] - r['res_mean']) / r['res_std'],
                      dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
        for r in train_residuals
    ]

    # LCAConv3D
    lca = LCAConv3D(
        out_neurons   = lca_cfg['features'],
        in_neurons    = 1,
        result_dir    = os.path.join(out_dir, 'lca_results'),
        kernel_size   = lca_cfg['kernel_size'],
        stride        = lca_cfg['stride'],
        lambda_       = lca_cfg['lambda_'],
        tau           = lca_cfg['tau'],
        lca_iters     = lca_cfg['lca_iters'],
        eta           = lca_cfg['learning_rate'],
        track_metrics = False,
        return_vars   = ['inputs', 'acts', 'recons', 'recon_errors'],
    ).to(dtype=dtype, device=device)

    if using_ddp:
        dist.broadcast(lca.weights.data, src=0)

    # COO sparse storage constants for LCA code
    cd = ceil(D_out / stride_lca)
    ch = ceil(H_out / stride_lca)
    cw = ceil(W_out / stride_lca)
    n_code_total  = lca_cfg['features'] * cd * ch * cw
    _index_bits   = int(np.ceil(np.log2(n_code_total + 1)))
    _bytes_per_nz = 4 + (_index_bits + 7) // 8

    if is_main:
        print(f"LCAConv3D      : {lca_cfg['features']} atoms | "
              f"kernel {lca_cfg['kernel_size']}³ | stride={stride_lca} | "
              f"λ={lca_cfg['lambda_']}")
        print(f"Code size      : {lca_cfg['features']} × {cd}×{ch}×{cw} = "
              f"{n_code_total} positions")
        print(f"Index bits     : {_index_bits}  →  bytes_per_nz={_bytes_per_nz}\n")

    # Training config
    max_epochs         = lca_cfg.get('max_epochs', 400)
    n_passes_per_epoch = lca_cfg.get('n_passes_per_epoch', 30)
    anneal_every       = lca_cfg.get('lambda_anneal_every', 1)
    anneal_step        = lca_cfg.get('lambda_anneal_step', 0.005)
    anneal_stop        = lca_cfg.get('lambda_anneal_stop', max_epochs)
    rel_err_target     = lca_cfg.get('rel_err_target', None)
    rel_err_ceiling    = lca_cfg.get('rel_err_ceiling', 0.05)
    stabilize_epochs   = lca_cfg.get('stabilize_epochs', 10)

    if is_main:
        print(f"Training       : {max_epochs} epochs × {n_passes_per_epoch} passes/epoch "
              f"× {len(train_residuals)} snapshot(s)")
        if rel_err_target is not None:
            print(f"Annealing      : rel_err-gated  "
                  f"target={rel_err_target}  ceiling={rel_err_ceiling}")
        else:
            print(f"Annealing      : time-based  stop={anneal_stop}  "
                  f"step={anneal_step}  every={anneal_every}")
        print()

    mode         = 'pre' if rel_err_target is not None else 'anneal'
    stab_count   = 0
    anneal_epoch = 0
    last_rel_err = 1.0
    best_comp    = 0.0

    all_hybrid_rel_err  = []
    all_lca_res_rel_err = []

    # Initialise with SVD-only state in case LCA never runs a full epoch
    vol0_hybrid_recon = train_residuals[0]['svd_recon'].copy()
    comp_ratio = bytes_in / bytes_svd if bytes_svd > 0 else float('inf')
    bpv        = bytes_svd * 8 / (D_out * H_out * W_out)
    hybrid_rel_err  = svd_rel_err
    active_nz = 0.0
    sparsity  = 0.0

    for epoch in range(max_epochs):
        t0 = time.time()

        # Annealing step at epoch start
        if rel_err_target is not None:
            if mode == 'anneal' and anneal_epoch % anneal_every == 0:
                if last_rel_err <= rel_err_ceiling:
                    lca.lambda_ += anneal_step
                    if is_main:
                        print(f"  [anneal] λ → {lca.lambda_:.3f}  "
                              f"(anneal epoch {anneal_epoch}/{anneal_stop})")
                else:
                    if is_main:
                        print(f"  [anneal] λ increment skipped — "
                              f"hybrid_rel_err={last_rel_err:.4f} > "
                              f"ceiling={rel_err_ceiling}")
        else:
            if epoch > 0 and anneal_epoch < anneal_stop \
                    and anneal_epoch % anneal_every == 0:
                lca.lambda_ += anneal_step
                if is_main:
                    print(f"  [anneal] λ → {lca.lambda_:.3f}")

        # n_passes_per_epoch forward + Hebbian passes, cycling through every
        # training snapshot in a random order each epoch. Metrics keep only
        # the LAST pass seen for each snapshot this epoch (matches the
        # original single-snapshot semantics exactly when there's only one).
        # Seeded by (epoch, pass) rather than global `random` so every DDP
        # rank shuffles identically without an extra broadcast — all ranks
        # must hit dist.all_reduce with the same volume in lockstep, since
        # they all hold the same weights.data and no DistributedSampler
        # splits work across ranks here.
        per_vol_metrics = [None] * len(residual_tensors)
        order = list(range(len(residual_tensors)))

        for pass_idx in range(n_passes_per_epoch):
            random.Random((epoch, pass_idx)).shuffle(order)
            for vi in order:
                residual_tensor = residual_tensors[vi]
                inputs_norm, code, recon_norm, recon_error_norm = lca(residual_tensor)
                lca.update_weights(code, recon_error_norm)
                if using_ddp:
                    dist.all_reduce(lca.weights.data, op=dist.ReduceOp.SUM)
                    lca.weights.data /= world_size
                    lca.normalize_weights()

                if is_main:
                    r = train_residuals[vi]
                    l1_cost = compute_l1_sparsity(code, lca.lambda_).item()
                    l2_cost = compute_l2_error(inputs_norm, recon_norm).item()
                    active_nz_v = float((code != 0).float().sum().item())

                    lca_recon_v    = recon_norm[0, 0].float().cpu().numpy() * r['res_std'] + r['res_mean']
                    hybrid_recon_v = r['svd_recon'] + lca_recon_v

                    lca_res_rel_err_v = float(
                        np.linalg.norm(lca_recon_v - r['residual']) /
                        (np.linalg.norm(r['residual']) + 1e-8)
                    )
                    hybrid_rel_err_v = float(
                        np.linalg.norm(hybrid_recon_v - r['input_vol']) /
                        (np.linalg.norm(r['input_vol']) + 1e-8)
                    )

                    per_vol_metrics[vi] = dict(
                        l1=l1_cost, l2=l2_cost, active_nz=active_nz_v,
                        lca_res_rel_err=lca_res_rel_err_v,
                        hybrid_rel_err=hybrid_rel_err_v,
                        hybrid_recon=hybrid_recon_v,
                    )

        # Metrics (rank 0 only — all ranks have identical weights) — mean
        # across training snapshots of each snapshot's most recent pass
        if is_main:
            active_nz       = float(np.mean([m['active_nz'] for m in per_vol_metrics]))
            sparsity        = 1.0 - active_nz / n_code_total
            l1_avg          = float(np.mean([m['l1'] for m in per_vol_metrics]))
            l2_avg          = float(np.mean([m['l2'] for m in per_vol_metrics]))
            lca_res_rel_err = float(np.mean([m['lca_res_rel_err'] for m in per_vol_metrics]))
            hybrid_rel_err  = float(np.mean([m['hybrid_rel_err'] for m in per_vol_metrics]))
            vol0_hybrid_recon = per_vol_metrics[0]['hybrid_recon']

            bytes_lca   = active_nz * _bytes_per_nz
            bytes_total = bytes_svd + bytes_lca
            comp_ratio  = bytes_in / bytes_total if bytes_total > 0 else float('inf')
            bpv         = bytes_total * 8 / (D_out * H_out * W_out)

            all_hybrid_rel_err.append(hybrid_rel_err)
            all_lca_res_rel_err.append(lca_res_rel_err)

            if mode == 'stabilize':
                mode_tag = f"  [stabilize {stab_count}/{stabilize_epochs}]"
            elif mode == 'anneal':
                mode_tag = f"  [anneal ep {anneal_epoch}/{anneal_stop}]"
            else:
                mode_tag = "  [pre-anneal]"

            print(
                f"Epoch {epoch:03d} | {time.time()-t0:.1f}s "
                f"({n_passes_per_epoch}p × {len(residual_tensors)}snap) | "
                f"Sparsity={sparsity:.3f}  Active={active_nz:.0f}/{n_code_total}  "
                f"LCA_res_err={lca_res_rel_err:.6f}  Hybrid_err={hybrid_rel_err:.6f}  "
                f"L2={l2_avg:.4f}  L1={l1_avg:.4f}  λ={lca.lambda_:.3f}  "
                f"comp_ratio={comp_ratio:.2f}x  BPV={bpv:.2f}" + mode_tag
            )

            torch.save(lca.state_dict(), os.path.join(models_dir, 'lca_hybrid.pth'))

            if hybrid_rel_err <= rel_err_ceiling and comp_ratio > best_comp:
                best_comp = comp_ratio
                torch.save(lca.state_dict(),
                           os.path.join(models_dir, 'lca_hybrid_best_compression.pth'))
                print(f"  [best] comp_ratio={best_comp:.2f}x  "
                      f"hybrid_rel_err={hybrid_rel_err:.6f}")

        last_rel_err = hybrid_rel_err

        # State machine
        if rel_err_target is not None:
            if mode in ('pre', 'anneal') and hybrid_rel_err <= rel_err_target:
                prev_mode  = mode
                mode       = 'stabilize'
                stab_count = 0
                reason = 'before first anneal' if prev_mode == 'pre' else 'mid-anneal'
                if is_main:
                    print(f"  [stabilize] hybrid_rel_err={hybrid_rel_err:.6f} <= "
                          f"{rel_err_target} — freezing λ for {stabilize_epochs} "
                          f"epochs ({reason})")
            elif mode == 'stabilize':
                stab_count += 1
                if stab_count >= stabilize_epochs:
                    if anneal_epoch >= anneal_stop:
                        if is_main:
                            print("  [done] annealing complete + stabilized — stopping")
                        break
                    mode       = 'anneal'
                    stab_count = 0
                    if is_main:
                        verb = 'Starting' if anneal_epoch == 0 else 'Resuming'
                        print(f"  [stabilize] done — {verb} λ annealing")

        if mode == 'anneal':
            anneal_epoch += 1

    # ------------------------------------------------------------------ #
    # Post-training output (rank 0 only)
    # ------------------------------------------------------------------ #
    if not is_main:
        if using_ddp:
            cleanup_ddp()
        return

    np.savez_compressed(
        os.path.join(models_dir, 'svd_basis.npz'),
        Vt=Vt[:k_lca],
        train_timesteps=np.array(train_timesteps),
        k=np.int32(k_lca),
    )
    print(f"\nSaved SVD basis → {os.path.join(models_dir, 'svd_basis.npz')}")
    print(f"Saved LCA model → {os.path.join(models_dir, 'lca_hybrid.pth')}")
    print("\nGenerating plots...")

    # Training curves
    if all_hybrid_rel_err:
        fig, (ax0, ax1_t) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax0.plot(all_lca_res_rel_err, label='LCA residual rel_err (mean over train snapshots)')
        ax0.set_ylabel('LCA residual rel_err'); ax0.legend(fontsize=8)
        ax1_t.plot(all_hybrid_rel_err, label='Hybrid (SVD+LCA) rel_err (mean over train snapshots)')
        if rel_err_target is not None:
            ax1_t.axhline(rel_err_target, color='r', linestyle='--', linewidth=0.8,
                          label=f'target={rel_err_target}')
        ax1_t.axhline(svd_rel_err, color='g', linestyle=':', linewidth=0.8,
                      label=f'SVD-only rel_err={svd_rel_err:.4f}')
        ax1_t.set_ylabel('Hybrid rel_err'); ax1_t.set_xlabel('Epoch')
        ax1_t.legend(fontsize=8)
        ax0.set_title(
            f'Hybrid SVD+LCA — Training Curves  |  '
            f'SVD k={k_lca}  SVD_err={svd_rel_err:.4f}  '
            f'{len(train_residuals)} training snapshot(s)'
        )
        plt.tight_layout()
        out = os.path.join(plots_dir, 'training_metrics.png')
        plt.savefig(out); plt.close(); print(f"Saved {out}")

    # Dictionary atoms
    weights = lca.get_weights().float().cpu().numpy()
    n_feat  = weights.shape[0]
    kD      = weights.shape[2]
    atoms   = weights[:, 0, kD // 2, :, :]
    cols    = int(np.ceil(np.sqrt(n_feat)))
    rows_a  = int(np.ceil(n_feat / cols))
    fig, ax_atoms = plt.subplots(rows_a, cols, figsize=(cols * 1.2, rows_a * 1.2))
    ax_atoms = np.array(ax_atoms).ravel()
    vmax_a = np.percentile(np.abs(atoms), 99)
    for i, ax in enumerate(ax_atoms):
        if i < n_feat:
            ax.imshow(atoms[i], cmap='RdBu_r', vmin=-vmax_a, vmax=vmax_a)
        ax.axis('off')
    fig.suptitle(
        f'Dictionary atoms — mid-plane slice  '
        f'({n_feat} atoms, kernel {lca_cfg["kernel_size"]}³)',
        fontsize=10
    )
    plt.tight_layout()
    out = os.path.join(plots_dir, 'dictionary_atoms.png')
    plt.savefig(out, dpi=150); plt.close(); print(f"Saved {out}")

    # Reconstruction: Input / SVD / Hybrid / Error — training snapshot 0
    _plot_reconstruction_grid(
        plots_dir, 'reconstruction.png',
        input_vol=train_residuals[0]['input_vol'],
        svd_recon=train_residuals[0]['svd_recon'],
        hybrid_recon=vol0_hybrid_recon,
        title=(f'Hybrid SVD+LCA  |  train t={train_residuals[0]["t"]}  SVD k={k_lca}  '
               f'SVD_err={train_residuals[0]["svd_rel_err"]:.4f}  '
               f'hybrid_err={hybrid_rel_err:.4f}  comp={comp_ratio:.2f}x  BPV={bpv:.2f}'),
    )

    # ------------------------------------------------------------------ #
    # === HELD-OUT EVALUATION — generalization to an unseen snapshot ===
    # SVD coefficients come from *projecting* onto the basis fit on
    # train_timesteps only (no re-fitting); the LCA dictionary is frozen.
    # This is the regime where beating SVD is a meaningful claim, since SVD
    # loses its "optimal for exactly this data" advantage on unseen data.
    # ------------------------------------------------------------------ #
    if eval_timestep is not None:
        print(f"\n{'='*60}")
        print(f"HELD-OUT EVALUATION  (t={eval_timestep}, not used in fitting)")
        print(f"{'='*60}")

        eval_vol, _, _ = load_normalized_volume(dcfg['h5_path'], dcfg['field_key'], eval_timestep)
        X_eval, means_eval, stds_eval, positions_eval, (nDe, nHe, nWe) = extract_tiled_patches(eval_vol, P)
        eval_shape = (nDe * P, nHe * P, nWe * P)
        if eval_shape != ref_shape:
            print(f"  [warn] eval tiled shape {eval_shape} != training shape {ref_shape} "
                  f"— metrics still valid (per-patch), spatial plot skipped.")
        input_vol_eval = eval_vol[:eval_shape[0], :eval_shape[1], :eval_shape[2]].copy()

        # Project held-out patches onto the SHARED basis (fit from training snapshots only)
        coeffs_eval    = X_eval @ Vt[:k_lca].T
        svd_recon_eval = reconstruct_volume(
            coeffs_eval, Vt[:k_lca], means_eval, stds_eval, positions_eval, eval_shape, P
        )
        svd_rel_err_eval = float(
            np.linalg.norm(input_vol_eval - svd_recon_eval) /
            (np.linalg.norm(input_vol_eval) + 1e-8)
        )

        residual_eval = (input_vol_eval - svd_recon_eval).astype(np.float32)
        res_mean_eval = float(residual_eval.mean())
        res_std_eval  = float(residual_eval.std()) + 1e-8
        residual_tensor_eval = torch.tensor(
            (residual_eval - res_mean_eval) / res_std_eval, dtype=dtype, device=device
        ).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            _, code_eval, recon_norm_eval, _ = lca(residual_tensor_eval)

        eDe, eHe, eWe = eval_shape
        n_code_eval = (lca_cfg['features'] * ceil(eDe / stride_lca)
                       * ceil(eHe / stride_lca) * ceil(eWe / stride_lca))
        active_nz_eval = float((code_eval != 0).float().sum().item())
        sparsity_eval  = 1.0 - active_nz_eval / n_code_eval

        lca_recon_eval    = recon_norm_eval[0, 0].float().cpu().numpy() * res_std_eval + res_mean_eval
        hybrid_recon_eval = svd_recon_eval + lca_recon_eval
        hybrid_rel_err_eval = float(
            np.linalg.norm(hybrid_recon_eval - input_vol_eval) /
            (np.linalg.norm(input_vol_eval) + 1e-8)
        )

        n_patches_eval   = len(positions_eval)
        bytes_in_eval    = eDe * eHe * eWe * 4
        bytes_svd_eval   = n_patches_eval * k_lca * 4
        bytes_lca_eval   = active_nz_eval * _bytes_per_nz
        bytes_total_eval = bytes_svd_eval + bytes_lca_eval
        comp_ratio_eval  = bytes_in_eval / bytes_total_eval if bytes_total_eval > 0 else float('inf')
        bpv_eval         = bytes_total_eval * 8 / (eDe * eHe * eWe)

        print(f"SVD-only  rel_err={svd_rel_err_eval:.6f}")
        print(f"Hybrid    rel_err={hybrid_rel_err_eval:.6f}  sparsity={sparsity_eval:.3f}  "
              f"active={active_nz_eval:.0f}/{n_code_eval}  "
              f"comp_ratio={comp_ratio_eval:.2f}x  BPV={bpv_eval:.2f}")
        print(f"Generalization gap (eval_hybrid_err − train_hybrid_err) = "
              f"{hybrid_rel_err_eval - hybrid_rel_err:+.6f}")

        eval_csv = os.path.join(out_dir, 'held_out_eval.csv')
        with open(eval_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'eval_timestep', 'train_timesteps', 'svd_k',
                'svd_rel_err', 'hybrid_rel_err', 'active_nz', 'sparsity',
                'comp_ratio', 'bpv', 'train_hybrid_rel_err',
            ])
            writer.writeheader()
            writer.writerow(dict(
                eval_timestep=eval_timestep, train_timesteps=str(train_timesteps), svd_k=k_lca,
                svd_rel_err=svd_rel_err_eval, hybrid_rel_err=hybrid_rel_err_eval,
                active_nz=active_nz_eval, sparsity=sparsity_eval,
                comp_ratio=comp_ratio_eval, bpv=bpv_eval,
                train_hybrid_rel_err=hybrid_rel_err,
            ))
        print(f"Saved {eval_csv}")

        if eval_shape == ref_shape:
            _plot_reconstruction_grid(
                plots_dir, 'reconstruction_eval.png',
                input_vol=input_vol_eval, svd_recon=svd_recon_eval,
                hybrid_recon=hybrid_recon_eval,
                title=(f'Hybrid SVD+LCA — HELD-OUT t={eval_timestep}  SVD k={k_lca}  '
                       f'SVD_err={svd_rel_err_eval:.4f}  hybrid_err={hybrid_rel_err_eval:.4f}  '
                       f'comp={comp_ratio_eval:.2f}x  BPV={bpv_eval:.2f}'),
            )

    print("\nDone.")
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    if using_ddp:
        cleanup_ddp()


if __name__ == '__main__':
    main()
