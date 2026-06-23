"""
Hybrid SVD + Global LCA compression on a single 3D simulation snapshot.

Pipeline
--------
  1. Load one timestep from HDF5, globally z-score.
  2. Extract non-overlapping tile_size³ tiles → truncated SVD(k_svd).
  3. Reconstruct SVD volume; residual = vol_norm − svd_recon.
  4. Train LCAConv3D on the FULL residual volume — one forward pass per epoch,
     no patch extraction, no DataLoader (suggestion 5: get rid of patching).
  5. Report combined compression: SVD coefficients + LCA sparse COO on residual.

Compression metric (both components exclude dictionary / basis overhead):
  bytes_svd  = n_tiles × k_svd × 4                (dense float32)
  bytes_lca  = active_nz × bytes_per_nz            (COO: float32 value + flat index)
  comp_ratio = (D×H×W×4) / (bytes_svd + bytes_lca)

Usage:
  python lca_sim_mldc_Hybrid.py [config_hybrid.yaml]
"""

import os
import shutil
import sys
import time
from datetime import datetime
from math import ceil

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from lcapt.lca import LCAConv3D
from lcapt.metric import compute_l1_sparsity, compute_l2_error


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
# SVD helpers
# ---------------------------------------------------------------------------

def _tile_iter(n_d, n_h, n_w):
    for di in range(n_d):
        for hi in range(n_h):
            for wi in range(n_w):
                yield di, hi, wi


def compute_svd_recon(vol_norm: np.ndarray, T: int, k: int):
    """Tile vol_norm into T³ blocks, compute truncated SVD, return reconstruction.

    Each tile is per-tile z-scored for SVD (matching LCA's internal normalization),
    then un-normalized when stitching the reconstruction back to global-norm space.

    Returns
    -------
    svd_recon  : (D, H, W) float32  reconstruction in global-norm space
    coeffs     : (n_tiles, k) float32  SVD coefficients (stored at encode time)
    Vt         : (k, T³) float32  right singular vectors (basis, stored once)
    means, stds : (n_tiles,) per-tile normalization parameters
    grid       : (n_d, n_h, n_w)
    """
    D, H, W = vol_norm.shape
    n_d, n_h, n_w = D // T, H // T, W // T
    n_tiles = n_d * n_h * n_w
    assert n_tiles > 0, f"tile_size={T} too large for volume {D}×{H}×{W}"

    X     = np.empty((n_tiles, T * T * T), dtype=np.float32)
    means = np.empty(n_tiles, dtype=np.float32)
    stds  = np.empty(n_tiles, dtype=np.float32)

    for idx, (di, hi, wi) in enumerate(_tile_iter(n_d, n_h, n_w)):
        tile = vol_norm[di*T:(di+1)*T, hi*T:(hi+1)*T, wi*T:(wi+1)*T]
        m = float(tile.mean())
        s = float(tile.std()) + 1e-8
        X[idx]     = ((tile - m) / s).ravel()
        means[idx] = m
        stds[idx]  = s

    k = min(k, n_tiles)
    # thin SVD — n_tiles << T³ so numpy is fast and memory-efficient
    U, s_vals, Vt = np.linalg.svd(X, full_matrices=False)
    Vt     = Vt[:k].astype(np.float32)
    coeffs = (U[:, :k] * s_vals[:k]).astype(np.float32)  # (n_tiles, k)

    X_hat = coeffs @ Vt  # (n_tiles, T³) in per-tile-normed space

    svd_recon = np.zeros((D, H, W), dtype=np.float32)
    for idx, (di, hi, wi) in enumerate(_tile_iter(n_d, n_h, n_w)):
        tile_global = X_hat[idx].reshape(T, T, T) * stds[idx] + means[idx]
        svd_recon[di*T:(di+1)*T, hi*T:(hi+1)*T, wi*T:(wi+1)*T] = tile_global

    return svd_recon, coeffs, Vt, means, stds, (n_d, n_h, n_w)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ------------------------------------------------------------------ #
    # Config
    # ------------------------------------------------------------------ #
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'config_hybrid.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    dcfg = cfg['data']
    scfg = cfg['svd']
    mcfg = cfg['model']
    tcfg = cfg['training']
    ocfg = cfg['output']
    

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype_str = tcfg.get('dtype', 'float32')
    dtype     = {'float16': torch.float16, 'bfloat16': torch.bfloat16}.get(dtype_str, torch.float32)

    # ------------------------------------------------------------------ #
    # Experiment directory
    # ------------------------------------------------------------------ #
    exp_dir    = os.path.join('experiments', 'hybrid_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    plots_dir  = os.path.join(exp_dir, 'plots')
    models_dir = os.path.join(exp_dir, 'models')
    os.makedirs(plots_dir,  exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    shutil.copy(cfg_path, os.path.join(exp_dir, 'config_hybrid.yaml'))
    _log        = open(os.path.join(exp_dir, 'run.log'), 'w')
    sys.stdout  = _Tee(sys.__stdout__, _log)
    sys.stderr  = _Tee(sys.__stderr__, _log)

    print(f"Experiment dir : {exp_dir}")
    print(f"Config         : {cfg_path}")
    print(f"Device         : {device}  dtype={dtype}\n")

    # ------------------------------------------------------------------ #
    # Load volume — one timestep, globally z-scored
    # ------------------------------------------------------------------ #
    with h5py.File(dcfg['h5_path'], 'r') as f:
        vol_raw = f[dcfg['field_key']][dcfg['timestep']]   # (D, H, W)

    vol_raw  = vol_raw.astype(np.float32)
    vol_mean = float(vol_raw.mean())
    vol_std  = float(vol_raw.std()) + 1e-8
    vol_norm = (vol_raw - vol_mean) / vol_std               # (D, H, W)
    D, H, W  = vol_norm.shape
    bytes_in = D * H * W * 4                                # float32 bytes for whole volume

    print(f"Volume         : {D}×{H}×{W}  field={dcfg['field_key']}  t={dcfg['timestep']}")
    print(f"Global mean    : {vol_mean:.4f}  std={vol_std:.4f}\n")

    # ------------------------------------------------------------------ #
    # SVD — computed once on the globally-normed volume
    # ------------------------------------------------------------------ #
    T     = dcfg['tile_size']
    k_svd = scfg['k']

    print(f"Computing SVD  : tile_size={T}  k={k_svd} ...")
    svd_recon, coeffs, Vt, tile_means, tile_stds, (n_d, n_h, n_w) = \
        compute_svd_recon(vol_norm, T, k_svd)

    n_tiles  = n_d * n_h * n_w
    k_actual = coeffs.shape[1]   # capped at n_tiles if k_svd > n_tiles

    svd_rel_err = float(np.linalg.norm(vol_norm - svd_recon) / (np.linalg.norm(vol_norm) + 1e-8))
    bytes_svd   = n_tiles * k_actual * 4   # dense float32, no index overhead

    print(f"SVD tiles      : {n_d}×{n_h}×{n_w} = {n_tiles}  tile={T}³  k={k_actual}")
    print(f"SVD rel_err    : {svd_rel_err:.6f}")
    print(f"SVD bytes      : {bytes_svd}  ({bytes_svd/bytes_in:.5f}× raw)")
    print(f"SVD-only comp  : {bytes_in/bytes_svd:.2f}x  (dense coefficients, basis excluded)\n")

    # Fixed residual that LCA will encode throughout all training epochs
    residual_np = (vol_norm - svd_recon).astype(np.float32)   # (D, H, W)
    residual_tensor = (
        torch.from_numpy(residual_np)
        .unsqueeze(0).unsqueeze(0)          # (1, 1, D, H, W)
        .to(dtype=dtype, device=device)
    )

    # Stats for undoing LCA's internal zero-mean / unit-var normalization
    res_mean = float(residual_tensor.mean())
    res_std  = float(residual_tensor.std()) + 1e-8

    # ------------------------------------------------------------------ #
    # LCA model — trained on full residual volume (no patching)
    # ------------------------------------------------------------------ #
    lca = LCAConv3D(
        out_neurons   = mcfg['features'],
        in_neurons    = mcfg['in_channels'],
        result_dir    = os.path.join(exp_dir, 'lca_results'),
        kernel_size   = mcfg['kernel_size'],
        stride        = mcfg['stride'],
        lambda_       = mcfg['lambda_'],
        tau           = mcfg['tau'],
        lca_iters     = mcfg['lca_iters'],
        eta           = mcfg['learning_rate'],
        track_metrics = False,
        return_vars   = ['inputs', 'acts', 'recons', 'recon_errors'],
    ).to(dtype=dtype, device=device)

    k = mcfg['kernel_size']
    print(f"LCAConv3D      : {mcfg['features']} atoms | kernel {k}³ | stride={mcfg['stride']} | λ={mcfg['lambda_']}")

    # ------------------------------------------------------------------ #
    # Compression constants — whole-volume COO sparse storage
    # ------------------------------------------------------------------ #
    stride       = mcfg['stride']
    D_out        = ceil(D / stride)
    H_out        = ceil(H / stride)
    W_out        = ceil(W / stride)
    n_code_total = mcfg['features'] * D_out * H_out * W_out
    _index_bits  = int(np.ceil(np.log2(n_code_total + 1)))
    _bytes_per_nz = 4 + (_index_bits + 7) // 8   # float32 value + packed flat index

    print(f"Code size      : {mcfg['features']} × {D_out}×{H_out}×{W_out} = {n_code_total} positions")
    print(f"Index bits     : {_index_bits}  →  bytes_per_nz={_bytes_per_nz}\n")

    # ------------------------------------------------------------------ #
    # Training state machine (same rel_err-gated logic as SingleSnapshot)
    # ------------------------------------------------------------------ #
    max_epochs        = tcfg['max_epochs']
    n_passes_per_epoch = tcfg.get('n_passes_per_epoch', 1)
    anneal_every      = tcfg['lambda_anneal_every']
    anneal_step       = tcfg['lambda_anneal_step']
    anneal_start      = tcfg.get('lambda_anneal_start', 0)
    anneal_stop       = tcfg.get('lambda_anneal_stop', max_epochs)
    rel_err_target    = tcfg.get('rel_err_target', None)
    rel_err_ceiling   = tcfg.get('rel_err_ceiling', 0.01)
    stabilize_epochs  = tcfg.get('stabilize_epochs', 10)

    print(f"Training       : {max_epochs} epochs × {n_passes_per_epoch} passes/epoch")
    if rel_err_target is not None:
        print(f"Annealing      : rel_err-gated  target={rel_err_target}  ceiling={rel_err_ceiling}")
    else:
        print(f"Annealing      : time-based  start={anneal_start}  stop={anneal_stop}  "
              f"step={anneal_step}  every={anneal_every}")
    print()

    mode            = 'pre' if rel_err_target is not None else 'anneal'
    stab_count      = 0
    anneal_epoch    = 0
    last_rel_err    = 0.0
    best_comp_ratio = 0.0

    all_hybrid_rel_err  = []
    all_lca_res_rel_err = []

    # Variables that persist to post-training section
    lca_recon_np    = np.zeros_like(vol_norm)
    hybrid_recon_np = np.zeros_like(vol_norm)
    comp_ratio      = 0.0
    bpv             = 0.0
    bytes_lca       = 0.0
    sparsity        = 0.0
    active_nz       = 0.0
    hybrid_rel_err  = float('nan')
    l1_cost = l2_cost = 0.0

    for epoch in range(max_epochs):
        t0 = time.time()

        # ---- annealing step at epoch start ----
        if rel_err_target is not None:
            if mode == 'anneal' and anneal_epoch % anneal_every == 0:
                if last_rel_err <= rel_err_ceiling:
                    lca.lambda_ += anneal_step
                    print(f"  [anneal] λ → {lca.lambda_:.3f}  (anneal epoch {anneal_epoch}/{anneal_stop})")
                else:
                    print(f"  [anneal] λ increment skipped — "
                          f"hybrid_rel_err={last_rel_err:.4f} > ceiling={rel_err_ceiling}")
        else:
            if epoch > 0 and anneal_start <= epoch < anneal_stop \
                    and (epoch - anneal_start) % anneal_every == 0:
                lca.lambda_ += anneal_step
                print(f"  [anneal] λ → {lca.lambda_:.3f}")

        # ---- n_passes_per_epoch forward+Hebbian passes on the fixed residual ----
        for _ in range(n_passes_per_epoch):
            inputs_norm, code, recon_norm, recon_error_norm = lca(residual_tensor)
            lca.update_weights(code, recon_error_norm)
        # metrics below use the last pass values

        # ---- metrics ----
        l1_cost  = compute_l1_sparsity(code, lca.lambda_).item()
        l2_cost  = compute_l2_error(inputs_norm, recon_norm).item()

        active_nz = float((code != 0).float().sum().item())
        sparsity  = 1.0 - active_nz / n_code_total

        # Undo LCA's internal normalization to get recon in global-norm space
        lca_recon_np    = recon_norm[0, 0].float().cpu().numpy() * res_std + res_mean

        # Hybrid (SVD + LCA) reconstruction
        hybrid_recon_np = svd_recon + lca_recon_np

        # Reconstruction quality
        lca_res_rel_err = float(
            np.linalg.norm(lca_recon_np - residual_np) /
            (np.linalg.norm(residual_np) + 1e-8)
        )
        hybrid_rel_err = float(
            np.linalg.norm(hybrid_recon_np - vol_norm) /
            (np.linalg.norm(vol_norm) + 1e-8)
        )

        # Compression
        bytes_lca   = active_nz * _bytes_per_nz
        bytes_total = bytes_svd + bytes_lca
        comp_ratio  = bytes_in / bytes_total if bytes_total > 0 else float('inf')
        bpv         = bytes_total * 8 / (D * H * W)

        all_hybrid_rel_err.append(hybrid_rel_err)
        all_lca_res_rel_err.append(lca_res_rel_err)

        epoch_time = time.time() - t0
        if mode == 'stabilize':
            mode_tag = f"  [stabilize {stab_count}/{stabilize_epochs}]"
        elif mode == 'anneal':
            mode_tag = f"  [anneal ep {anneal_epoch}/{anneal_stop}]"
        else:
            mode_tag = "  [pre-anneal]"

        print(
            f"Epoch {epoch:03d} | {epoch_time:.1f}s ({n_passes_per_epoch}p) | "
            f"Sparsity={sparsity:.3f}  Active={active_nz:.0f}/{n_code_total}  "
            f"LCA_res_err={lca_res_rel_err:.6f}  Hybrid_err={hybrid_rel_err:.6f}  "
            f"L2={l2_cost:.4f}  L1={l1_cost:.4f}  λ={lca.lambda_:.3f}  "
            f"comp_ratio={comp_ratio:.2f}x  BPV={bpv:.2f}"
            + mode_tag
        )

        torch.save(lca.state_dict(), os.path.join(models_dir, 'lca_hybrid.pth'))

        if hybrid_rel_err <= rel_err_ceiling and comp_ratio > best_comp_ratio:
            best_comp_ratio = comp_ratio
            torch.save(lca.state_dict(), os.path.join(models_dir, 'lca_hybrid_best_compression.pth'))
            print(f"  [best] comp_ratio={best_comp_ratio:.2f}x  hybrid_rel_err={hybrid_rel_err:.6f}")

        last_rel_err = hybrid_rel_err

        # ---- state machine ----
        if rel_err_target is not None:
            if mode in ('pre', 'anneal') and hybrid_rel_err <= rel_err_target:
                prev_mode  = mode
                mode       = 'stabilize'
                stab_count = 0
                reason = 'before first anneal' if prev_mode == 'pre' else 'mid-anneal'
                print(f"  [stabilize] hybrid_rel_err={hybrid_rel_err:.6f} <= {rel_err_target} — "
                      f"freezing λ for {stabilize_epochs} epochs ({reason})")
            elif mode == 'stabilize':
                stab_count += 1
                if stab_count >= stabilize_epochs:
                    if anneal_epoch >= anneal_stop:
                        print("  [done] annealing complete + stabilized — stopping training")
                        break
                    mode       = 'anneal'
                    stab_count = 0
                    verb = 'Starting' if anneal_epoch == 0 else 'Resuming'
                    print(f"  [stabilize] done — {verb} λ annealing")

        if mode == 'anneal':
            anneal_epoch += 1

    # ------------------------------------------------------------------ #
    # Save SVD basis for future encode/decode
    # ------------------------------------------------------------------ #
    np.savez_compressed(
        os.path.join(models_dir, 'svd_basis.npz'),
        Vt=Vt, coeffs=coeffs,
        tile_means=tile_means, tile_stds=tile_stds,
        vol_mean=np.float32(vol_mean), vol_std=np.float32(vol_std),
        n_d=np.int32(n_d), n_h=np.int32(n_h), n_w=np.int32(n_w),
        T=np.int32(T), k=np.int32(k_actual),
    )
    print(f"\nSaved SVD basis → {os.path.join(models_dir, 'svd_basis.npz')}")
    print(f"Saved LCA model → {os.path.join(models_dir, 'lca_hybrid.pth')}")

    # ------------------------------------------------------------------ #
    # Post-training plots
    # ------------------------------------------------------------------ #
    print("\nGenerating plots...")

    # --- Error / compression curves ---
    fig, axes_p = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes_p[0].plot(all_lca_res_rel_err, label='LCA residual rel_err')
    axes_p[0].set_ylabel('LCA residual rel_err')
    axes_p[0].legend(fontsize=8)
    axes_p[1].plot(all_hybrid_rel_err, label='Hybrid (SVD+LCA) rel_err')
    if rel_err_target is not None:
        axes_p[1].axhline(rel_err_target, color='r', linestyle='--', linewidth=0.8,
                          label=f'target={rel_err_target}')
    axes_p[1].axhline(svd_rel_err, color='g', linestyle=':', linewidth=0.8,
                      label=f'SVD-only rel_err={svd_rel_err:.4f}')
    axes_p[1].set_ylabel('Hybrid rel_err')
    axes_p[1].set_xlabel('Epoch')
    axes_p[1].legend(fontsize=8)
    axes_p[0].set_title('Hybrid SVD+LCA — Training Curves')
    plt.tight_layout()
    out = os.path.join(plots_dir, 'training_metrics.png')
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")

    # --- Dictionary atoms (mid-plane slice of each 3D kernel) ---
    weights = lca.get_weights().float().cpu().numpy()   # (features, 1, kD, kH, kW)
    n_feat  = weights.shape[0]
    kD      = weights.shape[2]
    atoms   = weights[:, 0, kD // 2, :, :]             # (features, kH, kW)
    cols    = int(np.ceil(np.sqrt(n_feat)))
    rows    = int(np.ceil(n_feat / cols))
    fig, ax_atoms = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
    ax_atoms = np.array(ax_atoms).ravel()
    vmax = np.percentile(np.abs(atoms), 99)
    for i, ax in enumerate(ax_atoms):
        if i < n_feat:
            ax.imshow(atoms[i], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.axis('off')
    fig.suptitle(f'Dictionary atoms — mid-plane slice  ({n_feat} atoms, kernel {k}³)', fontsize=10)
    plt.tight_layout()
    out = os.path.join(plots_dir, 'dictionary_atoms.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")

    # --- 4-row reconstruction plot (three orthogonal mid-plane slices) ---
    mid_d, mid_h, mid_w = D // 2, H // 2, W // 2

    slicers = [
        ('XY (z=mid)', lambda a: a[mid_d]),
        ('XZ (y=mid)', lambda a: a[:, mid_h, :]),
        ('YZ (x=mid)', lambda a: a[:, :, mid_w]),
    ]
    rows_data = [
        ('Input (vol_norm)',    vol_norm),
        ('SVD recon',          svd_recon),
        ('LCA residual recon', lca_recon_np),
        ('Hybrid (SVD+LCA)',   hybrid_recon_np),
    ]

    fig, ax_grid = plt.subplots(4, 3, figsize=(9, 12))
    for ri, (row_label, arr) in enumerate(rows_data):
        for ci, (plane_label, slicer) in enumerate(slicers):
            ax = ax_grid[ri, ci]
            sl = slicer(arr)
            vmax_sl = np.percentile(np.abs(sl), 99)
            ax.imshow(sl, cmap='RdBu_r', vmin=-vmax_sl, vmax=vmax_sl)
            ax.axis('off')
            if ri == 0:
                ax.set_title(plane_label, fontsize=8)
            if ci == 0:
                ax.set_ylabel(row_label, fontsize=7)

    fig.suptitle(
        f'Hybrid SVD+LCA  |  t={dcfg["timestep"]}  '
        f'hybrid_rel_err={hybrid_rel_err:.4f}  '
        f'comp_ratio={comp_ratio:.2f}x  BPV={bpv:.2f}',
        fontsize=9
    )
    plt.tight_layout()
    out = os.path.join(plots_dir, 'reconstruction_4row.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")

    # ------------------------------------------------------------------ #
    # Final summary
    # ------------------------------------------------------------------ #
    print(f"\n{'='*65}")
    print(f"SVD : k={k_actual}  tiles={n_tiles}  SVD_rel_err={svd_rel_err:.6f}  bytes={bytes_svd}")
    print(f"LCA : {mcfg['features']} atoms  stride={mcfg['stride']}  λ={lca.lambda_:.3f}")
    print(f"      Sparsity={sparsity:.3f}  Active={active_nz:.0f}/{n_code_total}  bytes={bytes_lca:.0f}")
    print(f"Combined : bytes_in={bytes_in}  total_coded={bytes_svd + bytes_lca:.0f}")
    print(f"           comp_ratio={comp_ratio:.2f}x  BPV={bpv:.2f}")
    print(f"           hybrid_rel_err={hybrid_rel_err:.6f}")
    print(f"{'='*65}")
    print("\nDone.")

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    _log.close()


if __name__ == '__main__':
    main()
