"""
LCA compression baseline — naked LCA (Rozell et al. 2008).

Implements the LCA dynamical system directly on flattened 3D patches:

    τ · du/dt = b - u - G·a       (membrane potential ODE)
    a         = T_λ(u)             (soft threshold)
    b         = Φᵀ s               (input drive)
    G         = Φᵀ Φ − I          (lateral inhibition)

where Φ is a (P³, M) dictionary, s is a flattened P³ patch, and a is the
M-dimensional sparse code.

No lcapt / LCAConv3D used — pure NumPy/PyTorch flat operations.
No dictionary learning — Φ is randomly initialised (unit-norm columns) or
loaded from a .npy checkpoint.

Sweeps lambda_ and reports the same metrics as svd_compression.py for
direct comparison (comp_coeff = P³ / avg_active, same COO formula, etc.).

Usage
-----
    python LCA.py config_simmldc.yaml
    python LCA.py config_simmldc.yaml --lambda-values 0.05 0.1 0.2 0.5
    python LCA.py config_simmldc.yaml --lambda-min 0.01 --lambda-max 2.0
    python LCA.py config_simmldc.yaml --atoms 256 --lca-iters 500
    python LCA.py config_simmldc.yaml --atoms-multiplier 4   # M = 4 × P³
    python LCA.py config_simmldc.yaml --svd-init             # Phi from SVD of X
    python LCA.py config_simmldc.yaml --patch-size 9         # override config patch_size
    python LCA.py config_simmldc.yaml --dict experiments/phi.npy
    python LCA.py config_simmldc.yaml --svd-bpv 3.5 --svd-rel-err 0.0098
    python LCA.py config_simmldc.yaml --output-dir results/lca_naked
"""

import argparse
import csv
import os
import shutil
import sys
import time
from datetime import datetime
from math import ceil, log2

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Patch utilities — mirrors svd_compression.py exactly
# ---------------------------------------------------------------------------

def extract_tiled_patches(vol: np.ndarray, patch_size: int):
    """
    Extract all non-overlapping patch_size³ tiles from a 3D volume.

    Tiles that do not fit are discarded (same as svd_compression.py).
    Per-patch normalisation: zero mean, unit variance.

    Returns
    -------
    X        : (n_patches, P³)  float32 — normalised, flattened patches
    means    : (n_patches,)
    stds     : (n_patches,)
    positions: list of (x, y, z) top-left corners
    grid     : (nD, nH, nW) tile counts per dimension
    """
    D, H, W = vol.shape
    P = patch_size
    nD, nH, nW = D // P, H // P, W // P
    positions = [
        (i*P, j*P, k*P)
        for i in range(nD)
        for j in range(nH)
        for k in range(nW)
    ]
    n     = len(positions)
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


def reconstruct_volume(
    recon_flat: np.ndarray,   # (n_patches, P³) — reconstructions in normalised space
    means: np.ndarray,
    stds: np.ndarray,
    positions: list,
    vol_shape: tuple,
    patch_size: int,
) -> np.ndarray:
    """Undo per-patch normalisation and paste tiles back — mirrors svd_compression.py."""
    D, H, W = vol_shape
    P = patch_size
    recon_vol = np.zeros((D, H, W), dtype=np.float32)
    for idx, (x, y, z) in enumerate(positions):
        patch = recon_flat[idx] * stds[idx] + means[idx]
        recon_vol[x:x+P, y:y+P, z:z+P] = patch.reshape(P, P, P)
    return recon_vol


def compute_metrics(
    input_vol: np.ndarray,
    recon_vol: np.ndarray,
    avg_active: float,   # mean non-zeros per patch — analog of k in SVD
    patch_size: int,
    M: int,              # number of dictionary atoms (index range for COO)
) -> dict:
    """
    Mirrors svd_compression.py compute_metrics.
    avg_active plays the role of k (average stored values per patch).
    """
    error     = input_vol - recon_vol
    mse       = float((error**2).mean())
    rmse      = float(np.sqrt(mse))
    rel_err   = float(rmse / (np.sqrt((input_vol**2).mean()) + 1e-8))
    sig_range = float(input_vol.max() - input_vol.min())
    psnr      = float(20 * np.log10(sig_range / (rmse + 1e-12)))

    P3 = patch_size**3

    # Coefficient-only  (P³/avg_active — same as SVD's comp_coeff = P³/k)
    comp_coeff = P3 / avg_active if avg_active > 0 else float('inf')
    bpv_coeff  = avg_active * 32 / P3       # bits per voxel, float32 values only

    # LCA-equivalent COO: index range = M (same formula as lca_sim_mldc_SingleSnaptshot.py)
    _index_bits   = int(ceil(log2(M + 1))) if M > 1 else 1
    _bytes_per_nz = 4 + (_index_bits + 7) // 8
    bytes_coo     = avg_active * _bytes_per_nz
    comp_lca_equiv = P3 * 4 / bytes_coo if bytes_coo > 0 else float('inf')
    bpv_lca_equiv  = bytes_coo * 8 / P3

    return dict(
        avg_active     = avg_active,
        rel_err        = rel_err,
        rmse           = rmse,
        psnr           = psnr,
        comp_coeff     = comp_coeff,
        bpv_coeff      = bpv_coeff,
        comp_lca_equiv = comp_lca_equiv,
        bpv_lca_equiv  = bpv_lca_equiv,
    )


# ---------------------------------------------------------------------------
# Dictionary initialisation
# ---------------------------------------------------------------------------

def init_dictionary(P3: int, M: int, seed: int = 42) -> np.ndarray:
    """Random Gaussian dictionary: (P³, M) unit-norm columns."""
    rng = np.random.default_rng(seed)
    Phi = rng.standard_normal((P3, M)).astype(np.float32)
    Phi /= np.linalg.norm(Phi, axis=0, keepdims=True) + 1e-12
    return Phi


def init_dictionary_svd(X: np.ndarray, M: int) -> np.ndarray:
    """
    Initialize Phi from the top singular vectors of the patch matrix X.

    Uses eigendecomposition of X.T @ X (P³×P³), which is efficient when
    P³ << n_patches.  With signed codes and lambda_=0 this gives reconstruction
    error equal to SVD truncated at min(M, P³) — the theoretical optimum.

    If M > P³, the extra atoms are random vectors orthogonalized against the
    SVD basis so they span the null space of X.
    """
    P3 = X.shape[1]
    k  = min(M, P3)

    XtX = (X.T.astype(np.float64) @ X.astype(np.float64)).astype(np.float32)
    _, V = np.linalg.eigh(XtX)          # ascending eigenvalues, columns = eigenvectors
    Phi  = V[:, -k:][:, ::-1].astype(np.float32)   # (P3, k), descending order

    if M > k:
        rng   = np.random.default_rng(42)
        extra = rng.standard_normal((P3, M - k)).astype(np.float32)
        extra -= Phi @ (Phi.T @ extra)               # orthogonalize vs SVD atoms
        extra /= np.linalg.norm(extra, axis=0, keepdims=True) + 1e-12
        Phi   = np.concatenate([Phi, extra], axis=1)

    return Phi


# ---------------------------------------------------------------------------
# Naked LCA  (Rozell et al. 2008, Section II)
# ---------------------------------------------------------------------------

def precompute_drives(
    X: np.ndarray,
    Phi_t: torch.Tensor,   # (P³, M) already on device
    batch_size: int,
) -> torch.Tensor:
    """
    Project all patches onto the dictionary: B = X @ Phi.

    B is independent of lambda_ — compute once and reuse across the sweep.
    Batched to keep GPU memory bounded for large P³.
    """
    n      = X.shape[0]
    device = Phi_t.device
    B_all  = torch.empty((n, Phi_t.shape[1]), dtype=torch.float32, device=device)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        B_all[start:end] = torch.from_numpy(X[start:end]).to(device) @ Phi_t
    return B_all


def lca_encode(
    B_all: torch.Tensor,   # (n_patches, M) — precomputed drives, already on device
    G_t: torch.Tensor,     # (M, M) — lateral inhibition, already on device
    lambda_: float,
    tau: float,
    lca_iters: int,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Run the LCA ODE from precomputed drives B.

        τ · Δu = B − U − A · G
        A = sign(U) · max(|U| − λ, 0)   [symmetric soft threshold]

    Signed threshold allows negative codes, matching SVD's representational
    capacity for zero-mean signals.  B = X @ Phi is precomputed once by
    precompute_drives() and reused across all lambda_ values.
    """
    n, M  = B_all.shape
    A_all = np.empty((n, M), dtype=np.float32)
    dt    = 1.0 / tau

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        B   = B_all[start:end]
        U   = torch.zeros_like(B)
        for _ in range(lca_iters):
            A = torch.sign(U) * torch.clamp(torch.abs(U) - lambda_, min=0.0)
            U = U + dt * (B - U - A @ G_t)
        A_all[start:end] = (
            torch.sign(U) * torch.clamp(torch.abs(U) - lambda_, min=0.0)
        ).cpu().numpy()

    return A_all


def lca_decode(A: np.ndarray, Phi: np.ndarray) -> np.ndarray:
    """Reconstruct flattened patches: x̂ = A @ Φᵀ  →  (n_patches, P³)."""
    return A @ Phi.T


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

#-----------------------------------------------------------------
# Usage: python LCA.py config_svd_lca.yaml --svd-init --atoms-multiplier 4 --patch-size 9 --lambda-max 2.0
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Naked LCA compression baseline')
    parser.add_argument('config', help='path to config_simmldc.yaml')
    parser.add_argument('--lambda-values', type=float, nargs='+', default=None,
                        help='explicit lambda_ values (default: auto log-sweep)')
    parser.add_argument('--lambda-min', type=float, default=0.01,
                        help='min lambda_ for auto sweep (default: 0.01)')
    parser.add_argument('--lambda-max', type=float, default=1.0,
                        help='max lambda_ for auto sweep (default: 2.0)')
    parser.add_argument('--n-lambda', type=int, default=20,
                        help='number of lambda_ values in auto sweep (default: 20)')
    parser.add_argument('--atoms', type=int, default=None,
                        help='number of dictionary atoms M (default: from config features)')
    parser.add_argument('--atoms-multiplier', type=int, default=None, metavar='K',
                        help='set M = K × P³ (e.g. 4 for 4× overcomplete); '
                             'ignored when --atoms is set')
    parser.add_argument('--patch-size', type=int, default=None,
                        help='override patch_size from config')
    parser.add_argument('--svd-init', action='store_true',
                        help='initialize Phi from SVD of the patch data '
                             '(optimal for lambda_=0; use with --svd-init for fair SVD comparison)')
    parser.add_argument('--lca-iters', type=int, default=None,
                        help='LCA ODE iterations (default: from config lca_iters)')
    parser.add_argument('--tau', type=float, default=None,
                        help='LCA time constant (default: from config tau)')
    parser.add_argument('--dict', default=None,
                        help='path to pre-computed dictionary .npy file, shape (P³, M)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='patch batch size for LCA inference (default: 256)')
    parser.add_argument('--svd-bpv', type=float, default=None,
                        help='SVD BPV result for comparison line on plots')
    parser.add_argument('--svd-rel-err', type=float, default=None,
                        help='SVD relative error for comparison line on plots')
    parser.add_argument('--output-dir', default=None,
                        help='output directory (default: experiments/lca_naked_TIMESTAMP)')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dcfg = cfg['data']
    mcfg = cfg['model']
    P    = args.patch_size or dcfg['patch_size']
    P3   = P**3
    if args.atoms:
        M = args.atoms
    elif args.atoms_multiplier:
        M = args.atoms_multiplier * P3
    else:
        M = mcfg['features']
    lca_iters = args.lca_iters or mcfg['lca_iters']
    tau       = args.tau      or mcfg['tau']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Output directory
    out_dir   = args.output_dir or os.path.join(
        'experiments', 'lca_naked_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
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

    if args.dict:
        dict_label = os.path.basename(args.dict)
    elif args.svd_init:
        dict_label = 'SVD init'
    else:
        dict_label = 'random init'
    print(f"Output dir  : {out_dir}")
    print(f"Config      : {args.config}")
    print(f"Patch size  : {P}³ = {P3:,} voxels")
    print(f"Atoms (M)   : {M}   (overcompleteness: {M/P3:.3f}x)")
    print(f"LCA         : iters={lca_iters}  tau={tau}  threshold=signed")
    print(f"Device      : {device}")
    print(f"Dictionary  : {dict_label}\n")

    # ------------------------------------------------------------------ #
    # Load volume  (same as svd_compression.py)
    # ------------------------------------------------------------------ #
    print(f"Loading {dcfg['h5_path']} — field={dcfg['field_key']} t={dcfg['timestep']} ...")
    with h5py.File(dcfg['h5_path'], 'r') as f:
        vol = f[dcfg['field_key']][dcfg['timestep']].astype(np.float32)

    vol = (vol - vol.mean()) / (vol.std() + 1e-8)   # global normalisation
    D, H, W = vol.shape
    print(f"Volume      : {D}×{H}×{W}\n")

    # ------------------------------------------------------------------ #
    # Extract non-overlapping patches  (same as svd_compression.py)
    # ------------------------------------------------------------------ #
    X, means, stds, positions, (nD, nH, nW) = extract_tiled_patches(vol, P)
    n_patches = len(positions)
    D_out, H_out, W_out = nD * P, nH * P, nW * P
    input_vol = vol[:D_out, :H_out, :W_out].copy()

    print(f"Tiles       : {nD}×{nH}×{nW} = {n_patches} patches "
          f"(covering {D_out}×{H_out}×{W_out} of {D}×{H}×{W})")
    print(f"Patch matrix X : {X.shape}  ({X.nbytes/1024/1024:.1f} MB)\n")

    # ------------------------------------------------------------------ #
    # Dictionary Φ  (P³, M)
    # ------------------------------------------------------------------ #
    if args.dict:
        Phi = np.load(args.dict).astype(np.float32)
        assert Phi.shape == (P3, M), \
            f"Dictionary shape mismatch: expected ({P3}, {M}), got {Phi.shape}"
        print(f"Loaded Φ from {args.dict}  shape={Phi.shape}\n")
    elif args.svd_init:
        print(f"Computing SVD of X ({X.shape}) to initialise Φ ...")
        t0  = time.time()
        Phi = init_dictionary_svd(X, M)
        phi_path = os.path.join(out_dir, 'phi_svd.npy')
        np.save(phi_path, Phi)
        print(f"  Φ shape={Phi.shape}  ({time.time()-t0:.1f}s)  saved → {phi_path}\n")
    else:
        Phi = init_dictionary(P3, M)
        phi_path = os.path.join(out_dir, 'phi_random.npy')
        np.save(phi_path, Phi)
        print(f"Initialised random Φ  shape={Phi.shape}  saved → {phi_path}\n")

    # Precompute Gram matrix G = ΦᵀΦ − I  (M, M)
    # This is the lateral inhibition kernel from Rozell et al. eq (9)
    print("Precomputing G = ΦᵀΦ − I ...")
    G = (Phi.T @ Phi).astype(np.float32)
    np.fill_diagonal(G, 0.0)   # equivalent to subtracting I
    print(f"G shape: {G.shape}  max off-diag: {np.abs(G).max():.4f}\n")

    # COO index cost (same formula as lca_sim_mldc_SingleSnaptshot.py, index range = M)
    _index_bits   = int(ceil(log2(M + 1))) if M > 1 else 1
    _bytes_per_nz = 4 + (_index_bits + 7) // 8
    print(f"Index bits  : {_index_bits}  →  {_bytes_per_nz} bytes/nz (COO, range M={M})\n")

    # Transfer Phi and G to GPU once — reused for all lambda values
    Phi_t = torch.from_numpy(Phi).to(device)
    G_t   = torch.from_numpy(G).to(device)

    # Precompute B = X @ Phi (input drives) once — independent of lambda_
    print("Precomputing B = X @ Φ  (once for all λ) ...")
    t0    = time.time()
    B_all = precompute_drives(X, Phi_t, args.batch_size)
    print(f"  shape={tuple(B_all.shape)}  "
          f"{B_all.numel()*4/1024/1024:.1f} MB on {device}  "
          f"({time.time()-t0:.1f}s)\n")

    # ------------------------------------------------------------------ #
    # Lambda sweep
    # ------------------------------------------------------------------ #
    if args.lambda_values:
        lambda_values = sorted(args.lambda_values)
    else:
        lambda_values = np.geomspace(args.lambda_min, args.lambda_max, args.n_lambda).tolist()

    if 0.0 not in lambda_values:
        lambda_values = [0.0] + lambda_values

    results = []
    print(f"{'lambda':>8}  {'avg_active':>12}  {'rel_err':>10}  {'PSNR(dB)':>10}  "
          f"{'Comp(coeff)':>13}  {'BPV(coeff)':>12}  "
          f"{'Comp(LCA-eq)':>14}  {'BPV(LCA-eq)':>13}  {'time(s)':>8}")
    print('-' * 120)

    for lam in lambda_values:
        t0 = time.time()
        A = lca_encode(B_all, G_t, lam, tau, lca_iters, args.batch_size)
        elapsed = time.time() - t0

        avg_active = float((A != 0).sum(axis=1).mean())

        if avg_active == 0:
            print(f"{lam:>8.4f}  {'0 (all silent)':>12}  — skipping")
            continue

        recon_flat = lca_decode(A, Phi)
        recon_vol  = reconstruct_volume(
            recon_flat, means, stds, positions, (D_out, H_out, W_out), P
        )
        m = compute_metrics(input_vol, recon_vol, avg_active, P, M)
        m['lambda_'] = lam
        results.append(m)

        print(f"{lam:>8.4f}  {avg_active:>12.1f}  {m['rel_err']:>10.6f}  {m['psnr']:>10.2f}  "
              f"{m['comp_coeff']:>13.2f}x  {m['bpv_coeff']:>12.3f}  "
              f"{m['comp_lca_equiv']:>14.2f}x  {m['bpv_lca_equiv']:>13.3f}  {elapsed:>8.1f}")

    print()

    if not results:
        print("No valid results — all lambda_ values silenced every neuron. "
              "Try --lambda-max with a smaller value.")
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        _log.close()
        return

    under_1pct = [r for r in results if r['rel_err'] <= 0.01]
    if under_1pct:
        best = max(under_1pct, key=lambda r: r['comp_coeff'])
        print(f"Best (rel_err ≤ 1%): λ={best['lambda_']:.4f}  "
              f"avg_active={best['avg_active']:.1f}  "
              f"rel_err={best['rel_err']:.4f}  "
              f"comp(coeff)={best['comp_coeff']:.2f}x  BPV(coeff)={best['bpv_coeff']:.3f}  "
              f"comp(LCA-eq)={best['comp_lca_equiv']:.2f}x  "
              f"BPV(LCA-eq)={best['bpv_lca_equiv']:.3f}")
    else:
        best = min(results, key=lambda r: r['rel_err'])
        print(f"Note: rel_err never reaches 1% — "
              f"best is λ={best['lambda_']:.4f}  avg_active={best['avg_active']:.1f}  "
              f"rel_err={best['rel_err']:.4f}  "
              f"comp(LCA-eq)={best['comp_lca_equiv']:.2f}x  "
              f"BPV(LCA-eq)={best['bpv_lca_equiv']:.3f}")

    lambdas        = [r['lambda_']        for r in results]
    avg_actives    = [r['avg_active']      for r in results]
    rel_errs       = [r['rel_err']         for r in results]
    bpv_coeffs     = [r['bpv_coeff']       for r in results]
    bpv_lca_equivs = [r['bpv_lca_equiv']   for r in results]

    # ------------------------------------------------------------------ #
    # Plot 1 — sparsity spectrum  (analog of SVD's singular value plot)
    # ------------------------------------------------------------------ #
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.semilogy(lambdas, avg_actives, 'o-', color='steelblue', markersize=4)
    ax1.set_xlabel('lambda_')
    ax1.set_ylabel('Avg active coefficients per patch (log)')
    ax1.set_title(f'LCA sparsity vs λ  (patch {P}³, {n_patches} patches, M={M})')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(best['lambda_'], color='red', linestyle='--', linewidth=0.8,
                label=f'best λ={best["lambda_"]:.3f}')
    ax1.legend(fontsize=8)

    ax2.semilogy(lambdas, rel_errs, 'o-', color='darkorange', markersize=4)
    ax2.axhline(0.01, color='red', linestyle=':', linewidth=1, label='1% target')
    if args.svd_rel_err is not None:
        ax2.axhline(args.svd_rel_err, color='purple', linestyle='--', linewidth=1,
                    label=f'SVD rel_err={args.svd_rel_err:.4f}')
    ax2.set_xlabel('lambda_')
    ax2.set_ylabel('Relative error (log)')
    ax2.set_title('Reconstruction error vs λ')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(plots_dir, 'lambda_sweep.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")

    # ------------------------------------------------------------------ #
    # Plot 2 — rel_err vs avg_active  (analog of SVD's Plot 2)
    # ------------------------------------------------------------------ #
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.semilogy(avg_actives, rel_errs, 'o-', color='steelblue', markersize=4,
                 label='rel_err')
    ax2.plot(avg_actives, [r['comp_coeff'] for r in results], 's--',
             color='darkorange', markersize=4, label='comp_coeff = P³/avg_active')

    ax1.axhline(0.01, color='red', linestyle=':', linewidth=1, label='1% target')
    if args.svd_rel_err is not None:
        ax1.axhline(args.svd_rel_err, color='purple', linestyle='--', linewidth=1,
                    label=f'SVD rel_err={args.svd_rel_err:.4f}')

    ax1.set_xlabel('Avg active coefficients per patch')
    ax1.set_ylabel('Relative reconstruction error (log)', color='steelblue')
    ax2.set_ylabel('Compression ratio (coeff only)', color='darkorange')
    ax1.set_title(f'LCA: quality vs sparsity  (patch {P}³, M={M}, {n_patches} patches)')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(plots_dir, 'rel_err_vs_active.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")

    # ------------------------------------------------------------------ #
    # Plot 3 — rate-distortion  (mirrors svd_compression.py Plot 3)
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(bpv_coeffs, rel_errs, 'o-', color='steelblue', markersize=5,
                label='LCA (coeff only, float32)')
    ax.semilogy(bpv_lca_equivs, rel_errs, '^:', color='darkorange', markersize=4,
                alpha=0.9, label='LCA (COO storage)')
    ax.axhline(0.01, color='red', linestyle=':', linewidth=1, label='1% error target')

    if args.svd_bpv is not None and args.svd_rel_err is not None:
        ax.scatter([args.svd_bpv], [args.svd_rel_err], marker='*', s=200,
                   color='green', zorder=5,
                   label=f'SVD ({args.svd_bpv:.2f} BPV, {args.svd_rel_err:.4f} err)')

    for r in results[::max(1, len(results)//8)]:
        ax.annotate(f"λ={r['lambda_']:.3f}", (r['bpv_coeff'], r['rel_err']),
                    textcoords='offset points', xytext=(4, 4), fontsize=7)

    ax.set_xlabel('Bits per voxel (BPV)')
    ax.set_ylabel('Relative reconstruction error (log scale)')
    ax.set_title(f'LCA Rate–Distortion  |  patch {P}³  |  M={M}  |  {dict_label}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(plots_dir, 'rate_distortion.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")

    # ------------------------------------------------------------------ #
    # Plot 4 — full-volume reconstruction at best lambda  (mirrors SVD Plot 4)
    # ------------------------------------------------------------------ #
    A_best     = lca_encode(B_all, G_t, best['lambda_'], tau, lca_iters, args.batch_size)
    recon_flat = lca_decode(A_best, Phi)
    recon_vol  = reconstruct_volume(
        recon_flat, means, stds, positions, (D_out, H_out, W_out), P
    )

    mD, mH, mW = D_out // 2, H_out // 2, W_out // 2
    plane_defs = [
        ('XY (z=mid)', input_vol[:, :, mW], recon_vol[:, :, mW]),
        ('XZ (y=mid)', input_vol[:, mH, :], recon_vol[:, mH, :]),
        ('YZ (x=mid)', input_vol[mD, :, :], recon_vol[mD, :, :]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(
        f'LCA Full-volume reconstruction  |  λ={best["lambda_"]:.4f}  '
        f'avg_active={best["avg_active"]:.1f}  '
        f'rel_err={best["rel_err"]:.4f}  '
        f'comp_coeff={best["comp_coeff"]:.1f}x  BPV={best["bpv_coeff"]:.3f}  '
        f'|  M={M}  {dict_label}',
        fontsize=9
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
    out = os.path.join(plots_dir, f'full_volume_lam{best["lambda_"]:.4f}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")

    # ------------------------------------------------------------------ #
    # Plot 5 — error maps  (mirrors svd_compression.py Plot 5)
    # ------------------------------------------------------------------ #
    error_vol = input_vol - recon_vol
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    plane_data = [
        ('XY', input_vol[:, :, mW],  error_vol[:, :, mW]),
        ('XZ', input_vol[:, mH, :],  error_vol[:, mH, :]),
        ('YZ', input_vol[mD, :, :],  error_vol[mD, :, :]),
    ]
    for ax, (lbl, inp_p, err_p) in zip(axes, plane_data):
        vmax = np.percentile(np.abs(inp_p), 99)
        ax.imshow(err_p, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                  origin='lower', aspect='equal')
        ax.set_title(f'LCA error  {lbl}', fontsize=9)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    svd_ref = '' if args.svd_rel_err is None else f'  vs SVD rel_err={args.svd_rel_err:.4f}'
    fig.suptitle(
        f'LCA error maps  |  λ={best["lambda_"]:.4f}  rel_err={best["rel_err"]:.4f}{svd_ref}',
        fontsize=9
    )
    plt.tight_layout()
    out = os.path.join(plots_dir, f'error_maps_lam{best["lambda_"]:.4f}.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")

    # ------------------------------------------------------------------ #
    # Save results CSV
    # ------------------------------------------------------------------ #
    csv_path = os.path.join(out_dir, 'lca_results.csv')
    with open(csv_path, 'w', newline='') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults table saved to {csv_path}")

    print(f"\n{'='*60}")
    print(f"LCA SUMMARY  (patch {P}³, {n_patches} tiles, t={dcfg['timestep']}, {dict_label})")
    print(f"{'='*60}")
    print(f"Atoms M            : {M}   (P³={P3}, overcomplete {M/P3:.3f}x)")
    print(f"LCA                : iters={lca_iters}  tau={tau}")
    print(f"Index bits (COO)   : {_index_bits}  →  {_bytes_per_nz} bytes/nz")
    if under_1pct:
        print(f"Best (rel_err ≤ 1%): λ={best['lambda_']:.4f}  "
              f"avg_active={best['avg_active']:.1f}  "
              f"comp(coeff)={best['comp_coeff']:.2f}x  BPV={best['bpv_coeff']:.3f}  "
              f"comp(LCA-eq)={best['comp_lca_equiv']:.2f}x  "
              f"BPV(LCA-eq)={best['bpv_lca_equiv']:.3f}")
    if args.svd_bpv:
        print(f"SVD reference      : BPV={args.svd_bpv:.3f}  rel_err={args.svd_rel_err}")

    print("\nDone.")
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    _log.close()


if __name__ == '__main__':
    main()
