"""
Multi-GPU DDP version of LCA.py — same sweep, parallelised across GPUs.

Phi is fixed (random or SVD-init), so DDP is pure data parallelism:
  - each rank encodes its own patch slice (the expensive step)
  - reconstructed volumes are all_reduced (SUM over non-overlapping patches)
  - avg_active is all_reduced; rank 0 computes metrics and plots

Expected speedup: ~N× on the lca_encode step (dominant cost).

Usage
-----
    # 4 GPUs:
    torchrun --nproc_per_node=4 lca_ddp.py config.yaml [same flags as LCA.py]

    # Single GPU (identical to LCA.py):
    python lca_ddp.py config.yaml [flags]
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
import torch.distributed as dist
import yaml

from LCA import (
    extract_tiled_patches,
    compute_metrics,
    init_dictionary,
    init_dictionary_svd,
    precompute_drives,
    lca_encode,
)


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


class _Tee:
    def __init__(self, *files): self.files = files
    def write(self, obj):
        for f in self.files: f.write(obj); f.flush()
    def flush(self):
        for f in self.files: f.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
# Usage: # 4 GPUs, SVD init, 4× overcomplete:
# torchrun --nproc_per_node=4 lca_ddp.py config_svd_lca.yaml --svd-init --atoms-multiplier 4 --patch-size 9


def main():
    # ------------------------------------------------------------------ #
    # DDP
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

    # ------------------------------------------------------------------ #
    # Args  (mirrors LCA.py exactly — drop-in replacement)
    # ------------------------------------------------------------------ #
    parser = argparse.ArgumentParser(description='LCA compression — multi-GPU (DDP)')
    parser.add_argument('config', help='path to config YAML')
    parser.add_argument('--lambda-values', type=float, nargs='+', default=None,
                        help='explicit lambda_ values (default: auto log-sweep)')
    parser.add_argument('--lambda-min',  type=float, default=0.01)
    parser.add_argument('--lambda-max',  type=float, default=1.0)
    parser.add_argument('--n-lambda',    type=int,   default=20)
    parser.add_argument('--atoms',       type=int,   default=None,
                        help='number of dictionary atoms M (default: from config features)')
    parser.add_argument('--atoms-multiplier', type=int, default=None, metavar='K',
                        help='set M = K × P³; ignored when --atoms is set')
    parser.add_argument('--patch-size',  type=int,   default=None,
                        help='override patch_size from config')
    parser.add_argument('--svd-init',    action='store_true',
                        help='initialize Phi from SVD of X (rank 0 computes, broadcasts)')
    parser.add_argument('--lca-iters',   type=int,   default=None)
    parser.add_argument('--tau',         type=float, default=None)
    parser.add_argument('--dict',        default=None,
                        help='path to pre-computed Phi .npy file')
    parser.add_argument('--batch-size',  type=int,   default=256,
                        help='patch batch size per rank for LCA inference')
    parser.add_argument('--svd-bpv',     type=float, default=None)
    parser.add_argument('--svd-rel-err', type=float, default=None)
    parser.add_argument('--output-dir',  default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dcfg = cfg['data']
    mcfg = cfg['model']
    P  = args.patch_size or dcfg['patch_size']
    P3 = P**3
    if args.atoms:
        M = args.atoms
    elif args.atoms_multiplier:
        M = args.atoms_multiplier * P3
    else:
        M = mcfg['features']
    lca_iters = args.lca_iters or mcfg['lca_iters']
    tau       = args.tau      or mcfg['tau']

    # ------------------------------------------------------------------ #
    # Experiment directory  (rank 0 generates name, broadcasts)
    # ------------------------------------------------------------------ #
    if is_main:
        base = args.output_dir or os.path.join(
            'experiments', 'lca_ddp_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        )
    else:
        base = None

    if using_ddp:
        container = [base]
        dist.broadcast_object_list(container, src=0)
        base = container[0]

    out_dir   = base
    plots_dir = os.path.join(out_dir, 'plots')

    if is_main:
        os.makedirs(plots_dir, exist_ok=True)
        shutil.copy(args.config, os.path.join(out_dir, 'config.yaml'))
        _log       = open(os.path.join(out_dir, 'run.log'), 'w')
        sys.stdout = _Tee(sys.__stdout__, _log)
        sys.stderr = _Tee(sys.__stderr__, _log)

    if using_ddp:
        dist.barrier()   # wait for rank 0 to finish creating dirs

    # ------------------------------------------------------------------ #
    # Volume  (all ranks load independently — read-only, fast)
    # ------------------------------------------------------------------ #
    with h5py.File(dcfg['h5_path'], 'r') as f:
        vol = f[dcfg['field_key']][dcfg['timestep']].astype(np.float32)
    vol = (vol - vol.mean()) / (vol.std() + 1e-8)
    D, H, W = vol.shape

    # ------------------------------------------------------------------ #
    # Patches  (all ranks extract all; each takes its contiguous slice)
    # ------------------------------------------------------------------ #
    X, means, stds, positions, (nD, nH, nW) = extract_tiled_patches(vol, P)
    n_patches              = len(positions)
    D_out, H_out, W_out    = nD * P, nH * P, nW * P
    input_vol              = vol[:D_out, :H_out, :W_out].copy()

    rank_slices = np.array_split(np.arange(n_patches), world_size)
    my_idx      = rank_slices[rank]
    X_local     = X[my_idx]
    pos_local   = [positions[i] for i in my_idx]
    means_local = means[my_idx]
    stds_local  = stds[my_idx]
    n_local     = len(my_idx)

    if is_main:
        dict_label = ('SVD init'                    if args.svd_init else
                      os.path.basename(args.dict)   if args.dict     else
                      'random init')
        print(f"Output dir  : {out_dir}")
        print(f"Config      : {args.config}")
        print(f"GPUs        : {world_size}")
        print(f"Patch size  : {P}³ = {P3:,} voxels")
        print(f"Atoms (M)   : {M}   ({M/P3:.2f}× overcomplete)")
        print(f"LCA         : iters={lca_iters}  tau={tau}  threshold=signed")
        print(f"Dictionary  : {dict_label}")
        print(f"Volume      : {D}×{H}×{W}")
        print(f"Tiles       : {nD}×{nH}×{nW} = {n_patches}  "
              f"(~{n_local} patches/rank)\n")

    # ------------------------------------------------------------------ #
    # Dictionary Φ  (rank 0 computes or loads, broadcasts Phi_t)
    # ------------------------------------------------------------------ #
    Phi_t = torch.empty((P3, M), dtype=torch.float32, device=device)

    if is_main:
        if args.dict:
            Phi_np = np.load(args.dict).astype(np.float32)
            assert Phi_np.shape == (P3, M), \
                f"Dict shape mismatch: expected ({P3},{M}), got {Phi_np.shape}"
            print(f"Loaded Φ from {args.dict}\n")
        elif args.svd_init:
            t0 = time.time()
            print(f"Computing SVD of X ({X.shape}) on rank 0 ...")
            Phi_np = init_dictionary_svd(X, M)
            np.save(os.path.join(out_dir, 'phi_svd.npy'), Phi_np)
            print(f"  done in {time.time()-t0:.1f}s  → phi_svd.npy\n")
        else:
            Phi_np = init_dictionary(P3, M)
            np.save(os.path.join(out_dir, 'phi_random.npy'), Phi_np)
            print(f"Initialised random Φ  shape={Phi_np.shape}  → phi_random.npy\n")
        Phi_t.copy_(torch.from_numpy(Phi_np))

    if using_ddp:
        dist.broadcast(Phi_t, src=0)

    Phi = Phi_t.cpu().numpy()   # all ranks have the same Phi after broadcast

    # ------------------------------------------------------------------ #
    # G = ΦᵀΦ − I  (all ranks compute from same Phi — no sync needed)
    # ------------------------------------------------------------------ #
    G = (Phi.T @ Phi).astype(np.float32)
    np.fill_diagonal(G, 0.0)
    G_t = torch.from_numpy(G).to(device)

    _index_bits   = int(ceil(log2(M + 1))) if M > 1 else 1
    _bytes_per_nz = 4 + (_index_bits + 7) // 8
    if is_main:
        print(f"G max off-diag : {np.abs(G).max():.4f}")
        print(f"Index bits     : {_index_bits}  →  {_bytes_per_nz} bytes/nz (COO)\n")

    # ------------------------------------------------------------------ #
    # Precompute B_local = X_local @ Φ  (one projection per rank)
    # ------------------------------------------------------------------ #
    if using_ddp:
        dist.barrier()
    t0      = time.time()
    B_local = precompute_drives(X_local, Phi_t, args.batch_size)
    if using_ddp:
        dist.barrier()
    if is_main:
        print(f"B_local precomputed  shape={tuple(B_local.shape)}  "
              f"({time.time()-t0:.1f}s across all ranks)\n")

    # ------------------------------------------------------------------ #
    # Lambda sweep
    # ------------------------------------------------------------------ #
    if args.lambda_values:
        lambda_values = sorted(args.lambda_values)
    else:
        lambda_values = np.geomspace(args.lambda_min, args.lambda_max,
                                     args.n_lambda).tolist()
    if 0.0 not in lambda_values:
        lambda_values = [0.0] + lambda_values

    results = []

    if is_main:
        print(f"{'lambda':>8}  {'avg_active':>12}  {'rel_err':>10}  {'PSNR(dB)':>10}  "
              f"{'Comp(coeff)':>13}  {'BPV(coeff)':>12}  "
              f"{'Comp(LCA-eq)':>14}  {'BPV(LCA-eq)':>13}  {'time(s)':>8}")
        print('-' * 120)

    for lam in lambda_values:
        if using_ddp:
            dist.barrier()
        t0 = time.time()

        # Each rank encodes its patch slice
        A_local = lca_encode(B_local, G_t, lam, tau, lca_iters, args.batch_size)

        # Decode on GPU: (n_local, M) @ (M, P3) — stays on device
        A_t          = torch.from_numpy(A_local).to(device)
        recon_flat_t = A_t @ Phi_t.T                           # (n_local, P3)
        recon_np     = recon_flat_t.cpu().numpy()

        # Fill local portion of the output volume (CPU loop, fast for tiled patches)
        recon_vol_local = np.zeros((D_out, H_out, W_out), dtype=np.float32)
        for i, (x, y, z) in enumerate(pos_local):
            patch = recon_np[i] * stds_local[i] + means_local[i]
            recon_vol_local[x:x+P, y:y+P, z:z+P] = patch.reshape(P, P, P)

        # All_reduce: SUM combines non-overlapping patch portions across all ranks
        recon_t = torch.from_numpy(recon_vol_local).to(device)
        if using_ddp:
            dist.all_reduce(recon_t, op=dist.ReduceOp.SUM)

        # All_reduce: total active count → average per patch
        active_t = torch.tensor(float((A_local != 0).sum()), device=device)
        if using_ddp:
            dist.all_reduce(active_t, op=dist.ReduceOp.SUM)
        avg_active = (active_t / n_patches).item()

        elapsed = time.time() - t0

        if avg_active == 0:
            if is_main:
                print(f"{lam:>8.4f}  {'0 (all silent)':>12}  — skipping")
            continue

        if is_main:
            recon_vol = recon_t.cpu().numpy()
            m = compute_metrics(input_vol, recon_vol, avg_active, P, M)
            m['lambda_'] = lam
            results.append(m)
            print(f"{lam:>8.4f}  {avg_active:>12.1f}  {m['rel_err']:>10.6f}  "
                  f"{m['psnr']:>10.2f}  "
                  f"{m['comp_coeff']:>13.2f}x  {m['bpv_coeff']:>12.3f}  "
                  f"{m['comp_lca_equiv']:>14.2f}x  {m['bpv_lca_equiv']:>13.3f}  "
                  f"{elapsed:>8.1f}")

    if is_main:
        print()

    # ------------------------------------------------------------------ #
    # Best lambda: rank 0 decides, broadcasts so all ranks re-encode
    # ------------------------------------------------------------------ #
    if is_main:
        if not results:
            print("No valid results — all lambdas silenced every neuron.")
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            _log.close()
            if using_ddp:
                cleanup_ddp()
            return

        under_1pct = [r for r in results if r['rel_err'] <= 0.01]
        if under_1pct:
            best     = max(under_1pct, key=lambda r: r['comp_coeff'])
            best_tag = f"Best (rel_err ≤ 1%): λ={best['lambda_']:.4f}  " \
                       f"avg_active={best['avg_active']:.1f}  " \
                       f"rel_err={best['rel_err']:.4f}  " \
                       f"comp(coeff)={best['comp_coeff']:.2f}x  " \
                       f"BPV(coeff)={best['bpv_coeff']:.3f}  " \
                       f"comp(LCA-eq)={best['comp_lca_equiv']:.2f}x  " \
                       f"BPV(LCA-eq)={best['bpv_lca_equiv']:.3f}"
        else:
            best     = min(results, key=lambda r: r['rel_err'])
            best_tag = (f"Note: rel_err never ≤ 1% — "
                        f"best λ={best['lambda_']:.4f}  "
                        f"avg_active={best['avg_active']:.1f}  "
                        f"rel_err={best['rel_err']:.4f}  "
                        f"comp(LCA-eq)={best['comp_lca_equiv']:.2f}x")
        print(best_tag)
        best_lam = best['lambda_']
    else:
        best_lam = None

    if using_ddp:
        container = [best_lam]
        dist.broadcast_object_list(container, src=0)
        best_lam = container[0]

    # ------------------------------------------------------------------ #
    # Re-encode at best lambda (all ranks) — needed for plots
    # ------------------------------------------------------------------ #
    if using_ddp:
        dist.barrier()

    A_best       = lca_encode(B_local, G_t, best_lam, tau, lca_iters, args.batch_size)
    A_t          = torch.from_numpy(A_best).to(device)
    recon_flat_t = A_t @ Phi_t.T
    recon_np     = recon_flat_t.cpu().numpy()

    recon_vol_local = np.zeros((D_out, H_out, W_out), dtype=np.float32)
    for i, (x, y, z) in enumerate(pos_local):
        patch = recon_np[i] * stds_local[i] + means_local[i]
        recon_vol_local[x:x+P, y:y+P, z:z+P] = patch.reshape(P, P, P)

    recon_t = torch.from_numpy(recon_vol_local).to(device)
    if using_ddp:
        dist.all_reduce(recon_t, op=dist.ReduceOp.SUM)

    # ------------------------------------------------------------------ #
    # Rank 0 only: plots, CSV, summary
    # ------------------------------------------------------------------ #
    if is_main:
        best_recon_vol = recon_t.cpu().numpy()

        lambdas        = [r['lambda_']      for r in results]
        avg_actives    = [r['avg_active']    for r in results]
        rel_errs       = [r['rel_err']       for r in results]
        bpv_coeffs     = [r['bpv_coeff']     for r in results]
        bpv_lca_equivs = [r['bpv_lca_equiv'] for r in results]

        # Plot 1 — lambda sweep
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.semilogy(lambdas, avg_actives, 'o-', color='steelblue', markersize=4)
        ax1.set_xlabel('lambda_')
        ax1.set_ylabel('Avg active coefficients per patch')
        ax1.set_title(f'LCA sparsity vs λ  (patch {P}³, {n_patches} patches, M={M})')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(best_lam, color='red', linestyle='--', linewidth=0.8,
                    label=f'best λ={best_lam:.3f}')
        ax1.legend(fontsize=8)
        ax2.semilogy(lambdas, rel_errs, 'o-', color='darkorange', markersize=4)
        ax2.axhline(0.01, color='red', linestyle=':', linewidth=1, label='1% target')
        if args.svd_rel_err is not None:
            ax2.axhline(args.svd_rel_err, color='purple', linestyle='--', linewidth=1,
                        label=f'SVD rel_err={args.svd_rel_err:.4f}')
        ax2.set_xlabel('lambda_')
        ax2.set_ylabel('Relative error')
        ax2.set_title('Reconstruction error vs λ')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        out = os.path.join(plots_dir, 'lambda_sweep.png')
        plt.savefig(out, dpi=150); plt.close(); print(f"Saved {out}")

        # Plot 2 — rel_err vs avg_active
        fig, ax1p = plt.subplots(figsize=(10, 5))
        ax2p = ax1p.twinx()
        ax1p.semilogy(avg_actives, rel_errs, 'o-', color='steelblue', markersize=4,
                      label='rel_err')
        ax2p.plot(avg_actives, [r['comp_coeff'] for r in results], 's--',
                  color='darkorange', markersize=4, label='comp_coeff = P³/avg_active')
        ax1p.axhline(0.01, color='red', linestyle=':', linewidth=1, label='1% target')
        if args.svd_rel_err is not None:
            ax1p.axhline(args.svd_rel_err, color='purple', linestyle='--', linewidth=1,
                         label=f'SVD rel_err={args.svd_rel_err:.4f}')
        ax1p.set_xlabel('Avg active coefficients per patch')
        ax1p.set_ylabel('Relative reconstruction error', color='steelblue')
        ax2p.set_ylabel('Compression ratio (coeff only)', color='darkorange')
        ax1p.set_title(f'LCA: quality vs sparsity  (patch {P}³, M={M})')
        lines1, labels1 = ax1p.get_legend_handles_labels()
        lines2, labels2 = ax2p.get_legend_handles_labels()
        ax1p.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')
        ax1p.grid(True, alpha=0.3)
        plt.tight_layout()
        out = os.path.join(plots_dir, 'rel_err_vs_active.png')
        plt.savefig(out, dpi=150); plt.close(); print(f"Saved {out}")

        # Plot 3 — rate-distortion
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.semilogy(bpv_coeffs, rel_errs, 'o-', color='steelblue', markersize=5,
                    label='LCA (coeff only, float32)')
        ax.semilogy(bpv_lca_equivs, rel_errs, '^:', color='darkorange', markersize=4,
                    alpha=0.9, label='LCA (COO storage)')
        ax.axhline(0.01, color='red', linestyle=':', linewidth=1, label='1% error target')
        if args.svd_bpv is not None and args.svd_rel_err is not None:
            ax.scatter([args.svd_bpv], [args.svd_rel_err], marker='*', s=200,
                       color='green', zorder=5,
                       label=f'SVD ({args.svd_bpv:.2f} BPV, {args.svd_rel_err:.4f})')
        for r in results[::max(1, len(results)//8)]:
            ax.annotate(f"λ={r['lambda_']:.3f}", (r['bpv_coeff'], r['rel_err']),
                        textcoords='offset points', xytext=(4, 4), fontsize=7)
        ax.set_xlabel('Bits per voxel (BPV)')
        ax.set_ylabel('Relative reconstruction error (log scale)')
        ax.set_title(f'LCA Rate–Distortion  |  patch {P}³  |  M={M}  |  '
                     f'{world_size} GPU(s)  |  {dict_label}')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = os.path.join(plots_dir, 'rate_distortion.png')
        plt.savefig(out, dpi=150); plt.close(); print(f"Saved {out}")

        # Plot 4 — full-volume reconstruction at best lambda
        mD, mH, mW = D_out // 2, H_out // 2, W_out // 2
        plane_defs = [
            ('XY (z=mid)', input_vol[:, :, mW], best_recon_vol[:, :, mW]),
            ('XZ (y=mid)', input_vol[:, mH, :], best_recon_vol[:, mH, :]),
            ('YZ (x=mid)', input_vol[mD, :, :], best_recon_vol[mD, :, :]),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        fig.suptitle(
            f'LCA Full-volume  |  λ={best_lam:.4f}  '
            f'avg_active={best["avg_active"]:.1f}  '
            f'rel_err={best["rel_err"]:.4f}  '
            f'comp={best["comp_coeff"]:.1f}x  BPV={best["bpv_coeff"]:.3f}  '
            f'M={M}  {dict_label}',
            fontsize=9
        )
        for col, (lbl, inp_p, rec_p) in enumerate(plane_defs):
            vmax = np.percentile(np.abs(inp_p), 99)
            for row, (data, row_lbl) in enumerate([(inp_p, 'Input'),
                                                    (rec_p, 'Reconstruction')]):
                ax = axes[row, col]
                im = ax.imshow(data, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                               origin='lower', aspect='equal')
                if row == 0:
                    ax.set_title(lbl, fontsize=9)
                if col == 0:
                    ax.set_ylabel(row_lbl, fontsize=9)
                ax.tick_params(left=False, bottom=False,
                               labelleft=False, labelbottom=False)
                plt.colorbar(im, ax=ax, shrink=0.85)
        plt.tight_layout()
        out = os.path.join(plots_dir, f'full_volume_lam{best_lam:.4f}.png')
        plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
        print(f"Saved {out}")

        # Plot 5 — error maps
        error_vol = input_vol - best_recon_vol
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
            ax.set_title(f'LCA error  {lbl}', fontsize=9)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
        svd_ref = ('' if args.svd_rel_err is None
                   else f'  vs SVD rel_err={args.svd_rel_err:.4f}')
        fig.suptitle(
            f'LCA error maps  |  λ={best_lam:.4f}  '
            f'rel_err={best["rel_err"]:.4f}{svd_ref}',
            fontsize=9
        )
        plt.tight_layout()
        out = os.path.join(plots_dir, f'error_maps_lam{best_lam:.4f}.png')
        plt.savefig(out, dpi=150); plt.close(); print(f"Saved {out}")

        # CSV
        csv_path = os.path.join(out_dir, 'lca_results.csv')
        with open(csv_path, 'w', newline='') as csvf:
            writer = csv.DictWriter(csvf, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults table → {csv_path}")

        # Summary
        print(f"\n{'='*60}")
        print(f"LCA SUMMARY  ({world_size} GPU(s), patch {P}³, {n_patches} tiles, "
              f"{dict_label})")
        print(f"{'='*60}")
        print(f"Atoms M   : {M}   (P³={P3}, {M/P3:.2f}× overcomplete)")
        print(f"LCA       : iters={lca_iters}  tau={tau}  threshold=signed")
        print(f"COO bytes : {_bytes_per_nz} bytes/nz  ({_index_bits} index bits)")
        print(best_tag)
        if args.svd_bpv:
            print(f"SVD ref   : BPV={args.svd_bpv:.3f}  rel_err={args.svd_rel_err}")

        print("\nDone.")
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        _log.close()

    if using_ddp:
        cleanup_ddp()


if __name__ == '__main__':
    main()
