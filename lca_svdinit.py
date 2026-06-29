"""
LCA with SVD-based dictionary initialisation for 3D simulation data.

The LCA dictionary is seeded with the top-k right singular vectors computed
on non-overlapping kernel_size³ patches of the input volume.  LCA then trains
on the full input volume with its normal Hebbian rule.

SVD is used ONLY as an initialisation tool — it contributes nothing to the
final compressed representation.  The compressed bitstream is purely:
    codes  — COO sparse (float32 value + flat index)
    atoms  — dense float32 dictionary  (features × kernel_size³)

Setting model.svd_init: false falls back to random initialisation so you can
run both with one config change to isolate the warm-start benefit.

Usage
-----
    python lca_svdinit.py config_svdinit.yaml
    torchrun --nproc_per_node=4 lca_svdinit.py config_svdinit.yaml
"""

import argparse
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
# SVD weight initialisation
# ---------------------------------------------------------------------------

def svd_init_weights(lca, vol, kernel_size, features, device, dtype):
    """
    Overwrite lca.weights with the top-k right singular vectors of the data.

    Non-overlapping kernel_size³ patches are extracted, mean/std normalised,
    and SVD is computed.  Each Vt row is reshaped to (1, P, P, P) and placed
    directly into lca.weights.  When k_svd < features the remaining atoms are
    initialised with small random values so they can still adapt.

    SVD atoms are already unit-norm (orthonormal rows of Vt), so the
    subsequent lca.normalize_weights() call is a no-op for those atoms but
    normalises the random padding.

    Returns (k_actual, n_patches) for logging.
    """
    P = kernel_size
    D, H, W = vol.shape
    nD, nH, nW = D // P, H // P, W // P
    n_patches = nD * nH * nW

    X = np.empty((n_patches, P**3), dtype=np.float32)
    idx = 0
    for di in range(nD):
        for hi in range(nH):
            for wi in range(nW):
                patch = vol[di*P:(di+1)*P, hi*P:(hi+1)*P, wi*P:(wi+1)*P].ravel().astype(np.float32)
                m = patch.mean()
                s = patch.std() + 1e-8
                X[idx] = (patch - m) / s
                idx += 1

    k = min(features, n_patches, P**3)

    if _HAVE_SKLEARN:
        _, _, Vt = randomized_svd(X, n_components=k, random_state=0)
    else:
        _, _, Vt_full = np.linalg.svd(X, full_matrices=False)
        Vt = Vt_full[:k]
    Vt = Vt.astype(np.float32)  # (k, P³)

    atoms = torch.tensor(Vt, dtype=dtype, device=device).reshape(k, 1, P, P, P)

    if k < features:
        pad = torch.randn(features - k, 1, P, P, P, dtype=dtype, device=device) * 0.01
        atoms = torch.cat([atoms, pad], dim=0)

    lca.weights.data.copy_(atoms)
    lca.normalize_weights()

    return k, n_patches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='LCA with SVD weight initialisation')
    parser.add_argument('config', help='path to config YAML')
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

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

    mcfg = cfg['model']
    tcfg = cfg['training']
    dcfg = cfg['data']

    use_svd_init = mcfg.get('svd_init', True)

    # Output directory (rank 0 creates it, all ranks receive the same path)
    if is_main:
        ts      = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        tag     = 'lca_svdinit_' if use_svd_init else 'lca_randinit_'
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

    if is_main:
        _log = open(os.path.join(out_dir, 'run.log'), 'w')
        sys.stdout = _Tee(sys.__stdout__, _log)
        sys.stderr = _Tee(sys.__stderr__, _log)

    # ------------------------------------------------------------------ #
    # Load and normalise volume — all ranks load independently
    # ------------------------------------------------------------------ #
    with h5py.File(dcfg['h5_path'], 'r') as f:
        vol = f[dcfg['field_key']][dcfg['timestep']].astype(np.float32)

    vol_mean = float(vol.mean())
    vol_std  = float(vol.std()) + 1e-8
    vol = (vol - vol_mean) / vol_std
    D, H, W = vol.shape

    # Truncate to multiples of kernel_size so SVD-init patches tile exactly.
    # LCA's convolutional forward is unaffected by the slight size reduction.
    P      = mcfg['kernel_size']
    D_out  = (D // P) * P
    H_out  = (H // P) * P
    W_out  = (W // P) * P
    input_vol = vol[:D_out, :H_out, :W_out].copy()

    if is_main:
        print(f"Output dir  : {out_dir}")
        print(f"Config      : {args.config}")
        print(f"Volume      : {D}×{H}×{W}  →  tiled region {D_out}×{H_out}×{W_out}")
        print(f"SVD backend : {'sklearn randomized_svd' if _HAVE_SKLEARN else 'numpy linalg.svd'}\n")

    # ------------------------------------------------------------------ #
    # Build LCA model
    # ------------------------------------------------------------------ #
    dtype_str = tcfg.get('dtype', 'float32')
    dtype     = {'float32': torch.float32, 'bfloat16': torch.bfloat16}[dtype_str]
    features  = mcfg['features']
    stride    = mcfg['stride']

    lca = LCAConv3D(
        out_neurons   = features,
        in_neurons    = 1,
        result_dir    = os.path.join(out_dir, 'lca_results'),
        kernel_size   = P,
        stride        = stride,
        lambda_       = mcfg['lambda_'],
        tau           = mcfg['tau'],
        lca_iters     = mcfg['lca_iters'],
        eta           = mcfg['learning_rate'],
        track_metrics = False,
        return_vars   = ['inputs', 'acts', 'recons', 'recon_errors'],
    ).to(dtype=dtype, device=device)

    # ------------------------------------------------------------------ #
    # SVD (or random) weight initialisation
    # All ranks compute SVD independently on the same data → identical result.
    # Broadcast from rank 0 afterwards for bit-exact safety.
    # ------------------------------------------------------------------ #
    if use_svd_init:
        if is_main:
            print(f"Computing SVD on {P}³ patches for weight init ...")
        k_init, n_init_patches = svd_init_weights(
            lca, input_vol, P, features, device, dtype
        )
        if is_main:
            nD_i = D_out // P
            print(f"  Patch grid  : {nD_i}³ = {n_init_patches} patches  "
                  f"(each {P}³ = {P**3} dims)")
            print(f"  SVD atoms   : top-{k_init} singular vectors → "
                  f"{'full feature set' if k_init == features else f'{features - k_init} atoms random-padded'}\n")
    else:
        if is_main:
            print("Random weight initialisation (baseline)\n")

    if using_ddp:
        dist.broadcast(lca.weights.data, src=0)

    # ------------------------------------------------------------------ #
    # COO sparse storage constants
    # ------------------------------------------------------------------ #
    cd = ceil(D_out / stride)
    ch = ceil(H_out / stride)
    cw = ceil(W_out / stride)
    n_code_total  = features * cd * ch * cw
    _index_bits   = int(np.ceil(np.log2(n_code_total + 1)))
    _bytes_per_nz = 4 + (_index_bits + 7) // 8   # float32 value + flat index

    bytes_in    = D_out * H_out * W_out * 4          # full volume float32
    bytes_atoms = features * P**3 * 4                 # dictionary, paid once

    if is_main:
        init_label = 'SVD' if use_svd_init else 'random'
        print(f"LCAConv3D   : {features} atoms | kernel {P}³ | "
              f"stride={stride} | λ={lca.lambda_} | init={init_label}")
        print(f"Code size   : {features} × {cd}×{ch}×{cw} = {n_code_total:,} positions")
        print(f"Index bits  : {_index_bits}  →  bytes_per_nz={_bytes_per_nz}")
        print(f"bytes_in    : {bytes_in:,}  ({bytes_in/1024/1024:.1f} MB)")
        print(f"bytes_atoms : {bytes_atoms:,}  (dictionary — amortised over all uses)\n")

    # ------------------------------------------------------------------ #
    # Fixed input tensor — same volume every pass
    # ------------------------------------------------------------------ #
    input_tensor = torch.tensor(
        input_vol, dtype=dtype, device=device,
    ).unsqueeze(0).unsqueeze(0)   # (1, 1, D_out, H_out, W_out)

    # ------------------------------------------------------------------ #
    # Training hyperparameters
    # ------------------------------------------------------------------ #
    max_epochs         = tcfg.get('max_epochs', 400)
    n_passes_per_epoch = tcfg.get('n_passes_per_epoch', 30)
    anneal_every       = tcfg.get('lambda_anneal_every', 1)
    anneal_step        = tcfg.get('lambda_anneal_step', 0.01)
    anneal_stop        = tcfg.get('lambda_anneal_stop', max_epochs)
    rel_err_target     = tcfg.get('rel_err_target', None)
    rel_err_ceiling    = tcfg.get('rel_err_ceiling', 0.05)
    stabilize_epochs   = tcfg.get('stabilize_epochs', 10)

    if is_main:
        print(f"Training    : {max_epochs} epochs × {n_passes_per_epoch} passes/epoch")
        if rel_err_target is not None:
            print(f"Annealing   : rel_err-gated  "
                  f"target={rel_err_target}  ceiling={rel_err_ceiling}")
        else:
            print(f"Annealing   : time-based  stop={anneal_stop}  "
                  f"step={anneal_step}  every={anneal_every}")
        print()

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    mode         = 'pre' if rel_err_target is not None else 'anneal'
    stab_count   = 0
    anneal_epoch = 0
    last_rel_err = 1.0
    best_comp    = 0.0

    all_rel_errs = []
    lca_recon_np = np.zeros_like(input_vol)
    active_nz    = 0.0
    sparsity     = 0.0
    comp_ratio   = 0.0
    comp_nodict  = 0.0
    bpv          = float('inf')
    rel_err      = 1.0

    for epoch in range(max_epochs):
        t0 = time.time()

        # λ annealing at epoch start
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
                              f"rel_err={last_rel_err:.4f} > ceiling={rel_err_ceiling}")
        else:
            if epoch > 0 and anneal_epoch < anneal_stop \
                    and anneal_epoch % anneal_every == 0:
                lca.lambda_ += anneal_step
                if is_main:
                    print(f"  [anneal] λ → {lca.lambda_:.3f}")

        # n_passes_per_epoch forward + Hebbian passes on the fixed input
        for _ in range(n_passes_per_epoch):
            inputs_norm, code, recon_norm, recon_error_norm = lca(input_tensor)
            lca.update_weights(code, recon_error_norm)
            if using_ddp:
                dist.all_reduce(lca.weights.data, op=dist.ReduceOp.SUM)
                lca.weights.data /= world_size
                lca.normalize_weights()

        # Metrics — rank 0 only; all ranks have identical weights
        if is_main:
            l1_cost = compute_l1_sparsity(code, lca.lambda_).item()
            l2_cost = compute_l2_error(inputs_norm, recon_norm).item()

            active_nz = float((code != 0).float().sum().item())
            sparsity  = 1.0 - active_nz / n_code_total

            lca_recon_np = recon_norm[0, 0].float().cpu().numpy()
            rel_err = float(
                np.linalg.norm(lca_recon_np - input_vol) /
                (np.linalg.norm(input_vol) + 1e-8)
            )

            bytes_code  = active_nz * _bytes_per_nz
            bytes_total = bytes_atoms + bytes_code
            comp_ratio  = bytes_in / bytes_total if bytes_total > 0 else float('inf')
            comp_nodict = bytes_in / bytes_code  if bytes_code  > 0 else float('inf')
            bpv         = bytes_total * 8 / (D_out * H_out * W_out)

            all_rel_errs.append(rel_err)

            if mode == 'stabilize':
                mode_tag = f"  [stabilize {stab_count}/{stabilize_epochs}]"
            elif mode == 'anneal':
                mode_tag = f"  [anneal ep {anneal_epoch}/{anneal_stop}]"
            else:
                mode_tag = "  [pre-anneal]"

            print(
                f"Epoch {epoch:03d} | {time.time()-t0:.1f}s ({n_passes_per_epoch}p) | "
                f"Sparsity={sparsity:.3f}  Active={active_nz:.0f}/{n_code_total}  "
                f"rel_err={rel_err:.6f}  L2={l2_cost:.4f}  L1={l1_cost:.4f}  "
                f"λ={lca.lambda_:.3f}  "
                f"comp={comp_ratio:.2f}x(+atoms) {comp_nodict:.2f}x(code only)  "
                f"BPV={bpv:.4f}" + mode_tag
            )

            torch.save(lca.state_dict(), os.path.join(models_dir, 'lca.pth'))

            if rel_err <= rel_err_ceiling and comp_ratio > best_comp:
                best_comp = comp_ratio
                torch.save(lca.state_dict(),
                           os.path.join(models_dir, 'lca_best_compression.pth'))
                print(f"  [best] comp={best_comp:.2f}x  rel_err={rel_err:.6f}")

        last_rel_err = rel_err

        # Annealing state machine
        if rel_err_target is not None:
            if mode in ('pre', 'anneal') and rel_err <= rel_err_target:
                prev_mode  = mode
                mode       = 'stabilize'
                stab_count = 0
                reason = 'before first anneal' if prev_mode == 'pre' else 'mid-anneal'
                if is_main:
                    print(f"  [stabilize] rel_err={rel_err:.6f} <= {rel_err_target} — "
                          f"freezing λ for {stabilize_epochs} epochs ({reason})")
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
        os.path.join(models_dir, 'lca_atoms.npz'),
        weights  = lca.get_weights().float().cpu().numpy(),
        vol_mean = np.float32(vol_mean),
        vol_std  = np.float32(vol_std),
    )
    print(f"\nSaved atoms → {os.path.join(models_dir, 'lca_atoms.npz')}")
    print(f"Saved model → {os.path.join(models_dir, 'lca.pth')}")
    print("\nGenerating plots...")

    # Training curve
    if all_rel_errs:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.semilogy(all_rel_errs, label='LCA rel_err', color='steelblue')
        if rel_err_target is not None:
            ax.axhline(rel_err_target, color='red', linestyle='--', linewidth=0.8,
                       label=f'target={rel_err_target}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Relative reconstruction error (log)')
        ax.set_title(
            f'LCA training  |  {"SVD" if use_svd_init else "random"} init  |  '
            f'{features} atoms  |  kernel {P}³  |  stride={stride}  |  '
            f'final comp={comp_ratio:.2f}x(+atoms)  BPV={bpv:.4f}'
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = os.path.join(plots_dir, 'training_metrics.png')
        plt.savefig(out, dpi=150); plt.close(); print(f"Saved {out}")

    # Dictionary atoms — mid-plane slice
    weights  = lca.get_weights().float().cpu().numpy()   # (features, 1, P, P, P)
    n_feat   = weights.shape[0]
    mid      = P // 2
    atoms_2d = weights[:, 0, mid, :, :]                  # (features, P, P)
    cols_a   = int(np.ceil(np.sqrt(n_feat)))
    rows_a   = int(np.ceil(n_feat / cols_a))
    fig, axes_a = plt.subplots(rows_a, cols_a, figsize=(cols_a * 1.2, rows_a * 1.2))
    axes_a = np.array(axes_a).ravel()
    vmax_a = np.percentile(np.abs(atoms_2d), 99)
    for i, ax in enumerate(axes_a):
        if i < n_feat:
            ax.imshow(atoms_2d[i], cmap='RdBu_r', vmin=-vmax_a, vmax=vmax_a)
        ax.axis('off')
    fig.suptitle(
        f'Dictionary atoms — mid-plane slice  ({n_feat} atoms, kernel {P}³)  '
        f'{"SVD" if use_svd_init else "random"} init',
        fontsize=10,
    )
    plt.tight_layout()
    out = os.path.join(plots_dir, 'dictionary_atoms.png')
    plt.savefig(out, dpi=150); plt.close(); print(f"Saved {out}")

    # Reconstruction panels: Input / LCA recon / Error  × 3 planes
    mD, mH, mW = D_out // 2, H_out // 2, W_out // 2
    slicers = [
        ('XY (z=mid)', lambda a: a[mD]),
        ('XZ (y=mid)', lambda a: a[:, mH, :]),
        ('YZ (x=mid)', lambda a: a[:, :, mW]),
    ]
    error_np = input_vol - lca_recon_np
    vmax_sig = float(np.percentile(np.abs(input_vol), 99))
    vmax_err = float(np.percentile(np.abs(error_np), 99))
    rows_data = [
        ('Input',     input_vol,    'RdBu_r', vmax_sig),
        ('LCA recon', lca_recon_np, 'RdBu_r', vmax_sig),
        ('Error',     error_np,     'bwr',     vmax_err),
    ]
    fig, ax_grid = plt.subplots(3, 3, figsize=(11, 10))
    for ri, (row_label, arr, cmap, vmax_r) in enumerate(rows_data):
        for ci, (plane_label, slicer) in enumerate(slicers):
            ax = ax_grid[ri, ci]
            im = ax.imshow(slicer(arr), cmap=cmap, vmin=-vmax_r, vmax=vmax_r)
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(plane_label, fontsize=9)
        ax_grid[ri, 0].set_ylabel(row_label, fontsize=8, labelpad=4)
        cbar = fig.colorbar(im, ax=list(ax_grid[ri, :]), shrink=0.82, pad=0.02)
        cbar.ax.tick_params(labelsize=7)
    fig.suptitle(
        f'LCA ({"SVD" if use_svd_init else "random"} init)  |  '
        f't={dcfg["timestep"]}  '
        f'rel_err={rel_err:.4f}  comp={comp_ratio:.2f}x(+atoms)  BPV={bpv:.4f}',
        fontsize=9, y=1.002,
    )
    plt.tight_layout()
    out = os.path.join(plots_dir, 'reconstruction.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close(); print(f"Saved {out}")

    print("\nDone.")
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    if using_ddp:
        cleanup_ddp()


if __name__ == '__main__':
    main()
