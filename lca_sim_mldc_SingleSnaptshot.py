"""
3D convolutional LCA dictionary learning on a single JHTDB simulation snapshot.

Loads one 3D pressure (or velocity) field from an HDF5 file, extracts random
3D patches, and trains LCAConv3D via Hebbian updates.

Supports single-GPU and multi-GPU training via torch.distributed (manual all_reduce).

Usage — single GPU:
    python lca_sim_mldc.py [config_simmldc.yaml]

Usage — N GPUs (e.g. 4):
    torchrun --nproc_per_node=4 lca_sim_mldc.py [config_simmldc.yaml]
"""

import argparse
import os
import shutil
import sys
import time
from datetime import datetime

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import yaml

from torch.utils.data import DataLoader, Dataset, DistributedSampler

from lcapt.lca import LCAConv3D
from lcapt.metric import compute_l1_sparsity, compute_l2_error


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def setup_ddp() -> tuple[int, int, int]:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), local_rank, dist.get_world_size()


def cleanup_ddp() -> None:
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Dataset — random 3D patch crops from a single HDF5 snapshot
# ---------------------------------------------------------------------------

class _HDF5PatchDataset(Dataset):
    """
    Loads one timestep from an HDF5 volume and serves random 3D crops.

    The HDF5 field is expected at key 'pressure' (or 'Vx'/'Vy'/'Vz') with
    shape (nt, nx, ny, nz).  Each __getitem__ returns a (1, P, P, P) patch
    drawn from a random location within the volume.

    n_patches sets the virtual epoch size — there is no fixed enumeration of
    patches; each call samples independently, giving effectively infinite
    augmentation from a single snapshot.
    """
    def __init__(self, h5_path: str, field_key: str, timestep: int,
                 patch_size: int, n_patches: int):
        with h5py.File(h5_path, 'r') as f:
            vol = f[field_key][timestep]          # (nx, ny, nz)  float32
        # Normalise to zero mean, unit variance so LCA starts in a stable range
        vol = (vol - vol.mean()) / (vol.std() + 1e-8)
        self.vol        = torch.from_numpy(vol.astype(np.float32))   # (nx, ny, nz)
        self.patch_size = patch_size
        self.n_patches  = n_patches
        nx, ny, nz      = vol.shape
        assert min(nx, ny, nz) >= patch_size, \
            f"Volume {vol.shape} smaller than patch_size={patch_size}"
        self.limits = (nx - patch_size, ny - patch_size, nz - patch_size)

    def __len__(self):
        return self.n_patches

    def __getitem__(self, _idx):
        x = torch.randint(0, self.limits[0] + 1, (1,)).item()
        y = torch.randint(0, self.limits[1] + 1, (1,)).item()
        z = torch.randint(0, self.limits[2] + 1, (1,)).item()
        p = self.patch_size
        patch = self.vol[x:x+p, y:y+p, z:z+p]   # (P, P, P)
        return patch.unsqueeze(0)                  # (1, P, P, P) — channel dim


# ---------------------------------------------------------------------------
# Logging helper
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
# Inference helper (Option 1 — context padding)
# ---------------------------------------------------------------------------

def _run_inference(args, cfg, lca, vol_np, device, dtype, out_dir):
    """
    Tile vol_np into infer_P³ tiles, inflate each by `pad` voxels on every
    side using real neighbouring data, run LCA, keep only the central P³ of
    the reconstruction.  Saves plots, metrics CSV, and run.log inside out_dir.
    """
    import csv

    mcfg   = cfg['model']
    stride = mcfg['stride']
    K      = mcfg['kernel_size']
    infer_P = args.infer_patch_size or cfg['data']['patch_size']

    if infer_P % stride != 0:
        raise ValueError(f"--infer-patch-size {infer_P} must be divisible by stride={stride}")

    if args.infer_pad is not None:
        pad = args.infer_pad
        if pad % stride != 0:
            raise ValueError(f"--infer-pad {pad} must be divisible by stride={stride}")
    else:
        pad = K // 2
        if pad % stride != 0:
            pad += stride - pad % stride   # round up to next multiple of stride

    if args.infer_lambda is not None:
        lca.lambda_ = args.infer_lambda

    plots_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    _log = open(os.path.join(out_dir, 'run.log'), 'w')

    def log(msg=''):
        line = msg + '\n'
        sys.__stdout__.write(line)
        _log.write(line)
        _log.flush()

    infer_full  = infer_P + 2 * pad
    D, H, W     = vol_np.shape
    nD, nH, nW  = D // infer_P, H // infer_P, W // infer_P
    D_out, H_out, W_out = nD * infer_P, nH * infer_P, nW * infer_P
    input_vol   = vol_np[:D_out, :H_out, :W_out].copy()

    pp = pad    // stride   # code positions to skip per side
    cp = infer_P // stride  # central code positions per side

    log("Inference — Option 1 (context padding)")
    log(f"  Source model   : {args.load_model}")
    log(f"  Volume         : {D}×{H}×{W}  →  tiled {D_out}×{H_out}×{W_out}")
    log(f"  Tile           : {infer_P}³  (trained on {cfg['data']['patch_size']}³)")
    log(f"  Padding        : {pad} voxels/side  →  inflated {infer_full}³")
    log(f"  Stride/kernel  : {stride} / {K}³")
    log(f"  Tiles          : {nD}×{nH}×{nW} = {nD*nH*nW}")
    log(f"  λ              : {lca.lambda_:.4f}")
    log()

    tiles        = [(di*infer_P, hi*infer_P, wi*infer_P)
                    for di in range(nD) for hi in range(nH) for wi in range(nW)]
    recon_vol    = np.zeros((D_out, H_out, W_out), dtype=np.float32)
    total_active = 0.0
    n_tiles      = len(tiles)

    bs = min(n_tiles, 32)
    with torch.no_grad():
        for start in range(0, n_tiles, bs):
            batch_coords = tiles[start:start + bs]
            patches = []
            for x, y, z in batch_coords:
                x0, x1 = max(0, x - pad), min(D, x + infer_P + pad)
                y0, y1 = max(0, y - pad), min(H, y + infer_P + pad)
                z0, z1 = max(0, z - pad), min(W, z + infer_P + pad)
                raw = vol_np[x0:x1, y0:y1, z0:z1]
                patches.append(
                    torch.from_numpy(
                        np.pad(raw, [
                            (pad - (x - x0), pad - (x1 - x - infer_P)),
                            (pad - (y - y0), pad - (y1 - y - infer_P)),
                            (pad - (z - z0), pad - (z1 - z - infer_P)),
                        ])
                    ).unsqueeze(0)
                )

            batch = torch.stack(patches).to(dtype=dtype, device=device)
            _, code, recon_batch, _ = lca(batch)

            recon_np = recon_batch.float().cpu().numpy()
            for i, (x, y, z) in enumerate(batch_coords):
                recon_vol[x:x+infer_P, y:y+infer_P, z:z+infer_P] = \
                    recon_np[i, 0, pad:pad+infer_P, pad:pad+infer_P, pad:pad+infer_P]

            central       = code[:, :, pp:pp+cp, pp:pp+cp, pp:pp+cp]
            total_active += (central != 0).float().sum(dim=(1, 2, 3, 4)).sum().item()

    avg_active = total_active / n_tiles

    # Compression metrics (central code only — matches the stored infer_P³ tile)
    _n_code_patch = mcfg['features'] * cp**3
    _index_bits   = int(np.ceil(np.log2(_n_code_patch + 1)))
    _bytes_per_nz = 4 + (_index_bits + 7) // 8
    bytes_sparse  = avg_active * _bytes_per_nz
    comp_coeff    = infer_P**3 / avg_active if avg_active > 0 else float('inf')
    comp_ratio    = infer_P**3 * 4 / bytes_sparse if bytes_sparse > 0 else float('inf')
    bpv           = bytes_sparse * 8 / infer_P**3

    # Quality metrics (full volume)
    err       = input_vol - recon_vol
    mse       = float((err**2).mean())
    rmse      = float(np.sqrt(mse))
    rel_err   = rmse / (float(np.sqrt((input_vol**2).mean())) + 1e-8)
    sig_range = float(input_vol.max() - input_vol.min())
    psnr      = float(20 * np.log10(sig_range / (rmse + 1e-12)))

    log(f"{'Metric':<26}  Value")
    log('-' * 42)
    log(f"{'rel_err':<26}  {rel_err:.6f}")
    log(f"{'RMSE':<26}  {rmse:.6f}")
    log(f"{'PSNR (dB)':<26}  {psnr:.2f}")
    log(f"{'avg_active (central)':<26}  {avg_active:.1f} / {_n_code_patch}")
    log(f"{'comp_coeff (P³/active)':<26}  {comp_coeff:.4f}x")
    log(f"{'comp_ratio (bytes)':<26}  {comp_ratio:.4f}x")
    log(f"{'BPV':<26}  {bpv:.3f}")
    log()

    csv_path = os.path.join(out_dir, 'inference_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'infer_patch_size', 'pad', 'lambda_', 'avg_active',
            'rel_err', 'rmse', 'psnr', 'comp_coeff', 'comp_ratio', 'bpv',
        ])
        writer.writeheader()
        writer.writerow(dict(
            infer_patch_size=infer_P, pad=pad, lambda_=lca.lambda_,
            avg_active=avg_active, rel_err=rel_err, rmse=rmse, psnr=psnr,
            comp_coeff=comp_coeff, comp_ratio=comp_ratio, bpv=bpv,
        ))
    log(f"Saved {csv_path}")

    mD, mH, mW = D_out // 2, H_out // 2, W_out // 2
    plane_defs = [
        ('XY (z=mid)', input_vol[:, :, mW], recon_vol[:, :, mW]),
        ('XZ (y=mid)', input_vol[:, mH, :], recon_vol[:, mH, :]),
        ('YZ (x=mid)', input_vol[mD, :, :], recon_vol[mD, :, :]),
    ]

    # Plot 1 — reconstruction
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(
        f'LCA inference (pad={pad})  |  tile={infer_P}³  λ={lca.lambda_:.3f}  '
        f'rel_err={rel_err:.4f}  Comp(P³)={comp_coeff:.2f}x  BPV={bpv:.2f}',
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
    p = os.path.join(plots_dir, f'reconstruction_tile{infer_P}_pad{pad}.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"Saved {p}")

    # Plot 2 — error maps
    fig, axes_row = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (lbl, inp_p, rec_p) in zip(axes_row, plane_defs):
        vmax = np.percentile(np.abs(inp_p), 99)
        im   = ax.imshow(inp_p - rec_p, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                         origin='lower', aspect='equal')
        ax.set_title(f'Error {lbl}', fontsize=9)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        plt.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle(f'Error maps  |  rel_err={rel_err:.4f}  PSNR={psnr:.1f} dB', fontsize=9)
    plt.tight_layout()
    p = os.path.join(plots_dir, f'error_maps_tile{infer_P}_pad{pad}.png')
    plt.savefig(p, dpi=150)
    plt.close()
    log(f"Saved {p}")

    log("\nDone.")
    _log.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
# Usage Inference: python lca_sim_mldc_SingleSnaptshot.py config_simmldc.yaml  --load-model experiments/simmldc_2026-06-29_13-41-19/models/lca_simmldc_best_compression.pth  --infer  --infer-patch-size 9   --infer-lambda 0.15

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
    # Args + Config
    # ------------------------------------------------------------------ #
    parser = argparse.ArgumentParser(description='LCAConv3D single-snapshot training')
    parser.add_argument('config', nargs='?', default='config_simmldc.yaml',
                        help='path to config YAML (default: config_simmldc.yaml)')
    parser.add_argument('--load-model', default=None, metavar='PATH',
                        help='path to a .pth checkpoint to pre-load the dictionary '
                             '(e.g. experiments/simmldc_.../models/lca_simmldc_best_compression.pth)')
    parser.add_argument('--infer', action='store_true',
                        help='skip training; run context-padded (Option 1) inference only '
                             '(requires --load-model)')
    parser.add_argument('--infer-patch-size', type=int, default=None, metavar='P',
                        help='tile size for inference; may differ from training patch_size '
                             '(must be divisible by stride; default: patch_size from config)')
    parser.add_argument('--infer-pad', type=int, default=None, metavar='N',
                        help='context padding voxels per side (must be divisible by stride); '
                             'default: auto = smallest multiple of stride ≥ kernel_size//2')
    parser.add_argument('--infer-lambda', type=float, default=None, metavar='LAM',
                        help='override lambda for inference (default: value from checkpoint)')
    args = parser.parse_args()
    cfg_path = args.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # ------------------------------------------------------------------ #
    # Experiment directory
    # ------------------------------------------------------------------ #
    if args.infer:
        if args.load_model is None:
            raise ValueError("--infer requires --load-model")
        # Place inference output inside the source experiment dir
        _src_exp   = os.path.dirname(os.path.dirname(os.path.abspath(args.load_model)))
        exp_dir    = os.path.join(_src_exp, 'inference_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        plots_dir  = os.path.join(exp_dir, 'plots')
        models_dir = None
        os.makedirs(plots_dir, exist_ok=True)
    else:
        if is_main:
            exp_dir = os.path.join(
                'experiments', 'simmldc_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            )
        else:
            exp_dir = None

        if using_ddp:
            container = [exp_dir]
            dist.broadcast_object_list(container, src=0)
            exp_dir = container[0]

        plots_dir  = os.path.join(exp_dir, 'plots')
        models_dir = os.path.join(exp_dir, 'models')

        if is_main:
            os.makedirs(plots_dir,  exist_ok=True)
            os.makedirs(models_dir, exist_ok=True)
            shutil.copy(cfg_path, os.path.join(exp_dir, 'config_simmldc.yaml'))
            _log       = open(os.path.join(exp_dir, 'run.log'), 'w')
            sys.stdout = _Tee(sys.__stdout__, _log)
            sys.stderr = _Tee(sys.__stderr__, _log)

        if using_ddp:
            dist.barrier()

    # ------------------------------------------------------------------ #
    # dtype
    # ------------------------------------------------------------------ #
    dtype_str = cfg['training'].get('dtype', 'float32')
    dtype = {'float16': torch.float16,
             'bfloat16': torch.bfloat16}.get(dtype_str, torch.float32)

    if is_main:
        print(f"Experiment dir : {exp_dir}")
        print(f"Config         : {cfg_path}")
        print(f"Device         : {device}  dtype={dtype}")
        print(f"GPUs           : {world_size}\n")

    # ------------------------------------------------------------------ #
    # Data — single 3D snapshot, served as random patches
    # ------------------------------------------------------------------ #
    dcfg = cfg['data']
    dset = _HDF5PatchDataset(
        h5_path    = dcfg['h5_path'],
        field_key  = dcfg['field_key'],
        timestep   = dcfg['timestep'],
        patch_size = dcfg['patch_size'],
        n_patches  = dcfg['n_patches'],
    )

    sampler = (
        DistributedSampler(dset, num_replicas=world_size, rank=rank, shuffle=True)
        if using_ddp else None
    )
    dataloader = DataLoader(
        dset,
        batch_size  = dcfg['batch_size'],
        shuffle     = (sampler is None),
        sampler     = sampler,
        num_workers = dcfg['num_workers'],
        pin_memory  = True,
        persistent_workers = dcfg['num_workers'] > 0,
    )

    if is_main:
        nx, ny, nz = dset.vol.shape
        print(f"Volume   : {nx}×{ny}×{nz}  |  field={dcfg['field_key']}  t={dcfg['timestep']}")
        print(f"Patches  : {dcfg['n_patches']} patches/epoch  |  size={dcfg['patch_size']}³")
        print(f"Batches  : {len(dataloader)}/GPU/epoch  |  "
              f"effective batch={dcfg['batch_size'] * world_size}\n")

    # ------------------------------------------------------------------ #
    # Model — LCAConv3D
    # ------------------------------------------------------------------ #
    mcfg = cfg['model']
    lca = LCAConv3D(
        out_neurons  = mcfg['features'],
        in_neurons   = mcfg['in_channels'],
        result_dir   = os.path.join(exp_dir, 'lca_results'),
        kernel_size  = mcfg['kernel_size'],
        stride       = mcfg['stride'],
        lambda_      = mcfg['lambda_'],
        tau          = mcfg['tau'],
        lca_iters    = mcfg['lca_iters'],
        eta          = mcfg['learning_rate'],
        track_metrics = False,
        return_vars  = ['inputs', 'acts', 'recons', 'recon_errors'],
    ).to(dtype=dtype, device=device)

    if args.load_model is not None:
        ckpt = torch.load(args.load_model, map_location=device)
        lca.load_state_dict(ckpt)
        if is_main:
            print(f"Loaded pre-trained dictionary: {args.load_model}")

    if using_ddp:
        dist.broadcast(lca.weights.data, src=0)

    lca_ddp   = lca
    lca_inner = lca

    # ------------------------------------------------------------------ #
    # Inference mode — skip training entirely
    # ------------------------------------------------------------------ #
    if args.infer:
        with h5py.File(dcfg['h5_path'], 'r') as fh:
            _raw = fh[dcfg['field_key']][dcfg['timestep']].astype(np.float32)
        vol_np = (_raw - _raw.mean()) / (_raw.std() + 1e-8)
        _run_inference(args, cfg, lca_inner, vol_np, device, dtype, exp_dir)
        if using_ddp:
            cleanup_ddp()
        return

    if is_main:
        k = mcfg['kernel_size']
        print(f"LCAConv3D : {mcfg['features']} atoms | "
              f"kernel {k}³ | stride {mcfg['stride']} | λ={mcfg['lambda_']}\n")

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    max_epochs       = cfg['training']['max_epochs']
    anneal_every     = cfg['training']['lambda_anneal_every']
    anneal_step      = cfg['training']['lambda_anneal_step']
    anneal_start     = cfg['training'].get('lambda_anneal_start', 0)   # fallback only
    anneal_stop      = cfg['training'].get('lambda_anneal_stop', max_epochs)
    rel_err_target   = cfg['training'].get('rel_err_target', None)
    rel_err_ceiling  = cfg['training'].get('rel_err_ceiling', 0.01)
    stabilize_epochs = cfg['training'].get('stabilize_epochs', 10)

    all_l2, all_l1, all_energy, all_rel_err = [], [], [], []

    # Compression constants — computed once from config (patch-level COO sparse storage)
    _P             = dcfg['patch_size']
    _n_code_patch  = mcfg['features'] * (_P // mcfg['stride'])**3   # code positions per patch
    _index_bits    = int(np.ceil(np.log2(_n_code_patch + 1)))
    _bytes_per_nz  = 4 + (_index_bits + 7) // 8   # float32 value + packed flat index
    _bytes_in      = _P**3 * 4                     # float32 input per patch (1 channel)

    # mode='pre'      : waiting for rel_err <= target before first anneal
    # mode='stabilize': λ frozen, counting stabilize_epochs (used before AND during annealing)
    # mode='anneal'   : λ increasing every anneal_every epochs
    # When rel_err_target is None: skip state machine, use fixed anneal_start/stop (legacy).
    mode              = 'pre' if rel_err_target is not None else 'anneal'
    stab_count        = 0
    anneal_epoch      = 0    # epochs spent in 'anneal' mode (used for anneal_every and anneal_stop)
    last_avg_rel_err  = 0.0  # previous epoch's rel_err — used by ceiling guard
    best_comp_ratio   = 0.0  # tracks highest compression seen while rel_err <= ceiling
    best_lambda       = mcfg['lambda_']

    for epoch in range(max_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        t0 = time.time()

        # ---- annealing step at epoch start ----
        if rel_err_target is not None:
            if mode == 'anneal' and anneal_epoch % anneal_every == 0:
                if last_avg_rel_err <= rel_err_ceiling:
                    lca_inner.lambda_ += anneal_step
                    if is_main:
                        print(f"  [anneal] λ → {lca_inner.lambda_:.3f}  "
                              f"(anneal epoch {anneal_epoch}/{anneal_stop})")
                else:
                    if is_main:
                        print(f"  [anneal] λ increment skipped — "
                              f"rel_err={last_avg_rel_err:.4f} > ceiling={rel_err_ceiling}")
        else:
            if epoch > 0 and anneal_start <= epoch < anneal_stop \
                    and (epoch - anneal_start) % anneal_every == 0:
                lca_inner.lambda_ += anneal_step
                if is_main:
                    print(f"  [anneal] λ → {lca_inner.lambda_:.3f}")

        ep_l2 = ep_l1 = ep_energy = ep_sparsity = ep_active = ep_rel_err = 0.0

        for patches in dataloader:
            patches = patches.to(dtype=dtype, device=device)  # (B, 1, P, P, P)
            inputs, code, recon, recon_error = lca_ddp(patches)

            lca_inner.update_weights(code, recon_error)

            if using_ddp:
                dist.all_reduce(lca_inner.weights.data, op=dist.ReduceOp.SUM)
                lca_inner.weights.data /= world_size
                lca_inner.normalize_weights()

            l1     = compute_l1_sparsity(code, lca_inner.lambda_).item()
            l2     = compute_l2_error(inputs, recon).item()
            energy = l2 + l1

            if is_main:
                all_l2.append(l2)
                all_l1.append(l1)
                all_energy.append(energy)
                all_rel_err.append(
                    ((inputs - recon).pow(2).sum(dim=(1, 2, 3, 4)) /
                     (inputs.pow(2).sum(dim=(1, 2, 3, 4)) + 1e-8)).mean().item()
                )

            # code shape: (B, features, D_out, H_out, W_out)
            n_total      = code.shape[1] * code.shape[2] * code.shape[3] * code.shape[4]
            ep_l2       += l2
            ep_l1       += l1
            ep_energy   += energy
            ep_sparsity += (code == 0).float().mean().item()
            ep_active   += (code != 0).float().sum(dim=(1, 2, 3, 4)).mean().item()
            ep_rel_err  += (
                (inputs - recon).pow(2).sum(dim=(1, 2, 3, 4)) /
                (inputs.pow(2).sum(dim=(1, 2, 3, 4)) + 1e-8)
            ).mean().item()

        nb         = len(dataloader)
        epoch_time = time.time() - t0

        avg_rel_err  = ep_rel_err / nb
        avg_active   = ep_active / nb
        bytes_sparse = avg_active * _bytes_per_nz
        comp_ratio   = _bytes_in / bytes_sparse if bytes_sparse > 0 else float('inf')
        comp_coeff   = _P**3 / avg_active if avg_active > 0 else float('inf')
        bpv          = bytes_sparse * 8 / _P**3

        if is_main:
            if mode == 'stabilize':
                mode_tag = f"  [stabilize {stab_count}/{stabilize_epochs}]"
            elif mode == 'anneal':
                mode_tag = f"  [anneal ep {anneal_epoch}/{anneal_stop}]"
            else:
                mode_tag = "  [pre-anneal]"
            print(
                f"Epoch {epoch:02d} | {epoch_time:.1f}s ({epoch_time/nb:.2f}s/batch) | "
                f"Sparsity: {ep_sparsity/nb:.3f}  "
                f"Active: {avg_active:.1f}/{n_total}  "
                f"Rel.err: {avg_rel_err:.6f}  "
                f"L2: {ep_l2/nb:.4f}  L1: {ep_l1/nb:.4f}  "
                f"Energy: {ep_energy/nb:.4f}  λ={lca_inner.lambda_:.3f}  "
                f"Comp(P³): {comp_coeff:.4f}x  CompRatio: {comp_ratio:.2f}x  BPV: {bpv:.2f}"
                + mode_tag
            )
            torch.save(lca_inner.state_dict(), os.path.join(models_dir, 'lca_simmldc.pth'))

            # best-compression checkpoint: highest comp_ratio seen while rel_err <= ceiling
            if avg_rel_err <= rel_err_ceiling and comp_ratio > best_comp_ratio:
                best_comp_ratio = comp_ratio
                best_lambda     = lca_inner.lambda_
                torch.save(lca_inner.state_dict(),
                           os.path.join(models_dir, 'lca_simmldc_best_compression.pth'))
                print(f"  [best] Comp(P³)={comp_coeff:.4f}x  CompRatio={best_comp_ratio:.2f}x  "
                      f"λ={best_lambda:.3f}  rel_err={avg_rel_err:.4f}  BPV={bpv:.2f}")

        last_avg_rel_err = avg_rel_err

        # ---- state machine (rel_err-gated) ----
        if rel_err_target is not None:
            if mode in ('pre', 'anneal') and avg_rel_err <= rel_err_target:
                prev_mode  = mode
                mode       = 'stabilize'
                stab_count = 0
                if is_main:
                    reason = 'before first anneal' if prev_mode == 'pre' else 'mid-anneal'
                    print(f"  [stabilize] rel_err={avg_rel_err:.6f} <= {rel_err_target} — "
                          f"freezing λ for {stabilize_epochs} epochs ({reason})")
            elif mode == 'stabilize':
                stab_count += 1
                if stab_count >= stabilize_epochs:
                    if anneal_epoch >= anneal_stop:
                        if is_main:
                            print("  [done] annealing complete + stabilized — stopping training")
                        break
                    mode       = 'anneal'
                    stab_count = 0
                    if is_main:
                        verb = 'Starting' if anneal_epoch == 0 else 'Resuming'
                        print(f"  [stabilize] done — {verb} λ annealing")

        if mode == 'anneal':
            anneal_epoch += 1

    # ------------------------------------------------------------------ #
    # Post-training output
    # ------------------------------------------------------------------ #
    if is_main:
        print("\nTraining complete.")

        sparsity = (code == 0).float().mean().item()
        active   = (code != 0).float().sum(dim=(1, 2, 3, 4)).mean().item()
        rel_err  = (
            (inputs - recon).pow(2).sum(dim=(1, 2, 3, 4)) /
            (inputs.pow(2).sum(dim=(1, 2, 3, 4)) + 1e-8)
        ).mean().item()
        k = mcfg['kernel_size']
        print(f"\n=== LCAConv3D ({mcfg['features']} atoms, kernel {k}³, λ={lca_inner.lambda_:.3f}) ===")
        comp_coeff_final = _P**3 / active if active > 0 else float('inf')
        print(f"  Sparsity (fraction zero):  {sparsity:.3f}")
        print(f"  Relative recon error:      {rel_err:.6f}")
        print(f"  Active coefficients/item:  {active:.1f} / {n_total}")
        print(f"  Comp(P³) [P³/active]: {comp_coeff_final:.4f}x")
        print(f"  Energy (L2 + L1):          {all_energy[-1]:.4f}  "
              f"(first batch: {all_energy[0]:.4f})")

        # Plot 1 — loss curves
        plot_start_epoch = cfg['output'].get('plot_start_epoch', 0)
        nb_plot          = len(dataloader)
        skip             = plot_start_epoch * nb_plot

        def _save_metrics_plot(data_lists, start_batch, title, filename):
            fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
            labels     = ['L2 Recon Error', 'L1 Sparsity', 'Total Energy', 'Relative Error']
            log_panels = {0, 2}   # L2 and Energy benefit from log scale
            for idx, (ax, values, label) in enumerate(zip(axes, data_lists, labels)):
                sliced = values[start_batch:]
                ax.plot(sliced)
                ax.set_ylabel(label)
                if idx in log_panels and any(v > 0 for v in sliced):
                    ax.set_yscale('log')
            if rel_err_target is not None:
                axes[3].axhline(rel_err_target, color='r', linestyle='--', linewidth=0.8,
                                label=f'target={rel_err_target}')
                axes[3].legend(fontsize=8)
            x0 = start_batch
            axes[-1].set_xlabel(f'Batch (across all epochs, starting batch {x0})')
            axes[0].set_title(title)
            plt.tight_layout()
            path = os.path.join(plots_dir, filename)
            plt.savefig(path)
            plt.close()
            print(f"Saved {path}")

        metric_lists = [all_l2, all_l1, all_energy, all_rel_err]
        _save_metrics_plot(metric_lists, 0,    'LCAConv3D — Training Metrics',
                           'training_metrics.png')
        if plot_start_epoch > 0:
            _save_metrics_plot(metric_lists, skip,
                               f'LCAConv3D — Training Metrics (from epoch {plot_start_epoch})',
                               'training_metrics_tail.png')

        # Plot 2 — dictionary atoms (mid-plane slice of each 3D kernel)
        # weights shape: (features, in_channels, kD, kH, kW)
        weights = lca_inner.get_weights().float().cpu().numpy()
        n_feat  = weights.shape[0]
        kD      = weights.shape[2]
        mid     = kD // 2                # show the central depth slice
        atoms   = weights[:, 0, mid, :, :]   # (features, kH, kW)

        cols = int(np.ceil(np.sqrt(n_feat)))
        rows = int(np.ceil(n_feat / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
        axes = np.array(axes).ravel()
        vmax = np.percentile(np.abs(atoms), 99)
        for i, ax in enumerate(axes):
            if i < n_feat:
                ax.imshow(atoms[i], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            ax.axis('off')
        fig.suptitle(f'Dictionary atoms — mid-plane slice  ({n_feat} atoms, kernel {k}³)',
                     fontsize=10)
        plt.tight_layout()
        out = os.path.join(plots_dir, 'dictionary_atoms.png')
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Saved {out}")

        # Plot 3 — full-volume reconstruction (3 orthogonal mid-plane slices)
        vol_np  = dset.vol.numpy()                  # (D, H, W), already normalised
        D, H, W = vol_np.shape
        nD, nH, nW = D // _P, H // _P, W // _P
        D_out, H_out, W_out = nD * _P, nH * _P, nW * _P

        input_vol = vol_np[:D_out, :H_out, :W_out].copy()
        recon_vol = np.zeros((D_out, H_out, W_out), dtype=np.float32)

        tiles = [(di*_P, hi*_P, wi*_P)
                 for di in range(nD) for hi in range(nH) for wi in range(nW)]

        with torch.no_grad():
            bs = min(len(tiles), 64)                    # fill GPU; ignore training batch_size
            for start in range(0, len(tiles), bs):
                batch_coords = tiles[start:start + bs]
                batch = torch.stack([
                    torch.from_numpy(vol_np[x:x+_P, y:y+_P, z:z+_P]).unsqueeze(0)
                    for x, y, z in batch_coords
                ]).to(dtype=dtype, device=device)       # (B, 1, P, P, P)
                _, _, recon_batch, _ = lca_inner(batch)
                recon_np = recon_batch.float().cpu().numpy()
                for i, (x, y, z) in enumerate(batch_coords):
                    recon_vol[x:x+_P, y:y+_P, z:z+_P] = recon_np[i, 0]

        mD, mH, mW = D_out // 2, H_out // 2, W_out // 2
        plane_defs = [
            ('XY (z=mid)', input_vol[:, :, mW],  recon_vol[:, :, mW]),
            ('XZ (y=mid)', input_vol[:, mH, :],  recon_vol[:, mH, :]),
            ('YZ (x=mid)', input_vol[mD, :, :],  recon_vol[mD, :, :]),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        fig.suptitle(
            f'LCA full-volume reconstruction  |  {mcfg["features"]} atoms  '
            f'λ={lca_inner.lambda_:.3f}  rel_err={avg_rel_err:.4f}  '
            f'CompRatio={comp_ratio:.2f}x  BPV={bpv:.2f}',
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
        out = os.path.join(plots_dir, 'full_volume_reconstruction.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {out}")

        print("\nDone.")
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        _log.close()

    if using_ddp:
        cleanup_ddp()


if __name__ == '__main__':
    main()
