"""
Convolutional LCA dictionary learning on CIFAR-10 using lcapt.
Supports single-GPU and multi-GPU training via torch.distributed (manual all_reduce).

Usage — single GPU:
    python lca_cifar_lcapt.py [config_lcapt.yaml]

Usage — N GPUs (e.g. 4):
    torchrun --nproc_per_node=4 lca_cifar_lcapt.py [config_lcapt.yaml]

References
----------
    Rozell et al. (2008), Neural Computation 20, 2526-2563.
    lcapt: https://github.com/lanl/lca-pytorch
"""

import glob
import os
import shutil
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import yaml
from PIL import Image

from torch.utils.data import DataLoader, Dataset, DistributedSampler

from lcapt.analysis import make_feature_grid
from lcapt.lca import LCAConv2D
from lcapt.metric import compute_l1_sparsity, compute_l2_error


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def setup_ddp() -> tuple[int, int, int]:
    """Initialise NCCL process group from env vars set by torchrun.
    Returns (rank, local_rank, world_size).
    """
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), local_rank, dist.get_world_size()


def cleanup_ddp() -> None:
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class _CIFARPNGDataset(Dataset):
    def __init__(self, image_glob):
        self.paths = sorted(glob.glob(image_glob))
        assert len(self.paths) > 0, f"No images found at {image_glob}"

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr.transpose(2, 0, 1))  # (C, H, W)


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
# Main
# ---------------------------------------------------------------------------

def main():
    # ------------------------------------------------------------------ #
    # DDP initialisation — gracefully falls back to single-GPU/CPU
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
    # Config
    # ------------------------------------------------------------------ #
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'config_lcapt.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # ------------------------------------------------------------------ #
    # Experiment directory
    # Rank 0 generates the timestamped name and broadcasts it so all
    # ranks write to the exact same directory regardless of clock skew.
    # ------------------------------------------------------------------ #
    if is_main:
        exp_dir = os.path.join(
            'experiments', 'lcapt_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
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
        shutil.copy(cfg_path, os.path.join(exp_dir, 'config_lcapt.yaml'))
        _log       = open(os.path.join(exp_dir, 'run.log'), 'w')
        sys.stdout = _Tee(sys.__stdout__, _log)
        sys.stderr = _Tee(sys.__stderr__, _log)

    # Let all ranks wait until rank 0 has created the directories
    if using_ddp:
        dist.barrier()

    # ------------------------------------------------------------------ #
    # dtype
    # ------------------------------------------------------------------ #
    dtype_str = cfg['training'].get('dtype', 'float32')
    if dtype_str == 'float16':
        dtype = torch.float16
    elif dtype_str == 'bfloat16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    if is_main:
        print(f"Experiment dir : {exp_dir}")
        print(f"Config         : {cfg_path}")
        print(f"Device         : {device}  dtype={dtype}")
        print(f"GPUs           : {world_size}\n")

    # ------------------------------------------------------------------ #
    # Data
    # Each GPU gets a non-overlapping slice via DistributedSampler.
    # Effective batch size across all GPUs = batch_size * world_size.
    # ------------------------------------------------------------------ #
    dset    = _CIFARPNGDataset(cfg['data']['image_glob'])
    sampler = (
        DistributedSampler(dset, num_replicas=world_size, rank=rank, shuffle=True)
        if using_ddp else None
    )
    dataloader = DataLoader(
        dset,
        batch_size=cfg['data']['batch_size'],
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg['data']['num_workers'],
        pin_memory=True,
        persistent_workers=cfg['data']['num_workers'] > 0,
    )

    if is_main:
        print(f"CIFAR-10 : {len(dset)} images | "
              f"{len(dataloader)} batches/GPU/epoch | "
              f"effective batch = {cfg['data']['batch_size'] * world_size}\n")

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    lca = LCAConv2D(
        out_neurons=cfg['model']['features'],
        in_neurons=cfg['model']['in_channels'],
        result_dir=os.path.join(exp_dir, 'lca_results'),
        kernel_size=cfg['model']['kernel_size'],
        stride=cfg['model']['stride'],
        lambda_=cfg['model']['lambda_'],
        tau=cfg['model']['tau'],
        lca_iters=cfg['model']['lca_iters'],
        eta=cfg['model']['learning_rate'],
        track_metrics=False,
        return_vars=['inputs', 'acts', 'recons', 'recon_errors'],
    ).to(dtype=dtype, device=device)

    # Broadcast initial weights from rank 0 so all replicas start identically
    if using_ddp:
        dist.broadcast(lca.weights.data, src=0)

    # LCAConv2D uses req_grad=False (Hebbian update, not backprop), so DDP
    # cannot wrap it (PyTorch rejects modules with no grad-tracked parameters).
    # Each rank runs its own forward pass; weights are kept in sync manually
    # via all_reduce after every Hebbian update.
    lca_ddp   = lca
    lca_inner = lca

    if is_main:
        print(f"LCAConv2D : {cfg['model']['features']} atoms | "
              f"kernel {cfg['model']['kernel_size']}×{cfg['model']['kernel_size']} | "
              f"stride {cfg['model']['stride']} | "
              f"λ={cfg['model']['lambda_']}\n")

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    epochs        = cfg['training']['epochs']
    anneal_every  = cfg['training']['lambda_anneal_every']
    anneal_step   = cfg['training']['lambda_anneal_step']
    anneal_start  = cfg['training'].get('lambda_anneal_start', 0)
    anneal_stop   = cfg['training'].get('lambda_anneal_stop', epochs)

    all_l2, all_l1, all_energy = [], [], []

    for epoch in range(epochs):
        # Ensures each epoch gets a different random shuffle on every GPU
        if sampler is not None:
            sampler.set_epoch(epoch)

        t0 = time.time()

        # Warmup phase: hold λ fixed until lambda_anneal_start epochs have passed,
        # then anneal every lambda_anneal_every epochs thereafter.
        if epoch > 0 and anneal_start <= epoch < anneal_stop and (epoch - anneal_start) % anneal_every == 0:
            lca_inner.lambda_ += anneal_step
            if is_main:
                print(f"  [anneal] λ → {lca_inner.lambda_:.3f}")

        ep_l2 = ep_l1 = ep_energy = ep_sparsity = ep_active = ep_rel_err = 0.0

        for images in dataloader:
            images = images.to(dtype=dtype, device=device)
            inputs, code, recon, recon_error = lca_ddp(images)

            # Hebbian weight update (in-place, bypasses autograd)
            lca_inner.update_weights(code, recon_error)

            # Manually average weights across all GPUs to keep replicas in sync.
            # update_weights already normalises, but we normalise again after
            # averaging because the average of unit-norm vectors is not unit-norm.
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

            n_total      = code.shape[1] * code.shape[2] * code.shape[3]
            ep_l2       += l2
            ep_l1       += l1
            ep_energy   += energy
            ep_sparsity += (code == 0).float().mean().item()
            ep_active   += (code != 0).float().sum(dim=(1, 2, 3)).mean().item()
            ep_rel_err  += (
                (inputs - recon).pow(2).sum(dim=(1, 2, 3)) /
                (inputs.pow(2).sum(dim=(1, 2, 3)) + 1e-8)
            ).mean().item()

        nb         = len(dataloader)
        epoch_time = time.time() - t0

        if is_main:
            print(
                f"Epoch {epoch:02d} | {epoch_time:.1f}s ({epoch_time/nb:.2f}s/batch) | "
                f"Sparsity: {ep_sparsity/nb:.3f}  "
                f"Active: {ep_active/nb:.1f}/{n_total}  "
                f"Rel.err: {ep_rel_err/nb:.6f}  "
                f"L2: {ep_l2/nb:.4f}  L1: {ep_l1/nb:.4f}  "
                f"Energy: {ep_energy/nb:.4f}  λ={lca_inner.lambda_:.3f}"
            )
            torch.save(lca_inner.state_dict(), os.path.join(models_dir, 'lca_cifar_lcapt.pth'))

    # ------------------------------------------------------------------ #
    # Post-training output — rank 0 only
    # ------------------------------------------------------------------ #
    if is_main:
        print("\nTraining complete.")

        sparsity = (code == 0).float().mean().item()
        active   = (code != 0).float().sum(dim=(1, 2, 3)).mean().item()
        rel_err  = (
            (inputs - recon).pow(2).sum(dim=(1, 2, 3)) /
            (inputs.pow(2).sum(dim=(1, 2, 3)) + 1e-8)
        ).mean().item()
        print(f"\n=== LCAConv2D ({cfg['model']['features']} atoms, "
              f"kernel {cfg['model']['kernel_size']}×{cfg['model']['kernel_size']}, "
              f"λ={lca_inner.lambda_:.3f}) ===")
        print(f"  Sparsity (fraction zero):  {sparsity:.3f}")
        print(f"  Relative recon error:      {rel_err:.6f}")
        print(f"  Active coefficients/item:  {active:.1f} / {n_total}")
        print(f"  Energy (L2 + L1):          {all_energy[-1]:.4f}  "
              f"(first batch: {all_energy[0]:.4f})")

        # Plot 1 — loss curves
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for ax, values, label in zip(
            axes, [all_l2, all_l1, all_energy],
            ['L2 Recon Error', 'L1 Sparsity', 'Total Energy']
        ):
            ax.plot(values)
            ax.set_ylabel(label)
        axes[-1].set_xlabel('Batch (across all epochs)')
        axes[0].set_title('LCAConv2D — Training Metrics')
        plt.tight_layout()
        out = os.path.join(plots_dir, 'training_metrics.png')
        plt.savefig(out)
        plt.close()
        print(f"Saved {out}")

        # Plot 2 — learned dictionary atoms
        weight_grid = make_feature_grid(lca_inner.get_weights())
        plt.figure(figsize=(10, 10))
        plt.imshow(weight_grid.float().cpu().numpy())
        plt.axis('off')
        plt.title(
            f'Learned dictionary atoms  ({cfg["model"]["features"]} features, '
            f'{cfg["model"]["kernel_size"]}×{cfg["model"]["kernel_size"]})'
        )
        plt.tight_layout()
        out = os.path.join(plots_dir, 'dictionary_atoms.png')
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Saved {out}")

        # Plot 3 — reconstruction examples (last batch)
        def to_rgb(tensor):
            arr = tensor.float().cpu().numpy().transpose(1, 2, 0)
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            return arr

        n = cfg['output']['n_images']
        fig, axes = plt.subplots(n, 3, figsize=(6, 2 * n))
        axes[0, 0].set_title('Input')
        axes[0, 1].set_title('Reconstruction')
        axes[0, 2].set_title('Recon Error')
        for i in range(n):
            axes[i, 0].imshow(to_rgb(recon[i] + recon_error[i]))
            axes[i, 1].imshow(to_rgb(recon[i]))
            axes[i, 2].imshow(to_rgb(recon_error[i]))
            for ax in axes[i]:
                ax.axis('off')
        plt.tight_layout()
        out = os.path.join(plots_dir, 'reconstructions.png')
        plt.savefig(out)
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
