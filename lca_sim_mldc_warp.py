"""
WARP-LCA: Warm-started LCA with a CNN predictor.

A small 3D CNN (WARPPredictor3D) predicts the initial LCA membrane state from
each input patch instead of starting the ODE from zero.  Better initialization
means the LCA converges in fewer iterations, improving both training speed and
reconstruction quality at the same sparsity.

Per-batch training procedure
-----------------------------
1. Predictor gradient step (backprop):
     pred_states = predictor(inputs_norm)
     recon_short  = lca.encode_grad(inputs_norm, pred_states, k_grad_iters)
     loss         = MSE(recon_short, inputs_norm)
     Adam update on predictor weights.

2. Hebbian LCA step (no backprop):
     pred_states = predictor(inputs_norm)          [no_grad]
     code, recon = lca(inputs, initial_states=pred_states.detach())
     lca.update_weights(code, recon_error)         [Hebbian]

The predictor is disabled for the first warmup_epochs so the LCA dictionary
can bootstrap before the predictor starts learning.

Compression pipeline and inference are identical to lca_sim_mldc_SingleSnaptshot.py;
the trained LCA weights can be loaded directly into the inference script.

Usage
-----
  python lca_sim_mldc_warp.py config_warp.yaml
  torchrun --nproc_per_node=4 lca_sim_mldc_warp.py config_warp.yaml
"""

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
import torch.nn as nn
import torch.nn.functional as F
import yaml

from torch.utils.data import DataLoader, Dataset, DistributedSampler

from lcapt.lca import LCAConv3D
from lcapt.metric import compute_l1_sparsity, compute_l2_error
from lcapt.preproc import make_zero_mean, make_unit_var
from lcapt.util import check_equal_shapes


# ---------------------------------------------------------------------------
# WARP components
# ---------------------------------------------------------------------------

class WARPPredictor3D(nn.Module):
    """
    Lightweight 3D CNN: input patch → initial LCA membrane state.

    The last conv uses stride=lca_stride so the output spatial dimensions
    exactly match the LCA code dimensions for any (patch_size, stride) pair
    where patch_size is divisible by stride.

    Input  shape : (B, in_channels, P, P, P)
    Output shape : (B, out_features, P//stride, P//stride, P//stride)
    """

    def __init__(
        self,
        in_channels: int,
        out_features: int,
        stride: int,
        hidden: tuple = (32, 64),
    ):
        super().__init__()
        h = hidden
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, h[0], 3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(h[0]),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv3d(h[0], h[1], 3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(h[1]),
            nn.LeakyReLU(0.1, inplace=True),

            # Strided conv: same spatial downsampling as the LCA conv
            nn.Conv3d(h[1], out_features, 3, stride=stride, padding=1),
            # No final activation — LCA membrane states are unbounded reals
        )
        # Near-zero init on last layer so predictor starts as an approximate
        # no-op and gradually departs from zero-initialization.
        nn.init.trunc_normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WARPLCAConv3D(LCAConv3D):
    """
    LCAConv3D subclass with two changes:

    1. _init_states does NOT detach the provided initial_states tensor,
       allowing gradients to flow back to the predictor during the predictor
       training step.  The caller is responsible for passing
       initial_states.detach() when gradient flow is not desired
       (e.g. the Hebbian update step).

    2. encode_grad runs the LCA Euler loop for n_iters steps while keeping
       gradients, and returns the reconstruction.  LCA weights are treated as
       constants (detached) so gradients flow only to the predictor.
    """

    def _init_states(
        self,
        input_drive: torch.Tensor,
        initial_states=None,
    ) -> torch.Tensor:
        if initial_states is None:
            return torch.zeros_like(input_drive, requires_grad=self.req_grad)
        check_equal_shapes(input_drive, initial_states)
        return initial_states   # preserve gradient connectivity

    def encode_grad(
        self,
        inputs_norm: torch.Tensor,
        initial_states: torch.Tensor,
        n_iters: int,
    ) -> torch.Tensor:
        """
        Run n_iters LCA Euler steps from initial_states keeping gradient
        connectivity to the predictor.  Returns the reconstruction.

        LCA weights are detached so no gradients accumulate on the dictionary.
        Backprop through tau=100, n_iters=20 retains ~82% of the gradient
        magnitude at initial_states; larger tau or fewer iters improve this.
        """
        w = self.weights.detach()
        input_drive  = self.compute_input_drive(inputs_norm, w)
        connectivity = self.compute_lateral_connectivity(w)
        states = initial_states

        for _ in range(n_iters):
            acts   = self.transfer(states)
            inhib  = self.lateral_competition(acts, connectivity)
            states = states + (1.0 / self.tau) * (input_drive - states - inhib)

        acts  = self.transfer(states)
        return self.compute_recon(acts, w)


# ---------------------------------------------------------------------------
# Dataset / Logging (identical to lca_sim_mldc_SingleSnaptshot.py)
# ---------------------------------------------------------------------------

class _HDF5PatchDataset(Dataset):
    def __init__(self, h5_path, field_key, timestep, patch_size, n_patches):
        with h5py.File(h5_path, 'r') as f:
            vol = f[field_key][timestep]
        vol = (vol - vol.mean()) / (vol.std() + 1e-8)
        self.vol        = torch.from_numpy(vol.astype(np.float32))
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
        return self.vol[x:x+p, y:y+p, z:z+p].unsqueeze(0)


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
# Main
# ---------------------------------------------------------------------------

def main():
    using_ddp = dist.is_available() and 'LOCAL_RANK' in os.environ
    if using_ddp:
        rank, local_rank, world_size = setup_ddp()
        device = torch.device(f'cuda:{local_rank}')
    else:
        rank = local_rank = 0
        world_size = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_main = (rank == 0)

    cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'config_warp.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # ---- Experiment directory ----
    if is_main:
        exp_dir = os.path.join(
            'experiments', 'warp_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
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
        shutil.copy(cfg_path, os.path.join(exp_dir, 'config_warp.yaml'))
        _log       = open(os.path.join(exp_dir, 'run.log'), 'w')
        sys.stdout = _Tee(sys.__stdout__, _log)
        sys.stderr = _Tee(sys.__stderr__, _log)

    if using_ddp:
        dist.barrier()

    dtype_str = cfg['training'].get('dtype', 'float32')
    dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16}.get(
        dtype_str, torch.float32
    )

    if is_main:
        print(f"Experiment dir : {exp_dir}")
        print(f"Config         : {cfg_path}")
        print(f"Device         : {device}  dtype={dtype}")
        print(f"GPUs           : {world_size}\n")

    # ---- Data ----
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
        batch_size         = dcfg['batch_size'],
        shuffle            = (sampler is None),
        sampler            = sampler,
        num_workers        = dcfg['num_workers'],
        pin_memory         = True,
        persistent_workers = dcfg['num_workers'] > 0,
    )

    if is_main:
        nx, ny, nz = dset.vol.shape
        print(f"Volume   : {nx}×{ny}×{nz}  |  field={dcfg['field_key']}  t={dcfg['timestep']}")
        print(f"Patches  : {dcfg['n_patches']} patches/epoch  |  size={dcfg['patch_size']}³")
        print(f"Batches  : {len(dataloader)}/GPU/epoch\n")

    # ---- Models ----
    mcfg = cfg['model']
    wcfg = cfg.get('warp', {})

    lca = WARPLCAConv3D(
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

    # Optional pretrained LCA weights (train predictor on top of existing dict)
    load_lca = wcfg.get('load_lca', None)
    if load_lca:
        lca.load_state_dict(torch.load(load_lca, map_location=device))
        if is_main:
            print(f"Loaded pretrained LCA from {load_lca}")

    if using_ddp:
        dist.broadcast(lca.weights.data, src=0)

    lca_inner = lca

    # Predictor
    pred_channels  = wcfg.get('pred_channels', [32, 64])
    pred_lr        = wcfg.get('pred_lr', 3e-4)
    k_grad         = wcfg.get('k_grad', 20)
    warmup_epochs  = wcfg.get('warmup_epochs', 10)
    pred_every     = wcfg.get('pred_every', 1)
    lca_iters_warm = wcfg.get('lca_iters_warm', mcfg['lca_iters'])

    predictor = WARPPredictor3D(
        in_channels  = mcfg['in_channels'],
        out_features = mcfg['features'],
        stride       = mcfg['stride'],
        hidden       = tuple(pred_channels),
    ).to(dtype=dtype, device=device)

    if using_ddp:
        for p in predictor.parameters():
            dist.broadcast(p.data, src=0)

    pred_optimizer = torch.optim.Adam(predictor.parameters(), lr=pred_lr)

    n_pred_params = sum(p.numel() for p in predictor.parameters())
    k = mcfg['kernel_size']
    if is_main:
        print(f"WARPLCAConv3D : {mcfg['features']} atoms | kernel {k}³ | "
              f"stride {mcfg['stride']} | λ={mcfg['lambda_']} | "
              f"lca_iters={mcfg['lca_iters']} (warm={lca_iters_warm})")
        print(f"WARPPredictor : {n_pred_params:,} params | "
              f"hidden={pred_channels} | k_grad={k_grad} | "
              f"warmup_epochs={warmup_epochs}\n")

    # ---- Training loop config ----
    max_epochs       = cfg['training']['max_epochs']
    anneal_every     = cfg['training']['lambda_anneal_every']
    anneal_step      = cfg['training']['lambda_anneal_step']
    anneal_start     = cfg['training'].get('lambda_anneal_start', 0)
    anneal_stop      = cfg['training'].get('lambda_anneal_stop', max_epochs)
    rel_err_target   = cfg['training'].get('rel_err_target', None)
    rel_err_ceiling  = cfg['training'].get('rel_err_ceiling', 0.01)
    stabilize_epochs = cfg['training'].get('stabilize_epochs', 10)

    all_l2, all_l1, all_energy, all_rel_err = [], [], [], []
    all_pred_loss = []    # epoch-level averages

    # Compression constants (COO sparse storage, patch level)
    _P            = dcfg['patch_size']
    _n_code_patch = mcfg['features'] * (_P // mcfg['stride'])**3
    _index_bits   = int(np.ceil(np.log2(_n_code_patch + 1)))
    _bytes_per_nz = 4 + (_index_bits + 7) // 8
    _bytes_in     = _P**3 * 4

    # State machine
    mode             = 'pre' if rel_err_target is not None else 'anneal'
    stab_count       = 0
    anneal_epoch     = 0
    last_avg_rel_err = 0.0
    best_comp_ratio  = 0.0
    best_lambda      = mcfg['lambda_']
    warp_active      = False   # enabled after warmup_epochs

    for epoch in range(max_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        t0 = time.time()

        # Activate predictor + switch to reduced lca_iters after warmup
        if not warp_active and epoch >= warmup_epochs:
            warp_active = True
            lca_inner.lca_iters = lca_iters_warm
            if is_main:
                print(f"  [warp] predictor activated at epoch {epoch}  "
                      f"lca_iters → {lca_iters_warm}")

        # Annealing step (state-machine variant or legacy fixed schedule)
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
        ep_pred_loss = 0.0
        n_pred_steps = 0
        n_code_patch = None

        for batch_idx, patches in enumerate(dataloader):
            patches = patches.to(dtype=dtype, device=device)   # (B, 1, P, P, P)

            # Normalize once — both predictor and LCA see the same input
            inputs_norm = make_unit_var(make_zero_mean(patches))

            # ----------------------------------------------------------
            # Step 1 — Predictor gradient update (truncated backprop)
            # ----------------------------------------------------------
            if warp_active and (batch_idx % pred_every == 0):
                predictor.train()
                pred_states = predictor(inputs_norm)
                recon_short = lca.encode_grad(inputs_norm, pred_states, k_grad)
                pred_loss   = F.mse_loss(recon_short, inputs_norm)

                pred_optimizer.zero_grad()
                pred_loss.backward()

                if using_ddp:
                    for p in predictor.parameters():
                        if p.grad is not None:
                            dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                            p.grad.data /= world_size

                pred_optimizer.step()
                ep_pred_loss += pred_loss.item()
                n_pred_steps += 1

            # ----------------------------------------------------------
            # Step 2 — Hebbian LCA update (full iters, warm start)
            # ----------------------------------------------------------
            predictor.eval()
            if warp_active:
                with torch.no_grad():
                    pred_states_hebb = predictor(inputs_norm)
                inputs_out, code, recon, recon_error = lca(
                    patches,
                    initial_states=pred_states_hebb.detach(),
                )
            else:
                inputs_out, code, recon, recon_error = lca(patches)

            lca_inner.update_weights(code, recon_error)

            if using_ddp:
                dist.all_reduce(lca_inner.weights.data, op=dist.ReduceOp.SUM)
                lca_inner.weights.data /= world_size
                lca_inner.normalize_weights()

            l1     = compute_l1_sparsity(code, lca_inner.lambda_).item()
            l2     = compute_l2_error(inputs_out, recon).item()
            energy = l2 + l1

            if is_main:
                all_l2.append(l2)
                all_l1.append(l1)
                all_energy.append(energy)
                all_rel_err.append(
                    ((inputs_out - recon).pow(2).sum(dim=(1, 2, 3, 4)) /
                     (inputs_out.pow(2).sum(dim=(1, 2, 3, 4)) + 1e-8)).mean().item()
                )

            if n_code_patch is None:
                n_code_patch = (code.shape[1] * code.shape[2]
                                * code.shape[3] * code.shape[4])

            ep_l2      += l2
            ep_l1      += l1
            ep_energy  += energy
            ep_sparsity += (code == 0).float().mean().item()
            ep_active  += (code != 0).float().sum(dim=(1, 2, 3, 4)).mean().item()
            ep_rel_err += (
                (inputs_out - recon).pow(2).sum(dim=(1, 2, 3, 4)) /
                (inputs_out.pow(2).sum(dim=(1, 2, 3, 4)) + 1e-8)
            ).mean().item()

        nb         = len(dataloader)
        epoch_time = time.time() - t0

        avg_rel_err   = ep_rel_err / nb
        avg_active    = ep_active  / nb
        avg_pred_loss = ep_pred_loss / n_pred_steps if n_pred_steps > 0 else 0.0
        bytes_sparse  = avg_active * _bytes_per_nz
        comp_ratio    = _bytes_in / bytes_sparse if bytes_sparse > 0 else float('inf')
        bpv           = bytes_sparse * 8 / _P**3

        if is_main:
            all_pred_loss.append(avg_pred_loss)

            if mode == 'stabilize':
                mode_tag = f"  [stabilize {stab_count}/{stabilize_epochs}]"
            elif mode == 'anneal':
                mode_tag = f"  [anneal ep {anneal_epoch}/{anneal_stop}]"
            else:
                mode_tag = "  [pre-anneal]"

            warp_tag = (f"  pred_loss={avg_pred_loss:.4f}"
                        if warp_active else "  [warmup]")

            print(
                f"Epoch {epoch:02d} | {epoch_time:.1f}s ({epoch_time/nb:.2f}s/batch) | "
                f"Sparsity: {ep_sparsity/nb:.3f}  Active: {avg_active:.1f}/{n_code_patch}  "
                f"Rel.err: {avg_rel_err:.6f}  L2: {ep_l2/nb:.4f}  L1: {ep_l1/nb:.4f}  "
                f"Energy: {ep_energy/nb:.4f}  λ={lca_inner.lambda_:.3f}  "
                f"CompRatio: {comp_ratio:.2f}x  BPV: {bpv:.2f}"
                + mode_tag + warp_tag
            )

            torch.save(lca_inner.state_dict(),
                       os.path.join(models_dir, 'warp_lca.pth'))
            torch.save(predictor.state_dict(),
                       os.path.join(models_dir, 'warp_predictor.pth'))

            if avg_rel_err <= rel_err_ceiling and comp_ratio > best_comp_ratio:
                best_comp_ratio = comp_ratio
                best_lambda     = lca_inner.lambda_
                torch.save(lca_inner.state_dict(),
                           os.path.join(models_dir, 'warp_lca_best.pth'))
                torch.save(predictor.state_dict(),
                           os.path.join(models_dir, 'warp_predictor_best.pth'))
                print(f"  [best] CompRatio={best_comp_ratio:.2f}x  "
                      f"λ={best_lambda:.3f}  rel_err={avg_rel_err:.4f}  BPV={bpv:.2f}")

        last_avg_rel_err = avg_rel_err

        # ---- state machine ----
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
            (inputs_out - recon).pow(2).sum(dim=(1, 2, 3, 4)) /
            (inputs_out.pow(2).sum(dim=(1, 2, 3, 4)) + 1e-8)
        ).mean().item()
        print(f"\n=== WARPLCAConv3D ({mcfg['features']} atoms, kernel {k}³, "
              f"λ={lca_inner.lambda_:.3f}) ===")
        print(f"  Sparsity (fraction zero):  {sparsity:.3f}")
        print(f"  Relative recon error:      {rel_err:.6f}")
        print(f"  Active coefficients/item:  {active:.1f} / {n_code_patch}")
        print(f"  Best compression ratio:    {best_comp_ratio:.2f}x  λ={best_lambda:.3f}")

        plot_start_epoch = cfg['output'].get('plot_start_epoch', 0)
        nb_plot          = len(dataloader)
        skip             = plot_start_epoch * nb_plot

        def _save_metrics_plot(data_lists, start_batch, title, filename):
            labels     = ['L2 Recon Error', 'L1 Sparsity', 'Total Energy', 'Relative Error']
            log_panels = {0, 2}
            fig, axes  = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
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
            axes[-1].set_xlabel(f'Batch (starting batch {start_batch})')
            axes[0].set_title(title)
            plt.tight_layout()
            path = os.path.join(plots_dir, filename)
            plt.savefig(path)
            plt.close()
            print(f"Saved {path}")

        metric_lists = [all_l2, all_l1, all_energy, all_rel_err]
        _save_metrics_plot(metric_lists, 0,
                           'WARP-LCA — Training Metrics', 'training_metrics.png')
        if plot_start_epoch > 0:
            _save_metrics_plot(metric_lists, skip,
                               f'WARP-LCA — Metrics (from epoch {plot_start_epoch})',
                               'training_metrics_tail.png')

        # Predictor loss (epoch-level)
        if any(v > 0 for v in all_pred_loss):
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(all_pred_loss, label='pred MSE loss')
            ax.axvline(warmup_epochs, color='gray', linestyle=':', linewidth=1,
                       label=f'predictor active (ep {warmup_epochs})')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Predictor MSE Loss')
            ax.set_title('WARP Predictor Training Loss')
            ax.set_yscale('log')
            ax.legend(fontsize=8)
            plt.tight_layout()
            out = os.path.join(plots_dir, 'predictor_loss.png')
            plt.savefig(out)
            plt.close()
            print(f"Saved {out}")

        # Dictionary atoms (mid-depth slice)
        weights = lca_inner.get_weights().float().cpu().numpy()
        n_feat  = weights.shape[0]
        mid     = weights.shape[2] // 2
        atoms   = weights[:, 0, mid, :, :]

        cols = int(np.ceil(np.sqrt(n_feat)))
        rows = int(np.ceil(n_feat / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
        axes = np.array(axes).ravel()
        vmax = np.percentile(np.abs(atoms), 99)
        for i, ax in enumerate(axes):
            if i < n_feat:
                ax.imshow(atoms[i], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            ax.axis('off')
        fig.suptitle(
            f'Dictionary atoms — mid-plane slice  ({n_feat} atoms, kernel {k}³)',
            fontsize=10
        )
        plt.tight_layout()
        out = os.path.join(plots_dir, 'dictionary_atoms.png')
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Saved {out}")

        # Reconstruction examples
        def mid_slice(t):
            return t.float().cpu().numpy()[0, t.shape[1] // 2]

        n = min(cfg['output']['n_images'], inputs_out.shape[0])
        fig, axes = plt.subplots(n, 3, figsize=(6, 2 * n))
        if n == 1:
            axes = axes[np.newaxis, :]
        axes[0, 0].set_title('Input (mid-slice)')
        axes[0, 1].set_title('Reconstruction')
        axes[0, 2].set_title('Recon Error')
        for i in range(n):
            inp_s  = mid_slice(recon[i] + recon_error[i])
            rec_s  = mid_slice(recon[i])
            err_s  = mid_slice(recon_error[i])
            vmax_i = np.percentile(np.abs(inp_s), 99)
            for ax, data in zip(axes[i], [inp_s, rec_s, err_s]):
                ax.imshow(data, cmap='RdBu_r', vmin=-vmax_i, vmax=vmax_i)
                ax.axis('off')
        fig.suptitle(
            f'WARP-LCA Reconstructions  |  t={dcfg["timestep"]}  '
            f'rel_err={avg_rel_err:.4f}  CompRatio={comp_ratio:.2f}x  BPV={bpv:.2f}',
            fontsize=9
        )
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
