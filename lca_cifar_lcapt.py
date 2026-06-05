"""
Convolutional LCA dictionary learning on CIFAR-10 using lcapt.
Re-implementation of lca-pytorch/examples/builtin_dictionary_learning_cifar.ipynb
as a reproducible script with config file, experiment logging, and saved plots.

Usage
-----
    python lca_cifar_lcapt.py [config_lcapt.yaml]

References
----------
    Rozell et al. (2008), Neural Computation 20, 2526-2563.
    lcapt: https://github.com/lanl/lca-pytorch
"""

import os
import shutil
import sys
import time
from datetime import datetime

import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from lcapt.analysis import make_feature_grid
from lcapt.lca import LCAConv2D
from lcapt.metric import compute_l1_sparsity, compute_l2_error

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'config_lcapt.yaml'
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

# ---------------------------------------------------------------------------
# Experiment directory
# ---------------------------------------------------------------------------

exp_dir   = os.path.join('experiments', 'lcapt_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
plots_dir = os.path.join(exp_dir, 'plots')
models_dir = os.path.join(exp_dir, 'models')
os.makedirs(plots_dir,  exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
shutil.copy(cfg_path, os.path.join(exp_dir, 'config_lcapt.yaml'))


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


_log = open(os.path.join(exp_dir, 'run.log'), 'w')
sys.stdout = _Tee(sys.__stdout__, _log)
sys.stderr = _Tee(sys.__stderr__, _log)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype  = torch.float16 if cfg['training']['dtype'] == 'float16' else torch.float32

print(f"Experiment dir: {exp_dir}")
print(f"Config:         {cfg_path}")
print(f"Device:         {device}  dtype={dtype}\n")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class _CIFARPNGDataset(Dataset):
    def __init__(self, image_glob):
        self.paths = sorted(glob.glob(image_glob))
        assert len(self.paths) > 0, f"No images found at {image_glob}"

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        arr = np.array(img, dtype=np.float32) / 255.0   # [0, 1], shape (H, W, C)
        return torch.from_numpy(arr.transpose(2, 0, 1))  # → (C, H, W)


dset = _CIFARPNGDataset(cfg['data']['image_glob'])
dataloader = DataLoader(
    dset,
    batch_size=cfg['data']['batch_size'],
    shuffle=True,
    num_workers=cfg['data']['num_workers'],
    pin_memory=torch.cuda.is_available(),
    persistent_workers=cfg['data']['num_workers'] > 0,
)
print(f"CIFAR-10: {len(dset)} images, {len(dataloader)} batches/epoch\n")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

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

print(f"LCAConv2D: {cfg['model']['features']} atoms, "
      f"kernel {cfg['model']['kernel_size']}x{cfg['model']['kernel_size']}, "
      f"stride {cfg['model']['stride']}, "
      f"λ={cfg['model']['lambda_']}\n")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

epochs         = cfg['training']['epochs']
anneal_every   = cfg['training']['lambda_anneal_every']
anneal_step    = cfg['training']['lambda_anneal_step']
print_freq     = cfg['training']['print_freq']

# track per-batch metrics for final loss plot
all_l2, all_l1, all_energy = [], [], []

for epoch in range(epochs):
    t0 = time.time()

    # lambda annealing: increase threshold every N epochs → sparser codes over time
    if epoch > 0 and epoch % anneal_every == 0:
        lca.lambda_ += anneal_step
        print(f"  [anneal] λ → {lca.lambda_:.3f}")

    ep_l2 = ep_l1 = ep_energy = ep_sparsity = ep_active = ep_rel_err = 0.0

    for images in dataloader:
        images = images.to(dtype=dtype, device=device)
        inputs, code, recon, recon_error = lca(images)
        lca.update_weights(code, recon_error)

        l1     = compute_l1_sparsity(code, lca.lambda_).item()
        l2     = compute_l2_error(inputs, recon).item()
        energy = l2 + l1
        all_l2.append(l2)
        all_l1.append(l1)
        all_energy.append(energy)

        n_total     = code.shape[1] * code.shape[2] * code.shape[3]
        ep_l2       += l2
        ep_l1       += l1
        ep_energy   += energy
        ep_sparsity += (code == 0).float().mean().item()
        ep_active   += (code != 0).float().sum(dim=(1, 2, 3)).mean().item()
        ep_rel_err  += (
            (inputs - recon).pow(2).sum(dim=(1, 2, 3)) /
            (inputs.pow(2).sum(dim=(1, 2, 3)) + 1e-8)
        ).mean().item()

    nb = len(dataloader)
    epoch_time = time.time() - t0
    print(f"Epoch {epoch:02d} | {epoch_time:.1f}s ({epoch_time/nb:.2f}s/batch) | "
          f"Sparsity: {ep_sparsity/nb:.3f}  "
          f"Active: {ep_active/nb:.1f}/{n_total}  "
          f"Rel.err: {ep_rel_err/nb:.6f}  "
          f"L2: {ep_l2/nb:.4f}  L1: {ep_l1/nb:.4f}  Energy: {ep_energy/nb:.4f}  "
          f"λ={lca.lambda_:.3f}")

    # save checkpoint after every epoch (overwrites — keeps only latest)
    torch.save(lca.state_dict(), os.path.join(models_dir, 'lca_cifar_lcapt.pth'))

print("\nTraining complete.")

# Final summary — same metrics as LCA_torch.py (evaluated on last batch)
n_total  = code.shape[1] * code.shape[2] * code.shape[3]
sparsity = (code == 0).float().mean().item()
active   = (code != 0).float().sum(dim=(1, 2, 3)).mean().item()
rel_err  = (
    (inputs - recon).pow(2).sum(dim=(1, 2, 3)) /
    (inputs.pow(2).sum(dim=(1, 2, 3)) + 1e-8)
).mean().item()
print(f"\n=== LCAConv2D ({cfg['model']['features']} atoms, "
      f"kernel {cfg['model']['kernel_size']}×{cfg['model']['kernel_size']}, "
      f"λ={lca.lambda_:.3f}) ===")
print(f"  Sparsity (fraction zero):  {sparsity:.3f}")
print(f"  Relative recon error:      {rel_err:.6f}")
print(f"  Active coefficients/item:  {active:.1f} / {n_total}")
print(f"  Energy (L2 + L1):          {all_energy[-1]:.4f}  "
      f"(first batch: {all_energy[0]:.4f})")

# ---------------------------------------------------------------------------
# Plot 1: loss curves
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
for ax, values, label in zip(axes, [all_l2, all_l1, all_energy],
                              ['L2 Recon Error', 'L1 Sparsity', 'Total Energy']):
    ax.plot(values)
    ax.set_ylabel(label)
ax.set_xlabel('Batch (across all epochs)')
axes[0].set_title('LCAConv2D — Training Metrics')
plt.tight_layout()
out = os.path.join(plots_dir, 'training_metrics.png')
plt.savefig(out)
plt.close()
print(f"Saved {out}")

# ---------------------------------------------------------------------------
# Plot 2: learned dictionary atoms
# ---------------------------------------------------------------------------

weight_grid = make_feature_grid(lca.get_weights())
plt.figure(figsize=(10, 10))
plt.imshow(weight_grid.float().cpu().numpy())
plt.axis('off')
plt.title(f'Learned dictionary atoms  ({cfg["model"]["features"]} features, '
          f'{cfg["model"]["kernel_size"]}×{cfg["model"]["kernel_size"]})')
plt.tight_layout()
out = os.path.join(plots_dir, 'dictionary_atoms.png')
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved {out}")

# ---------------------------------------------------------------------------
# Plot 3: reconstruction examples (last batch)
# ---------------------------------------------------------------------------

n = cfg['output']['n_images']
fig, axes = plt.subplots(n, 3, figsize=(6, 2 * n))
axes[0, 0].set_title('Input')
axes[0, 1].set_title('Reconstruction')
axes[0, 2].set_title('Recon Error')

def to_rgb(tensor):
    arr = tensor.float().cpu().numpy().transpose(1, 2, 0)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr

for i in range(n):
    # recon_error = input − recon, so input = recon + recon_error
    inp = (recon[i] + recon_error[i])
    axes[i, 0].imshow(to_rgb(inp))
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
