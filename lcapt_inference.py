"""
lcapt inference — loads a saved LCAConv2D (lca_cifar_lcapt.pth) and runs sparse coding
on a fresh random batch of CIFAR-10 test images.

Automatically reads the final λ from the experiment's run.log and switches the
image glob to the test split.  Results are saved inside the experiment directory
under an inference/ subfolder (timestamped to allow multiple runs).

Usage
-----
    python lcapt_inference.py <path/to/models/lca_cifar_lcapt.pth>
"""

import glob
import os
import re
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from lcapt.lca import LCAConv2D
from lcapt.metric import compute_l1_sparsity, compute_l2_error

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

if len(sys.argv) < 2:
    print("Usage: python lcapt_inference.py <path/to/models/lca_cifar_lcapt.pth>")
    sys.exit(1)

pth_path = sys.argv[1]

# Derive experiment directory: models/ is one level below the experiment root
exp_dir = os.path.dirname(os.path.dirname(os.path.abspath(pth_path)))

# ---------------------------------------------------------------------------
# Load and patch config
#   - switch image_glob to test split
#   - replace lambda_ with the final trained value from run.log
# ---------------------------------------------------------------------------

cfg_path = os.path.join(exp_dir, 'config_lcapt.yaml')
if not os.path.exists(cfg_path):
    print(f"Error: config not found at {cfg_path}")
    sys.exit(1)

with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

# 1. Switch to test set
cfg['data']['image_glob'] = 'cifar-10-images/test/*/*.png'

# 2. Read final lambda from run.log
log_path = os.path.join(exp_dir, 'run.log')
final_lambda = None
if os.path.exists(log_path):
    with open(log_path) as f:
        content = f.read()
    # Last occurrence of λ=<value> in the log (appears on every epoch line)
    matches = re.findall(r'λ=([0-9]+\.[0-9]+)', content)
    if matches:
        final_lambda = float(matches[-1])

if final_lambda is not None:
    cfg['model']['lambda_'] = final_lambda
else:
    print(f"Warning: could not parse final λ from {log_path}, using config value {cfg['model']['lambda_']}")

# ---------------------------------------------------------------------------
# Inference output directory — inside the experiment folder
# ---------------------------------------------------------------------------

inf_dir   = os.path.join(exp_dir, 'inference', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
plots_dir = os.path.join(inf_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)

# Save the patched config so the inference run is fully reproducible
patched_cfg_path = os.path.join(inf_dir, 'config_lcapt.yaml')
with open(patched_cfg_path, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


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


_log = open(os.path.join(inf_dir, 'run.log'), 'w')
sys.stdout = _Tee(sys.__stdout__, _log)
sys.stderr = _Tee(sys.__stderr__, _log)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype  = torch.float16 if cfg['training']['dtype'] == 'float16' else torch.float32

print(f"Experiment dir:  {exp_dir}")
print(f"Inference dir:   {inf_dir}")
print(f"Model:           {pth_path}")
print(f"Config:          {cfg_path}")
print(f"Device:          {device}  dtype={dtype}")
print(f"λ (final):       {cfg['model']['lambda_']}")
print(f"Image glob:      {cfg['data']['image_glob']}\n")

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

lca = LCAConv2D(
    out_neurons=cfg['model']['features'],
    in_neurons=cfg['model']['in_channels'],
    result_dir=os.path.join(inf_dir, 'lca_results'),
    kernel_size=cfg['model']['kernel_size'],
    stride=cfg['model']['stride'],
    lambda_=cfg['model']['lambda_'],
    tau=cfg['model']['tau'],
    lca_iters=cfg['model']['lca_iters'],
    return_vars=['inputs', 'acts', 'recons', 'recon_errors'],
).to(dtype=dtype, device=device)

lca.load_state_dict(torch.load(pth_path, map_location=device, weights_only=True))
lca.eval()

print(f"Loaded LCAConv2D  weights shape: {lca.weights.shape}\n")

# ---------------------------------------------------------------------------
# Load test images
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
        return torch.from_numpy(arr.transpose(2, 0, 1))

dset   = _CIFARPNGDataset(cfg['data']['image_glob'])
loader = DataLoader(dset, batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=0)
images = next(iter(loader)).to(dtype=dtype, device=device)
print(f"Test set: {len(dset)} images  |  batch shape: {images.shape}\n")

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

with torch.no_grad():
    inputs, code, recon, recon_error = lca(images)

n_total  = code.shape[1] * code.shape[2] * code.shape[3]
sparsity = (code == 0).float().mean().item()
active   = (code != 0).float().sum(dim=(1, 2, 3)).mean().item()
l1       = compute_l1_sparsity(code, lca.lambda_).item()
l2       = compute_l2_error(inputs, recon).item()
rel_err  = (
    (inputs - recon).pow(2).sum(dim=(1, 2, 3)) /
    (inputs.pow(2).sum(dim=(1, 2, 3)) + 1e-8)
).mean().item()

print(f"=== LCAConv2D inference  (λ={lca.lambda_:.3f}) ===")
print(f"  Sparsity (fraction zero):  {sparsity:.3f}")
print(f"  Relative recon error:      {rel_err:.6f}")
print(f"  Active coefficients/item:  {active:.1f} / {n_total}")
print(f"  L2 recon error:            {l2:.4f}")
print(f"  L1 sparsity cost:          {l1:.4f}")
print(f"  Total energy (L2+L1):      {l2+l1:.4f}")
print()

# ---------------------------------------------------------------------------
# Reconstruction plot
# ---------------------------------------------------------------------------

n = cfg['output']['n_images']

def to_rgb(tensor):
    arr = tensor.float().cpu().numpy().transpose(1, 2, 0)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr

fig, axes = plt.subplots(n, 3, figsize=(6, 2 * n))
axes[0, 0].set_title('Input')
axes[0, 1].set_title('Reconstruction')
axes[0, 2].set_title('Recon Error')

for i in range(n):
    inp = recon[i] + recon_error[i]
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
