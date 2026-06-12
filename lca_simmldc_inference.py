"""
Inference for the 3D LCA model trained by lca_sim_mldc_SingleSnaptshot.py.

Automatically reads the final λ from run.log and runs inference on a batch of
random 3D patches from the HDF5 volume.  Optionally evaluates on a different
timestep than training to test generalisation.

Results are saved inside the experiment directory under inference/<datetime>/.

Usage
-----
    # same timestep as training
    python lca_simmldc_inference.py <path/to/models/lca_simmldc.pth>

    # different timestep (e.g. t=20) to test generalisation
    python lca_simmldc_inference.py <path/to/models/lca_simmldc.pth> 20
"""

import os
import re
import sys
from datetime import datetime

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from torch.utils.data import DataLoader

from lcapt.lca import LCAConv3D
from lcapt.metric import compute_l1_sparsity, compute_l2_error

# ---------------------------------------------------------------------------
# Full-volume reconstruction
# ---------------------------------------------------------------------------

def reconstruct_full_volume(lca, h5_path, field_key, timestep,
                             patch_size, device, dtype, tile_batch=16):
    """
    Tile the full 128³ volume into non-overlapping patch_size³ blocks, encode
    each with LCA, then assemble the reconstructions back into a volume.

    LCA internally normalises each patch (zero-mean, unit-var).  We undo that
    normalization when stitching so that the assembled reconstruction lives in
    the same coordinate system as the global-normalized input volume.

    Returns
    -------
    input_vol : np.ndarray (D, H, W)  globally-normalised input
    recon_vol : np.ndarray (D, H, W)  assembled reconstruction
    error_vol : np.ndarray (D, H, W)  per-voxel error
    stats     : dict  sparsity, rel_err, n_tiles, active_total, code_total
    """
    # Load and globally normalise — same as _HDF5PatchDataset
    with h5py.File(h5_path, 'r') as f:
        vol = f[field_key][timestep].astype(np.float32)
    vol = (vol - vol.mean()) / (vol.std() + 1e-8)

    D, H, W = vol.shape
    P       = patch_size
    nD, nH, nW = D // P, H // P, W // P          # tiles per dim (floor)
    D_out   = nD * P                               # cropped size (ignore edge strip if any)
    H_out   = nH * P
    W_out   = nW * P

    recon_vol = np.zeros((D_out, H_out, W_out), dtype=np.float32)
    input_vol = vol[:D_out, :H_out, :W_out].copy()

    # Collect tile positions
    positions = [(i*P, j*P, k*P)
                 for i in range(nD) for j in range(nH) for k in range(nW)]
    n_tiles = len(positions)

    total_sparsity = 0.0
    total_active   = 0
    total_code     = 0

    lca.eval()
    with torch.no_grad():
        for start in range(0, n_tiles, tile_batch):
            batch_pos = positions[start:start + tile_batch]

            # Build batch tensor and record per-tile normalisation stats
            tiles, means, stds = [], [], []
            for (x, y, z) in batch_pos:
                patch = torch.from_numpy(vol[x:x+P, y:y+P, z:z+P])
                m = patch.mean().item()
                s = patch.std().item() + 1e-8
                tiles.append(patch.unsqueeze(0))   # (1, P, P, P)
                means.append(m)
                stds.append(s)

            batch = torch.stack(tiles).to(dtype=dtype, device=device)  # (B,1,P,P,P)
            _, code_b, recon_b, _ = lca(batch)

            for bi, (x, y, z) in enumerate(batch_pos):
                # Undo LCA's per-patch normalisation to restore global scale
                recon_global = recon_b[bi, 0].float().cpu().numpy() * stds[bi] + means[bi]
                recon_vol[x:x+P, y:y+P, z:z+P] = recon_global

                total_sparsity += (code_b[bi] == 0).float().mean().item()
                total_active   += int((code_b[bi] != 0).sum().item())
                total_code     += code_b[bi].numel()

    error_vol = input_vol - recon_vol
    rel_err   = float(np.sqrt((error_vol**2).mean()) /
                      (np.sqrt((input_vol**2).mean()) + 1e-8))

    stats = {
        'n_tiles':      n_tiles,
        'sparsity':     total_sparsity / n_tiles,
        'active_total': total_active,
        'code_total':   total_code,
        'rel_err':      rel_err,
    }
    return input_vol, recon_vol, error_vol, stats


# ---------------------------------------------------------------------------
# Import the dataset class from the training script
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.dirname(__file__))
from lca_sim_mldc_SingleSnaptshot import _HDF5PatchDataset

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

if len(sys.argv) < 2:
    print("Usage: python lca_simmldc_inference.py <models/lca_simmldc.pth> [timestep]")
    sys.exit(1)

pth_path = sys.argv[1]
override_timestep = int(sys.argv[2]) if len(sys.argv) > 2 else None

# Experiment directory is two levels up from models/
exp_dir = os.path.dirname(os.path.dirname(os.path.abspath(pth_path)))

# ---------------------------------------------------------------------------
# Load and patch config
#   - replace lambda_ with the final trained value from run.log
#   - optionally override timestep for generalisation testing
# ---------------------------------------------------------------------------

cfg_path = os.path.join(exp_dir, 'config_simmldc.yaml')
if not os.path.exists(cfg_path):
    print(f"Error: config not found at {cfg_path}")
    sys.exit(1)

with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

# Read final lambda from run.log
log_path = os.path.join(exp_dir, 'run.log')
final_lambda = None
if os.path.exists(log_path):
    with open(log_path) as f:
        content = f.read()
    matches = re.findall(r'λ=([0-9]+\.[0-9]+)', content)
    if matches:
        final_lambda = float(matches[-1])

if final_lambda is not None:
    cfg['model']['lambda_'] = final_lambda
else:
    print(f"Warning: could not parse final λ from {log_path}, "
          f"using config value {cfg['model']['lambda_']}")

train_timestep = cfg['data']['timestep']
if override_timestep is not None:
    cfg['data']['timestep'] = override_timestep

# ---------------------------------------------------------------------------
# Inference output directory — inside the experiment folder
# ---------------------------------------------------------------------------

inf_dir   = os.path.join(exp_dir, 'inference', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
plots_dir = os.path.join(inf_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)

patched_cfg_path = os.path.join(inf_dir, 'config_simmldc.yaml')
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
dtype  = torch.float32

print(f"Experiment dir:  {exp_dir}")
print(f"Inference dir:   {inf_dir}")
print(f"Model:           {pth_path}")
print(f"λ (final):       {cfg['model']['lambda_']}")
print(f"Train timestep:  {train_timestep}")
print(f"Infer timestep:  {cfg['data']['timestep']}"
      + ("  ← generalisation test" if override_timestep is not None else "  (same as training)"))
print(f"Device:          {device}\n")

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

mcfg = cfg['model']
lca  = LCAConv3D(
    out_neurons   = mcfg['features'],
    in_neurons    = mcfg['in_channels'],
    result_dir    = os.path.join(inf_dir, 'lca_results'),
    kernel_size   = mcfg['kernel_size'],
    stride        = mcfg['stride'],
    lambda_       = mcfg['lambda_'],
    tau           = mcfg['tau'],
    lca_iters     = mcfg['lca_iters'],
    return_vars   = ['inputs', 'acts', 'recons', 'recon_errors'],
).to(dtype=dtype, device=device)

lca.load_state_dict(torch.load(pth_path, map_location=device, weights_only=True))
lca.eval()

k = mcfg['kernel_size']
print(f"Loaded LCAConv3D  weights shape: {lca.weights.shape}\n")

# ---------------------------------------------------------------------------
# Load patches
# ---------------------------------------------------------------------------

dcfg = cfg['data']
dset = _HDF5PatchDataset(
    h5_path    = dcfg['h5_path'],
    field_key  = dcfg['field_key'],
    timestep   = dcfg['timestep'],
    patch_size = dcfg['patch_size'],
    n_patches  = dcfg['batch_size'],   # one batch is enough for inference
)
loader  = DataLoader(dset, batch_size=dcfg['batch_size'], shuffle=True, num_workers=0)
patches = next(iter(loader)).to(dtype=dtype, device=device)

nx, ny, nz = dset.vol.shape
print(f"Volume   : {nx}×{ny}×{nz}  field={dcfg['field_key']}  t={dcfg['timestep']}")
print(f"Patches  : {patches.shape}  (batch, channel, D, H, W)\n")

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

with torch.no_grad():
    inputs, code, recon, recon_error = lca(patches)

n_total  = code.shape[1] * code.shape[2] * code.shape[3] * code.shape[4]
sparsity = (code == 0).float().mean().item()
active   = (code != 0).float().sum(dim=(1, 2, 3, 4)).mean().item()
l1       = compute_l1_sparsity(code, lca.lambda_).item()
l2       = compute_l2_error(inputs, recon).item()
rel_err  = (
    (inputs - recon).pow(2).sum(dim=(1, 2, 3, 4)) /
    (inputs.pow(2).sum(dim=(1, 2, 3, 4)) + 1e-8)
).mean().item()

_patch_index_bits   = int(np.ceil(np.log2(n_total + 1)))
_patch_bytes_per_nz = 4 + (_patch_index_bits + 7) // 8
_patch_bytes_in     = dcfg['patch_size']**3 * 4
_patch_bytes_sparse = active * _patch_bytes_per_nz
patch_comp_ratio    = _patch_bytes_in / _patch_bytes_sparse if _patch_bytes_sparse > 0 else float('inf')
patch_bpv           = _patch_bytes_sparse * 8 / dcfg['patch_size']**3

print(f"=== LCAConv3D inference  (λ={lca.lambda_:.3f}) ===")
print(f"  Sparsity (fraction zero):  {sparsity:.3f}")
print(f"  Relative recon error:      {rel_err:.6f}")
print(f"  Active coefficients/item:  {active:.1f} / {n_total}")
print(f"  L2 recon error:            {l2:.4f}")
print(f"  L1 sparsity cost:          {l1:.4f}")
print(f"  Total energy (L2+L1):      {l2+l1:.4f}")
print(f"  Compression ratio (patch): {patch_comp_ratio:.2f}x  BPV: {patch_bpv:.2f}")
print()

# ---------------------------------------------------------------------------
# Plot 1 — reconstruction examples (three mid-plane slices per sample)
# ---------------------------------------------------------------------------

def mid_slices(t):
    """Return XY, XZ, YZ mid-plane slices from a (C, D, H, W) tensor."""
    arr = t.float().cpu().numpy()[0]   # (D, H, W)
    return arr[arr.shape[0]//2, :, :], arr[:, arr.shape[1]//2, :], arr[:, :, arr.shape[2]//2]

n = min(cfg['output']['n_images'], inputs.shape[0])
plane_labels = ['XY (mid-D)', 'XZ (mid-H)', 'YZ (mid-W)']

fig, axes = plt.subplots(n * 3, 3, figsize=(7, 2.2 * n * 3))
for row_block in range(n):
    inp_planes  = mid_slices(recon[row_block] + recon_error[row_block])
    rec_planes  = mid_slices(recon[row_block])
    err_planes  = mid_slices(recon_error[row_block])
    vmax = np.percentile(np.abs(inp_planes[0]), 99)

    for pi, (inp_p, rec_p, err_p, lbl) in enumerate(
            zip(inp_planes, rec_planes, err_planes, plane_labels)):
        row = row_block * 3 + pi
        for col, (data, title) in enumerate([
            (inp_p, f'Input [{lbl}]'),
            (rec_p, 'Reconstruction'),
            (err_p, 'Recon Error'),
        ]):
            ax = axes[row, col]
            ax.imshow(data, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            if row_block == 0:
                ax.set_title(title, fontsize=8)
            ax.axis('off')

plt.suptitle(
    f'3D patch reconstructions  |  t={dcfg["timestep"]}  λ={lca.lambda_:.3f}  '
    f'rel_err={rel_err:.4f}  CompRatio={patch_comp_ratio:.2f}x  BPV={patch_bpv:.2f}',
    fontsize=9)
plt.tight_layout()
out = os.path.join(plots_dir, 'reconstructions.png')
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved {out}")

# ---------------------------------------------------------------------------
# Plot 2 — dictionary atoms (mid-plane slice of each 3D kernel)
# ---------------------------------------------------------------------------

weights = lca.get_weights().float().cpu().numpy()   # (features, 1, kD, kH, kW)
n_feat  = weights.shape[0]
mid     = weights.shape[2] // 2
atoms   = weights[:, 0, mid, :, :]                  # (features, kH, kW)

cols = int(np.ceil(np.sqrt(n_feat)))
rows = int(np.ceil(n_feat / cols))
fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
axes = np.array(axes).ravel()
vmax = np.percentile(np.abs(atoms), 99)
for i, ax in enumerate(axes):
    if i < n_feat:
        ax.imshow(atoms[i], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.axis('off')
fig.suptitle(f'Dictionary atoms — mid-plane  ({n_feat} atoms, kernel {k}³)', fontsize=9)
plt.tight_layout()
out = os.path.join(plots_dir, 'dictionary_atoms.png')
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved {out}")

# ---------------------------------------------------------------------------
# Plot 3 — activation distribution
# ---------------------------------------------------------------------------

acts_flat = code[code != 0].float().cpu().numpy()
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(acts_flat, bins=80, log=True, color='steelblue', edgecolor='none')
ax.set_xlabel('Activation value')
ax.set_ylabel('Count (log scale)')
ax.set_title(f'Non-zero activation distribution  (sparsity={sparsity:.3f})')
ax.grid(True, alpha=0.3)
out = os.path.join(plots_dir, 'activation_distribution.png')
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved {out}")

# ---------------------------------------------------------------------------
# Full-volume reconstruction
# ---------------------------------------------------------------------------

print("Reconstructing full volume ...")
input_vol, recon_vol, error_vol, vstats = reconstruct_full_volume(
    lca,
    h5_path    = dcfg['h5_path'],
    field_key  = dcfg['field_key'],
    timestep   = dcfg['timestep'],
    patch_size = dcfg['patch_size'],
    device     = device,
    dtype      = dtype,
)

D, H, W = input_vol.shape
print(f"\n=== Full-volume reconstruction  ({D}×{H}×{W}) ===")
print(f"  Tiles:                     {vstats['n_tiles']}  "
      f"({D//dcfg['patch_size']}×{H//dcfg['patch_size']}×{W//dcfg['patch_size']})")
print(f"  Sparsity (mean over tiles): {vstats['sparsity']:.3f}")
print(f"  Relative recon error:       {vstats['rel_err']:.6f}")
print(f"  Active coefficients total:  {vstats['active_total']:,} / {vstats['code_total']:,}")
nonzero_pct = 100.0 * vstats['active_total'] / vstats['code_total']
print(f"  Non-zero fraction:          {nonzero_pct:.2f}%")
print()

# ---------------------------------------------------------------------------
# Compression metrics
# ---------------------------------------------------------------------------
voxels    = D * H * W
bytes_in  = voxels * 4          # raw float32

# Sparse COO storage: each non-zero = float32 value (4 bytes) + flat index.
# Index addresses the full code tensor:
#   features × (D/P) × (H/P) × (W/P) × (P/stride)³ positions
n_code_positions = (mcfg['features'] *
                    (D // dcfg['patch_size']) *
                    (H // dcfg['patch_size']) *
                    (W // dcfg['patch_size']) *
                    (dcfg['patch_size'] // mcfg['stride'])**3)
index_bits   = int(np.ceil(np.log2(n_code_positions + 1)))
bytes_per_nz = 4 + (index_bits + 7) // 8      # value + index
n_nonzero    = vstats['active_total']
bytes_sparse = n_nonzero * bytes_per_nz

# Quality metrics
rmse         = float(np.sqrt((error_vol**2).mean()))
signal_range = float(input_vol.max() - input_vol.min())
psnr         = float(20 * np.log10(signal_range / (rmse + 1e-12)))
bpv          = (bytes_sparse * 8) / voxels
comp_ratio   = bytes_in / bytes_sparse if bytes_sparse > 0 else float('inf')

print(f"  --- Compression metrics ---")
print(f"  Input size      (float32):  {bytes_in/1024:.1f} KB")
print(f"  Sparse code     (est.):     {bytes_sparse/1024:.1f} KB  "
      f"({index_bits}-bit index + 32-bit value per non-zero)")
print(f"  Compression ratio:          {comp_ratio:.2f}×")
print(f"  Bits per voxel  (BPV):      {bpv:.3f}  (baseline = 32 bpv)")
print(f"  RMSE:                       {rmse:.6f}")
print(f"  PSNR:                       {psnr:.2f} dB")
print(f"  Relative L2 error:          {vstats['rel_err']:.6f}")
print()

# Plot 4 — full-volume mid-plane slices
fig, axes = plt.subplots(3, 3, figsize=(13, 12))
mid = [D // 2, H // 2, W // 2]
plane_defs = [
    ('XY  (z=mid)', input_vol[:, :, mid[2]],  recon_vol[:, :, mid[2]],  error_vol[:, :, mid[2]]),
    ('XZ  (y=mid)', input_vol[:, mid[1], :],  recon_vol[:, mid[1], :],  error_vol[:, mid[1], :]),
    ('YZ  (x=mid)', input_vol[mid[0], :, :],  recon_vol[mid[0], :, :],  error_vol[mid[0], :, :]),
]
col_titles = ['Input (global norm)', 'Reconstruction', 'Recon Error']

for row, (plane_lbl, inp_p, rec_p, err_p) in enumerate(plane_defs):
    vmax = np.percentile(np.abs(inp_p), 99)
    for col, (data, ctitle) in enumerate(zip([inp_p, rec_p, err_p], col_titles)):
        ax = axes[row, col]
        im = ax.imshow(data, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
        if row == 0:
            ax.set_title(ctitle, fontsize=9)
        ax.set_ylabel(plane_lbl, fontsize=8)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        plt.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle(
    f'Full-volume reconstruction  |  t={dcfg["timestep"]}  λ={lca.lambda_:.3f}  '
    f'rel_err={vstats["rel_err"]:.4f}  CompRatio={comp_ratio:.2f}x  BPV={bpv:.2f}  '
    f'sparsity={vstats["sparsity"]:.3f}',
    fontsize=9)
plt.tight_layout()
out = os.path.join(plots_dir, 'full_volume_reconstruction.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {out}")

# Plot 5 — clean three-plane view: input (top) vs reconstruction (bottom)
plane_data = [
    ('XY  (z=mid)', input_vol[:, :, mid[2]],  recon_vol[:, :, mid[2]]),
    ('XZ  (y=mid)', input_vol[:, mid[1], :],  recon_vol[:, mid[1], :]),
    ('YZ  (x=mid)', input_vol[mid[0], :, :],  recon_vol[mid[0], :, :]),
]

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.suptitle(
    f'Input vs Reconstruction — three planes  |  t={dcfg["timestep"]}  λ={lca.lambda_:.3f}  '
    f'rel_err={vstats["rel_err"]:.4f}  CompRatio={comp_ratio:.2f}x  BPV={bpv:.2f}',
    fontsize=10)

row_labels = ['Input', 'Reconstruction']
for col, (plane_lbl, inp_p, rec_p) in enumerate(plane_data):
    vmax = np.percentile(np.abs(inp_p), 99)
    for row, (data, row_lbl) in enumerate([(inp_p, 'Input'), (rec_p, 'Reconstruction')]):
        ax = axes[row, col]
        im = ax.imshow(data, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       origin='lower', aspect='equal')
        if row == 0:
            ax.set_title(plane_lbl, fontsize=10)
        if col == 0:
            ax.set_ylabel(row_lbl, fontsize=10)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        plt.colorbar(im, ax=ax, shrink=0.85, label='p (norm.)')

plt.tight_layout()
out = os.path.join(plots_dir, 'full_volume_planes.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {out}")

print("\nDone.")
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
_log.close()
