# Single Simulation Single Snapshop
We take a single shop from a single simulation and do a few forward passes to optimize the Phi
### scripts used 
- lca_sim_mldc_SingleSnaptshot.py
- config_simmldc.yaml
### Key Design Decisions lca_sim_mldc_SingleSnaptshot.py
- **Patch extraction**: Each `__getitem__` draws a new random 32³ crop from the single 128³ volume, giving effectively unlimited augmentation. `n_patches=2000` sets the virtual epoch size.
- **Normalization**: Volume is z-scored on load (zero mean, unit variance) so LCA's internal normalization works correctly with scalar pressure data.
- **`in_channels=1`**: Pressure is a scalar field, unlike the 3-channel RGB images in the CIFAR version.
- **Dictionary atom plots**: 3D kernels are visualized as their central depth slice (`kD//2`), the standard way to display 3D filters.
- **`patch_size=32`, `stride=4`**: Output code is `(32/4)³ = 8³ = 512` positions/patch with 64 atoms → 32,768 total code values per patch.
- **LCAConv3D**: Same API as LCAConv2D but takes `(B, C, D, H, W)` input. Multi-GPU sync via manual `all_reduce` after each Hebbian update, identical to the CIFAR pipeline.
- **Full-volume reconstruction**: The inference script tiles the 128³ volume into non-overlapping 32³ patches (4×4×4 = 64 tiles), runs LCA on each batch, then undoes LCA's per-patch normalization (`recon_global = recon_lca × patch_std + patch_mean`) before stitching back to 128³. Tile-boundary discontinuities are visible as a faint grid artifact in the reconstruction.
- **Compression metrics**: Sparse COO storage assigns each non-zero a float32 value (4 bytes) + a flat index into the full code tensor of `features × (D/P) × (H/P) × (W/P) × (P/stride)³` positions. With 2,097,152 addressable positions, each index costs 22 bits (3 bytes), giving 7 bytes per non-zero. Standard quality/rate metrics reported: PSNR (dB), RMSE, bits-per-voxel (BPV, baseline = 32), and compression ratio vs raw float32.
- **rel_err-gated training state machine**: Training proceeds through three modes — `pre` (train freely until `rel_err ≤ rel_err_target`), `stabilize` (freeze λ for `stabilize_epochs` epochs to let the model consolidate), and `anneal` (increase λ by `lambda_anneal_step` every `lambda_anneal_every` epochs). The cycle repeats until `lambda_anneal_stop` annealing epochs have elapsed and the final stabilization completes, at which point training stops automatically. `max_epochs` acts as a safety cap only.
- **`rel_err_ceiling` guard**: At the start of each annealing epoch, the previous epoch's `avg_rel_err` is checked against `rel_err_ceiling` (default 0.01 = 1%). If it exceeds the ceiling, the λ increment is skipped and logged — ensuring λ can only grow when the model is actually below the error budget.
- **Best-compression checkpoint**: Every epoch where `avg_rel_err ≤ rel_err_ceiling` and the patch-level compression ratio is a new high, `lca_simmldc_best_compression.pth` is saved alongside the regular `lca_simmldc.pth`. This captures the highest compression achieved without ever violating the error budget, even if the final epoch overshoots it.

### Results
**Experiment:** `simmldc_2026-06-11_13-28-15`  
**Config:** 64 atoms, kernel 7³, stride 4, patch 32³, λ warmup 0.05→0.55 (ep15–40, hold ep40–54), 2000 patches/epoch, t=15

| Metric | Value |
|---|---|
| Sparsity (fraction zero) | 88.7% |
| Relative recon error | 0.98% |
| Active coefficients/patch | 3,718 / 32,768 |
| L2 recon error | 160.13 |
| L1 sparsity cost | 1,092.46 |
| Final λ | 0.55 |

**Observations:**
- Reconstruction fidelity is excellent at 0.98% relative error — significantly better than the CIFAR case, likely because a single pressure snapshot is a much more homogeneous and structured field than natural images.
- Dictionary atoms (mid-plane slices of 7³ kernels) show diverse gradient and edge-like filters oriented along all three spatial directions, consistent with the smooth, slowly-varying pressure structures in isotropic turbulence.
- Recon error column in the reconstruction plots is nearly blank, confirming the sparse code is capturing most of the variance.

### WARP-LCA: Warm-started LCA with CNN Predictor
#### scripts used
- lca_sim_mldc_warp.py
- config_warp.yaml

#### Method
WARP-LCA (from [arxiv 2410.18794](https://arxiv.org/abs/2410.18794)) replaces the standard zero initialization of the LCA membrane potential with a learned warm start predicted by a small CNN. Per-batch training alternates between two steps:
1. **Predictor gradient step**: run predictor → K truncated LCA steps → MSE reconstruction loss → Adam update on predictor weights (backprop via truncated BPTT through the LCA ODE).
2. **Hebbian step**: run predictor (no grad) → full LCA iterations with warm initial state → standard Hebbian weight update on the dictionary.

#### Architecture
- **WARPPredictor3D**: 3-layer 3D CNN (Conv3d 1→32→64→features). The last layer uses `stride=lca_stride` and `kernel=3, padding=1` to produce output spatial dimensions that exactly match the LCA code tensor for any valid (patch_size, stride) pair. ~278K parameters for features=128.
- **WARPLCAConv3D**: subclass of `LCAConv3D` with two overrides: (1) `_init_states` does not detach the provided initial state, enabling gradient flow from predictor to LCA ODE; (2) `encode_grad(inputs_norm, initial_states, n_iters)` runs n_iters Euler steps with LCA weights detached, so gradients flow only to the predictor.

#### Key Design Decisions
- **Near-zero last-layer init**: predictor final Conv3d initialized with `std=0.01` so it starts as an approximate no-op and warms up without disrupting early LCA training.
- **`warmup_epochs=10`**: LCA trains normally for 10 epochs to bootstrap the dictionary before the predictor activates. When the predictor activates, `lca_iters` drops from 600 to `lca_iters_warm=200` since the warm start converges faster.
- **`k_grad=20`**: 20 backprop steps through the LCA ODE. With `tau=100`, the gradient magnitude at the initial state retains ~82% of its value (`(1-1/tau)^k_grad = 0.99^20 ≈ 0.82`), providing a strong learning signal.
- **Gradient isolation**: `encode_grad` detaches `lca.weights` so gradients accumulate only on predictor parameters, not the Hebbian dictionary.
- **Multi-GPU**: predictor gradients are manually all-reduced after each `backward()` call, consistent with the existing manual all-reduce pattern for LCA weights.

#### Expected Benefits
- Fewer LCA iterations at inference (warm start → faster convergence) without changing the compression format.
- Better sparse codes: warm initialization avoids poor local minima of the LCA energy, improving reconstruction quality at the same sparsity level.
- Pretrained LCA weights can be loaded via `warp.load_lca` to train the predictor on top of an already-converged dictionary.

#### Config Highlights (`config_warp.yaml`)
| Parameter | Value | Notes |
|---|---|---|
| `warp.pred_channels` | [32, 64] | hidden dims in predictor |
| `warp.pred_lr` | 3e-4 | Adam lr for predictor |
| `warp.k_grad` | 20 | LCA steps for predictor backprop |
| `warp.warmup_epochs` | 10 | plain LCA before predictor activates |
| `warp.lca_iters_warm` | 200 | LCA iters once predictor is active |
| `model.lca_iters` | 600 | LCA iters during warmup phase |


### SVD Compression Baseline
#### scripts used
- svd_compression.py

#### Method
Truncated SVD compression applied to non-overlapping 3D patches from a single HDF5 snapshot. No learning — SVD is computed analytically from the patches themselves in a single pass.

**Pipeline per patch:**
1. Extract all non-overlapping `P³` tiles from the volume (same tiling as LCA inference).
2. Normalize each patch: zero mean, unit variance (matching LCA's internal normalization).
3. Stack into matrix `X` of shape `(n_patches, P³)`.
4. Compute truncated SVD: `U, s, Vt = SVD(X, k_max)` where `k_max = n_patches`.
5. Per-patch coefficients: `c = U[:, :k] * s[:k]` — shape `(n_patches, k)`.
6. Reconstruct: `x_norm_hat = c @ Vt[:k]`, then undo per-patch normalization.

**Compression model:**
- Storage per patch: `k × 4` bytes (k dense float32 coefficients — no index overhead, unlike LCA's COO format).
- Basis overhead: `k × P³ × 4` bytes total, amortised over all patches.
- Compression ratio (coefficients only): `comp_ratio = P³ / k`.

#### Key Design Decisions
- **`k_max = n_patches`**: SVD on a `(n_patches, P³)` matrix yields at most `n_patches` meaningful components. For a 256³ volume with P=48: 5³=125 tiles → max 125 components. The compression ratio is structurally limited by tile count.
- **No sklearn required**: falls back to `numpy.linalg.svd` automatically. Since `n_patches << P³` (short-and-wide matrix), numpy SVD is fast and only computes `n_patches` singular values.
- **Per-patch normalization**: matches LCA's internal `make_zero_mean` / `make_unit_var` for a fair comparison.
- **Three compression metrics reported**: coefficients-only (`comp_coeff = P³/k`), amortised-basis (`comp_total`), and the LCA-equivalent metric (`comp_lca_equiv`) — see below. The gap between coeff-only and amortised-basis is large for small tile counts and negligible for large datasets.
- **`comp_lca_equiv` — apples-to-apples LCA comparison**: applies the exact COO sparse-storage formula from `lca_sim_mldc_SingleSnaptshot.py` to SVD's k dense coefficients. Each coefficient is treated as if it needed a flat index into [0, k), so `bytes_per_coeff = 4 + ceil(ceil(log2(k+1))/8)` and `comp_lca_equiv = (P³×4) / (k × bytes_per_coeff)`. This is the metric to read alongside LCA's `comp_ratio`; both exclude model overhead (dictionary / basis) and use the same byte-counting formula. Residual differences reflect two factors: (1) SVD always stores all k coefficients while LCA stores only `avg_active` non-zeros (sparsity advantage to LCA), and (2) SVD's index range is k (small → 1 byte/coeff) vs LCA's `features × (P//stride)³` (large → 3 bytes/nz), so SVD pays slightly less index overhead per stored value.
- **Sweep output**: sweeps all k values from 1 to k_max (logarithmically spaced), reports rel_err, PSNR, comp_ratio, BPV for each, and identifies the best k subject to `rel_err ≤ 1%`.

#### Outputs
- `singular_values.png` — singular value spectrum (log) + cumulative explained variance
- `rel_err_vs_k.png` — reconstruction error and compression ratio vs k
- `rate_distortion.png` — rel_err vs BPV (with optional LCA comparison point)
- `full_volume_k{k}.png` — three-plane reconstruction at best k
- `svd_results.csv` — full metrics table for all k values

#### Usage
```bash
# Basic sweep
python svd_compression.py config_simmldc.yaml

# With LCA reference point for rate-distortion plot
python svd_compression.py config_simmldc.yaml --lca-bpv 3.5 --lca-rel-err 0.0098

# Specific k values
python svd_compression.py config_simmldc.yaml --k-values 5 10 20 50
```

#### Notes on comparison with LCA
- SVD coefficients are **dense** (all k stored) vs LCA coefficients which are **sparse** (only non-zeros stored + index). SVD needs no index overhead (4 bytes/coeff vs LCA's 7 bytes/non-zero).
- SVD gives the **optimal linear basis** (minimum MSE for a given rank) — it is the theoretical ceiling for any linear patch-based method.
- LCA can potentially beat SVD in compression ratio because its sparse code discards small coefficients entirely, while SVD keeps all k dense.
- The rate-distortion plot (`rate_distortion.png`) shows directly where LCA sits relative to the SVD Pareto front.


### Notes on meetings June 22 / 2026
- DO SVD to compare with LLCA
- Try spare dict initialization (AI or sckit-learn)
- Get rip of patching

---

## Hybrid SVD+LCA

Three architectural directions that move LCA toward globally-aware representations, addressing its main weakness on smooth, globally-correlated pressure fields where SVD dominates.

#### SVD-initialized dictionary
Initialize `lca.weights` from the right singular vectors `Vt[:features]` of the patch matrix (computed once, as in `svd_compression.py`) instead of random truncated-normal. This puts LCA at the globally-optimal linear starting point. Hebbian updates then refine the dictionary toward atoms that also support sparse codes.

Hypothesis: LCA may converge to higher sparsity when initialized at the SVD basis because the basis already explains most variance — residuals after projecting onto the top modes are smaller and naturally sparser. Implements the meeting-note item "Try sparse dict initialization".

```python
# After SVD(X, k_max) — Vt shape: (k_max, P³)
lca_init = torch.from_numpy(Vt[:mcfg['features']]).reshape(
    mcfg['features'], 1, k, k, k   # reshape to (features, in_ch, kD, kH, kW)
)
lca.weights.data.copy_(lca_init)
lca.normalize_weights()
```

#### Get rid of patching
Run LCA (or a learned global transform) directly on the full 3D volume without patch extraction. Without patching the receptive field spans the whole volume, capturing long-range correlations that make SVD efficient. This is the meeting-note item "Get rid of patching" and the architectural analogue of what SVD does: find global basis vectors rather than local 9³ kernels.

Concretely: replace `LCAConv3D` with a fully-connected dictionary `(N_voxels, M)` and learn sparse codes via Hebbian updates on the full volume. At convergence this recovers the sparse PCA / dictionary-learning solution, which is a generalization of SVD that allows non-orthogonal, overcomplete bases.

#### Hybrid: SVD global modes + LCA sparse residual
Use SVD to remove the dominant low-rank structure, then run LCA only on the residual. The residual is less globally correlated and potentially sparse in a local convolutional basis.

Storage: `k` dense SVD coefficients (4 bytes each, flat) + sparse LCA code of the residual (COO). If the residual is 10× sparser than the original signal, the combined scheme can beat standalone SVD at the same reconstruction quality.

```python
# Encode
svd_coeffs  = patch_norm @ Vt[:k].T          # (k,) — k dense floats
residual    = patch_norm - svd_coeffs @ Vt[:k]
lca_code    = LCA(residual)                   # sparse COO on the residual

# Decode
patch_norm_hat = svd_coeffs @ Vt[:k] + LCA_recon(lca_code)
```

---

#### Additional Tweaks to improve LCA compression after training

These require no architectural changes and can be applied to any already-trained LCA checkpoint.

**1. Entropy-code the sparse output**
The COO 7-bytes/nz model assumes flat storage. Active LCA indices cluster spatially (features fire in connected regions), so lossless entropy coding of the serialized code tensor compresses both the value and index streams significantly. Apply as a post-processing step at inference time — no retraining needed.

```python
import zlib
compressed = zlib.compress(code_coo_bytes, level=6)
effective_comp_ratio = raw_bytes / len(compressed)
```
Typical gain on structured sparse codes: **1.5–4×** additional compression over raw COO for free.

**2. Quantize coefficient values**
The float32 value per non-zero (4 bytes) can be reduced without retraining. float16 halves the value cost with negligible quality impact; int8 with a per-patch scale factor reduces it to 1 byte at ~0.5% additional quantization error.

| Format | bytes/value | bytes_per_nz | Gain vs float32 |
|---|---|---|---|
| float32 (current) | 4 | 7 | — |
| float16 | 2 | 5 | 1.4× |
| int8 + scale | 1 | 4 | 1.75× |

**3. Increase stride**
Larger stride reduces code positions per patch (18³ → 9³ for stride 4→8), directly reducing `avg_active` at the same sparsity fraction and proportionally improving `comp_ratio`. Needs retraining. The quality tradeoff must be measured empirically.

| stride | n_code_patch | bytes_per_nz | Estimated comp_ratio (at ~96% sparsity) |
|---|---|---|---|
| 4 (current) | 746,496 | 7 | ~6.7× |
| 8 | 93,312 | 7 | ~53× |
| 12 | 27,648 | 6 | ~90× |


# Single Simulation Multiple Snapshops

# Multiple Simulations 
