# Theoretical Considerations

## SVD on rectangular vs square patch matrices

In `svd_compression.py` the data matrix `X` has shape `(n_patches, P³)`.  With `patch_size=27` and a 256³ volume this is **(729 × 19,683) — short and wide**: far fewer patches than patch dimensions.

**Compression ratio is P³/k regardless of X's shape.**
The Eckart-Young theorem guarantees the rank-k SVD truncation is the *optimal* k-dimensional approximation of X in L2.  Whether X is 729×19,683 or 19,683×19,683, for the same k you store k coefficients instead of P³ values per patch.  The square structure of the matrix does not change that ratio.

> **Eckart-Young theorem (1936):** Among all rank-k matrices, the truncated SVD gives the one closest to the original matrix in Frobenius norm (and in spectral norm).  Concretely, if `X = U Σ Vt` and `X_k = U_k Σ_k Vt_k` (top-k singular values only), then for *any* other rank-k matrix `B`:
> ```
> ‖X − X_k‖_F  ≤  ‖X − B‖_F
> ```
> The truncated SVD is not just *a* good rank-k approximation — it is the **uniquely best one**.  No other choice of k basis vectors and k coefficients can reconstruct X with lower MSE.  This is why SVD is the theoretical ceiling for any linear patch-based compression method: PCA, random projections, hand-crafted wavelets are all at best equal to SVD at the same k.  It also explains why forcing a square submatrix cannot help — that is just a specific rank-k approximation, and Eckart-Young already beats every alternative.

**What the matrix shape does affect.**
The rank of X is bounded by `min(n_patches, P³)`.  With the current short-wide regime, Vt has at most 729 rows — a **729-dimensional subspace** of the 19,683-dimensional patch space (3.7% coverage).  A square matrix with `n_patches = P³` would give a *complete* orthogonal basis spanning all of patch space.  The distinction matters for **generalization** (compressing unseen patches / other timesteps), not for compressing the exact patches X was built from.

**For a single-volume compression task** the current rectangular SVD is already optimal — it finds the best 729 directions for *this* data.  For a **universal dictionary** (multi-volume or streaming) you want `n_patches >> P³`, which requires smaller P.  With P=9 and a 256³ volume: 28³ ≈ 21,952 patches of 729 dimensions each → X is `(21,952 × 729)` — tall and thin, statistically stable, complete basis.

**What actually determines compression quality: singular value decay.**
Fast decay (few large singular values dominate) means small k is sufficient.  Fast decay happens when patches have strong spatial correlations (smooth fields like pressure → yes) and P is large enough to capture the field's correlation length.  Forcing a square submatrix by dropping columns from X strictly discards information and cannot beat the full rectangular SVD.

| Regime | X shape | Basis | Good for |
|---|---|---|---|
| n_patches < P³ (current, P=27) | short and wide | incomplete — spans n_patches directions | single-volume compression (optimal for this data) |
| n_patches ≈ P³ | square | complete orthogonal basis | generalizable dictionary |
| n_patches > P³ (e.g., P=9) | tall and thin | complete, statistically stable | multi-volume / streaming compression |

**Practical takeaway:** to move from the underdetermined (short-wide) to the overdetermined (tall-thin) regime without collecting more data, reduce P until `(D//P)³ ≥ P³`, i.e., `P ≤ D^(1/2)`.  For D=256 that is `P ≤ 16`.  This is also the regime where SVD-initialized LCA (using Vt rows as atom seeds) is most meaningful, since the learned basis is complete and generalizes to unseen patches.

## Alternatives to SVD rank-k approximation

Eckart-Young is tight within its own constraints: **linear**, **rank-k** (dense coefficients), **L2 loss**, **fixed bits per coefficient**.  Every method that beats it attacks one of those four constraints.

### 1. Sparsity — break the "dense k coefficients" constraint

SVD always stores k coefficients per patch.  Sparse coding stores M > k atoms in the dictionary but activates only s ≪ M per patch.  If the signal is sparse in the learned dictionary you store far fewer numbers at the same reconstruction error.

- **K-SVD** (Aharon et al. 2006): alternates between OMP sparse coding and SVD-based dictionary atom updates.  The dictionary is non-orthogonal and overcomplete.  At the same bit budget as SVD it achieves lower error; at the same error it uses fewer non-zeros.
- **LCA / LASSO / Basis Pursuit**: same idea — L1-penalized sparse codes in an overcomplete dictionary.
- **Why it can beat Eckart-Young**: Eckart-Young guarantees optimality among *all rank-k matrices*, but a sparse code in an overcomplete dictionary is not rank-k — it lives in a union of low-dimensional subspaces, a richer structure that L2 rank-k cannot access.

LCA is precisely this approach.  The open question is whether LCA's Hebbian dictionary is as good as K-SVD's analytically updated one.

### 2. Non-linearity — break the "linear" constraint

A smooth pressure field lives on a low-dimensional manifold in voxel space; linear projections (PCA/SVD) can only slice it with hyperplanes.  Any non-linear encoder can follow that manifold instead.

- **Autoencoders**: encoder E(x) → z, decoder D(z) → x̂.  A nonlinear encoder with the same bottleneck dimension k beats PCA/SVD because it bends the projection to follow the data manifold.
- **Variational autoencoders (VAE)**: same, with a learned prior over the latent that enables entropy coding.
- **Nonlinear ICA / normalizing flows**: learn invertible nonlinear transforms; the Jacobian structure enables exact likelihood.

The cost: these require training data, have no closed-form solution, and are slow to train.

### 3. Entropy coding — break the "fixed bits per coefficient" constraint

Eckart-Young says nothing about how many bits each coefficient costs.  SVD naively stores all k coefficients at 32 bits each.  In reality SVD coefficients are not uniformly distributed — the first few dominate and the rest are small.  Entropy coding exploits this.

- **Quantize + Huffman / arithmetic code**: allocate more bits to high-variance coefficients, fewer to small ones.  JPEG does exactly this with DCT coefficients.
- **Rate-distortion optimal bit allocation**: allocate bits proportional to log(σᵢ²) — the same variance that SVD's singular values encode.
- **Combined (transform + quantize + entropy code)**: how JPEG 2000, BPG, and VVC work; dramatically outperforms raw SVD with dense float32 storage.
- **ZFP**: purpose-built for floating-point scientific data — fits a polynomial per block (like a local SVD), quantizes, and entropy codes.  Achieves significantly better rate-distortion than raw SVD on smooth fields.

This is the cheapest win available on an already-trained LCA: entropy-coding the sparse COO stream costs nothing in retraining and typically yields 1.5–4× additional compression (already noted in the tweaks section below).

### 4. Better loss — break the "L2" constraint

Eckart-Young minimizes MSE.  MSE ignores long-range correlations and is not the right metric for many applications.

- **Perceptual loss (LPIPS, SSIM)**: minimizing perceptual distance rather than pixel-wise L2 yields better-looking reconstructions at the same bit rate — used in learned image compression.
- **GAN-based compression**: a discriminator forces the reconstruction onto the data manifold.  At high compression ratios this looks sharper than L2 but may hallucinate fine-scale structure.
- **Physics-informed loss**: for simulation data, adding a PDE residual term penalizes reconstructions that violate known physics, which can guide the model toward physically consistent low-bit-rate codes.

### 5. Implicit neural representations (INR) — bypass the patch paradigm entirely

Instead of a linear transform, represent the field as a neural network f(x, y, z) → value.  The network weights are the compressed representation.  No patches, no dictionary — the network *is* the code.

- **SIREN** (Implicit Neural Representations with Periodic Activations): a small MLP with sinusoidal activations fits smooth fields very well.
- **COIN / COIN++**: overfit a tiny network to a single volume; the weights are the bitstream.
- **NeRF-style for 3D fields**: demonstrated for turbulence data (Han & Wang 2022, Shen et al. 2023).

For a smooth pressure field, a 50K-parameter network can represent a 256³ = 16.7M voxel volume at competitive quality — a 330× compression in parameter count before quantization.  Decoding is slow (one network forward pass per voxel query), but for offline storage that may be acceptable.

### 6. Wavelet / multi-scale transforms — better energy compaction structure

Wavelets exploit multi-scale structure across the whole volume.  They are a fixed (non-data-adaptive) transform, so they cannot beat SVD in the Eckart-Young sense for *this specific data*.  But their structured sparsity pattern enables far more efficient entropy coding, so in practice (at a fixed bit budget after coding) wavelets + arithmetic coding often outperform SVD + dense float32 storage.

- **3D DWT (discrete wavelet transform)**: decomposes the volume into coarse + detail subbands.  Smooth fields concentrate almost all energy in the lowest-frequency subband; high-frequency coefficients are near zero and can be heavily quantized or discarded.

### Summary

| Method | Constraint broken | Beats SVD when… |
|---|---|---|
| Sparse coding (K-SVD, LCA) | dense coefficients | signal is sparse in an overcomplete dictionary |
| Nonlinear autoencoder | linearity | data lies on a curved low-dimensional manifold |
| Transform + entropy coding (ZFP, wavelet) | fixed bits/coefficient | coefficient distribution is highly non-uniform |
| GAN / perceptual / physics-informed loss | L2 objective | physical quality matters more than MSE |
| INR (SIREN, COIN) | patch paradigm entirely | field is globally smooth, decode speed is unimportant |

For the pressure field the most actionable paths are: entropy-coding the LCA sparse codes (free, no retraining), and INR as a complementary experiment — the field is globally smooth and that is exactly where INR methods are most competitive against patch-based linear methods.

---

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
**Config:** 64 atoms, kernel 7³, stride 4, patch 32³, λ warmup 0.05→0.55 (ep15–40, hold ep40–54), 

**Observations:**
- LLCA lca_sim_mldc_SingleSnapshot -> for fix kernel (3) bigger patch size, lower rel_err, bigger comp(vals)
-  

**On the structural difference between LCA kernel_size and SVD patch_size:**

SVD and convolutional LCA operate at fundamentally different granularities, even when they share the same `patch_size`:

- **SVD** decomposes the entire P³ patch as one unit. Each basis vector (row of Vt) is P³-dimensional; each patch receives k dense scalar coefficients.
- **LCA (convolutional, kernel_size=9, stride=3, patch_size=27):** each atom is 9³=729-dimensional and slides across the 27³ patch at (27/3)³=729 positions. Each patch can produce up to `features × 729` coefficients, a sparse subset of which are active.

These are two different levels of abstraction: SVD captures global patch structure; convolutional LCA captures local sub-patch features that repeat across spatial positions.

**For a direct apples-to-apples comparison with SVD**, set `kernel_size = patch_size = stride` (non-overlapping, fully-connected mode):

```yaml
kernel_size: 27   # odd ✓; atoms are 27³ = 19,683-dim — same dimensionality as SVD basis vectors
stride:      27   # one code position per patch; no spatial sliding
patch_size:  27   # training crop = one kernel application
```

This gives each patch exactly `features` possible activations (sparse subset active), directly comparable to SVD's k dense coefficients. The compression metric `P³ / avg_active` then matches the form of SVD's `P³ / k`.

The current `kernel_size=9` configuration is a richer but structurally different representation: atoms are 729-voxel local features shared across many spatial positions. It can capture repeating local structure more parameter-efficiently but cannot be read as a direct SVD analogue.

| Config | Atom size | Positions/patch | Structure | SVD-comparable? |
|---|---|---|---|---|
| `k=27, s=27` | 27³ = 19,683 | 1 | fully-connected | ✓ |
| `k=9, s=3` (current) | 9³ = 729 | 9³ = 729 | convolutional | ✗ (different level) |
| `k=9, s=9` | 9³ = 729 | 3³ = 27 | non-overlapping conv | partial |

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

# Notes from Meeting 

- Laso svd
- Smaller patches
- Sparce coding