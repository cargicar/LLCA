# LCA — Locally Competitive Algorithm

PyTorch implementation of the Locally Competitive Algorithm for sparse coding, based on:

> Rozell, C.J., Johnson, D.H., Baraniuk, R.G., & Olshausen, B.A. (2008).
> *Sparse Coding via Thresholding and Local Competition in Neural Circuits.*
> Neural Computation, 20, 2526–2563.

---

## What's included

### `LCA` — core inference module

Takes a fixed dictionary and runs the ODE dynamics (Eq. 3.1 from the paper) via Euler integration. Each node has an internal state `u_m(t)` governed by:

```
τ · u̇_m(t) = b_m(t) − u_m(t) − Σ G_{m,n} · a_n(t)
```

where `b_m = ⟨φ_m, s⟩` is the driving input, the second term is leaky decay, and the third is lateral inhibition from active neighbours. Supports both thresholding variants and optional energy tracking.

### `soft_threshold` / `hard_threshold` — thresholding functions

Two variants matching the paper:

- **SLCA (soft threshold)** — zero below λ, then linear above. Minimises an ℓ₁ cost function; equivalent to BPDN/LASSO.
- **HLCA (hard threshold)** — zero below λ, then identity above. Minimises an ℓ₀-like cost; more aggressively sparse.

### `LCAWithDictionaryLearning` — dictionary learning wrapper

Extends LCA with alternating minimisation: inference runs without gradients (LCA step), then the dictionary Φ is updated via SGD on the reconstruction loss `½‖s − Φa‖²`. Dictionary columns are re-normalised to unit norm after each update.

---

## Installation

```bash
pip install torch
```

No other dependencies required.

---

## Quick start

```python
import torch
from lca import LCA

# Build an overcomplete dictionary (N=64 input dim, M=256 atoms)
Phi = torch.randn(64, 256)
Phi = Phi / Phi.norm(dim=0, keepdim=True)   # unit-norm columns

lca = LCA(Phi, lam=0.1, threshold='soft', tau=10.0, n_iter=300)

s = torch.randn(16, 64)       # batch of 16 signals
a, s_hat = lca(s)             # a: sparse codes (16,256),  s_hat: reconstruction (16,64)

print(f"Sparsity:   {lca.sparsity(a):.3f}")
print(f"Recon error: {lca.reconstruction_error(s, s_hat):.6f}")
```

### With dictionary learning

```python
from lca import LCAWithDictionaryLearning

lca_dl = LCAWithDictionaryLearning(
    n_features=64, n_atoms=256,
    lam=0.1, threshold='soft',
    tau=10.0, n_iter=200, dict_lr=1e-3
)

for batch in dataloader:
    a, s_hat = lca_dl(batch)   # inference + dictionary update in one call
```

---

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `lam` | Threshold / sparsity trade-off λ. Larger → sparser codes. | `0.1` |
| `threshold` | `'soft'` (SLCA) or `'hard'` (HLCA) | `'soft'` |
| `tau` | Neural time constant. Larger → slower, more stable convergence. | `10.0` |
| `n_iter` | Number of Euler integration steps. | `300` |
| `dt` | Euler step size. Should satisfy `dt < tau`. | `1.0` |
| `track_energy` | If `True`, forward returns energy at each step as a third output. | `False` |

---

## Demo output

Running `python lca.py` on a batch of known 5-sparse signals (N=64, M=256):

```
=== SLCA (soft threshold) ===
  Sparsity (fraction zero):  0.982
  Relative recon error:      0.028528
  Active coefficients/item:  4.5 / 256
  Energy (first→last iter):  0.8576 → 0.2289

=== HLCA (hard threshold) ===
  Sparsity (fraction zero):  0.983
  Relative recon error:      0.001165
  Active coefficients/item:  4.5 / 256
  Energy (first→last iter):  0.8576 → 0.0233

=== Support recovery (Jaccard, higher=better) ===
  SLCA: 0.818
  HLCA: 0.829
```

Both variants achieve ~98% sparsity, activating only ~4–5 atoms per signal on a 256-atom dictionary — matching the true support size of k=5. HLCA achieves substantially lower reconstruction error (0.001 vs 0.028) by more aggressively committing to a sparse support. Support recovery (Jaccard ~0.82) confirms both variants correctly identify most of the true non-zero atoms.

---

## Key design notes

- The Gram matrix `G = ΦᵀΦ − I` is pre-computed once and stored as a buffer. The diagonal is zeroed so nodes don't self-inhibit.
- Driving inputs `b = s @ Phi` compute all inner products `⟨φ_m, s⟩` in a single matmul.
- Lateral inhibition `a @ G.T` is naturally one-way: inactive nodes (where `a = 0`) contribute nothing, matching the paper's energy-efficient inhibition property.
- The module is fully batched and device-agnostic — works on CPU and GPU via standard `.to(device)`.

## Copy data from RNET
rsync -avz --progress FPC/ hyperion:/data/mldc/
