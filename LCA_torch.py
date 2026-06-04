"""
Locally Competitive Algorithm (LCA) — PyTorch implementation
Based on: Rozell et al., "Sparse Coding via Thresholding and Local Competition
          in Neural Circuits", Neural Computation 20, 2526-2563 (2008)

Two variants are implemented:
  - SLCA : soft-thresholding  → minimises ℓ₁ cost  (equivalent to BPDN/LASSO)
  - HLCA : hard-thresholding  → minimises ℓ₀-like cost (more aggressive sparsity)

Quick-start
-----------
    from lca import LCA
    import torch

    # Random overcomplete dictionary (N=64 input dim, M=128 atoms)
    dictionary = torch.randn(64, 128)
    dictionary = dictionary / dictionary.norm(dim=0, keepdim=True)   # unit-norm atoms

    lca = LCA(dictionary, lam=0.1, threshold='soft', tau=10.0, n_iter=300)

    s = torch.randn(16, 64)   # batch of 16 signals
    a, recon = lca(s)         # a: sparse codes (16,128),  recon: (16,64)
"""

import torch
import torch.nn as nn
from typing import Literal, Tuple


# ---------------------------------------------------------------------------
# Threshold functions
# ---------------------------------------------------------------------------

def soft_threshold(u: torch.Tensor, lam: float) -> torch.Tensor:
    """Soft (shrinkage) threshold: max(|u| - λ, 0) * sign(u).
    Corresponds to ℓ₁ cost function C(a) = |a|.
    """
    return torch.sign(u) * torch.relu(u.abs() - lam)


def hard_threshold(u: torch.Tensor, lam: float) -> torch.Tensor:
    """Hard threshold: u if |u| > λ, else 0.
    Corresponds to ℓ₀-like cost function C(a) = λ²/2 · 𝟙(|a|>λ).
    """
    return u * (u.abs() > lam).float()


# ---------------------------------------------------------------------------
# Core LCA module
# ---------------------------------------------------------------------------

class LCA(nn.Module):
    """Locally Competitive Algorithm for sparse coding.

    Parameters
    ----------
    dictionary : Tensor, shape (n_features, n_atoms)
        The dictionary Φ. Columns should be unit-norm atoms.
    lam : float
        Threshold / sparsity trade-off λ. Larger → sparser codes.
    threshold : 'soft' | 'hard'
        Which thresholding function to use (SLCA vs HLCA).
    tau : float
        Neural time constant τ (controls integration speed).
    n_iter : int
        Number of Euler integration steps.
    dt : float
        Step size for Euler integration. Should satisfy dt < tau.
    track_energy : bool
        If True, forward also returns the energy E at each step (inference only).
    learn_dict : bool
        If True, the dictionary is updated via gradient descent each forward pass.
    dict_lr : float
        Learning rate for dictionary update (used only when learn_dict=True).
    """

    def __init__(
        self,
        dictionary: torch.Tensor,
        lam: float = 0.1,
        threshold: Literal['soft', 'hard'] = 'soft',
        tau: float = 10.0,
        n_iter: int = 300,
        dt: float = 1.0,
        track_energy: bool = False,
        learn_dict: bool = False,
        dict_lr: float = 1e-3,
    ):
        super().__init__()

        self.lam = lam
        self.tau = tau
        self.n_iter = n_iter
        self.dt = dt
        self.track_energy = track_energy
        self.learn_dict = learn_dict
        self.dict_lr = dict_lr

        if threshold == 'soft':
            self.T = soft_threshold
        elif threshold == 'hard':
            self.T = hard_threshold
        else:
            raise ValueError(f"threshold must be 'soft' or 'hard', got '{threshold}'")

        if learn_dict:
            # Learnable: gradient will flow through Phi, G recomputed each forward
            self.Phi = nn.Parameter(dictionary.float())
        else:
            # Fixed: stored as buffer, G pre-computed once
            self.register_buffer('Phi', dictionary.float())
            G = self.Phi.T @ self.Phi
            G = G - torch.eye(G.shape[0], device=G.device)
            self.register_buffer('G', G)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run LCA inference (and optionally update the dictionary).

        Parameters
        ----------
        s : Tensor, shape (..., N)

        Returns
        -------
        a : Tensor, shape (..., M)  — sparse coefficients
        s_hat : Tensor, shape (..., N)  — reconstruction ŝ = Φ·a
        energies : list[float]  — only returned when track_energy=True and learn_dict=False
        """
        if self.learn_dict:
            return self._forward_with_learning(s)
        return self._forward_inference(s)

    def _forward_inference(self, s: torch.Tensor):
        b = s @ self.Phi
        u = torch.zeros_like(b)
        energies = [] if self.track_energy else None

        for _ in range(self.n_iter):
            a = self.T(u, self.lam)
            inhibition = a @ self.G.T
            u = u + self.dt * (b - u - inhibition) / self.tau
            if self.track_energy:
                energies.append(self._energy(s, a).mean().item())

        a = self.T(u, self.lam)
        s_hat = a @ self.Phi.T
        if self.track_energy:
            return a, s_hat, energies
        return a, s_hat

    def _forward_with_learning(self, s: torch.Tensor):
        Phi = self.Phi

        # Inference — no gradients through the ODE
        with torch.no_grad():
            G = Phi.T @ Phi - torch.eye(Phi.shape[1], device=Phi.device)
            b = s @ Phi
            u = torch.zeros_like(b)
            for _ in range(self.n_iter):
                a = self.T(u, self.lam)
                u = u + self.dt * (b - u - a @ G.T) / self.tau
            a = self.T(u, self.lam)

        # Dictionary update — gradient flows only through Phi here
        s_hat = a @ Phi.T
        recon_loss = 0.5 * (s - s_hat).pow(2).mean()
        recon_loss.backward()

        with torch.no_grad():
            self.Phi.data -= self.dict_lr * self.Phi.grad
            self.Phi.grad.zero_()
            self._normalise_dict()

        return a.detach(), s_hat.detach()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalise_dict(self):
        self.Phi.data = self.Phi.data / self.Phi.data.norm(dim=0, keepdim=True).clamp(min=1e-8)

    def _energy(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        s_hat = a @ self.Phi.T
        recon_loss = 0.5 * (s - s_hat).pow(2).sum(dim=-1)
        if self.T is soft_threshold:
            sparsity_cost = self.lam * a.abs().sum(dim=-1)
        else:
            sparsity_cost = (self.lam ** 2 / 2) * (a.abs() > self.lam).float().sum(dim=-1)
        return recon_loss + sparsity_cost

    @property
    def n_features(self) -> int:
        return self.Phi.shape[0]

    @property
    def n_atoms(self) -> int:
        return self.Phi.shape[1]

    def sparsity(self, a: torch.Tensor) -> float:
        return (a == 0).float().mean().item()

    def reconstruction_error(self, s: torch.Tensor, s_hat: torch.Tensor, relative: bool = True) -> float:
        mse = (s - s_hat).pow(2).mean().item()
        if relative:
            mse /= (s.pow(2).mean().item() + 1e-8)
        return mse


# ---------------------------------------------------------------------------
# Demo / usage example
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import glob
    import random
    from PIL import Image
    import numpy as np

    torch.manual_seed(42)
    random.seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {device}\n")

    # ------------------------------------------------------------------ #
    # 1. Load a batch of random CIFAR-10 images as signals
    #    Each 32x32x3 image is flattened to a vector of length N=3072
    # ------------------------------------------------------------------ #
    N = 3072   # 32 x 32 x 3
    M = 5000    # dictionary atoms (undercomplete for speed; increase for overcomplete)
    BATCH = 32
    niter= 500
    learn_dict = True
    learning_rate = 5e-2

    image_paths = glob.glob('cifar-10-images/*/*.png')
    sampled = random.sample(image_paths, BATCH)

    images = []
    for path in sampled:
        img = Image.open(path).convert('RGB')
        arr = np.array(img, dtype=np.float32) / 255.0   # [0, 1]
        arr = arr - arr.mean()                            # zero-mean per image
        images.append(arr.flatten())

    s = torch.tensor(np.stack(images), device=device)    # (BATCH, 3072)
    print(f"Loaded {BATCH} CIFAR-10 images, signal shape: {s.shape}\n")

    # ------------------------------------------------------------------ #
    # 2. Build a random overcomplete dictionary
    # ------------------------------------------------------------------ #
    Phi = torch.randn(N, M, device=device)
    Phi = Phi / Phi.norm(dim=0, keepdim=True)   # unit-norm columns

    # ------------------------------------------------------------------ #
    # 3. Run SLCA (soft threshold)
    # ------------------------------------------------------------------ #
    slca = LCA(Phi, lam=0.1, threshold='soft', tau=10.0, n_iter=niter,
               track_energy=True).to(device)
    a_soft, s_hat_soft, energies_soft = slca(s)

    print("=== SLCA (soft threshold) ===")
    print(f"  Sparsity (fraction zero):  {slca.sparsity(a_soft):.3f}")
    print(f"  Relative recon error:      {slca.reconstruction_error(s, s_hat_soft):.6f}")
    print(f"  Active coefficients/item:  {(a_soft != 0).float().sum(dim=1).mean():.1f} / {M}")
    print(f"  Energy (first→last iter):  {energies_soft[0]:.4f} → {energies_soft[-1]:.4f}\n")

    # ------------------------------------------------------------------ #
    # 4. Run HLCA (hard threshold)
    # ------------------------------------------------------------------ #
    hlca = LCA(Phi, lam=0.1, threshold='hard', tau=10.0, n_iter=niter,
               track_energy=True).to(device)
    a_hard, s_hat_hard, energies_hard = hlca(s)

    print("=== HLCA (hard threshold) ===")
    print(f"  Sparsity (fraction zero):  {hlca.sparsity(a_hard):.3f}")
    print(f"  Relative recon error:      {hlca.reconstruction_error(s, s_hat_hard):.6f}")
    print(f"  Active coefficients/item:  {(a_hard != 0).float().sum(dim=1).mean():.1f} / {M}")
    print(f"  Energy (first→last iter):  {energies_hard[0]:.4f} → {energies_hard[-1]:.4f}\n")

    # ------------------------------------------------------------------ #
    # 5. Reconstruction error comparison
    # ------------------------------------------------------------------ #
    # (Support recovery is skipped: no ground-truth sparse codes for real images)
    print("=== Reconstruction error comparison ===")
    print(f"  SLCA relative MSE: {slca.reconstruction_error(s, s_hat_soft):.6f}")
    print(f"  HLCA relative MSE: {hlca.reconstruction_error(s, s_hat_hard):.6f}")
    print()

    # ------------------------------------------------------------------ #
    # 6. Plot original vs reconstructed images
    # ------------------------------------------------------------------ #
    import matplotlib.pyplot as plt

    def plot_loss(losses, title="Loss", xlabel="Step", ylabel="MSE", filename="loss.png"):
        plt.figure()
        plt.plot(losses)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(filename)
        plt.close()
        print(f"Saved {filename}")

    def plot_reconstructions(s, s_hat_soft, s_hat_hard, n_images=4, filename="reconstructions.png", same_T=False):
        def to_image(vec):
            img = vec.cpu().numpy().reshape(32, 32, 3)
            img = img - img.min()
            img = img / (img.max() + 1e-8)
            return img

        _, axes = plt.subplots(n_images, 3, figsize=(6, 2 * n_images))
        axes[0, 0].set_title("Original")
        if same_T:
            axes[0, 1].set_title("LCA_inference")
            axes[0, 2].set_title("LCA_dictionary_learning")
        else:
            axes[0, 1].set_title("SLCA recon")
            axes[0, 2].set_title("HLCA recon")

        for i in range(n_images):
            for ax, vec in zip(axes[i], [s[i], s_hat_soft[i], s_hat_hard[i]]):
                ax.imshow(to_image(vec))
                ax.axis('off')

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Saved {filename}")

    plot_reconstructions(s, s_hat_soft, s_hat_hard, filename="reconstructions_inference.png")

    # ------------------------------------------------------------------ #
    # 7. (Optional) Dictionary learning demo — tiny example
    # ------------------------------------------------------------------ #
    if learn_dict:
        steps = 2000
        print(f"=== Dictionary learning ({steps} SGD steps, tiny demo) ===")
        lca_dl = LCA(
            Phi, lam=0.1, threshold='hard',
            tau=10.0, n_iter=500, dt=1.0, dict_lr=learning_rate,
            learn_dict=True
        ).to(device)

        losses = []
        for step in range(steps):
            a_dl, s_hat_dl = lca_dl(s)
            err = (s - s_hat_dl).pow(2).mean().item()
            losses.append(err)
            if step % 100 == 0:
                print(f"  step {step:02d} | recon MSE = {err:.6f} | "
                    f"active = {(a_dl != 0).float().sum(dim=1).mean():.1f}")

        plot_loss(losses, title="Dictionary Learning — Reconstruction MSE", filename="dict_learning_loss.png")

        plot_reconstructions(s, s_hat_hard, s_hat_dl, filename="reconstructions_dictionary_learning.png", same_T=True)

    print("\nDone.")