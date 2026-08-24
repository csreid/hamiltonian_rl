"""Sanity check: can HardConcreteGate recover a small true dimensionality?

Generates data with a known small true dimensionality (`true_dim`) rendered
linearly into a much larger observed space (`observed_dim`), trains a
minimal gated autoencoder on it with the same gate-training recipe used in
experiments/pendulum_offline.py (separate gate LR, linear L0-weight warmup),
and checks whether the gate actually converges to ~true_dim open dims that
correlate with the true generating factors -- not just that effective_dim
shrinks. Isolates the gate mechanism from the full pendulum pipeline's
confounds (LSTM dynamics, image reconstruction, curriculum schedules).

Usage:
    uv run python experiments/toy_gate_recovery.py
    uv run python experiments/toy_gate_recovery.py --variational --kl-weight 1e-3
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hamilton_rl.models import HardConcreteGate


class ToyGatedAutoencoder(nn.Module):
    def __init__(self, observed_dim: int, latent_dim: int, variational: bool):
        super().__init__()
        self.variational = variational
        self.mu_head = nn.Linear(observed_dim, latent_dim)
        self.logvar_head = nn.Linear(observed_dim, latent_dim) if variational else None
        self.gate = HardConcreteGate(latent_dim)
        self.decoder = nn.Linear(latent_dim, observed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (recon, mu, kl) -- kl is a zero scalar in deterministic mode."""
        mu = self.mu_head(x)
        if self.variational:
            logvar = self.logvar_head(x)
            latent = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
        else:
            latent = mu
            kl = latent.new_zeros(())
        latent_gated = latent * self.gate((x.shape[0],))
        recon = self.decoder(latent_gated)
        return recon, mu, kl


def make_toy_data(
    true_dim: int, observed_dim: int, n_samples: int, noise_std: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (x, z, A): x = z @ A.T + noise, the observed/true factors/projection."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(observed_dim, true_dim, generator=g)
    z = torch.randn(n_samples, true_dim, generator=g)
    x = z @ A.T + noise_std * torch.randn(n_samples, observed_dim, generator=g)
    return x, z, A


def best_matching_correlations(z: torch.Tensor, latent: torch.Tensor) -> list[tuple[int, float]]:
    """For each column of z, the (latent dim index, |Pearson r|) of its best match."""
    z_np = z.numpy()
    latent_np = latent.numpy()
    results = []
    for i in range(z_np.shape[1]):
        corrs = [
            abs(np.corrcoef(z_np[:, i], latent_np[:, j])[0, 1])
            for j in range(latent_np.shape[1])
        ]
        best_j = int(np.argmax(corrs))
        results.append((best_j, corrs[best_j]))
    return results


def main_impl(
    true_dim: int,
    observed_dim: int,
    latent_dim: int,
    n_samples: int,
    noise_std: float,
    epochs: int,
    lr: float,
    gate_weight: float,
    gate_warmup_epochs: int,
    gate_lr_mult: float,
    batch_size: int,
    seed: int,
    out: str,
    variational: bool,
    kl_weight: float,
) -> None:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x, z, _A = make_toy_data(true_dim, observed_dim, n_samples, noise_std, seed)
    n_val = max(1, n_samples // 5)
    x_train, x_val = x[:-n_val].to(device), x[-n_val:].to(device)
    z_val = z[-n_val:]

    model = ToyGatedAutoencoder(observed_dim, latent_dim, variational).to(device)

    gate_params = list(model.gate.parameters())
    gate_param_ids = {id(p) for p in gate_params}
    other_params = [p for p in model.parameters() if id(p) not in gate_param_ids]
    optimizer = torch.optim.Adam([
        {"params": other_params, "lr": lr},
        {"params": gate_params, "lr": lr * gate_lr_mult},
    ])

    print(f"{'epoch':>6} {'recon':>10} {'kl':>10} {'l0':>10} {'gate_w':>8} {'eff_dim':>8}")
    for epoch in range(epochs):
        model.train()
        gate_weight_epoch = gate_weight * (
            min(1.0, epoch / gate_warmup_epochs) if gate_warmup_epochs > 0 else 1.0
        )

        perm = torch.randperm(x_train.shape[0], device=device)
        total_recon = total_kl = total_l0 = 0.0
        n_batches = 0
        for i in range(0, x_train.shape[0], batch_size):
            batch = x_train[perm[i:i + batch_size]]
            recon, _mu, kl = model(batch)
            recon_loss = F.mse_loss(recon, batch)
            l0 = model.gate.l0_penalty()
            loss = recon_loss + kl_weight * kl + gate_weight_epoch * l0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_recon += recon_loss.item()
            total_kl += kl.item()
            total_l0 += l0.item()
            n_batches += 1

        if epoch % max(1, epochs // 20) == 0 or epoch == epochs - 1:
            print(
                f"{epoch:>6} {total_recon / n_batches:>10.4f} {total_kl / n_batches:>10.4f} "
                f"{total_l0 / n_batches:>10.3f} {gate_weight_epoch:>8.4f} "
                f"{model.gate.effective_dim():>8.3f}"
            )

    model.eval()
    with torch.no_grad():
        recon_val, mu_val, _kl = model(x_val)
        val_recon_loss = F.mse_loss(recon_val, x_val).item()

    effective_dim = model.gate.effective_dim()
    gate_probs = torch.sigmoid(model.gate.log_alpha.detach().cpu())
    order = torch.argsort(gate_probs, descending=True)

    k = max(1, round(effective_dim))
    top_k_dims = order[:k].tolist()
    matches = best_matching_correlations(z_val, mu_val.cpu()[:, top_k_dims])

    print(f"\nHeld-out reconstruction MSE: {val_recon_loss:.4f}")
    print(f"effective_dim ≈ {effective_dim:.2f} (true_dim = {true_dim})")
    print(f"top-{k} gated dims (by open prob): {top_k_dims}")
    print(f"{'true factor':>12} {'best latent (local idx)':>26} {'|corr|':>8}")
    for i, (j, corr) in enumerate(matches):
        print(f"{i:>12} {top_k_dims[j]:>26} {corr:>8.4f}")

    recovered = sum(1 for _j, corr in matches if corr > 0.9)
    print(
        f"\nInterpretation: {recovered}/{true_dim} true factors have |corr| > 0.9 with a "
        f"gate-selected dim, and effective_dim ({effective_dim:.2f}) is "
        f"{'close to' if abs(effective_dim - true_dim) < 1.5 else 'far from'} true_dim "
        f"({true_dim}). Recovery is confirmed only if both hold -- a matching effective_dim "
        "with weak correlations would mean the gate is opening the *right number* of dims "
        "without opening the *right* ones."
    )

    std = mu_val.cpu().std(dim=0)
    plot_order = torch.argsort(std, descending=True)
    dim_h = latent_dim
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(max(8, dim_h * 0.35), 7), squeeze=True
    )
    parts = ax1.violinplot(mu_val.cpu()[:, plot_order].numpy(), showmedians=False, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("tab:blue")
        pc.set_alpha(0.6)
    ax1.set_xticks(np.arange(1, dim_h + 1))
    ax1.set_xticklabels([str(i.item()) for i in plot_order], fontsize=6 if dim_h > 16 else 8)
    ax1.set_xlabel("latent dim (sorted by std, descending)")
    ax1.set_ylabel("value")
    ax1.set_title("Held-out latent (mu) value distribution")

    ax2.bar(np.arange(1, dim_h + 1), gate_probs[plot_order].numpy(), color="tab:orange")
    ax2.set_xticks(np.arange(1, dim_h + 1))
    ax2.set_xticklabels([str(i.item()) for i in plot_order], fontsize=6 if dim_h > 16 else 8)
    ax2.set_xlabel("latent dim (same order as above)")
    ax2.set_ylabel("sigmoid(log_alpha)")
    ax2.set_ylim(0, 1)
    ax2.set_title(f"Gate open probability (effective_dim ≈ {effective_dim:.2f}, true_dim = {true_dim})")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")


@click.command()
@click.option("--true-dim", type=int, default=3, show_default=True)
@click.option("--observed-dim", type=int, default=64, show_default=True)
@click.option("--latent-dim", type=int, default=16, show_default=True)
@click.option("--n-samples", type=int, default=4000, show_default=True)
@click.option("--noise-std", type=float, default=0.05, show_default=True)
@click.option("--epochs", type=int, default=2000, show_default=True)
@click.option("--lr", type=float, default=1e-2, show_default=True)
@click.option("--gate-weight", type=float, default=0.02, show_default=True,
              help="L0 penalty weight on the expected number of active gate dims")
@click.option("--gate-warmup-epochs", type=int, default=200, show_default=True,
              help="Epochs over which --gate-weight ramps linearly from 0")
@click.option("--gate-lr-mult", type=float, default=0.1, show_default=True,
              help="Multiplier on --lr for the gate's log_alpha parameters")
@click.option("--batch-size", type=int, default=256, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--out", default="toy_gate_recovery.png", show_default=True)
@click.option("--variational", is_flag=True, default=False, show_default=True,
              help="Use a VAE-style reparameterized latent instead of the deterministic default")
@click.option("--kl-weight", type=float, default=1e-3, show_default=True,
              help="Weight on the standard-normal KL term (only used with --variational)")
def main(**kwargs):
    """Test whether HardConcreteGate recovers a known small true dimensionality."""
    main_impl(**kwargs)


if __name__ == "__main__":
    main()
