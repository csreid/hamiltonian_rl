"""Pairwise Markov-consistency probe for the Phase 1 LSTM encoder.

test_markov_property.py asks "does h_t stabilize as I reveal more of the
*same* history?" — a necessary check, but it can't see path-dependence,
since every window it compares is a prefix of one fixed trajectory.

This script asks the complementary question directly: if the true state is
Markov and the encoder has learned to summarize exactly that state, then two
frames that are close in true phase space should have close h vectors
*regardless of which trajectory (policy, history) produced them*. So it
collects a pool of transitions from several policies, encodes every frame
with its real (full-history) h_t via forward_all, and compares:

    delta_phase(i, j)  = || phase(i) - phase(j) ||   (cos, sin, theta_dot/max_speed)
    delta_hidden(i, j) = || h(i) - h(j) ||  (and 1 - cos_sim(h(i), h(j)))

restricted to pairs drawn from *different* episodes, so a match can't be
explained away by temporal smoothness within one rollout.

Usage:
    uv run python experiments/test_markov_pairwise.py \\
        --checkpoint models/pendulum_offline_phase1/<run>/checkpoint.pt \\
        --episodes-per-policy 40 --max-steps 120 --n-pairs 200000
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from data.pendulum import (
    _MAX_SPEED,
    collect_random_trajectories,
    collect_spin_trajectories,
    collect_val_trajectories,
    collect_zero_trajectories,
)
from hamilton_rl.checkpoint import load_world_model

_POLICIES = {
    "random": collect_random_trajectories,
    "spin": collect_spin_trajectories,
    "energy": collect_val_trajectories,
    "zero": collect_zero_trajectories,
}


@torch.no_grad()
def encode_all(
    model, frames: torch.Tensor, device: torch.device, chunk: int = 8
) -> torch.Tensor:
    """h_t for every frame of every episode, via real full-history forward_all.

    frames: (N, T+1, C, H, W) -> (N, T+1, latent_dim)
    Chunked over N to keep peak memory bounded on modest GPUs.
    """
    out = []
    for lo in range(0, frames.shape[0], chunk):
        mu_all, _ = model.autoencoder.encoder.forward_all(frames[lo:lo + chunk].to(device))
        out.append(mu_all.cpu())
    return torch.cat(out, dim=0)


def collect_pool(
    episodes_per_policy: int, max_steps: int, img_size: int
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Collect a mixed-policy pool of episodes.

    Returns:
        frames:   (N, T+1, C, H, W)
        states:   (N, T+1, 3) -- (cos theta, sin theta, theta_dot)
        traj_ids: (N,) int -- episode index, distinct across all policies
    """
    all_frames, all_states = [], []
    for name, fn in _POLICIES.items():
        eps = fn(n_episodes=episodes_per_policy, img_size=img_size, max_steps=max_steps)
        for f, _actions, s in eps:
            all_frames.append(f)
            all_states.append(s)
    frames = torch.stack(all_frames)  # (N, T+1, C, H, W)
    states = torch.stack(all_states)  # (N, T+1, 3)
    traj_ids = np.arange(frames.shape[0])
    return frames, states, traj_ids


def normalized_phase(states: torch.Tensor) -> torch.Tensor:
    """(cos theta, sin theta, theta_dot) with theta_dot scaled to ~[-1, 1]."""
    out = states.clone()
    out[..., 2] = out[..., 2] / _MAX_SPEED
    return out


def sample_cross_traj_pairs(
    n_episodes: int, steps_per_episode: int, n_pairs: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Flat-index pairs (i, j) drawn from different episodes.

    Flat index = episode_idx * steps_per_episode + t.
    """
    idx_i = np.empty(n_pairs, dtype=np.int64)
    idx_j = np.empty(n_pairs, dtype=np.int64)
    filled = 0
    while filled < n_pairs:
        batch = n_pairs - filled
        ei = rng.integers(0, n_episodes, size=batch)
        ej = rng.integers(0, n_episodes, size=batch)
        keep = ei != ej
        n_keep = int(keep.sum())
        ti = rng.integers(0, steps_per_episode, size=n_keep)
        tj = rng.integers(0, steps_per_episode, size=n_keep)
        idx_i[filled:filled + n_keep] = ei[keep] * steps_per_episode + ti
        idx_j[filled:filled + n_keep] = ej[keep] * steps_per_episode + tj
        filled += n_keep
    return idx_i, idx_j


def main_impl(
    checkpoint: str,
    episodes_per_policy: int,
    max_steps: int,
    img_size: int,
    n_pairs: int,
    neighbor_pct: float,
    n_bins: int,
    seed: int,
    out: str,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_world_model(checkpoint, device=device)

    frames, states, _traj_ids = collect_pool(episodes_per_policy, max_steps, img_size)
    n_episodes, steps_per_episode = frames.shape[0], frames.shape[1]
    print(f"Collected {n_episodes} episodes x {steps_per_episode} steps "
          f"= {n_episodes * steps_per_episode} transitions")

    h_all = encode_all(model, frames, device)  # (N, T+1, latent_dim)
    phase_all = normalized_phase(states)  # (N, T+1, 3)

    h_flat = h_all.reshape(-1, h_all.shape[-1])
    phase_flat = phase_all.reshape(-1, phase_all.shape[-1])

    rng = np.random.default_rng(seed)
    idx_i, idx_j = sample_cross_traj_pairs(n_episodes, steps_per_episode, n_pairs, rng)

    hi, hj = h_flat[idx_i], h_flat[idx_j]
    pi, pj = phase_flat[idx_i], phase_flat[idx_j]

    delta_phase = (pi - pj).norm(dim=-1).numpy()
    delta_hidden = (hi - hj).norm(dim=-1).numpy()
    cos_sim = F.cosine_similarity(hi, hj, dim=-1).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1) binned conditional view: delta_hidden distribution as a function of delta_phase
    ax = axes[0]
    bin_edges = np.quantile(delta_phase, np.linspace(0, 1, n_bins + 1))
    bin_edges[-1] += 1e-6
    bin_idx = np.digitize(delta_phase, bin_edges) - 1
    centers, medians, q25, q75 = [], [], [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        centers.append(delta_phase[mask].mean())
        medians.append(np.median(delta_hidden[mask]))
        q25.append(np.percentile(delta_hidden[mask], 25))
        q75.append(np.percentile(delta_hidden[mask], 75))
    ax.fill_between(centers, q25, q75, alpha=0.3, label="IQR")
    ax.plot(centers, medians, marker="o", label="median")
    ax.set_xlabel("delta phase-space (cross-trajectory pairs)")
    ax.set_ylabel("delta hidden ||h_i - h_j||")
    ax.set_title("h-distance vs. phase-distance (binned)")
    ax.legend(fontsize=8)

    # 2) neighbor vs. random histogram
    ax = axes[1]
    thresh = np.percentile(delta_phase, neighbor_pct)
    is_neighbor = delta_phase <= thresh
    ax.hist(delta_hidden[is_neighbor], bins=50, density=True, alpha=0.6,
             label=f"phase-neighbors (bottom {neighbor_pct:.0f}%, n={is_neighbor.sum()})")
    ax.hist(delta_hidden, bins=50, density=True, alpha=0.6,
             label=f"all cross-traj pairs (n={len(delta_hidden)})")
    ax.set_xlabel("delta hidden ||h_i - h_j||")
    ax.set_ylabel("density")
    ax.set_title("Neighbors in phase space vs. random pairs")
    ax.legend(fontsize=8)

    # 3) same comparison in cosine-similarity space
    ax = axes[2]
    ax.hist(cos_sim[is_neighbor], bins=50, density=True, alpha=0.6,
             label="phase-neighbors")
    ax.hist(cos_sim, bins=50, density=True, alpha=0.6, label="all cross-traj pairs")
    ax.set_xlabel("cos similarity(h_i, h_j)")
    ax.set_ylabel("density")
    ax.set_title("Hidden-state cosine similarity")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")

    print(
        f"\nphase-neighbor threshold (bottom {neighbor_pct:.0f}%): delta_phase <= {thresh:.4f}"
    )
    print(
        f"delta_hidden | neighbors: median={np.median(delta_hidden[is_neighbor]):.4f}  "
        f"all: median={np.median(delta_hidden):.4f}"
    )
    print(
        "\nInterpretation: if h is Markov, the 'phase-neighbors' distribution in "
        "panels 2-3 should sit clearly apart from the 'all pairs' distribution "
        "(small delta_hidden / high cos-sim) even though every pair here comes from "
        "a *different* episode. Overlapping distributions mean closeness in true "
        "phase space does not predict closeness in h -- i.e. h carries path-dependent "
        "information beyond the physical state."
    )


@click.command()
@click.option("--checkpoint", required=True, help="Path to a Phase 1 (or full) world-model checkpoint.pt")
@click.option("--episodes-per-policy", type=int, default=40, show_default=True,
              help="Episodes collected per policy (random/spin/energy/zero)")
@click.option("--max-steps", type=int, default=120, show_default=True)
@click.option("--img-size", type=int, default=64, show_default=True)
@click.option("--n-pairs", type=int, default=200_000, show_default=True,
              help="Cross-trajectory (i, j) pairs to sample")
@click.option("--neighbor-pct", type=float, default=5.0, show_default=True,
              help="Percentile of delta_phase treated as 'close in phase space'")
@click.option("--n-bins", type=int, default=20, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--out", default="markov_pairwise.png", show_default=True)
def main(checkpoint, episodes_per_policy, max_steps, img_size, n_pairs, neighbor_pct, n_bins, seed, out):
    """Test whether h is Markov: close true states -> close h, across different trajectories."""
    main_impl(
        checkpoint=checkpoint,
        episodes_per_policy=episodes_per_policy,
        max_steps=max_steps,
        img_size=img_size,
        n_pairs=n_pairs,
        neighbor_pct=neighbor_pct,
        n_bins=n_bins,
        seed=seed,
        out=out,
    )


if __name__ == "__main__":
    main()
