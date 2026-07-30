"""Probe the Phase 1 LSTM encoder for the Markov property.

The encoder (FlexLSTMEncoder.forward_all) is a *causal* LSTM whose hidden
state is reset to zero at the start of every call — it never sees anything
before the first frame of whatever tensor it's given. That makes it a clean
lever for this test: encode the same rollout, at the same target timestep t,
from windows of different lengths (frames[t-L+1 : t+1] for several L) and
compare the resulting h_t vectors.

If the true state is Markov and the encoder has learned to summarize exactly
that state (e.g. angle + angular velocity, recoverable from ~2 frames), h_t
should converge to essentially the same vector as soon as L is large enough
to see velocity — additional history shouldn't matter. If h_t keeps drifting
as L grows, the encoder is folding non-Markovian information into h_t
(startup transients, policy-dependent history, integration drift, ...), and
Phase 2's dynamics model — which only ever sees single steps h_t -> h_{t+1}
— has no way to recover whatever that extra context was encoding.

Usage:
    uv run python experiments/test_markov_property.py \\
        --checkpoint models/pendulum_offline_phase1/<run>/checkpoint.pt \\
        --policy random --timesteps 20,50,100 --context-lengths 2,3,5,10,19,40
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import click
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from data.pendulum import (
    collect_random_trajectories,
    collect_spin_trajectories,
    collect_val_trajectories,
)
from hamilton_rl.checkpoint import load_world_model

_POLICIES = {
    "random": collect_random_trajectories,
    "spin": collect_spin_trajectories,
    "energy": collect_val_trajectories,
}


@torch.no_grad()
def encode_window(model, frames: torch.Tensor, t: int, L: int, device: torch.device) -> torch.Tensor:
    """Encode frames[t-L+1 : t+1] fresh (zero LSTM state) and return h_t = mu at the last step."""
    lo = max(0, t - L + 1)
    ctx = frames[lo:t + 1].unsqueeze(0).to(device)  # (1, min(L, t+1), C, H, W)
    mu_all, _ = model.autoencoder.encoder.forward_all(ctx)
    return mu_all[0, -1].cpu()


@torch.no_grad()
def probe_timestep(
    model,
    frames: torch.Tensor,
    t: int,
    context_lengths: list[int],
    device: torch.device,
) -> dict[int, torch.Tensor]:
    """h_t encoded from each context length in context_lengths (skipping L > t+1)."""
    return {
        L: encode_window(model, frames, t, L, device)
        for L in context_lengths
        if L <= t + 1
    }


def main_impl(
    checkpoint: str,
    policy: str,
    max_steps: int,
    img_size: int,
    timesteps: list[int],
    context_lengths: list[int],
    out: str,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_world_model(checkpoint, device=device)

    collect_fn = _POLICIES[policy]
    frames, _actions, _states = collect_fn(n_episodes=1, img_size=img_size, max_steps=max_steps)[0]

    context_lengths = sorted(set(context_lengths))
    max_t = frames.shape[0] - 1
    timesteps = [t for t in timesteps if t <= max_t]
    if not timesteps:
        raise click.ClickException(f"No requested timestep <= last available frame index {max_t}")

    fig, ax = plt.subplots(figsize=(7, 5))
    print(f"{'t':>5} {'L':>5} {'||h_L - h_ref||':>16} {'rel. norm':>10} {'cos sim':>10}")

    for t in timesteps:
        per_L = probe_timestep(model, frames, t, context_lengths, device)
        Ls = sorted(per_L)
        ref_L = Ls[-1]  # longest available context = "most history" baseline
        h_ref = per_L[ref_L]

        rel_norms, cos_sims = [], []
        for L in Ls:
            h = per_L[L]
            diff = (h - h_ref).norm().item()
            rel = diff / (h_ref.norm().item() + 1e-8)
            cos = F.cosine_similarity(h.unsqueeze(0), h_ref.unsqueeze(0)).item()
            rel_norms.append(rel)
            cos_sims.append(cos)
            print(f"{t:>5} {L:>5} {diff:>16.5f} {rel:>10.4f} {cos:>10.4f}")

        ax.plot(Ls, rel_norms, marker="o", label=f"t={t} (ref L={ref_L})")

    ax.set_xlabel("context length L (frames fed to encoder)")
    ax.set_ylabel(f"||h_L - h_ref|| / ||h_ref||")
    ax.set_title(f"Markov probe: h_t vs. context length ({policy} policy)")
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")
    print(
        "\nInterpretation: curves flattening near 0 as L grows past ~2-3 frames "
        "is consistent with the Markov property (short context already fixes h_t). "
        "Curves that keep drifting (no plateau) mean h_t depends on how much history "
        "it saw -- i.e. it is encoding more than the Markov state."
    )


@click.command()
@click.option("--checkpoint", required=True, help="Path to a Phase 1 (or full) world-model checkpoint.pt")
@click.option("--policy", type=click.Choice(sorted(_POLICIES)), default="random", show_default=True)
@click.option("--max-steps", type=int, default=200, show_default=True)
@click.option("--img-size", type=int, default=64, show_default=True)
@click.option("--timesteps", default="20,50,100,150", show_default=True,
              help="Comma-separated timestep indices t to probe")
@click.option("--context-lengths", default="2,3,5,10,19,40,80", show_default=True,
              help="Comma-separated context lengths L to encode h_t from")
@click.option("--out", default="markov_probe.png", show_default=True)
def main(checkpoint, policy, max_steps, img_size, timesteps, context_lengths, out):
    """Test whether h_t (Phase 1 LSTM encoder output) is Markov in context length."""
    main_impl(
        checkpoint=checkpoint,
        policy=policy,
        max_steps=max_steps,
        img_size=img_size,
        timesteps=[int(x) for x in timesteps.split(",")],
        context_lengths=[int(x) for x in context_lengths.split(",")],
        out=out,
    )


if __name__ == "__main__":
    main()
