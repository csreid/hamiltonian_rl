"""Offline Pendulum world-model training — three-phase regimen.

Phase 1 (phase1 subcommand):
    Train the LSTM autoencoder (encoder + f_psi + decoder) for reconstruction.
    Loss = MSE(decoder(f_psi(z)[:q_dim]), frame) + kl_weight * KL.
    After training, saves the raw training episodes (frames, actions,
    ground-truth state) to episodes_cache.pt in the run directory, both for
    Phase 2 (which re-encodes fresh windows through the frozen encoder each
    batch — see below) and for later analysis.
    Saves a unified world-model checkpoint with dynamics=None.

Phase 2 (phase2 subcommand):
    Load the cached episodes. Train a new HamiltonianFlowModel (Phi + H + J/R/B)
    that maps h_t → (q, p) such that Hamiltonian dynamics hold:

        L_tf  = MSE(phi^{-1}(RK4(phi(h_t), u_t)),  h_{t+1})       [teacher-forced, all t]
        L_cl  = MSE(phi^{-1}(RK4^k(phi(h_seed), u)), h_{seed+k})  [closed-loop, seq_len steps]

    Each batch samples a random window [s, s+ctx+seq_len) from the full
    episode and re-encodes just that window through the frozen Phase-1
    encoder (fresh LSTM state at s, ctx frames of context) so the seed h
    matches what the encoder actually produces at inference time, while
    still covering every part of the episode over the course of training —
    not just the first ctx+seq_len frames.

    Architecture and data params (latent_dim, img_size, etc.) are loaded
    automatically from the Phase 1 checkpoint — no need to re-specify them.
    Saves a complete world-model checkpoint (autoencoder + dynamics): the one
    file the dashboard needs.

Phase 3 (phase3 subcommand):
    Load a complete Phase 2 world model and finetune everything end-to-end
    (encoder + f_psi + decoder + phi + H + R/B) through the full dreaming
    pipeline. The primary objective is pixel-space: dreamed frames (encode
    context → phi → Hamiltonian rollout → phi⁻¹ → f_psi → decoder) against
    ground-truth frames — the first time the quantity we actually care about
    at inference is optimised directly, and the only place the encoder can be
    pulled toward producing a *predictable* h (Phases 1-2 only measure the
    representation/dynamics mismatch; they can't fix it).

    Guard rails against representation collapse (with the encoder unfrozen, a
    latent-space loss whose target is also encoder output is minimised by
    shrinking h):
      - the Phase-1 reconstruction loss stays on as an anchor — the encoder
        must keep explaining the pixels, so it can't throw information away;
      - the h-space teacher-forced/closed-loop losses are kept as denser
        signals but with stop-grad targets, so gradients flow only through
        the prediction path;
      - much lower default LR than Phases 1-2, per-module param groups, and
        --freeze-physics / --freeze-encoder switches for constrained runs.
    h is used deterministically (h = mu, no reparameterization/KL), matching
    how the encoder is consumed in Phase 2 and at inference.

Inference (dreaming) — same pipeline after Phase 2 or Phase 3:
    h_0  = encoder(frame_0..frame_{ctx-1})              [Phase 1 encoder]
    q, p = Phi(h_{ctx-1})                               [Phase 2 forward]
    q_k, p_k = HamiltonianRollout(q, p, actions)        [Phase 2 dynamics]
    h_k  = Phi^{-1}(q_k, p_k)                          [Phase 2 inverse]
    q_dec= f_psi(h_k)[:q_dim]                           [Phase 1 f_psi]
    frame= decoder(q_dec)                               [Phase 1 decoder]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.interpolate import griddata
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.pendulum import (
    PendulumDataset,
    collect_data,
    collect_random_trajectories,
    collect_spin_trajectories,
    collect_val_trajectories,
    collect_zero_trajectories_targeted,
    _DRAG_COEFF,
    _G,
    _MAX_SPEED,
)
from hamilton_rl.checkpoint import load_world_model, make_run_dir
from hamilton_rl.models import HamiltonianFlowModel, TemporalAutoencoder, WorldModel


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _log_latent_variance(
    qs: torch.Tensor,
    ps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean per-dim variance of q and p, as 0-dim tensors (no host sync)."""
    q_dim = qs.shape[-1]
    q_var = qs.detach().reshape(-1, q_dim).var(dim=0).mean()
    p_var = ps.detach().reshape(-1, q_dim).var(dim=0).mean()
    return q_var, p_var


def _annotate_frame(frame: torch.Tensor, text: str) -> torch.Tensor:
    img = Image.fromarray((frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    draw.text((2, 2), text, fill=(255, 255, 0))
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0


def _flatten_hparams(hparams: dict, prefix: str = "") -> dict:
    """Flatten nested dicts and cast to TensorBoard-hparam-safe scalar types.

    ``add_hparams`` only accepts int/float/str/bool values, so nested dicts
    (e.g. phase2's embedded ``phase1_config``) are inlined with a prefixed
    key, and anything else (lists, Paths, None, ...) is stringified.
    """
    flat = {}
    for k, v in hparams.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(_flatten_hparams(v, prefix=f"{key}."))
        elif isinstance(v, (int, float, str, bool)):
            flat[key] = v
        else:
            flat[key] = str(v)
    return flat


def _log_hparams_text(writer: SummaryWriter, hparams: dict, tag: str = "hparams") -> None:
    """Dump the full (possibly nested) hparams dict as readable text at step 0."""
    lines = [f"- **{k}**: {v}" for k, v in hparams.items()]
    writer.add_text(tag, "\n".join(lines), 0)


def _log_hparams_table(
    writer: SummaryWriter,
    hparams: dict,
    final_metrics: dict[str, float],
) -> None:
    """Log flattened hparams + final metrics to the TB HParams tab for cross-run filtering."""
    writer.add_hparams(_flatten_hparams(hparams), final_metrics, run_name=".")

def _plot_phase_space_coverage(
    episodes: list,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Scatter every sampled (θ, θ̇) transition in ``episodes`` to eyeball state-space coverage."""
    thetas, theta_dots = [], []
    for _, _, states in episodes:
        thetas.append(torch.atan2(states[:, 1], states[:, 0]))
        theta_dots.append(states[:, 2])
    theta_all = torch.cat(thetas).numpy()
    theta_dot_all = torch.cat(theta_dots).numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    ax.scatter(theta_all, theta_dot_all, s=2, alpha=0.25, linewidths=0, color="tab:blue")
    ax.set_xlabel("θ (rad)")
    ax.set_ylabel("θ̇ (rad/s)")
    ax.set_title(f"Training data phase-space coverage (N={len(theta_all):,})")
    fig.tight_layout()

    return fig


def _collect_energy_grid_episodes(
    resolution: int = 20,
    img_size: int = 64,
    damping: float = 0.0,
    drag: float = _DRAG_COEFF,
    context_frames: int = 5,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Collect the zero-action grid-seeded episodes for the energy-landscape plot.

    These are pure physics rollouts (independent of the model being trained),
    so they're collected once up front and reused across every validation
    step rather than re-simulated each time. Each episode's *final* state
    (context_frames - 1 zero-action steps in) covers the standard grid of
    (theta, theta_dot) — via `collect_zero_trajectories_targeted`, which
    backward-solves for the right seed rather than just seeding the grid
    directly, since a plain zero-action rollout from the grid would let the
    highest-energy points (near theta=0, the unstable equilibrium, at high
    angular velocity) swing far away from theta=0 within just a few steps,
    leaving that corner of the grid unsampled.

    The backward solve is exact everywhere (including the high-energy corner
    near theta=0) since there's no hard velocity clip to saturate against
    anymore — see `collect_zero_trajectories_targeted`'s docstring.

    Rolling forward at all (rather than encoding a single still frame) is
    necessary because the causal LSTM encoder needs real motion to infer
    velocity from, exactly as at train/inference time.
    """
    return collect_zero_trajectories_targeted(
        n_episodes=resolution * resolution,
        img_size=img_size,
        max_steps=context_frames - 1,
        damping=damping,
        drag=drag,
    )


@torch.no_grad()
def _collect_grid_qp_samples(
    model: WorldModel,
    episodes: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device | None = None,
) -> dict:
    """Encode pre-collected grid episodes (see `_collect_energy_grid_episodes`)
    through the full pipeline (encoder -> phi) to get learned (q, p).

    The ground-truth energy is evaluated at the state actually reached at the
    end of each episode's zero-action rollout, not the raw seed, since that's
    the state the encoder's output corresponds to.

    Returns a dict with q, p (each (N, q_dim), learned latents at the end of
    context), H_true (N,), theta (N,), theta_dot (N,) — the latter two being
    the true post-rollout state, for plotting in physical coordinates.
    """
    if device is None:
        device = next(model.autoencoder.parameters()).device

    frames = torch.stack([frames for frames, _, _ in episodes]).to(device)  # (N, ctx, C, H, W)
    states = torch.stack([states for _, _, states in episodes])             # (N, ctx, 3)

    mu_all, _ = model.autoencoder.encoder.forward_all(frames)
    h_last = mu_all[:, -1]  # (N, latent_dim)
    q, p = model.dynamics.encode(h_last)

    final_state = states[:, -1]
    theta = torch.atan2(final_state[:, 1], final_state[:, 0])
    theta_dot = final_state[:, 2]
    # Canonical convention matching the env's EOM (ṗ = 1.5·g·sin θ, see
    # data/pendulum.py): T = θ̇²/2 pairs with V = 1.5·g·(1 + cos θ).
    H_true = 0.5 * theta_dot**2 + 1.5 * _G * (1.0 + torch.cos(theta))

    return {"q": q.cpu(), "p": p.cpu(), "H_true": H_true, "theta": theta, "theta_dot": theta_dot}


@torch.no_grad()
def _plot_learned_energy_landscape(
    model: WorldModel,
    episodes: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    landscape_resolution: int = 200,
    min_vel: float | None = None,
    max_vel: float | None = None,
    device: torch.device | None = None,
) -> plt.Figure:
    """Compare learned H(q, p) against true pendulum energy on a phase-space grid.

    Three panels, side by side:
      1. Ground truth: the true energy, swept densely from its closed form
         (`landscape_resolution`, independent of how many grid points were
         actually sampled/encoded through the model).
      2. Interpolated: the learned H at the sampled (theta, theta_dot) points,
         interpolated onto that same dense grid (via `scipy.griddata`) so it
         can be compared shape-for-shape against panel 1.
      3. The same interpolated surface with the measured (sampled) points
         overlaid, to show what data the interpolation was built from and
         where it's extrapolating past the convex hull of the samples (NaN,
         left transparent).

    The learned H uses its own independent color scale from the true energy
    — H is only ever constrained through its gradient during training, so
    it's identifiable only up to an unknown affine offset/scale, and forcing
    a shared numeric range would misleadingly imply the absolute values
    should match. Separate colorbars let you compare *pattern* — where
    energy is high/low — by matching colors, without implying agreement in
    magnitude.

    `min_vel`/`max_vel` default to the *sampled* episodes' own
    theta_dot range (with a small margin) rather than a fixed constant — the
    env's old hard velocity clip is gone (replaced by a soft quadratic drag,
    see `data.pendulum._DRAG_COEFF`), so there's no longer a physical bound
    to assume; deriving the range from the data avoids silently cropping real
    high-speed samples out of the heatmap.
    """
    model.eval()
    if device is None:
        device = next(model.autoencoder.parameters()).device

    samples = _collect_grid_qp_samples(model, episodes, device=device)
    H_learned = model.dynamics.hamiltonian(
        samples["q"].to(device), samples["p"].to(device)
    ).cpu().numpy()
    theta = samples["theta"].cpu().numpy()
    theta_dot = samples["theta_dot"].cpu().numpy()
    H_true = samples["H_true"].cpu().numpy()

    if min_vel is None or max_vel is None:
        vel_margin = 1.1
        max_vel = float(np.abs(theta_dot).max()) * vel_margin
        min_vel = -max_vel

    theta_dense = torch.linspace(-torch.pi, torch.pi, landscape_resolution)
    theta_dot_dense = torch.linspace(min_vel, max_vel, landscape_resolution)
    grid_theta, grid_theta_dot = torch.meshgrid(theta_dense, theta_dot_dense, indexing="xy")
    H_true_dense = (0.5 * grid_theta_dot**2 + 1.5 * _G * (1.0 + torch.cos(grid_theta))).numpy()

    H_learned_dense = griddata(
        points=np.stack([theta, theta_dot], axis=-1),
        values=H_learned,
        xi=(grid_theta.numpy(), grid_theta_dot.numpy()),
        method="cubic",
    )

    extent = [-np.pi, np.pi, min_vel, max_vel]
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharex=True, sharey=True)

    im0 = axes[0].imshow(
        H_true_dense, origin="lower", aspect="auto", extent=extent, cmap="viridis",
    )
    axes[0].set_title("Ground truth")
    fig.colorbar(im0, ax=axes[0], label="H_true", pad=0.02)

    im1 = axes[1].imshow(
        H_learned_dense, origin="lower", aspect="auto", extent=extent, cmap="viridis",
    )
    axes[1].set_title("Learned H (interpolated)")
    fig.colorbar(im1, ax=axes[1], label="H_learned", pad=0.02)

    im2 = axes[2].imshow(
        H_learned_dense, origin="lower", aspect="auto", extent=extent, cmap="viridis",
    )
    axes[2].scatter(
        theta, theta_dot, c=H_learned, cmap="viridis", vmin=im2.norm.vmin, vmax=im2.norm.vmax,
        s=30, edgecolors="white", linewidths=0.6,
    )
    axes[2].set_title("Learned H (interpolated + measured points)")
    fig.colorbar(im2, ax=axes[2], label="H_learned", pad=0.02)

    for ax in axes:
        ax.set_xlabel("θ (rad)")
    axes[0].set_ylabel("θ̇ (rad/s)")

    r = np.corrcoef(H_true, H_learned)[0, 1]
    fig.suptitle(f"True energy vs. learned H, Pearson r={r:.3f}")
    fig.tight_layout()

    return fig

# ---------------------------------------------------------------------------
# Phase 1: autoencoder training
# ---------------------------------------------------------------------------


def _train_epoch_phase1(
    model: TemporalAutoencoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    kl_weight: float,
    free_bits: float,
    grad_clip: float,
    device: torch.device,
    temporal_reg_weight: float = 0.0,
    temporal_scale: float = 0.01,
    sparsity_weight: float = 0.0,
    max_context_len: int = 0,
    deterministic: bool = False,
) -> dict[str, float]:
    """Reconstruction-only epoch: encoder + f_psi + decoder, no Hamiltonian.

    Two prediction targets from the causal (forward-only) LSTM:
      - h_t → current frame   (reconstruction signal)
      - h_t → next frame      (predictive signal; h_t has seen only 0..t)

    deterministic=True is an ablation against the VAE machinery: skip the
    reparameterization noise and KL term entirely and train h as a plain
    (non-probabilistic) autoencoder latent. logvar_head is still computed by
    the encoder (unused) so model architecture/checkpoints stay identical
    either way — only the training-time treatment of h changes.
    """
    model.train()
    total_recon = total_recon_next = total_kl = total_temporal = total_sparsity = total_loss = 0.0

    for frames, actions, _ in loader:
        frames = frames.to(device)    # (B, T+1, C, H, W)
        B_size = frames.shape[0]
        if max_context_len >= 2:
            max_L = min(max_context_len, frames.shape[1])
            L = int(torch.randint(2, max_L + 1, (1,)).item())
            frames = frames[:, :L]
        T_full = frames.shape[1] - 1
        q_dim = model.latent_dim // 2

        mu_all, logvar_all = model.encoder.forward_all(frames)

        if deterministic:
            z_all = mu_all
        else:
            logvar_all = logvar_all.clamp(-10, 2)
            z_all = mu_all + torch.randn_like(mu_all) * (0.5 * logvar_all).exp()

        def _decode(z: torch.Tensor, B: int, T: int):
            s = model.f_psi(z.reshape(B * T, -1))
            return model.decoder(s[:, :q_dim]).reshape(B, T, *frames.shape[2:])

        # Current frame reconstruction
        pred_curr = _decode(z_all, B_size, T_full + 1)
        recon = F.mse_loss(pred_curr, frames)

        # Next-frame prediction: h_t + a_t → frame_{t+1}
        h_curr = z_all[:, :-1].reshape(B_size * T_full, -1)   # (B*T, latent_dim)
        a_curr = actions[:, :T_full].to(device=device, dtype=frames.dtype).reshape(B_size * T_full, 1)
        pred_next = model.next_frame_decoder(h_curr, a_curr).reshape(B_size, T_full, *frames.shape[2:])
        recon_next = F.mse_loss(pred_next, frames[:, 1:])

        if deterministic:
            kl = torch.zeros((), device=device)
        else:
            def _kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
                return (
                    (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()))
                    .clamp(min=free_bits)
                    .sum(dim=-1)
                    .mean()
                )

            kl = _kl(mu_all, logvar_all)

        loss = recon + recon_next + kl_weight * kl

        # Sparsity regulariser: L1 on the latent mean pushes irrelevant
        # dimensions to exactly 0 (unlike the KL term, which only pulls
        # every dimension toward the unit prior). A sparse h also encourages
        # markovian-ness for phase-2 dynamics learning by discouraging the
        # encoder from spreading state across dimensions that aren't needed.
        if sparsity_weight > 0:
            sparsity = mu_all.abs().sum(dim=-1).mean()
            loss = loss + sparsity_weight * sparsity
            total_sparsity = total_sparsity + sparsity.detach()

        # Temporal metric regulariser: random pairs should be at least
        # temporal_scale * |t1 - t2| apart in h-space.  One-sided so we only
        # penalise being too close, not too far.
        if temporal_reg_weight > 0:
            T_seq = mu_all.shape[1]
            t1 = torch.randint(T_seq, (T_seq,), device=device)
            t2 = torch.randint(T_seq, (T_seq,), device=device)
            dt = (t1 - t2).abs().float()                   # (T_seq,)
            h1 = mu_all[:, t1]                             # (B, T_seq, D)
            h2 = mu_all[:, t2]                             # (B, T_seq, D)
            dist = torch.norm(h1 - h2, dim=-1)             # (B, T_seq)
            temporal_reg = F.relu(temporal_scale * dt - dist).mean()
            loss = loss + temporal_reg_weight * temporal_reg
            total_temporal = total_temporal + temporal_reg.detach()

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # Accumulate as tensors — .item() per batch forces a GPU sync each time;
        # a single sync at epoch end is enough.
        total_recon = total_recon + recon.detach()
        total_recon_next = total_recon_next + recon_next.detach()
        total_kl = total_kl + kl.detach()
        total_loss = total_loss + loss.detach()

    n = len(loader)
    return {
        "phase1/loss": float(total_loss) / n,
        "phase1/recon": float(total_recon) / n,
        "phase1/recon_next": float(total_recon_next) / n,
        "phase1/kl": float(total_kl) / n,
        "phase1/temporal_reg": float(total_temporal) / n,
        "phase1/sparsity": float(total_sparsity) / n,
    }


@torch.no_grad()
def _eval_loss_phase1(
    model: TemporalAutoencoder,
    val_trajs: list,
    device: torch.device,
    chunk_size: int = 4,
) -> dict[str, float]:
    """Per-frame reconstruction loss, batched over trajectories.

    Trajectories are processed chunk_size at a time (all val trajs share a
    length, so they stack) — the decoder activations bound the chunk size.
    """
    model.eval()
    q_dim = model.latent_dim // 2
    frames_all = torch.stack([t[0] for t in val_trajs])  # (N, T+1, C, H, W)
    total_perframe = 0.0
    for i in range(0, len(frames_all), chunk_size):
        frames = frames_all[i:i + chunk_size].to(device)  # (n, T+1, C, H, W)
        n_chunk, T1 = frames.shape[:2]
        mu_all, _ = model.encoder.forward_all(frames)
        s_all = model.f_psi(mu_all.reshape(n_chunk * T1, -1))
        pred = model.decoder(s_all[:, :q_dim])
        mse = F.mse_loss(pred, frames.reshape(n_chunk * T1, *frames.shape[2:]))
        total_perframe += mse.item() * n_chunk
    return {"phase1/val_recon": total_perframe / len(val_trajs)}


@torch.no_grad()
def _log_reconstruction_lstm_video(
    model: TemporalAutoencoder,
    val_traj: tuple,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    tag: str = "val/reconstruction_lstm",
    fps: int = 10,
) -> None:
    model.eval()
    frames, actions, _ = val_traj
    ctx = frames.unsqueeze(0).to(device)
    q_dim = model.latent_dim // 2

    mu_all, _ = model.encoder.forward_all(ctx)
    s_all = model.f_psi(mu_all.squeeze(0))
    z_dec = s_all[:, :q_dim]
    recon = model.decoder(z_dec).cpu()

    gt = frames
    gt_ann = torch.stack([_annotate_frame(gt[i], f"{i}") for i in range(len(gt))])
    recon_ann = torch.stack([_annotate_frame(recon[i].clamp(0, 1), f"{i}") for i in range(len(recon))])
    side_by_side = torch.cat([gt_ann, recon_ann], dim=3).unsqueeze(0)
    writer.add_video(tag, (side_by_side.clamp(0, 1) * 255).byte(), epoch, fps=fps)


@torch.no_grad()
def _log_latent_distribution_phase1(
    model: TemporalAutoencoder,
    val_traj_sets: list,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    tag: str = "val/latent_distribution",
) -> None:
    """Violin plot of per-dimension h_t values, sorted by std descending.

    Purpose is to visualize latent sparsity: a well-factored encoding of the
    pendulum's 2D phase space should concentrate variance in a small handful
    of dims (ideally ~2-3) while the rest collapse to near-zero, unused
    dims — the more of the tail is flat, the sparser (better) the code.
    """
    model.eval()
    all_h = []
    for val_trajs, _label in val_traj_sets:
        for frames, actions, states in val_trajs:
            ctx = frames.unsqueeze(0).to(device)
            mu_all, _ = model.encoder.forward_all(ctx)
            all_h.append(mu_all.squeeze(0).cpu())
    h_pool = torch.cat(all_h, dim=0)  # (N, dim_h)

    std = h_pool.std(dim=0)
    order = torch.argsort(std, descending=True)
    h_sorted = h_pool[:, order].numpy()
    dim_h = h_sorted.shape[1]

    fig, ax = plt.subplots(figsize=(max(8, dim_h * 0.35), 4))
    parts = ax.violinplot(h_sorted, showmedians=False, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("tab:blue")
        pc.set_alpha(0.6)
    ax.set_xticks(np.arange(1, dim_h + 1))
    ax.set_xticklabels([str(i.item()) for i in order], fontsize=6 if dim_h > 16 else 8)
    ax.set_xlabel("h dimension (sorted by std, descending)")
    ax.set_ylabel("value")
    ax.set_title(f"LSTM latent value distribution, held-out trajectories (epoch {epoch + 1})")
    fig.tight_layout()
    writer.add_figure(tag, fig, epoch)
    plt.close(fig)


@torch.no_grad()
def _log_markov_pairwise_probe_phase1(
    model: TemporalAutoencoder,
    val_traj_sets: list,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    tag: str = "val/markov_pairwise_probe",
    n_pairs: int = 50_000,
    neighbor_pct: float = 5.0,
    n_bins: int = 20,
    seed: int = 0,
) -> None:
    """Cross-trajectory Markov-consistency probe (see experiments/test_markov_pairwise.py).

    Pools every val trajectory across all policies, encodes each one fully
    (real full-history h_t via forward_all), then samples pairs of frames
    drawn from *different* episodes and compares closeness in true phase
    space (cos θ, sin θ, θ̇) against closeness in h. If h is Markov, pairs
    that are close in phase space should be close in h regardless of which
    trajectory/policy produced them — overlapping "neighbor" vs. "random
    pair" distributions below mean h carries path-dependent information
    beyond the physical state, something Phase 2's single-step dynamics
    model has no way to recover.

    This only checks one direction (phase-neighbors → close in h, i.e.
    "tearing": physically adjacent states thrown apart). It says nothing
    about the reverse — h-neighbors that are actually far apart in phase
    space ("folding": physically distant states collapsed together, e.g.
    pendulum-up and pendulum-down sharing a latent code because both render
    as a thin centered rod). The `fold_score` scalars and the last two
    panels below check that reverse direction directly.
    """
    model.eval()
    h_list, phase_list, traj_id_list = [], [], []
    traj_id = 0
    for val_trajs, _label in val_traj_sets:
        for frames, _actions, states in val_trajs:
            ctx = frames.unsqueeze(0).to(device)
            mu_all, _ = model.encoder.forward_all(ctx)
            h_list.append(mu_all.squeeze(0).cpu())
            phase = states.float().clone()
            phase[:, 2] = phase[:, 2] / _MAX_SPEED
            phase_list.append(phase)
            traj_id_list.append(torch.full((frames.shape[0],), traj_id, dtype=torch.long))
            traj_id += 1

    if traj_id < 2:
        return  # need at least two trajectories for a cross-trajectory pair

    h_all = torch.cat(h_list, dim=0)
    phase_all = torch.cat(phase_list, dim=0)
    traj_ids = torch.cat(traj_id_list, dim=0).numpy()
    n = h_all.shape[0]

    rng = np.random.default_rng(seed)
    idx_i = np.empty(n_pairs, dtype=np.int64)
    idx_j = np.empty(n_pairs, dtype=np.int64)
    filled = 0
    while filled < n_pairs:
        batch = n_pairs - filled
        ci = rng.integers(0, n, size=batch)
        cj = rng.integers(0, n, size=batch)
        keep = traj_ids[ci] != traj_ids[cj]
        n_keep = int(keep.sum())
        idx_i[filled:filled + n_keep] = ci[keep]
        idx_j[filled:filled + n_keep] = cj[keep]
        filled += n_keep

    hi, hj = h_all[idx_i], h_all[idx_j]
    pi, pj = phase_all[idx_i], phase_all[idx_j]
    delta_phase = (pi - pj).norm(dim=-1).numpy()
    delta_hidden = (hi - hj).norm(dim=-1).numpy()
    cos_sim = F.cosine_similarity(hi, hj, dim=-1).numpy()
    theta_i = torch.atan2(pi[:, 1], pi[:, 0]).numpy()
    theta_j = torch.atan2(pj[:, 1], pj[:, 0]).numpy()

    fig, axes = plt.subplots(1, 5, figsize=(24, 4.5))

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
    ax.plot(centers, medians, marker="o", markersize=3, label="median")
    ax.set_xlabel("Δ phase-space (cross-trajectory pairs)")
    ax.set_ylabel("Δ hidden ||h_i - h_j||")
    ax.set_title("h-distance vs. phase-distance")
    ax.legend(fontsize=8)

    ax = axes[1]
    thresh = np.percentile(delta_phase, neighbor_pct)
    is_neighbor = delta_phase <= thresh
    ax.hist(delta_hidden[is_neighbor], bins=50, density=True, alpha=0.6,
            label=f"phase-neighbors (n={is_neighbor.sum()})")
    ax.hist(delta_hidden, bins=50, density=True, alpha=0.6,
            label=f"all pairs (n={len(delta_hidden)})")
    ax.set_xlabel("Δ hidden ||h_i - h_j||")
    ax.set_title(f"Neighbors (bottom {neighbor_pct:.0f}%) vs. random")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.hist(cos_sim[is_neighbor], bins=50, density=True, alpha=0.6, label="phase-neighbors")
    ax.hist(cos_sim, bins=50, density=True, alpha=0.6, label="all pairs")
    ax.set_xlabel("cos similarity(h_i, h_j)")
    ax.set_title("Hidden-state cosine similarity")
    ax.legend(fontsize=8)

    # --- Reverse direction: h-neighbors that are actually far apart in phase
    # space ("folding"). Same bottom-neighbor_pct% construction as above, but
    # thresholded on delta_hidden instead of delta_phase.
    h_thresh = np.percentile(delta_hidden, neighbor_pct)
    is_h_neighbor = delta_hidden <= h_thresh
    fold_median = float(np.median(delta_phase[is_h_neighbor]))
    fold_p95 = float(np.percentile(delta_phase[is_h_neighbor], 95))

    ax = axes[3]
    ax.hist(delta_phase[is_h_neighbor], bins=50, density=True, alpha=0.6,
            label=f"h-neighbors (n={is_h_neighbor.sum()})")
    ax.hist(delta_phase, bins=50, density=True, alpha=0.6, label="all pairs")
    ax.set_xlabel("Δ phase-space ||p_i - p_j||")
    ax.set_title(f"Fold check: phase-distance of h-neighbors (bottom {neighbor_pct:.0f}%)")
    ax.legend(fontsize=8)

    # Localize *where* the worst folded pairs sit in phase space: h-neighbors
    # whose phase distance is itself in the top decile of the h-neighbor
    # population, i.e. the pairs h thinks are close but truly aren't.
    ax = axes[4]
    fold_thresh = np.percentile(delta_phase[is_h_neighbor], 90)
    folded = is_h_neighbor & (delta_phase >= fold_thresh)
    ax.scatter(theta_i, theta_j, s=2, alpha=0.15, color="gray", label="all pairs")
    sc = ax.scatter(theta_i[folded], theta_j[folded], s=6, c=delta_phase[folded],
                     cmap="viridis", label=f"worst folds (n={folded.sum()})")
    fig.colorbar(sc, ax=ax, label="Δ phase", pad=0.02)
    ax.set_xlabel("θ_i (rad)")
    ax.set_ylabel("θ_j (rad)")
    ax.set_title("Where folds occur (worst h-neighbor / phase-distant pairs)")
    ax.legend(fontsize=8, markerscale=3)

    fig.suptitle(f"Markov pairwise probe, cross-trajectory (epoch {epoch + 1})")
    fig.tight_layout()
    writer.add_figure(tag, fig, epoch)
    plt.close(fig)

    writer.add_scalar(
        f"{tag}_median_delta_h/neighbors", float(np.median(delta_hidden[is_neighbor])), epoch,
    )
    writer.add_scalar(
        f"{tag}_median_delta_h/all_pairs", float(np.median(delta_hidden)), epoch,
    )
    writer.add_scalar(f"{tag}_fold_score/median", fold_median, epoch)
    writer.add_scalar(f"{tag}_fold_score/p95", fold_p95, epoch)


@torch.no_grad()
def _log_latent_scatter_phase1(
    model: TemporalAutoencoder,
    val_traj_sets: list,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    tag: str = "val/latent_regression",
) -> None:
    """val_traj_sets: list of (val_trajs, policy_label) pairs.

    A single linear probe is fit on the pooled train split (even-indexed
    trajectories across all policies) and evaluated on the pooled val split
    (odd-indexed), so all policies share one regression. Points are colored
    by the policy that produced them.
    """
    model.eval()
    per_policy_s, per_policy_st = {}, {}
    for val_trajs, label in val_traj_sets:
        all_s, all_st = [], []
        for frames, actions, states in val_trajs:
            ctx = frames.unsqueeze(0).to(device)
            mu_all, _ = model.encoder.forward_all(ctx)
            s_all = model.f_psi(mu_all.squeeze(0)).cpu()
            all_s.append(s_all)
            all_st.append(states.float())
        per_policy_s[label] = all_s
        per_policy_st[label] = all_st

    # Hold out entire trajectories for validation rather than splitting each
    # rollout in half — a temporal split would leak information (adjacent
    # frames within a trajectory are highly correlated).
    train_s = torch.cat([s for all_s in per_policy_s.values() for s in all_s[0::2]], dim=0)
    train_st = torch.cat([st for all_st in per_policy_st.values() for st in all_st[0::2]], dim=0)
    A = torch.linalg.lstsq(train_s, train_st).solution

    val_pred, val_true = {}, {}
    for label in per_policy_s:
        val_s = torch.cat(per_policy_s[label][1::2], dim=0)
        val_st = torch.cat(per_policy_st[label][1::2], dim=0)
        val_pred[label] = (val_s @ A).numpy()
        val_true[label] = val_st.numpy()

    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, name in enumerate(["cos(θ)", "sin(θ)", "θ̇ (rad/s)"]):
        all_true_i, all_pred_i = [], []
        for j, label in enumerate(val_pred):
            true_i, pred_i = val_true[label][:, i], val_pred[label][:, i]
            axes[i].scatter(true_i, pred_i, s=2, alpha=0.3, color=colors[j % len(colors)], label=label, linewidths=0)
            all_true_i.append(true_i)
            all_pred_i.append(pred_i)
        true_i = np.concatenate(all_true_i)
        pred_i = np.concatenate(all_pred_i)
        lo, hi = min(true_i.min(), pred_i.min()), max(true_i.max(), pred_i.max())
        axes[i].plot([lo, hi], [lo, hi], "r--", linewidth=0.8)
        axes[i].set_xlabel(f"True {name}")
        axes[i].set_ylabel(f"Predicted {name}")
        ss_res = ((true_i - pred_i) ** 2).sum()
        ss_tot = ((true_i - true_i.mean()) ** 2).sum()
        axes[i].set_title(f"{name}  R²={1 - ss_res / (ss_tot + 1e-8):.3f}")
    axes[0].legend(markerscale=4, fontsize=8)
    fig.suptitle(f"Latent → state regression, held-out trajectories (epoch {epoch + 1})")
    fig.tight_layout()
    writer.add_figure(tag, fig, epoch)
    plt.close(fig)


@torch.no_grad()
def _log_h_state_regression_coeffs_phase1(
    model: TemporalAutoencoder,
    val_trajs: list,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    tag: str = "val/h_state_regression_coeffs",
) -> None:
    """Bar charts of bidirectional linear-probe coefficients between raw h_t and state.

    Unlike `_log_latent_scatter_phase1` (which probes s_all = f_psi(h_t)), this
    probes h_t = mu_all directly — the exact quantity Phase 2 re-encodes
    from windows of the cached episode frames. `_log_latent_scatter_phase1`'s R² only shows whether state is
    *recoverable* from h (h → state); it says nothing about whether h contains
    additional non-Markovian content (history/burn-in transient, VAE noise
    dims) that Phase 2's dynamics model has no way to predict, since Phase 2
    must reproduce every dim of h. The state → h direction below is one probe
    of the reverse: coefficients that don't line up with the h → state ones
    flag h-dims whose variance is not well explained by state.

    Both directions collapse to a (dim_h, 3) coefficient matrix, drawn as
    dim_h groups of 3 bars (cos θ, sin θ, θ̇):
      - h → state: row i of the forward lstsq solution — how much h_i
        contributes to predicting each state component.
      - state → h: the reverse lstsq solution (3, dim_h), transposed so row i
        holds the three coefficients of the simple regression fit onto h_i
        alone.
    """
    model.eval()
    all_h, all_st = [], []
    for frames, actions, states in val_trajs:
        ctx = frames.unsqueeze(0).to(device)
        mu_all, _ = model.encoder.forward_all(ctx)
        all_h.append(mu_all.squeeze(0).cpu())
        all_st.append(states.float())

    h_pool = torch.cat(all_h, dim=0)
    st_pool = torch.cat(all_st, dim=0)

    # h → state: (dim_h, 3); row i = h_i's contribution to each state comp.
    A = torch.linalg.lstsq(h_pool, st_pool).solution

    # state → h: (3, dim_h) -> transpose to (dim_h, 3) so row i holds the
    # three state coefficients of the regression fit onto h_i alone.
    B_t = torch.linalg.lstsq(st_pool, h_pool).solution.T

    dim_h = A.shape[0]
    labels = ["cos(θ)", "sin(θ)", "θ̇"]
    x = np.arange(dim_h)
    width = 0.25

    fig, axes = plt.subplots(2, 1, figsize=(max(8, dim_h * 0.35), 7))
    for ax, mat, title in (
        (axes[0], A.numpy(), "h → state  (row i: h_i's coefficient predicting each state comp)"),
        (axes[1], B_t.numpy(), "state → h  (row i: state coefficients predicting h_i)"),
    ):
        for j, label in enumerate(labels):
            ax.bar(x + (j - 1) * width, mat[:, j], width, label=label)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("h dimension")
        ax.set_ylabel("coefficient")
        ax.set_title(title)
        ax.set_xticks(x)
        if dim_h > 16:
            ax.tick_params(axis="x", labelsize=6)
        ax.legend(fontsize=8)
    fig.suptitle(f"h ↔ state linear-probe coefficients, held-out trajectories (epoch {epoch + 1})")
    fig.tight_layout()
    writer.add_figure(tag, fig, epoch)
    plt.close(fig)


@torch.no_grad()
def _log_cnn_feature_distribution_phase1(
    model: TemporalAutoencoder,
    val_traj_sets: list,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    tag: str = "val/cnn_feature_distribution",
) -> None:
    """Violin plot of per-dimension CNN-feature values, sorted by std descending.

    Same purpose as `_log_latent_distribution_phase1` but probing
    `encoder.frame_cnn` output directly, one stage before the LSTM. Comparing
    the two tells you whether any latent collapse originates in the per-frame
    conv or is introduced downstream by the LSTM/KL training.
    """
    model.eval()
    all_feat = []
    for val_trajs, _label in val_traj_sets:
        for frames, _actions, _states in val_trajs:
            all_feat.append(model.encoder.frame_cnn(frames.to(device)).cpu())
    feat_pool = torch.cat(all_feat, dim=0)

    std = feat_pool.std(dim=0)
    order = torch.argsort(std, descending=True)
    feat_sorted = feat_pool[:, order].numpy()
    dim_feat = feat_sorted.shape[1]

    fig, ax = plt.subplots(figsize=(max(8, dim_feat * 0.08), 4))
    parts = ax.violinplot(feat_sorted, showmedians=False, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("tab:orange")
        pc.set_alpha(0.6)
    if dim_feat <= 64:
        ax.set_xticks(np.arange(1, dim_feat + 1))
        ax.set_xticklabels([str(i.item()) for i in order], fontsize=6 if dim_feat > 16 else 8)
    else:
        ax.set_xticks([])
    ax.set_xlabel("CNN feature dimension (sorted by std, descending)")
    ax.set_ylabel("value")
    ax.set_title(f"Per-frame CNN feature distribution, held-out trajectories (epoch {epoch + 1})")
    fig.tight_layout()
    writer.add_figure(tag, fig, epoch)
    plt.close(fig)


@torch.no_grad()
def _log_cnn_feature_regression_phase1(
    model: TemporalAutoencoder,
    val_traj_sets: list,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    tag: str = "val/cnn_feature_regression",
) -> None:
    """Linear probe from per-frame CNN features to (cos θ, sin θ).

    Mirrors `_log_latent_scatter_phase1` but on `encoder.frame_cnn` output and
    without θ̇ — a single static frame carries no velocity information, so
    including it would only measure the probe's inability to see something
    that was never there. High R² here means position is already linearly
    recoverable before the LSTM even runs; a low R² here contrasted against a
    high R² for h → state would point at the conv itself, not the LSTM, as
    the source of any position-encoding failure.
    """
    model.eval()
    per_policy_feat, per_policy_theta = {}, {}
    for val_trajs, label in val_traj_sets:
        all_feat, all_theta = [], []
        for frames, _actions, states in val_trajs:
            all_feat.append(model.encoder.frame_cnn(frames.to(device)).cpu())
            all_theta.append(states[:, :2].float())  # (cos θ, sin θ)
        per_policy_feat[label] = all_feat
        per_policy_theta[label] = all_theta

    train_feat = torch.cat([f for all_f in per_policy_feat.values() for f in all_f[0::2]], dim=0)
    train_theta = torch.cat([t for all_t in per_policy_theta.values() for t in all_t[0::2]], dim=0)
    A = torch.linalg.lstsq(train_feat, train_theta).solution

    val_pred, val_true = {}, {}
    for label in per_policy_feat:
        val_feat = torch.cat(per_policy_feat[label][1::2], dim=0)
        val_theta = torch.cat(per_policy_theta[label][1::2], dim=0)
        val_pred[label] = (val_feat @ A).numpy()
        val_true[label] = val_theta.numpy()

    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for i, name in enumerate(["cos(θ)", "sin(θ)"]):
        all_true_i, all_pred_i = [], []
        for j, label in enumerate(val_pred):
            true_i, pred_i = val_true[label][:, i], val_pred[label][:, i]
            axes[i].scatter(true_i, pred_i, s=2, alpha=0.3, color=colors[j % len(colors)], label=label, linewidths=0)
            all_true_i.append(true_i)
            all_pred_i.append(pred_i)
        true_i = np.concatenate(all_true_i)
        pred_i = np.concatenate(all_pred_i)
        lo, hi = min(true_i.min(), pred_i.min()), max(true_i.max(), pred_i.max())
        axes[i].plot([lo, hi], [lo, hi], "r--", linewidth=0.8)
        axes[i].set_xlabel(f"True {name}")
        axes[i].set_ylabel(f"Predicted {name}")
        ss_res = ((true_i - pred_i) ** 2).sum()
        ss_tot = ((true_i - true_i.mean()) ** 2).sum()
        axes[i].set_title(f"{name}  R²={1 - ss_res / (ss_tot + 1e-8):.3f}")
    axes[0].legend(markerscale=4, fontsize=8)
    fig.suptitle(f"CNN feature → position regression, held-out trajectories (epoch {epoch + 1})")
    fig.tight_layout()
    writer.add_figure(tag, fig, epoch)
    plt.close(fig)


@torch.no_grad()
def _log_cnn_feature_fold_probe_phase1(
    model: TemporalAutoencoder,
    val_traj_sets: list,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    tag: str = "val/cnn_feature_fold_probe",
    n_pairs: int = 50_000,
    neighbor_pct: float = 5.0,
    seed: int = 0,
) -> None:
    """Fold check on per-frame CNN features, mirroring the reverse-direction
    panels of `_log_markov_pairwise_probe_phase1` but one stage earlier.

    Unlike the h-space probe, pairs aren't restricted to cross-trajectory —
    a single frame's CNN embedding has no temporal/path dependence to guard
    against, so any two frames anywhere in the val pool are fair game. Only
    position matters (θ, via circular distance) since a static frame carries
    no velocity. If the θ ↔ -θ / θ ↔ π-θ mirror lines already show up here,
    the fold originates in the per-frame conv (or the render itself), not in
    LSTM/VAE training dynamics.
    """
    model.eval()
    feat_list, theta_list = [], []
    for val_trajs, _label in val_traj_sets:
        for frames, _actions, states in val_trajs:
            feat_list.append(model.encoder.frame_cnn(frames.to(device)).cpu())
            theta_list.append(torch.atan2(states[:, 1].float(), states[:, 0].float()))

    feat_all = torch.cat(feat_list, dim=0)
    theta_all = torch.cat(theta_list, dim=0)
    n = feat_all.shape[0]

    rng = np.random.default_rng(seed)
    idx_i = rng.integers(0, n, size=n_pairs)
    idx_j = rng.integers(0, n, size=n_pairs)

    fi, fj = feat_all[idx_i], feat_all[idx_j]
    ti, tj = theta_all[idx_i].numpy(), theta_all[idx_j].numpy()
    delta_feat = (fi - fj).norm(dim=-1).numpy()
    delta_theta = np.abs(np.angle(np.exp(1j * (ti - tj))))  # circular distance in [0, pi]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    feat_thresh = np.percentile(delta_feat, neighbor_pct)
    is_feat_neighbor = delta_feat <= feat_thresh
    fold_median = float(np.median(delta_theta[is_feat_neighbor]))
    fold_p95 = float(np.percentile(delta_theta[is_feat_neighbor], 95))
    ax.hist(delta_theta[is_feat_neighbor], bins=50, density=True, alpha=0.6,
            label=f"feat-neighbors (n={is_feat_neighbor.sum()})")
    ax.hist(delta_theta, bins=50, density=True, alpha=0.6, label="all pairs")
    ax.set_xlabel("Δθ (circular, rad)")
    ax.set_title(f"Fold check: angular distance of CNN-feat-neighbors (bottom {neighbor_pct:.0f}%)")
    ax.legend(fontsize=8)

    ax = axes[1]
    bin_edges = np.linspace(0, delta_feat.max(), 21)
    bin_idx = np.digitize(delta_feat, bin_edges) - 1
    centers, medians = [], []
    for b in range(len(bin_edges) - 1):
        mask = bin_idx == b
        if not mask.any():
            continue
        centers.append(delta_feat[mask].mean())
        medians.append(np.median(delta_theta[mask]))
    ax.plot(centers, medians, marker="o", markersize=3)
    ax.set_xlabel("Δ CNN feature ||f_i - f_j||")
    ax.set_ylabel("median Δθ")
    ax.set_title("Δθ vs. feature-distance")

    ax = axes[2]
    fold_thresh = np.percentile(delta_theta[is_feat_neighbor], 90)
    folded = is_feat_neighbor & (delta_theta >= fold_thresh)
    ax.scatter(ti, tj, s=2, alpha=0.15, color="gray", label="all pairs")
    sc = ax.scatter(ti[folded], tj[folded], s=6, c=delta_theta[folded], cmap="viridis",
                     label=f"worst folds (n={folded.sum()})")
    fig.colorbar(sc, ax=ax, label="Δθ", pad=0.02)
    ax.set_xlabel("θ_i (rad)")
    ax.set_ylabel("θ_j (rad)")
    ax.set_title("Where CNN-feature folds occur")
    ax.legend(fontsize=8, markerscale=3)

    fig.suptitle(f"CNN feature fold probe (epoch {epoch + 1})")
    fig.tight_layout()
    writer.add_figure(tag, fig, epoch)
    plt.close(fig)

    writer.add_scalar(f"{tag}_fold_score/median", fold_median, epoch)
    writer.add_scalar(f"{tag}_fold_score/p95", fold_p95, epoch)


# ---------------------------------------------------------------------------
# Episode cache (between phases)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Phase 2: dynamics training
# ---------------------------------------------------------------------------


@torch.enable_grad()
def _energy_balance_loss(
    dyn_model: HamiltonianFlowModel,
    q_all: torch.Tensor,
    p_all: torch.Tensor,
    actions_win: torch.Tensor,
) -> torch.Tensor:
    """Port-Hamiltonian energy-balance consistency along encoded real trajectories.

    Chain rule plus antisymmetry of J gives, along any trajectory of the model
    class dz/dt = (J - R(z)) ∇H + B u:

        dH/dt = -∇ₚHᵀ R_pp(z) ∇ₚH + ∇ₚHᵀ B u

    This is an identity of the model class, so the true (H, R, B) must satisfy
    it along the *data*. The loss penalises the discrete mismatch between
    H(z_{t+1}) - H(z_t) and dt times the trapezoid of the right-hand side over
    the frame gap (u_t zero-order-held across it). Every unit of energy H
    loses along real trajectories must then be invoiced to the model's own
    dissipation R_pp(z), and every unit gained to the input power — so H and R
    cannot buy rollout contraction with decay the data doesn't show. The
    identity holds for any PSD R(z): state-dependent (drag-like) damping is
    constrained exactly, not approximated.

    z is detached, so the balance shapes H, R, and B only; if it also reached
    phi, the coordinates could warp to make the books balance instead of the
    physics. It is a consistency constraint, not conservation — H ≡ const
    satisfies it trivially — so it supplements the prediction losses and can
    never replace them.

    q_all/p_all: (B, W, q_dim) encoded states; actions_win: (B, W-1) controls.
    """
    B_size, W, q_dim = q_all.shape
    z = torch.cat([q_all, p_all], dim=-1).reshape(B_size * W, 2 * q_dim).detach()
    z = z.requires_grad_(True)
    H = dyn_model.hamiltonian(z[:, :q_dim], z[:, q_dim:])  # (B*W,)
    grad_H = torch.autograd.grad(H.sum(), z, create_graph=True)[0]
    g_p = grad_H[:, q_dim:]
    if dyn_model._has_dissipation:
        diss = -(g_p * dyn_model._apply_R_pp(z, g_p)).sum(-1)
    else:
        diss = torch.zeros_like(H)
    H = H.reshape(B_size, W)
    diss = diss.reshape(B_size, W)
    g_p = g_p.reshape(B_size, W, q_dim)
    # B u enters only the p block, so its power is ∇ₚHᵀBu.
    Bu = actions_win.reshape(B_size, W - 1, -1) @ dyn_model.get_B().T  # (B, W-1, q_dim)
    power_left = diss[:, :-1] + (g_p[:, :-1] * Bu).sum(-1)
    power_right = diss[:, 1:] + (g_p[:, 1:] * Bu).sum(-1)
    lhs = H[:, 1:] - H[:, :-1]
    rhs = 0.5 * dyn_model.dt * (power_left + power_right)
    return F.mse_loss(lhs, rhs)


def _train_epoch_phase2(
    dyn_model: HamiltonianFlowModel,
    encoder: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
    device: torch.device,
    seq_len: int,
    seed_ctx_len: int,
    logdet_weight: float,
    l1_weight: float = 0.0,
    teacher_force_weight: float = 1.0,
    closed_loop_weight: float = 1.0,
    closed_loop_gamma: float = 1.0,
    structural_reg_weight: float = 0.0,
    energy_balance_weight: float = 0.0,
    huber_delta: float = 0.0,
    h_noise_std: float = 0.0,
    h_noise_scale: torch.Tensor | None = None,
) -> dict[str, float]:
    """Dynamics epoch: joint teacher-forced + closed-loop rollout.

    Each batch samples one random window [s, s+ctx+T) — shared across the
    batch — from the raw episode frames (ctx = seed_ctx_len, T = seq_len
    clamped to the episode length) and re-encodes only that window through
    the frozen Phase-1 encoder, with a fresh LSTM state at s. This matches
    what the encoder actually produces from a short real context at
    inference time (rather than carrying full episode history from t=0),
    while covering every part of the episode over the course of training as
    s varies from batch to batch — not just the first ctx+seq_len frames.

    Teacher-forced: for every consecutive pair (h_t, h_{t+1}) within the
    window, take one RK4 step from (q_t, p_t) and compare the decoded
    prediction to h_{t+1}. All steps are independent so they are batched as
    (B*T, q_dim) — no Python loop, no sequential graph depth.

    Closed-loop: starting from (q, p) at the end of context (local index
    ctx-1), roll T Hamiltonian steps without re-encoding and compare each
    decoded prediction to the corresponding freshly-encoded h. Gradients
    from the closed-loop loss flow back through phi at the seed alongside
    those from the teacher-forced objective.

    Per-step errors within the rollout are combined with exponential decay
    closed_loop_gamma**t (t=0 at the first post-seed step), normalised so the
    weights sum to 1.  closed_loop_gamma=1.0 recovers a plain mean over all T
    steps. Discounting later steps keeps gradient pressure on getting the
    near-term dynamics right, rather than being dominated by large late-step
    errors that mostly reflect compounding of earlier (possibly correct)
    error rather than a wrong local step.

    Logdet regulariser is applied over all ctx+T encoded timesteps in the
    window, so its strength stays constant as seq_len grows.

    If huber_delta > 0, the per-element closed-loop error is a Huber loss
    scaled by 2 so it equals the squared error for |e| <= delta (keeping
    cl_loss and its EMA curriculum gate on the plain-MSE scale) but grows only
    linearly beyond. Late rollout steps that have drifted out of phase then
    stop dominating the batch gradient — pressure that otherwise rewards
    contractive (spuriously damped) dynamics.

    If energy_balance_weight > 0, adds the port-Hamiltonian energy-balance
    consistency loss (see _energy_balance_loss) over the encoded window. With
    h-noise augmentation active, the balance is evaluated on a clean no-grad
    re-encoding: on jittered z, H's spread across the jitter joins the loss
    floor, rewarding exactly the flattened H the term exists to prevent.

    If h_noise_std > 0, zero-mean Gaussian noise is added to the h values fed
    into phi (both teacher-forced and closed-loop seeds), while the prediction
    targets stay clean.  This is denoising-style augmentation: the model learns
    to map jittered latents back onto the dynamics manifold, which improves
    closed-loop stability where rollout error accumulates.  When h_noise_scale
    (per-dim std of the data) is given, h_noise_std is a multiplier on each
    dimension's spread; otherwise it is an absolute std applied uniformly.
    """
    dyn_model.train()
    encoder.eval()
    total_dynamics = total_tf = total_cl = total_logdet_reg = 0.0
    total_q_var = total_p_var = total_hamiltonian_l1 = total_grad_H_norm = total_struct_reg = 0.0
    total_energy_balance = 0.0
    q_dim = dyn_model.latent_dim // 2
    ctx = seed_ctx_len

    for frames, actions, _states in loader:
        actions = actions.to(device)  # (B, T_full)
        B_size = frames.shape[0]
        T_full = actions.shape[1]

        # --- Sample one random window [s, s+ctx+T), shared across the batch ---
        T = max(min(seq_len, T_full - ctx + 1), 1)
        W = ctx + T  # frames in the window
        max_s = T_full + 1 - W  # T_full+1 total frames, indices 0..T_full
        s = int(torch.randint(0, max_s + 1, (1,)).item()) if max_s > 0 else 0

        frames_win = frames[:, s:s + W].to(device)    # (B, W, C, H, W)
        actions_win = actions[:, s:s + W - 1]          # (B, W-1)

        # --- Re-encode just this window with a fresh LSTM state at s ---
        with torch.no_grad():
            h_all, _ = encoder.forward_all(frames_win)  # (B, W, latent_dim)
        D = h_all.shape[-1]

        # --- Encode the window through phi in one batched call ---
        # Augmentation: jitter the inputs to phi but keep h_all (the targets)
        # clean, so the model learns to denoise toward the dynamics manifold.
        # h_noise_scale (per-dim std) makes the std relative to each dimension's
        # spread; without it the std is absolute and uniform across dims.
        if h_noise_std > 0:
            std = h_noise_std if h_noise_scale is None else h_noise_std * h_noise_scale
            h_in = h_all + torch.randn_like(h_all) * std
        else:
            h_in = h_all
        h_flat = h_in.reshape(B_size * W, D)
        s_flat, log_det_flat = dyn_model.phi.forward_with_logdet(h_flat)
        q_all = s_flat[:, :q_dim].reshape(B_size, W, q_dim)  # (B, W, q_dim)
        p_all = s_flat[:, q_dim:].reshape(B_size, W, q_dim)
        log_det_all = log_det_flat.reshape(B_size, W)          # (B, W)
        logdet_metric = log_det_all.detach().pow(2).mean()     # save before backward

        logdet_reg = logdet_weight * log_det_all.pow(2).mean()

        # --- Teacher-forced loss: one batched RK4 step at every t in the window ---
        # All steps are independent — reshape to (B*T_tf, q_dim) for one forward pass.
        T_tf = W - 1
        q_tf = q_all[:, :T_tf].reshape(B_size * T_tf, q_dim)
        p_tf = p_all[:, :T_tf].reshape(B_size * T_tf, q_dim)
        a_tf = actions_win.reshape(B_size * T_tf, 1)
        q_tf_next, p_tf_next = dyn_model.controlled_step(q_tf, p_tf, a_tf)
        h_tf_pred = dyn_model.decode(q_tf_next, p_tf_next)
        h_tf_target = h_all[:, 1:].reshape(B_size * T_tf, D)
        tf_loss = F.mse_loss(h_tf_pred, h_tf_target)

        # --- Closed-loop rollout from the end of context (local index ctx-1) ---
        k = ctx - 1
        q, p = q_all[:, k], p_all[:, k]
        q_k_log, p_k_log = q.detach(), p.detach()  # save before graph is freed
        qs_steps, ps_steps = [], []
        for t in range(T):
            q, p = dyn_model.controlled_step(q, p, actions_win[:, k + t:k + t + 1])
            qs_steps.append(q)
            ps_steps.append(p)
        # The decode never feeds back into the rollout, so all T states go
        # through phi^{-1} in one batched call instead of T tiny ones.
        q_traj = torch.stack(qs_steps, dim=1)  # (B, T, q_dim)
        p_traj = torch.stack(ps_steps, dim=1)
        h_cl_pred = dyn_model.decode(
            q_traj.reshape(B_size * T, q_dim), p_traj.reshape(B_size * T, q_dim)
        )
        h_cl_target = h_all[:, k + 1:k + 1 + T]
        h_cl_pred = h_cl_pred.reshape(B_size, T, D)
        if huber_delta > 0:
            # 2x huber == squared error for |e| <= delta, linear beyond — same
            # scale as plain MSE where the EMA curriculum gate operates.
            elem_err = 2.0 * F.huber_loss(
                h_cl_pred, h_cl_target, reduction="none", delta=huber_delta
            )
        else:
            elem_err = (h_cl_pred - h_cl_target).pow(2)
        per_step_loss = elem_err.mean(dim=(0, 2))  # (T,) — mean over batch and latent dims
        step_weights = closed_loop_gamma ** torch.arange(T, device=device, dtype=per_step_loss.dtype)
        cl_loss = (per_step_loss * step_weights).sum() / step_weights.sum()

        loss = logdet_reg + teacher_force_weight * tf_loss + closed_loop_weight * cl_loss

        if energy_balance_weight > 0:
            if h_noise_std > 0:
                # Clean re-encode: the balance must see the data manifold, not
                # the jitter (see docstring).
                with torch.no_grad():
                    s_clean = dyn_model.phi(h_all.reshape(B_size * W, D))
                q_eb = s_clean[:, :q_dim].reshape(B_size, W, q_dim)
                p_eb = s_clean[:, q_dim:].reshape(B_size, W, q_dim)
            else:
                q_eb, p_eb = q_all, p_all
            eb_loss = _energy_balance_loss(dyn_model, q_eb, p_eb, actions_win)
            loss = loss + energy_balance_weight * eb_loss
            total_energy_balance = total_energy_balance + eb_loss.detach()

        if l1_weight > 0:
            l1_loss = sum(param.abs().sum() for param in dyn_model.hamiltonian.parameters())
            loss = loss + l1_weight * l1_loss
            total_hamiltonian_l1 = total_hamiltonian_l1 + l1_loss.detach()

        if structural_reg_weight > 0 and dyn_model.learn_structure:
            # J is a fixed buffer in HamiltonianFlowModel — only R is learned,
            # so penalizing ‖J‖² would just add a constant 2·q_dim to the loss.
            if dyn_model.state_dep_r:
                # Penalize R at the batch's own phase-space points (detached so
                # the penalty shapes r_net, not phi), mean over samples so the
                # weight is comparable to the constant-R Frobenius penalty.
                z_r = torch.cat([q_all, p_all], dim=-1).reshape(-1, D).detach()
                struct_reg = dyn_model.get_R_pp(z_r).pow(2).sum(dim=(-2, -1)).mean()
            else:
                struct_reg = dyn_model.get_R_pp().pow(2).sum()
            loss = loss + structural_reg_weight * struct_reg
            total_struct_reg = total_struct_reg + struct_reg.detach()

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(dyn_model.parameters(), grad_clip)
        optimizer.step()

        # Gradient of H norm — use the saved seed point, no re-encoding needed
        with torch.enable_grad():
            z_eval = torch.cat([q_k_log, p_k_log], dim=-1).requires_grad_(True)
            H_eval = dyn_model.hamiltonian(z_eval[:, :q_dim], z_eval[:, q_dim:]).sum()
            grad_eval = torch.autograd.grad(H_eval, z_eval)[0]
            total_grad_H_norm = total_grad_H_norm + grad_eval.norm(dim=-1).mean().detach()

        # Accumulate as tensors — .item() per batch forces a GPU sync each time;
        # a single sync at epoch end is enough.
        total_logdet_reg = total_logdet_reg + logdet_metric
        total_tf = total_tf + tf_loss.detach()
        total_cl = total_cl + cl_loss.detach()
        total_dynamics = total_dynamics + loss.detach()
        with torch.no_grad():
            q_var, p_var = _log_latent_variance(
                torch.cat([q_k_log.unsqueeze(1), q_traj], dim=1),
                torch.cat([p_k_log.unsqueeze(1), p_traj], dim=1),
            )
            total_q_var = total_q_var + q_var
            total_p_var = total_p_var + p_var

    n = len(loader)
    return {
        "phase2/dynamics": float(total_dynamics) / n,
        "phase2/tf_loss": float(total_tf) / n,
        "phase2/cl_loss": float(total_cl) / n,
        "phase2/logdet_reg": float(total_logdet_reg) / n,
        "phase2/q_var": float(total_q_var) / n,
        "phase2/p_var": float(total_p_var) / n,
        "phase2/hamiltonian_l1": float(total_hamiltonian_l1) / n,
        "phase2/grad_H_norm": float(total_grad_H_norm) / n,
        "phase2/struct_reg": float(total_struct_reg) / n,
        "phase2/energy_balance": float(total_energy_balance) / n,
    }


def _encode_val_h(
    phase1_model: TemporalAutoencoder,
    val_trajs: list,
    device: torch.device,
    enc_chunk: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode all val trajectories to h in chunks: (N, T+1, D) h, (N, T) actions.

    All val trajs share a length so they stack; chunking bounds the encoder's
    activation memory while keeping the per-call batch large.
    """
    frames_all = torch.stack([t[0] for t in val_trajs])              # (N, T+1, C, H, W)
    actions_all = torch.stack([t[1] for t in val_trajs]).to(device)  # (N, T)
    h_chunks = []
    for i in range(0, len(frames_all), enc_chunk):
        mu_all, _ = phase1_model.encoder.forward_all(frames_all[i:i + enc_chunk].to(device))
        h_chunks.append(mu_all)
    return torch.cat(h_chunks, dim=0), actions_all.float()


@torch.no_grad()
def _eval_loss_phase2(
    world_model: WorldModel,
    val_trajs: list,
    device: torch.device,
) -> dict[str, float]:
    """Validation losses for the dynamics model, batched over trajectories.

    The closed-loop loss is evaluated over the *full* available horizon
    (T_full - 1 steps) regardless of the training seq_len curriculum, so the
    metric is comparable across epochs.  If it tracked the curriculum length
    instead, every seq_len bump would lengthen the rollout and inject a
    spurious step-wise increase into the curve.

    All trajectories roll out together (B = N) and every rollout state is
    decoded in a single phi^{-1} call — the sequential bottleneck is only the
    T dynamics steps, not T×N tiny kernel launches.
    """
    world_model.eval()
    phase1_model, dyn_model = world_model.autoencoder, world_model.dynamics
    q_dim = dyn_model.latent_dim // 2

    h_all, actions_all = _encode_val_h(phase1_model, val_trajs, device)
    N, T_seq, D = h_all.shape
    T_full = actions_all.shape[1]

    q_flat, p_flat = dyn_model.encode(h_all.reshape(N * T_seq, D))
    q_all = q_flat.reshape(N, T_seq, q_dim)
    p_all = p_flat.reshape(N, T_seq, q_dim)

    q_teacher = q_all[:, :T_full].reshape(N * T_full, q_dim)
    p_teacher = p_all[:, :T_full].reshape(N * T_full, q_dim)
    q_next, p_next = dyn_model.controlled_step(
        q_teacher, p_teacher, actions_all.reshape(N * T_full, 1)
    )
    h_teacher_pred = dyn_model.decode(q_next, p_next)
    h_teacher_target = h_all[:, 1:].reshape(N * T_full, D)
    teacher_forced = F.mse_loss(h_teacher_pred, h_teacher_target).item()

    n_rollout_steps = T_full - 1
    closed_loop = 0.0
    if n_rollout_steps > 0:
        q, p = q_all[:, 1], p_all[:, 1]
        qs_steps, ps_steps = [], []
        for t in range(n_rollout_steps):
            q, p = dyn_model.controlled_step(q, p, actions_all[:, 1 + t: 2 + t])
            qs_steps.append(q)
            ps_steps.append(p)
        q_traj = torch.stack(qs_steps, dim=1).reshape(N * n_rollout_steps, q_dim)
        p_traj = torch.stack(ps_steps, dim=1).reshape(N * n_rollout_steps, q_dim)
        h_cl_pred = dyn_model.decode(q_traj, p_traj)
        h_cl_target = h_all[:, 2:2 + n_rollout_steps].reshape(N * n_rollout_steps, D)
        closed_loop = F.mse_loss(h_cl_pred, h_cl_target).item()

    return {
        "phase2/val_tf_loss": teacher_forced,
        "phase2/val_cl_loss": closed_loop,
    }


@torch.no_grad()
def _log_structural_matrices_phase2(
    dyn_model: HamiltonianFlowModel,
    writer: SummaryWriter,
    epoch: int,
) -> None:
    """With state-dependent R, the logged R is the z = 0 baseline (r_net bias)."""
    J = dyn_model.get_J().cpu()
    R = dyn_model.get_R().cpu()
    R_name = "R(z=0)" if dyn_model.state_dep_r else "R"
    writer.add_scalar("phase2/structure/J_frob", J.pow(2).sum().sqrt().item(), epoch)
    writer.add_scalar("phase2/structure/R_frob", R.pow(2).sum().sqrt().item(), epoch)
    writer.add_histogram("phase2/structure/R_eigenvalues", torch.linalg.eigvalsh(R), epoch)
    for name, title, mat in (("J", "J", J), ("R", R_name, R)):
        fig, ax = plt.subplots(figsize=(4, 4))
        m = mat.numpy()
        vmax = max(abs(m.max()), abs(m.min()), 1e-6)
        im = ax.imshow(m, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"{title} (epoch {epoch + 1})")
        fig.tight_layout()
        writer.add_figure(f"phase2/structure/{name}", fig, epoch)
        plt.close(fig)


def _plot_dissipation_landscape(
    model: WorldModel,
    episodes: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    damping: float,
    drag: float,
    device: torch.device | None = None,
) -> plt.Figure:
    """Compare the learned energy-dissipation rate against the true one on the phase-space grid.

    The port-Hamiltonian energy balance gives the learned model's dissipation
    rate at a state as −Ḣ = ∇ₚHᵀ R_pp(z) ∇ₚH (zero control).  The env's true
    rate is damping·θ̇² + drag·|θ̇|³ — the drag term is what a constant R
    cannot express, since its rate is forced to be quadratic in ∇ₚH.

    Like the energy-landscape plot, H (hence its gradient and R's scale) is
    only identified up to affine/scale factors, so the two panels use
    independent color scales and agreement is summarized by Pearson r on the
    log-rates (log because both rates span decades from the origin outward;
    a tiny floor keeps the θ̇ ≈ 0 samples finite).
    """
    model.eval()
    if device is None:
        device = next(model.autoencoder.parameters()).device
    dyn = model.dynamics

    samples = _collect_grid_qp_samples(model, episodes, device=device)
    q = samples["q"].to(device)
    p = samples["p"].to(device)
    with torch.enable_grad():
        z = torch.cat([q, p], dim=-1).requires_grad_(True)
        q_dim = dyn.latent_dim // 2
        H_val = dyn.hamiltonian(z[:, :q_dim], z[:, q_dim:]).sum()
        g_p = torch.autograd.grad(H_val, z)[0][:, q_dim:]
    with torch.no_grad():
        R_pp = dyn.get_R_pp(z.detach() if dyn.state_dep_r else None)
        Rg = g_p @ R_pp if R_pp.dim() == 2 else (R_pp @ g_p.unsqueeze(-1)).squeeze(-1)
        rate_learned = (g_p * Rg).sum(dim=-1).cpu().numpy()

    theta = samples["theta"].numpy()
    theta_dot = samples["theta_dot"].numpy()
    rate_true = damping * theta_dot**2 + drag * np.abs(theta_dot) ** 3

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, rate, label in (
        (axes[0], rate_true, "true rate: damping·θ̇² + drag·|θ̇|³"),
        (axes[1], rate_learned, "learned rate: ∇ₚHᵀ R_pp(z) ∇ₚH"),
    ):
        sc = ax.scatter(theta, theta_dot, c=rate, cmap="magma", s=30)
        fig.colorbar(sc, ax=ax, label="−Ḣ", pad=0.02)
        ax.set_xlabel("θ (rad)")
        ax.set_ylabel("θ̇ (rad/s)")
        ax.set_title(label)

    eps = 1e-8
    log_true = np.log10(rate_true + eps)
    log_learned = np.log10(np.maximum(rate_learned, 0.0) + eps)
    r = np.corrcoef(log_true, log_learned)[0, 1] if np.ptp(rate_true) > 0 else float("nan")
    axes[2].scatter(log_true, log_learned, s=10, alpha=0.6)
    axes[2].set_xlabel("log₁₀ true rate")
    axes[2].set_ylabel("log₁₀ learned rate")
    axes[2].set_title(f"log-rate agreement, Pearson r={r:.3f}")

    fig.suptitle("Energy-dissipation rate: true vs. learned R")
    fig.tight_layout()
    return fig


@torch.no_grad()
def _log_dreamed_video_phase2(
    world_model: WorldModel,
    val_traj: tuple,
    writer: SummaryWriter,
    epoch: int,
    seq_len: int,
    context_frames: int = 5,
    tag: str = "val/dreamed_phase2",
    fps: int = 10,
) -> None:
    """Log a dreamed rollout alongside ground truth.

    Context: context_frames frames fed through the Phase 1 LSTM encoder → h.
    Rollout: seq_len Hamiltonian steps in phase space, decoded back to pixels
             via phi^{-1} → f_psi → decoder.
    """
    world_model.eval()
    frames, actions, _ = val_traj

    dreamed = world_model.dream(frames, actions, n_context=context_frames, n_steps=seq_len)
    n_steps = len(dreamed)
    if n_steps == 0:
        return

    gt = frames[context_frames:context_frames + n_steps]  # (n_steps, C, H, W)

    gt_ann = torch.stack([
        _annotate_frame(gt[i], f"gt {context_frames + i}") for i in range(len(gt))
    ])
    dream_ann = torch.stack([
        _annotate_frame(dreamed[i].clamp(0, 1), f"dr {context_frames + i}") for i in range(len(dreamed))
    ])
    side_by_side = torch.cat([gt_ann, dream_ann], dim=3).unsqueeze(0)
    writer.add_video(tag, (side_by_side.clamp(0, 1) * 255).byte(), epoch, fps=fps)


@torch.no_grad()
def _log_phase_space_regression_phase2(
    world_model: WorldModel,
    val_trajs: list,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    tag: str = "val/phase_space_regression",
) -> None:
    """Probe whether closed-loop rolled-out (q, p) linearly encodes true state.

    Mirrors `_log_latent_scatter_phase1`, but the (q, p) sequence being probed
    comes from rolling the learned Hamiltonian dynamics forward in closed loop
    (as in `_log_dreamed_video_phase2`) rather than one-step teacher-forced
    encoding — closed loop is what dreaming/planning actually uses. Trajectories
    are held out whole rather than split in time: rollout error compounds over
    the horizon, so a temporal split would confound "does the regression
    generalize" with "how far has this rollout drifted."
    """
    world_model.eval()
    phase1_model, dyn_model = world_model.autoencoder, world_model.dynamics

    h_all, actions_all = _encode_val_h(phase1_model, val_trajs, device)
    N = h_all.shape[0]
    T_full = actions_all.shape[1]
    n_steps = T_full - 1
    if n_steps <= 0 or N < 2:
        return

    # Roll all trajectories forward together (B = N)
    q, p = dyn_model.encode(h_all[:, 1])
    qp_steps = []
    for t in range(n_steps):
        q, p = dyn_model.controlled_step(q, p, actions_all[:, 1 + t: 2 + t])
        qp_steps.append(torch.cat([q, p], dim=-1))
    all_qp = torch.stack(qp_steps, dim=1).cpu()                        # (N, n_steps, D)
    all_st = torch.stack([t[2] for t in val_trajs]).float()[:, 2:2 + n_steps]  # (N, n_steps, 3)
    D = all_qp.shape[-1]

    train_qp = all_qp[0::2].reshape(-1, D)
    train_st = all_st[0::2].reshape(-1, all_st.shape[-1])
    val_qp = all_qp[1::2].reshape(-1, D)
    val_st = all_st[1::2].reshape(-1, all_st.shape[-1])
    val_idx = (
        torch.arange(n_steps, dtype=torch.float32).repeat(all_qp[1::2].shape[0]).numpy()
    )

    A = torch.linalg.lstsq(train_qp, train_st).solution
    st_pred = (val_qp @ A).numpy()
    st_true = val_st.numpy()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    sc = None
    for i, name in enumerate(["cos(θ)", "sin(θ)", "θ̇ (rad/s)"]):
        true_i, pred_i = st_true[:, i], st_pred[:, i]
        sc = axes[i].scatter(true_i, pred_i, c=val_idx, cmap="viridis", s=2, alpha=0.4, linewidths=0)
        lo, hi = min(true_i.min(), pred_i.min()), max(true_i.max(), pred_i.max())
        axes[i].plot([lo, hi], [lo, hi], "r--", linewidth=0.8)
        axes[i].set_xlabel(f"True {name}")
        axes[i].set_ylabel(f"Predicted {name}")
        ss_res = ((true_i - pred_i) ** 2).sum()
        ss_tot = ((true_i - true_i.mean()) ** 2).sum()
        axes[i].set_title(f"{name}  R²={1 - ss_res / (ss_tot + 1e-8):.3f}")
    fig.colorbar(sc, ax=axes, label="rollout step", fraction=0.03, pad=0.02)
    fig.suptitle(
        f"Closed-loop (q,p) → state regression, held-out trajectories (epoch {epoch + 1})"
    )
    writer.add_figure(tag, fig, epoch)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Phase 3: end-to-end finetuning
# ---------------------------------------------------------------------------


def _train_epoch_phase3(
    world_model: WorldModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
    device: torch.device,
    seq_len: int,
    seed_ctx_len: int,
    logdet_weight: float,
    recon_weight: float,
    pixel_cl_weight: float,
    h_tf_weight: float,
    h_cl_weight: float,
    closed_loop_gamma: float,
    energy_balance_weight: float = 0.0,
    huber_delta: float = 0.0,
    decode_stride: int = 1,
) -> dict[str, float]:
    """End-to-end epoch: every module trains through the full dreaming pipeline.

    Window sampling matches Phase 2 (one random [s, s+ctx+T) window shared
    across the batch, fresh LSTM state at s), but the encoder now runs WITH
    gradients and h = mu is used deterministically (no reparameterization/KL),
    matching how the encoder is consumed in Phase 2 and at inference.

    Losses:
      recon     pixel MSE decoding the encoded h at each window frame — the
                anti-collapse anchor: the encoder must keep explaining the
                pixels, so it cannot shrink h to make dynamics trivially easy.
      pixel_cl  the dreaming objective: closed-loop rollout from the end of
                context, decoded to pixels, vs. ground-truth frames.
      tf_h/cl_h the Phase-2 h-space losses, kept as denser training signal
                but with STOP-GRAD targets — with the encoder unfrozen, a
                loss of the form MSE(pred, encoder(frames)) whose gradient
                also flows into the target is minimised by collapsing h;
                detaching the target restricts gradients to the prediction
                path. (The targets still drift as the encoder updates across
                steps; recon + pixel_cl are what anchor that slow direction.)
      logdet    same near-volume-preservation regulariser on phi as Phase 2.

    closed_loop_gamma discounts later rollout steps exactly as in Phase 2.
    huber_delta > 0 switches the pixel closed-loop per-element error to a
    2x-scaled Huber (== squared error below delta, linear above — same scale
    as MSE where the EMA curriculum gate operates), so phase-drifted late
    steps stop dominating the gradient; the h-space losses stay MSE (they are
    stop-grad auxiliaries on a different error scale). energy_balance_weight
    > 0 adds the port-Hamiltonian energy-balance consistency loss (see
    _energy_balance_loss) on detached encoded states, shaping H/R/B only.
    decode_stride > 1 decodes only every decode_stride-th window frame (recon)
    and rollout step (pixel_cl) to bound decoder activation memory at long
    curriculum horizons; the h-space losses still cover every step.
    """
    world_model.train()
    ae = world_model.autoencoder
    dyn = world_model.dynamics
    q_dim = dyn.latent_dim // 2
    ctx = seed_ctx_len
    total_loss = total_recon = total_pix_cl = total_tf = total_cl_h = 0.0
    total_logdet = total_q_var = total_p_var = total_energy_balance = 0.0

    for frames, actions, _states in loader:
        actions = actions.to(device)  # (B, T_full)
        B_size = frames.shape[0]
        T_full = actions.shape[1]

        # --- Sample one random window [s, s+ctx+T), shared across the batch ---
        T = max(min(seq_len, T_full - ctx + 1), 1)
        W = ctx + T
        max_s = T_full + 1 - W
        s = int(torch.randint(0, max_s + 1, (1,)).item()) if max_s > 0 else 0

        frames_win = frames[:, s:s + W].to(device)    # (B, W, C, H, W)
        actions_win = actions[:, s:s + W - 1]          # (B, W-1)

        # --- Encode with grad; deterministic h = mu ---
        h_all, _ = ae.encoder.forward_all(frames_win)  # (B, W, D)
        D = h_all.shape[-1]
        h_flat = h_all.reshape(B_size * W, D)

        # --- Reconstruction anchor ---
        rec_idx = torch.arange(0, W, decode_stride, device=device)
        h_rec = h_all[:, rec_idx].reshape(B_size * len(rec_idx), D)
        s_psi = ae.f_psi(h_rec)
        recon_pred = ae.decoder(s_psi[:, :q_dim])
        recon_target = frames_win[:, rec_idx].reshape(
            B_size * len(rec_idx), *frames_win.shape[2:]
        )
        recon = F.mse_loss(recon_pred, recon_target)

        # --- phi over the whole window, with logdet regulariser ---
        s_flat, log_det_flat = dyn.phi.forward_with_logdet(h_flat)
        q_all = s_flat[:, :q_dim].reshape(B_size, W, q_dim)
        p_all = s_flat[:, q_dim:].reshape(B_size, W, q_dim)
        logdet_metric = log_det_flat.detach().pow(2).mean()  # save before backward
        logdet_reg = logdet_weight * log_det_flat.pow(2).mean()

        # --- Teacher-forced h-space step at every t (stop-grad targets) ---
        T_tf = W - 1
        q_tf = q_all[:, :T_tf].reshape(B_size * T_tf, q_dim)
        p_tf = p_all[:, :T_tf].reshape(B_size * T_tf, q_dim)
        a_tf = actions_win.reshape(B_size * T_tf, 1)
        q_tf_next, p_tf_next = dyn.controlled_step(q_tf, p_tf, a_tf)
        h_tf_pred = dyn.decode(q_tf_next, p_tf_next)
        tf_loss = F.mse_loss(h_tf_pred, h_all[:, 1:].reshape(B_size * T_tf, D).detach())

        # --- Closed-loop rollout from the end of context (local index ctx-1) ---
        k = ctx - 1
        q, p = q_all[:, k], p_all[:, k]
        q_k_log, p_k_log = q.detach(), p.detach()  # save before graph is freed
        qs_steps, ps_steps = [], []
        for t in range(T):
            q, p = dyn.controlled_step(q, p, actions_win[:, k + t:k + t + 1])
            qs_steps.append(q)
            ps_steps.append(p)
        q_traj = torch.stack(qs_steps, dim=1)  # (B, T, q_dim)
        p_traj = torch.stack(ps_steps, dim=1)
        h_cl_pred = dyn.decode(
            q_traj.reshape(B_size * T, q_dim), p_traj.reshape(B_size * T, q_dim)
        ).reshape(B_size, T, D)

        step_weights = closed_loop_gamma ** torch.arange(
            T, device=device, dtype=h_cl_pred.dtype
        )

        # h-space closed-loop (stop-grad targets), every step
        h_cl_target = h_all[:, k + 1:k + 1 + T].detach()
        per_step_h = (h_cl_pred - h_cl_target).pow(2).mean(dim=(0, 2))  # (T,)
        cl_h = (per_step_h * step_weights).sum() / step_weights.sum()

        # Pixel closed-loop — the dreaming objective
        pix_idx = torch.arange(0, T, decode_stride, device=device)
        h_pix = h_cl_pred[:, pix_idx].reshape(B_size * len(pix_idx), D)
        s_pix = ae.f_psi(h_pix)
        frames_cl_pred = ae.decoder(s_pix[:, :q_dim]).reshape(
            B_size, len(pix_idx), *frames_win.shape[2:]
        )
        frames_cl_target = frames_win[:, k + 1:k + 1 + T][:, pix_idx]
        if huber_delta > 0:
            # 2x huber == squared error for |e| <= delta, linear beyond — same
            # scale as plain MSE where the EMA curriculum gate operates.
            pix_err = 2.0 * F.huber_loss(
                frames_cl_pred, frames_cl_target, reduction="none", delta=huber_delta
            )
        else:
            pix_err = (frames_cl_pred - frames_cl_target).pow(2)
        per_step_pix = pix_err.mean(dim=(0, 2, 3, 4))
        w_pix = step_weights[pix_idx]
        pix_cl = (per_step_pix * w_pix).sum() / w_pix.sum()

        loss = (
            recon_weight * recon
            + logdet_reg
            + h_tf_weight * tf_loss
            + h_cl_weight * cl_h
            + pixel_cl_weight * pix_cl
        )

        if energy_balance_weight > 0:
            eb_loss = _energy_balance_loss(dyn, q_all, p_all, actions_win)
            loss = loss + energy_balance_weight * eb_loss
            total_energy_balance = total_energy_balance + eb_loss.detach()

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(world_model.parameters(), grad_clip)
        optimizer.step()

        # Accumulate as tensors — .item() per batch forces a GPU sync each time;
        # a single sync at epoch end is enough.
        total_loss = total_loss + loss.detach()
        total_recon = total_recon + recon.detach()
        total_pix_cl = total_pix_cl + pix_cl.detach()
        total_tf = total_tf + tf_loss.detach()
        total_cl_h = total_cl_h + cl_h.detach()
        total_logdet = total_logdet + logdet_metric
        with torch.no_grad():
            q_var, p_var = _log_latent_variance(
                torch.cat([q_k_log.unsqueeze(1), q_traj.detach()], dim=1),
                torch.cat([p_k_log.unsqueeze(1), p_traj.detach()], dim=1),
            )
            total_q_var = total_q_var + q_var
            total_p_var = total_p_var + p_var

    n = len(loader)
    return {
        "phase3/loss": float(total_loss) / n,
        "phase3/recon": float(total_recon) / n,
        "phase3/cl_pixel": float(total_pix_cl) / n,
        "phase3/tf_h": float(total_tf) / n,
        "phase3/cl_h": float(total_cl_h) / n,
        "phase3/logdet_reg": float(total_logdet) / n,
        "phase3/q_var": float(total_q_var) / n,
        "phase3/p_var": float(total_p_var) / n,
        "phase3/energy_balance": float(total_energy_balance) / n,
    }


@torch.no_grad()
def _eval_loss_phase3(
    world_model: WorldModel,
    val_trajs: list,
    device: torch.device,
    dec_chunk: int = 256,
) -> dict[str, float]:
    """Pixel-space closed-loop rollout loss over the full horizon.

    Mirrors _eval_loss_phase2's closed-loop rollout (full T_full - 1 steps,
    all trajectories rolled together, fixed horizon so the metric is
    comparable across epochs) but scores in pixel space through f_psi +
    decoder — the quantity Phase 3 actually optimises. Decoding is chunked to
    bound decoder activation memory.
    """
    world_model.eval()
    ae, dyn = world_model.autoencoder, world_model.dynamics
    q_dim = dyn.latent_dim // 2

    h_all, actions_all = _encode_val_h(ae, val_trajs, device)
    N = h_all.shape[0]
    T_full = actions_all.shape[1]
    n_steps = T_full - 1
    if n_steps <= 0:
        return {}

    q, p = dyn.encode(h_all[:, 1])
    qs_steps, ps_steps = [], []
    for t in range(n_steps):
        q, p = dyn.controlled_step(q, p, actions_all[:, 1 + t: 2 + t])
        qs_steps.append(q)
        ps_steps.append(p)
    q_traj = torch.stack(qs_steps, dim=1).reshape(N * n_steps, q_dim)
    p_traj = torch.stack(ps_steps, dim=1).reshape(N * n_steps, q_dim)
    h_cl = dyn.decode(q_traj, p_traj)  # (N * n_steps, D)

    gt = torch.stack([t[0] for t in val_trajs])[:, 2:2 + n_steps]  # CPU (N, n, C, H, W)
    gt_flat = gt.reshape(N * n_steps, *gt.shape[2:])

    total_sq = 0.0
    for i in range(0, N * n_steps, dec_chunk):
        s_psi = ae.f_psi(h_cl[i:i + dec_chunk])
        pred = ae.decoder(s_psi[:, :q_dim])
        total_sq += F.mse_loss(
            pred, gt_flat[i:i + dec_chunk].to(device), reduction="sum"
        ).item()
    return {"phase3/val_pixel_cl": total_sq / gt_flat.numel()}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.group()
def cli():
    """Offline Pendulum world-model training (three phases)."""
    pass


@cli.command("phase1")
@click.option("--resume-from", type=str, default=None,
              help="Path to a checkpoint (.pt) whose autoencoder weights to warm-start "
                   "from; training still writes to a fresh run dir, and the optimizer "
                   "and epoch count both restart from scratch")
# data
@click.option("--n-episodes", type=int, default=200, show_default=True)
@click.option("--img-size", type=int, default=64, show_default=True)
@click.option("--epsilon", type=float, default=0.1, show_default=True,
              help="Fraction of steps with random uniform action")
@click.option("--energy-k", type=float, default=1.0, show_default=True,
              help="Gain for energy-pumping controller")
@click.option("--max-steps", type=int, default=200, show_default=True,
              help="Number of steps per episode")
@click.option("--damping", type=float, default=0.0, show_default=True,
              help="Linear viscous damping coefficient")
# model architecture
@click.option("--pos-ch", type=int, default=8, show_default=True)
@click.option("--feat-dim", type=int, default=256, show_default=True)
@click.option("--latent-dim", type=int, default=32, show_default=True)
@click.option("--lstm-layers", type=int, default=1, show_default=True,
              help="Number of stacked LSTM layers in the encoder (lstm encoder only)")
@click.option("--encoder-type", type=click.Choice(["lstm", "framestack"]), default="lstm",
              show_default=True,
              help="How h_t is built from frames: 'lstm' (causal LSTM hidden state over "
                   "the whole context) or 'framestack' (memoryless function of the two "
                   "most recent frames — momentum is identifiable from two consecutive "
                   "frames, so in theory no longer history is needed)")
# training
@click.option("--epochs", type=int, default=3000, show_default=True)
@click.option("--batch-size", type=int, default=8, show_default=True)
@click.option("--lr", type=float, default=1e-4, show_default=True)
@click.option("--kl-weight", type=float, default=1e-3, show_default=True)
@click.option("--free-bits", type=float, default=0.5, show_default=True)
@click.option("--deterministic", is_flag=True, default=False, show_default=True,
              help="Ablation: skip VAE reparameterization/KL entirely and train h "
                   "as a plain deterministic autoencoder latent (kl-weight/free-bits "
                   "are ignored when set).")
@click.option("--grad-clip", type=float, default=1.0, show_default=True)
@click.option("--max-context-len", type=int, default=0, show_default=True,
              help="Max frames fed to LSTM per batch step (0 = full sequence). "
                   "Sampled uniformly from [2, max-context-len] each step.")
@click.option("--temporal-reg-weight", type=float, default=0.1, show_default=True,
              help="Temporal metric regulariser weight (0 to disable)")
@click.option("--temporal-scale", type=float, default=0.01, show_default=True,
              help="Expected h-space distance per timestep")
@click.option("--sparsity-weight", type=float, default=0.0, show_default=True,
              help="L1 penalty on latent mean, pushes irrelevant dims to 0 (0 to disable)")
@click.option("--ema-alpha", type=float, default=0.99, show_default=True)
@click.option("--convergence-patience", type=int, default=0, show_default=True,
              help="Epochs of stable EMA before stopping; 0 disables")
@click.option("--convergence-threshold", type=float, default=1e-4, show_default=True)
# logging
@click.option("--log-every", type=int, default=5, show_default=True)
@click.option("--val-every", type=int, default=10, show_default=True,
              help="Epochs between validation plots (0 to disable)")
@click.option("--n-val-episodes", type=int, default=-1, show_default=True,
              help="Val episodes per type (-1 = n_episodes // 2)")
@click.option("--val-max-steps", type=int, default=0, show_default=True,
              help="Steps per val episode (0 = 2x --max-steps)")
@click.option("--checkpoint-every", type=int, default=10, show_default=True)
def phase1_cmd(**kwargs):
    """Phase 1: train the LSTM autoencoder (encoder + f_psi + decoder)."""
    assert kwargs["img_size"] % 8 == 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # autotune conv algos for our fixed shapes
    print(f"Device: {device}")

    writer = SummaryWriter(comment="_pendulum_offline_phase1")
    run_dir = make_run_dir("pendulum_offline_phase1")

    n_val_episodes = kwargs["n_val_episodes"]
    if n_val_episodes < 0:
        n_val_episodes = kwargs["n_episodes"] // 2
    n_val = n_val_episodes if kwargs["val_every"] > 0 else 0
    val_steps = kwargs["val_max_steps"] or kwargs["max_steps"] * 2

    # Mix three collection policies for training, same as the val split below —
    # an epsilon-random/energy-pumping controller alone traces a near-1D energy
    # band and under-covers the phase space.
    n_energy = kwargs["n_episodes"] // 3
    n_random = kwargs["n_episodes"] // 3
    n_spin = kwargs["n_episodes"] - n_energy - n_random
    print(
        f"\nCollecting {kwargs['n_episodes']} train episodes "
        f"({n_energy} energy-pump, {n_random} random, {n_spin} spin)..."
    )
    episodes = (
        collect_data(
            n_episodes=n_energy,
            img_size=kwargs["img_size"],
            epsilon=kwargs["epsilon"],
            energy_k=kwargs["energy_k"],
            max_steps=kwargs["max_steps"],
            damping=kwargs["damping"],
        )
        + collect_random_trajectories(
            n_episodes=n_random,
            img_size=kwargs["img_size"],
            max_steps=kwargs["max_steps"],
            damping=kwargs["damping"],
        )
        + collect_spin_trajectories(
            n_episodes=n_spin,
            img_size=kwargs["img_size"],
            max_steps=kwargs["max_steps"],
            damping=kwargs["damping"],
        )
    )

    # Save the raw training episodes (frames, actions, ground-truth state) right
    # away — Phase 2 re-encodes windows of these through the frozen encoder each
    # batch, and this cache also doubles as a dataset for later analysis. Saved
    # before training starts so it survives even if training is interrupted.
    episodes_cache_path = run_dir / "episodes_cache.pt"
    torch.save(episodes, episodes_cache_path)
    print(f"Saved episode cache ({len(episodes)} episodes) to {episodes_cache_path}")

    val_energy, val_random, val_spin = [], [], []
    if n_val > 0:
        print(f"Collecting {n_val} val episodes per type ({val_steps} steps each)...")
        val_energy = collect_val_trajectories(
            n_episodes=n_val, img_size=kwargs["img_size"],
            max_steps=val_steps, energy_k=kwargs["energy_k"], damping=kwargs["damping"],
        )
        val_random = collect_random_trajectories(
            n_episodes=n_val, img_size=kwargs["img_size"],
            max_steps=val_steps, damping=kwargs["damping"],
        )
        val_spin = collect_spin_trajectories(
            n_episodes=n_val, img_size=kwargs["img_size"],
            max_steps=val_steps, damping=kwargs["damping"],
        )

    coverage_fig = _plot_phase_space_coverage(episodes)
    writer.add_figure("data/phase_space_coverage", coverage_fig, 0)
    plt.close(coverage_fig)

    dataset = PendulumDataset(episodes)
    loader = DataLoader(
        dataset, batch_size=kwargs["batch_size"], shuffle=True,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    print(f"Dataset: {len(dataset)} episodes")

    model = TemporalAutoencoder(
        latent_dim=kwargs["latent_dim"],
        feat_dim=kwargs["feat_dim"],
        pos_ch=kwargs["pos_ch"],
        img_size=kwargs["img_size"],
        control_dim=1,
        num_layers=kwargs["lstm_layers"],
        encoder_type=kwargs["encoder_type"],
    ).to(device)
    print(f"Phase 1 model parameters: {sum(p.numel() for p in model.parameters()):,}")

    if kwargs["resume_from"]:
        print(f"Resuming autoencoder weights from {kwargs['resume_from']}...")
        resume_model = load_world_model(kwargs["resume_from"], device)
        model.load_state_dict(resume_model.autoencoder.state_dict())
        del resume_model

    optimizer = torch.optim.Adam(model.parameters(), lr=kwargs["lr"])

    # How the training episodes were collected — saved into the checkpoint so
    # Phase 2 and the dashboard can reproduce matching episodes.
    data_config = {k: kwargs[k] for k in (
        "n_episodes", "img_size", "epsilon", "energy_k", "max_steps", "damping",
    )}
    world_model = WorldModel(model, dynamics=None, data_config=data_config)

    hparams = {k: v for k, v in kwargs.items()}
    _log_hparams_text(writer, hparams)
    _log_hparams_table(writer, hparams, {})
    best_loss = float("inf")
    ema_loss = None
    converge_streak = 0

    print("\n=== Phase 1: reconstruction training ===")
    for epoch in tqdm(range(kwargs["epochs"]), desc="Phase 1"):
        metrics = _train_epoch_phase1(
            model=model,
            loader=loader,
            optimizer=optimizer,
            kl_weight=kwargs["kl_weight"],
            free_bits=kwargs["free_bits"],
            grad_clip=kwargs["grad_clip"],
            device=device,
            temporal_reg_weight=kwargs["temporal_reg_weight"],
            temporal_scale=kwargs["temporal_scale"],
            sparsity_weight=kwargs["sparsity_weight"],
            max_context_len=kwargs["max_context_len"],
            deterministic=kwargs["deterministic"],
        )

        alpha = kwargs["ema_alpha"]
        prev_ema = ema_loss
        ema_loss = (
            metrics["phase1/loss"]
            if ema_loss is None
            else alpha * ema_loss + (1.0 - alpha) * metrics["phase1/loss"]
        )

        if prev_ema is not None and kwargs["convergence_patience"] > 0:
            rel_change = abs(ema_loss - prev_ema) / (abs(prev_ema) + 1e-8)
            if rel_change < kwargs["convergence_threshold"]:
                converge_streak += 1
                if converge_streak >= kwargs["convergence_patience"]:
                    tqdm.write(
                        f"  Phase 1 converged at epoch {epoch + 1}"
                        f" (EMA Δ={rel_change:.2e}, streak={converge_streak})"
                    )
                    break
            else:
                converge_streak = 0

        if (epoch + 1) % kwargs["log_every"] == 0:
            for k, v in metrics.items():
                writer.add_scalar(k, v, epoch)
            writer.add_scalar("phase1/ema_loss", ema_loss, epoch)
            tqdm.write(
                f"  epoch {epoch + 1:4d}"
                f"  loss={metrics['phase1/loss']:.4f}"
                f"  ema={ema_loss:.4f}"
                f"  recon={metrics['phase1/recon']:.4f}"
                f"  next={metrics['phase1/recon_next']:.4f}"
                f"  kl={metrics['phase1/kl']:.4f}"
                f"  tc={metrics['phase1/temporal_reg']:.4f}"
            )

        if kwargs["val_every"] > 0 and (epoch + 1) % kwargs["val_every"] == 0:
            policy_val_trajs = (
                (val_energy, "energy_pump"),
                (val_random, "random"),
                (val_spin, "spin"),
            )
            for val_trajs, label in policy_val_trajs:
                if not val_trajs:
                    continue
                val_metrics = _eval_loss_phase1(model, val_trajs, device)
                for k, v in val_metrics.items():
                    writer.add_scalar(f"{k}/{label}", v, epoch)
                _log_reconstruction_lstm_video(
                    model=model, val_traj=val_trajs[0],
                    device=device, writer=writer, epoch=epoch,
                    tag=f"val/reconstruction_lstm/{label}",
                )
                if len(val_trajs) >= 2:
                    _log_h_state_regression_coeffs_phase1(
                        model=model, val_trajs=val_trajs,
                        device=device, writer=writer, epoch=epoch,
                        tag=f"val/h_state_regression_coeffs/{label}",
                    )
            scatter_sets = [(vt, label) for vt, label in policy_val_trajs if len(vt) >= 2]
            if scatter_sets:
                _log_latent_scatter_phase1(
                    model=model, val_traj_sets=scatter_sets,
                    device=device, writer=writer, epoch=epoch,
                )
                _log_latent_distribution_phase1(
                    model=model, val_traj_sets=scatter_sets,
                    device=device, writer=writer, epoch=epoch,
                )
                _log_cnn_feature_distribution_phase1(
                    model=model, val_traj_sets=scatter_sets,
                    device=device, writer=writer, epoch=epoch,
                )
                _log_cnn_feature_regression_phase1(
                    model=model, val_traj_sets=scatter_sets,
                    device=device, writer=writer, epoch=epoch,
                )
                _log_cnn_feature_fold_probe_phase1(
                    model=model, val_traj_sets=scatter_sets,
                    device=device, writer=writer, epoch=epoch,
                )
            _log_markov_pairwise_probe_phase1(
                model=model, val_traj_sets=policy_val_trajs,
                device=device, writer=writer, epoch=epoch,
            )
            _log_reconstruction_lstm_video(
                model=model, val_traj=episodes[0],
                device=device, writer=writer, epoch=epoch,
                tag="train/reconstruction_lstm",
            )

        if (
            kwargs["checkpoint_every"] > 0
            and (epoch + 1) % kwargs["checkpoint_every"] == 0
            and metrics["phase1/loss"] < best_loss
        ):
            world_model.save(run_dir, "best", hparams, metrics, epoch)
            best_loss = metrics["phase1/loss"]

    # Always save final checkpoint
    world_model.save(run_dir, "final", hparams, metrics, epoch)

    print(f"\nTo run Phase 2:\n  uv run python experiments/pendulum_offline.py phase2 --phase1-run {run_dir}")

    writer.close()
    print("\nDone. Run: tensorboard --logdir runs")
    os._exit(0)


@cli.command("phase2")
# input — architecture + data params are loaded from the Phase 1 checkpoint
@click.option("--phase1-run", type=str, required=True,
              help="Path to a Phase 1 run directory; loads best.pt (arch/data config + "
                   "autoencoder weights) and episodes_cache.pt for training episodes")
@click.option("--phase1-checkpoint", type=str, default=None,
              help="Override the Phase 1 checkpoint (default: {phase1-run}/best.pt, "
                   "falling back to final.pt)")
@click.option("--episode-cache", type=str, default=None,
              help="Override the episode cache path (default: {phase1-run}/episodes_cache.pt)")
@click.option("--resume-from", type=str, default=None,
              help="Path to a Phase 2 checkpoint (.pt) whose dynamics weights to warm-start "
                   "from; training still writes to a fresh run dir, and the optimizer "
                   "and epoch count both restart from scratch")
# dynamics model
@click.option("--dt", type=float, default=0.05, show_default=True,
              help="Integration step size (should match the env frame interval)")
@click.option("--separable/--no-separable", default=True, show_default=True,
              help="Use a separable Hamiltonian H = T(p) + V(q); required for --integrator leapfrog")
@click.option("--learn-structure/--no-learn-structure", default=True, show_default=True,
              help="Learn R/B matrices; --no-learn-structure fixes R from the data damping, B=1")
@click.option("--state-dep-r/--no-state-dep-r", default=False, show_default=True,
              help="Parameterize the dissipation R as a function of the latent phase-space "
                   "point z = (q, p) via a small MLP (R_pp(z) = L(z)L(z)ᵀ, PSD everywhere) "
                   "instead of a constant matrix — needed for state-dependent dissipation "
                   "like the env's quadratic drag. Requires --learn-structure.")
@click.option("--integrator", type=click.Choice(["rk4", "leapfrog"]), default="leapfrog",
              show_default=True,
              help="Dynamics integrator: 'leapfrog' (symplectic Strang split, requires "
                   "separable H) or 'rk4' (classic 4-stage, works for any structure)")
# training
@click.option("--epochs", type=int, default=3000, show_default=True)
@click.option("--batch-size", type=int, default=8, show_default=True)
@click.option("--lr", type=float, default=1e-4, show_default=True)
@click.option("--structural-lr", type=float, default=1e-2, show_default=True,
              help="Learning rate for the R/B structure matrices (only with --learn-structure)")
@click.option("--grad-clip", type=float, default=1.0, show_default=True)
@click.option("--quadratic-t/--no-quadratic-t", default=True, show_default=True,
              help="Kinetic energy as a PSD quadratic form T(p) = ½ pᵀM⁻¹p with learned "
                   "constant mass (convex, T(0)=0, even in p) instead of a free MLP; "
                   "requires separable H")
@click.option("--logdet-weight", type=float, default=1e-3, show_default=True,
              help="Weight on log|det J_Phi|^2 regulariser; keeps flow near-volume-preserving")
@click.option("--l1-weight", type=float, default=0.0, show_default=True,
              help="L1 penalty on Hamiltonian network weights; encourages simpler dynamics")
@click.option("--structural-reg-weight", type=float, default=0.0, show_default=True,
              help="Frobenius norm penalty on the learned dissipation R (Phase 2, learn-structure only); prevents it from growing unbounded")
@click.option("--teacher-force-weight", type=float, default=1.0, show_default=True,
              help="Weight on teacher-forced 1-step loss (set 0 to disable)")
@click.option("--closed-loop-weight", type=float, default=1.0, show_default=True,
              help="Weight on closed-loop rollout loss (set 0 to disable)")
@click.option("--closed-loop-gamma", type=float, default=1.0, show_default=True,
              help="Exponential discount applied to closed-loop rollout steps "
                   "(weight = gamma**t, t=0 at the first post-seed step, normalised "
                   "to sum to 1). 1.0 = plain mean over all steps (no discounting); "
                   "<1.0 downweights later, more error-compounded steps.")
@click.option("--huber-delta", type=float, default=0.2, show_default=True,
              help="Huber threshold (h units) for the closed-loop per-element error: "
                   "quadratic (= MSE scale) below delta, linear above, so phase-drifted "
                   "late rollout steps stop dominating the gradient and rewarding "
                   "contractive dynamics. 0 = plain MSE.")
@click.option("--energy-balance-weight", type=float, default=1.0, show_default=True,
              help="Weight on the port-Hamiltonian energy-balance consistency loss: "
                   "H's change along encoded real trajectories must match the "
                   "dissipation -gradpH^T R(z) gradpH plus input power the model "
                   "itself claims. Anchors H and R at every timestep independent of "
                   "rollout horizon. 0 disables.")
@click.option("--h-noise-std", type=float, default=0.0, show_default=True,
              help="Zero-mean Gaussian noise added to h inputs (augmentation; targets "
                   "stay clean), as a multiplier on each h-dim's spread across the cache. "
                   "e.g. 0.05 = 5% of each dim's std. 0 disables.")
@click.option("--seed-ctx-len", type=int, default=3, show_default=True,
              help="Frames of context re-encoded fresh (LSTM state reset) before each "
                   "closed-loop seed, matching the context length the encoder actually "
                   "sees at inference; should match/be within Phase 1's --max-context-len. "
                   "The window's start point is randomised across the whole episode each "
                   "batch, so training covers every part of the episode, not just the start.")
@click.option("--seq-len-start", type=int, default=5, show_default=True,
              help="Initial closed-loop rollout length for curriculum")
@click.option("--max-seq-len", type=int, default=0, show_default=True,
              help="Cap on the curriculum's closed-loop rollout length "
                   "(0 = no cap: grow to the full episode length)")
@click.option("--seq-len-advance-threshold", type=float, default=0.005, show_default=True,
              help="Closed-loop EMA loss below which rollout length advances by 1")
@click.option("--ema-alpha", type=float, default=0.99, show_default=True)
@click.option("--convergence-patience", type=int, default=0, show_default=True,
              help="Epochs of stable EMA before stopping; 0 disables")
@click.option("--convergence-threshold", type=float, default=1e-4, show_default=True)
# logging
@click.option("--log-every", type=int, default=5, show_default=True)
@click.option("--val-every", type=int, default=10, show_default=True,
              help="Epochs between dreaming video logs (0 to disable; requires Phase 1 model)")
@click.option("--n-val-episodes", type=int, default=-1, show_default=True,
              help="Val episodes per type (-1 = phase1 n_episodes // 2)")
@click.option("--val-max-steps", type=int, default=0, show_default=True,
              help="Steps per val episode (0 = 2x phase1 max_steps)")
@click.option("--val-context-frames", type=int, default=5, show_default=True,
              help="Context frames fed to encoder before dreaming rollout")
@click.option("--checkpoint-every", type=int, default=10, show_default=True)
def phase2_cmd(**kwargs):
    """Phase 2: train Hamiltonian flow dynamics on precomputed latents."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # autotune conv algos (val encoder)
    print(f"Device: {device}")

    run1 = Path(kwargs["phase1_run"])
    if kwargs["phase1_checkpoint"]:
        phase1_ckpt = Path(kwargs["phase1_checkpoint"])
    else:
        phase1_ckpt = run1 / "best.pt"
        if not phase1_ckpt.exists():
            phase1_ckpt = run1 / "final.pt"
    if not phase1_ckpt.exists():
        raise click.UsageError(
            f"No Phase 1 checkpoint found in {run1} (expected best.pt or final.pt)."
        )
    episode_cache_path = kwargs["episode_cache"] or str(run1 / "episodes_cache.pt")

    writer = SummaryWriter(comment="_pendulum_offline_phase2")
    run_dir = make_run_dir("pendulum_offline_phase2")

    # Load the Phase 1 world model (autoencoder kept alive for dreaming video logs)
    print(f"Loading Phase 1 checkpoint from {phase1_ckpt}...")
    world_model = load_world_model(phase1_ckpt, device)
    phase1_model = world_model.autoencoder
    data_cfg = world_model.data_config
    print("Phase 1 config: " + ", ".join(
        f"{k}={v}" for k, v in {**phase1_model.config, **data_cfg}.items()
    ))

    # Load episode cache
    if not Path(episode_cache_path).exists():
        raise click.UsageError(
            f"Episode cache not found at {episode_cache_path}. "
            "Re-run Phase 1 to regenerate it, or pass --episode-cache explicitly."
        )
    print(f"Loading episode cache from {episode_cache_path}...")
    episodes = torch.load(episode_cache_path, weights_only=False)

    # Collect val episodes (only if dreaming logs are enabled)
    train_sample_trajs = []
    val_energy, val_random, val_spin = [], [], []
    energy_grid_episodes = []
    if kwargs["val_every"] > 0:
        n_val = kwargs["n_val_episodes"]
        if n_val < 0:
            n_val = data_cfg.get("n_episodes", 200) // 2
        val_steps = kwargs["val_max_steps"] or data_cfg.get("max_steps", 200) * 2
        img_size = data_cfg.get("img_size", 64)
        damping = data_cfg.get("damping", 0.0)
        print(f"Collecting {n_val} val episodes per type ({val_steps} steps each)...")
        val_energy = collect_val_trajectories(
            n_episodes=n_val, img_size=img_size,
            max_steps=val_steps, energy_k=data_cfg.get("energy_k", 1.0),
            damping=damping,
        )
        val_random = collect_random_trajectories(
            n_episodes=n_val, img_size=img_size,
            max_steps=val_steps, damping=damping,
        )
        val_spin = collect_spin_trajectories(
            n_episodes=n_val, img_size=img_size,
            max_steps=val_steps, damping=damping,
        )
        print("Collecting 3 training-distribution episodes for video logging...")
        train_sample_trajs = collect_data(
            n_episodes=3,
            img_size=img_size,
            epsilon=data_cfg.get("epsilon", 0.1),
            energy_k=data_cfg.get("energy_k", 1.0),
            max_steps=data_cfg.get("max_steps", 200),
            damping=damping,
        )
        print("Collecting grid episodes for energy-landscape logging...")
        energy_grid_episodes = _collect_energy_grid_episodes(
            img_size=img_size, damping=damping,
            drag=data_cfg.get("drag", _DRAG_COEFF),
            context_frames=kwargs["val_context_frames"],
        )

    latent_dim = phase1_model.config["latent_dim"]
    print(f"Latent dim from Phase 1 config: {latent_dim}")

    # Per-dim std of h across the training episodes — used to scale augmentation
    # noise so --h-noise-std acts as a multiplier on each dimension's spread.
    # Encoded with full episode history (not windowed) purely as a one-off scale
    # estimate; it doesn't need to match the windowed encoding used in training.
    h_noise_scale = None
    if kwargs["h_noise_std"] > 0:
        with torch.no_grad():
            h_all_full, _ = _encode_val_h(phase1_model, episodes, device)
            h_noise_scale = h_all_full.reshape(-1, latent_dim).std(dim=0)
        print(
            f"h noise: std={kwargs['h_noise_std']} × per-dim spread "
            f"(mean dim std={h_noise_scale.mean().item():.4f})"
        )
        del h_all_full

    episode_dataset = PendulumDataset(episodes)
    episode_loader = DataLoader(
        episode_dataset, batch_size=kwargs["batch_size"], shuffle=True, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    print(f"Episode dataset: {len(episode_dataset)} episodes")

    dyn_model = HamiltonianFlowModel(
        latent_dim=latent_dim,
        control_dim=1,
        separable=kwargs["separable"],
        learn_structure=kwargs["learn_structure"],
        dt=kwargs["dt"],
        damping=data_cfg.get("damping", 0.0),
        integrator=kwargs["integrator"],
        quadratic_t=kwargs["quadratic_t"],
        state_dep_r=kwargs["state_dep_r"],
    ).to(device)
    print(f"Phase 2 model parameters: {sum(p.numel() for p in dyn_model.parameters()):,}")

    if kwargs["resume_from"]:
        print(f"Resuming dynamics weights from {kwargs['resume_from']}...")
        resume_model = load_world_model(kwargs["resume_from"], device)
        if resume_model.dynamics is None:
            raise click.UsageError(
                f"{kwargs['resume_from']} has no dynamics weights (Phase 1-only checkpoint)."
            )
        dyn_model.load_state_dict(resume_model.dynamics.state_dict())
        del resume_model

    world_model.dynamics = dyn_model

    if kwargs["learn_structure"]:
        optimizer = torch.optim.Adam([
            {
                "params": (
                    list(dyn_model.phi.parameters())
                    + list(dyn_model.hamiltonian.parameters())
                ),
                "lr": kwargs["lr"],
            },
            {
                "params": dyn_model.structural_parameters(),
                "lr": kwargs["structural_lr"],
            },
        ])
    else:
        optimizer = torch.optim.Adam(dyn_model.parameters(), lr=kwargs["lr"])

    # Store both phase2 CLI kwargs and the phase1 arch/data params that govern the run
    hparams = {
        **kwargs,
        "phase1_config": {**phase1_model.config, **data_cfg},
    }
    _log_hparams_text(writer, hparams)
    _log_hparams_table(writer, hparams, {})

    episode_len = episodes[0][1].shape[0]  # actions per episode
    full_seq_len = episode_len - kwargs["seed_ctx_len"] + 1  # max T given seed_ctx_len
    if kwargs["max_seq_len"] > 0:
        full_seq_len = min(full_seq_len, kwargs["max_seq_len"])
    seq_len = min(kwargs["seq_len_start"], full_seq_len)
    ema_loss = None
    ema_cl = None   # separate EMA for closed-loop loss — gates seq_len curriculum
    best_loss = float("inf")
    converge_streak = 0

    print("\n=== Phase 2: dynamics flow training ===")
    for epoch in tqdm(range(kwargs["epochs"]), desc="Phase 2"):
        metrics = _train_epoch_phase2(
            dyn_model=dyn_model,
            encoder=phase1_model.encoder,
            loader=episode_loader,
            optimizer=optimizer,
            grad_clip=kwargs["grad_clip"],
            device=device,
            seq_len=seq_len,
            seed_ctx_len=kwargs["seed_ctx_len"],
            logdet_weight=kwargs["logdet_weight"],
            l1_weight=kwargs["l1_weight"],
            teacher_force_weight=kwargs["teacher_force_weight"],
            closed_loop_weight=kwargs["closed_loop_weight"],
            closed_loop_gamma=kwargs["closed_loop_gamma"],
            structural_reg_weight=kwargs["structural_reg_weight"],
            energy_balance_weight=kwargs["energy_balance_weight"],
            huber_delta=kwargs["huber_delta"],
            h_noise_std=kwargs["h_noise_std"],
            h_noise_scale=h_noise_scale,
        )

        alpha = kwargs["ema_alpha"]
        prev_ema = ema_loss
        ema_loss = (
            metrics["phase2/dynamics"]
            if ema_loss is None
            else alpha * ema_loss + (1.0 - alpha) * metrics["phase2/dynamics"]
        )
        ema_cl = (
            metrics["phase2/cl_loss"]
            if ema_cl is None
            else alpha * ema_cl + (1.0 - alpha) * metrics["phase2/cl_loss"]
        )

        if prev_ema is not None and kwargs["convergence_patience"] > 0:
            rel_change = abs(ema_loss - prev_ema) / (abs(prev_ema) + 1e-8)
            if rel_change < kwargs["convergence_threshold"]:
                converge_streak += 1
                if converge_streak >= kwargs["convergence_patience"]:
                    tqdm.write(
                        f"  Phase 2 converged at epoch {epoch + 1}"
                        f" (EMA Δ={rel_change:.2e}, streak={converge_streak})"
                    )
                    break
            else:
                converge_streak = 0

        # Gate seq_len curriculum on cl_loss EMA so the teacher-forced
        # component (always T_full steps) doesn't drown out the signal.
        if ema_cl < kwargs["seq_len_advance_threshold"] and seq_len < full_seq_len:
            seq_len += 1

        if (epoch + 1) % kwargs["log_every"] == 0:
            for k, v in metrics.items():
                writer.add_scalar(k, v, epoch)
            writer.add_scalar("phase2/seq_len", seq_len, epoch)
            writer.add_scalar("phase2/ema_loss", ema_loss, epoch)
            writer.add_scalar("phase2/ema_cl", ema_cl, epoch)
            if kwargs["learn_structure"]:
                writer.add_scalar(
                    "phase2/structure/B_norm",
                    dyn_model.get_B().norm().item(),
                    epoch,
                )
            tqdm.write(
                f"  epoch {epoch + 1:4d}"
                f"  seq_len={seq_len:3d}"
                f"  loss={metrics['phase2/dynamics']:.4f}"
                f"  tf={metrics['phase2/tf_loss']:.4f}"
                f"  cl={metrics['phase2/cl_loss']:.4f}"
                f"  ema_cl={ema_cl:.4f}"
                f"  logdet={metrics['phase2/logdet_reg']:.4f}"
                f"  q_var={metrics['phase2/q_var']:.4f}"
                f"  p_var={metrics['phase2/p_var']:.4f}"
            )

        if kwargs["val_every"] > 0 and (epoch + 1) % kwargs["val_every"] == 0:
            _log_structural_matrices_phase2(dyn_model=dyn_model, writer=writer, epoch=epoch)
            energy_fig = _plot_learned_energy_landscape(
                world_model, energy_grid_episodes, device=device,
            )
            writer.add_figure("val/energy_landscape", energy_fig, epoch)
            plt.close(energy_fig)
            if dyn_model._has_dissipation:
                dissipation_fig = _plot_dissipation_landscape(
                    world_model,
                    energy_grid_episodes,
                    damping=data_cfg.get("damping", 0.0),
                    drag=data_cfg.get("drag", _DRAG_COEFF),
                    device=device,
                )
                writer.add_figure("val/dissipation_landscape", dissipation_fig, epoch)
                plt.close(dissipation_fig)
            for val_trajs, label in (
                (val_energy, "energy_pump"),
                (val_random, "random"),
                (val_spin, "spin"),
            ):
                if val_trajs:
                    val_loss_metrics = _eval_loss_phase2(
                        world_model=world_model,
                        val_trajs=val_trajs,
                        device=device,
                    )
                    for k, v in val_loss_metrics.items():
                        writer.add_scalar(f"{k}/{label}", v, epoch)
                    _log_phase_space_regression_phase2(
                        world_model=world_model,
                        val_trajs=val_trajs,
                        device=device,
                        writer=writer,
                        epoch=epoch,
                        tag=f"val/phase_space_regression/{label}",
                    )
                    _log_dreamed_video_phase2(
                        world_model=world_model,
                        val_traj=val_trajs[0],
                        writer=writer,
                        epoch=epoch,
                        seq_len=seq_len,
                        context_frames=kwargs["val_context_frames"],
                        tag=f"val/dreamed_phase2/{label}",
                    )
            for i, train_traj in enumerate(train_sample_trajs):
                _log_dreamed_video_phase2(
                    world_model=world_model,
                    val_traj=train_traj,
                    writer=writer,
                    epoch=epoch,
                    seq_len=seq_len,
                    context_frames=kwargs["val_context_frames"],
                    tag=f"train/dreamed_phase2/sample_{i}",
                )

        if (
            kwargs["checkpoint_every"] > 0
            and (epoch + 1) % kwargs["checkpoint_every"] == 0
            and metrics["phase2/dynamics"] < best_loss
        ):
            world_model.save(run_dir, "best", hparams, metrics, epoch)
            best_loss = metrics["phase2/dynamics"]

    world_model.save(run_dir, "final", hparams, metrics, epoch)

    print(f"\nTo run Phase 3:\n  uv run python experiments/pendulum_offline.py phase3 --phase2-run {run_dir}")

    writer.close()
    print("\nDone. Run: tensorboard --logdir runs")
    os._exit(0)


@cli.command("phase3")
# input — everything (arch, data params, both sets of weights) comes from the
# Phase 2 checkpoint; the episode cache is found via the Phase 1 run recorded
# in its hparams
@click.option("--phase2-run", type=str, required=True,
              help="Path to a Phase 2 run directory; loads best.pt (full world model: "
                   "autoencoder + dynamics weights and configs)")
@click.option("--phase2-checkpoint", type=str, default=None,
              help="Override the Phase 2 checkpoint (default: {phase2-run}/best.pt, "
                   "falling back to final.pt)")
@click.option("--episode-cache", type=str, default=None,
              help="Override the episode cache path (default: episodes_cache.pt in the "
                   "Phase 1 run recorded in the Phase 2 checkpoint's hparams)")
@click.option("--resume-from", type=str, default=None,
              help="Path to a Phase 3 checkpoint (.pt) whose full world-model weights "
                   "(autoencoder + dynamics) to warm-start from; config and the episode "
                   "cache still resolve through --phase2-run, training writes to a fresh "
                   "run dir, and the optimizer and epoch count both restart from scratch")
# training
@click.option("--epochs", type=int, default=1000, show_default=True)
@click.option("--batch-size", type=int, default=8, show_default=True,
              help="May need lowering at long curriculum horizons — unlike Phase 2, the "
                   "encoder and decoder graphs are alive through the whole window")
@click.option("--lr", type=float, default=1e-5, show_default=True,
              help="Finetuning LR for f_psi/decoder/phi/H — deliberately much lower than "
                   "Phases 1-2 so end-to-end co-adaptation refines rather than wrecks "
                   "the structure each phase already learned")
@click.option("--encoder-lr", type=float, default=None,
              help="LR for the encoder (default: same as --lr)")
@click.option("--structural-lr", type=float, default=1e-4, show_default=True,
              help="LR for the R/B structure matrices (learn-structure checkpoints only)")
@click.option("--freeze-encoder", is_flag=True, default=False, show_default=True,
              help="Keep the Phase-1 encoder frozen; finetunes decoder/flows/dynamics "
                   "only (useful when the fold/Markov probes say the representation "
                   "is fine and the win is decoder drift-tolerance)")
@click.option("--freeze-physics", is_flag=True, default=False, show_default=True,
              help="Freeze H and R/B; finetunes only encoder/f_psi/decoder/phi, provably "
                   "preserving the learned energy landscape and dissipation")
@click.option("--grad-clip", type=float, default=1.0, show_default=True)
# loss weights
@click.option("--recon-weight", type=float, default=1.0, show_default=True,
              help="Weight on the Phase-1-style reconstruction anchor (collapse guard; "
                   "0 disables at your own risk)")
@click.option("--pixel-cl-weight", type=float, default=1.0, show_default=True,
              help="Weight on the pixel-space closed-loop (dreaming) loss")
@click.option("--h-tf-weight", type=float, default=1.0, show_default=True,
              help="Weight on the h-space teacher-forced loss (stop-grad targets)")
@click.option("--h-cl-weight", type=float, default=1.0, show_default=True,
              help="Weight on the h-space closed-loop loss (stop-grad targets)")
@click.option("--logdet-weight", type=float, default=1e-3, show_default=True,
              help="Weight on log|det J_Phi|^2 regulariser; keeps flow near-volume-preserving")
@click.option("--closed-loop-gamma", type=float, default=1.0, show_default=True,
              help="Exponential discount over closed-loop rollout steps (both h-space "
                   "and pixel), as in Phase 2")
@click.option("--huber-delta", type=float, default=0.1, show_default=True,
              help="Huber threshold (pixel units) for the pixel closed-loop per-element "
                   "error: quadratic (= MSE scale) below delta, linear above, so "
                   "phase-drifted late steps stop dominating the gradient; h-space "
                   "losses stay MSE. 0 = plain MSE.")
@click.option("--energy-balance-weight", type=float, default=1.0, show_default=True,
              help="Weight on the port-Hamiltonian energy-balance consistency loss "
                   "(as Phase 2), on detached encoded states so it shapes H/R/B only. "
                   "0 disables.")
@click.option("--decode-stride", type=int, default=1, show_default=True,
              help="Decode only every Nth window frame (recon) and rollout step "
                   "(pixel loss) to bound decoder memory at long horizons; h-space "
                   "losses still cover every step")
# curriculum
@click.option("--seed-ctx-len", type=int, default=3, show_default=True,
              help="Frames of context encoded before each closed-loop seed (as Phase 2)")
@click.option("--seq-len-start", type=int, default=5, show_default=True,
              help="Initial closed-loop rollout length for curriculum")
@click.option("--max-seq-len", type=int, default=0, show_default=True,
              help="Cap on the curriculum's closed-loop rollout length "
                   "(0 = no cap: grow to the full episode length)")
@click.option("--seq-len-advance-threshold", type=float, default=2e-3, show_default=True,
              help="Pixel closed-loop EMA loss below which rollout length advances by 1 "
                   "(pixel MSE scale — not comparable to Phase 2's h-space threshold)")
@click.option("--ema-alpha", type=float, default=0.99, show_default=True)
@click.option("--convergence-patience", type=int, default=0, show_default=True,
              help="Epochs of stable EMA before stopping; 0 disables")
@click.option("--convergence-threshold", type=float, default=1e-4, show_default=True)
# logging
@click.option("--log-every", type=int, default=5, show_default=True)
@click.option("--val-every", type=int, default=10, show_default=True,
              help="Epochs between validation logs (0 to disable)")
@click.option("--n-val-episodes", type=int, default=-1, show_default=True,
              help="Val episodes per type (-1 = phase1 n_episodes // 2)")
@click.option("--val-max-steps", type=int, default=0, show_default=True,
              help="Steps per val episode (0 = 2x phase1 max_steps)")
@click.option("--val-context-frames", type=int, default=5, show_default=True,
              help="Context frames fed to encoder before dreaming rollout")
@click.option("--checkpoint-every", type=int, default=10, show_default=True)
def phase3_cmd(**kwargs):
    """Phase 3: finetune the whole world model end-to-end through the dream pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")

    run2 = Path(kwargs["phase2_run"])
    if kwargs["phase2_checkpoint"]:
        phase2_ckpt = Path(kwargs["phase2_checkpoint"])
    else:
        phase2_ckpt = run2 / "best.pt"
        if not phase2_ckpt.exists():
            phase2_ckpt = run2 / "final.pt"
    if not phase2_ckpt.exists():
        raise click.UsageError(
            f"No Phase 2 checkpoint found in {run2} (expected best.pt or final.pt)."
        )

    # Peek at the checkpoint payload for the Phase 1 run path (episode cache
    # default) and to fail early on a Phase 1-only checkpoint.
    payload = torch.load(phase2_ckpt, map_location="cpu", weights_only=True)
    if payload.get("dynamics") is None:
        raise click.UsageError(
            f"{phase2_ckpt} has no dynamics weights (Phase 1-only checkpoint); "
            "Phase 3 needs a complete Phase 2 world model."
        )
    phase1_run = (payload.get("hparams") or {}).get("phase1_run")
    del payload

    if kwargs["episode_cache"]:
        episode_cache_path = Path(kwargs["episode_cache"])
    elif phase1_run:
        episode_cache_path = Path(phase1_run) / "episodes_cache.pt"
    else:
        raise click.UsageError(
            f"{phase2_ckpt} records no phase1_run in its hparams; "
            "pass --episode-cache explicitly."
        )
    if not episode_cache_path.exists():
        raise click.UsageError(
            f"Episode cache not found at {episode_cache_path}. "
            "Pass --episode-cache explicitly."
        )

    writer = SummaryWriter(comment="_pendulum_offline_phase3")
    run_dir = make_run_dir("pendulum_offline_phase3")

    print(f"Loading Phase 2 checkpoint from {phase2_ckpt}...")
    world_model = load_world_model(phase2_ckpt, device)
    phase1_model = world_model.autoencoder
    dyn_model = world_model.dynamics
    data_cfg = world_model.data_config
    print("World-model config: " + ", ".join(
        f"{k}={v}" for k, v in {**phase1_model.config, **dyn_model.config, **data_cfg}.items()
    ))

    if kwargs["resume_from"]:
        print(f"Resuming world-model weights from {kwargs['resume_from']}...")
        resume_model = load_world_model(kwargs["resume_from"], device)
        if resume_model.dynamics is None:
            raise click.UsageError(
                f"{kwargs['resume_from']} has no dynamics weights (Phase 1-only "
                "checkpoint); Phase 3 needs a complete world model to resume from."
            )
        phase1_model.load_state_dict(resume_model.autoencoder.state_dict())
        dyn_model.load_state_dict(resume_model.dynamics.state_dict())
        del resume_model

    print(f"Loading episode cache from {episode_cache_path}...")
    episodes = torch.load(episode_cache_path, weights_only=False)

    # Collect val episodes (only if validation logs are enabled)
    train_sample_trajs = []
    val_energy, val_random, val_spin = [], [], []
    energy_grid_episodes = []
    if kwargs["val_every"] > 0:
        n_val = kwargs["n_val_episodes"]
        if n_val < 0:
            n_val = data_cfg.get("n_episodes", 200) // 2
        val_steps = kwargs["val_max_steps"] or data_cfg.get("max_steps", 200) * 2
        img_size = data_cfg.get("img_size", 64)
        damping = data_cfg.get("damping", 0.0)
        print(f"Collecting {n_val} val episodes per type ({val_steps} steps each)...")
        val_energy = collect_val_trajectories(
            n_episodes=n_val, img_size=img_size,
            max_steps=val_steps, energy_k=data_cfg.get("energy_k", 1.0),
            damping=damping,
        )
        val_random = collect_random_trajectories(
            n_episodes=n_val, img_size=img_size,
            max_steps=val_steps, damping=damping,
        )
        val_spin = collect_spin_trajectories(
            n_episodes=n_val, img_size=img_size,
            max_steps=val_steps, damping=damping,
        )
        print("Collecting 3 training-distribution episodes for video logging...")
        train_sample_trajs = collect_data(
            n_episodes=3,
            img_size=img_size,
            epsilon=data_cfg.get("epsilon", 0.1),
            energy_k=data_cfg.get("energy_k", 1.0),
            max_steps=data_cfg.get("max_steps", 200),
            damping=damping,
        )
        print("Collecting grid episodes for energy-landscape logging...")
        energy_grid_episodes = _collect_energy_grid_episodes(
            img_size=img_size, damping=damping,
            drag=data_cfg.get("drag", _DRAG_COEFF),
            context_frames=kwargs["val_context_frames"],
        )

    episode_dataset = PendulumDataset(episodes)
    episode_loader = DataLoader(
        episode_dataset, batch_size=kwargs["batch_size"], shuffle=True, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    print(f"Episode dataset: {len(episode_dataset)} episodes")

    # Freezing + per-module param groups. requires_grad_(False) (not just
    # optimizer exclusion) so autograd skips the frozen subgraphs entirely.
    if kwargs["freeze_encoder"]:
        phase1_model.encoder.requires_grad_(False)
    if kwargs["freeze_physics"]:
        dyn_model.hamiltonian.requires_grad_(False)
        if dyn_model.learn_structure:
            for prm in dyn_model.structural_parameters():
                prm.requires_grad_(False)

    encoder_lr = kwargs["encoder_lr"] if kwargs["encoder_lr"] is not None else kwargs["lr"]
    # next_frame_decoder is deliberately absent: no Phase 3 loss touches it.
    groups = []
    if not kwargs["freeze_encoder"]:
        groups.append({"params": list(phase1_model.encoder.parameters()), "lr": encoder_lr})
    groups.append({
        "params": (
            list(phase1_model.f_psi.parameters())
            + list(phase1_model.decoder.parameters())
            + list(dyn_model.phi.parameters())
        ),
        "lr": kwargs["lr"],
    })
    if not kwargs["freeze_physics"]:
        groups.append({"params": list(dyn_model.hamiltonian.parameters()), "lr": kwargs["lr"]})
        if dyn_model.learn_structure:
            groups.append({
                "params": dyn_model.structural_parameters(),
                "lr": kwargs["structural_lr"],
            })
    optimizer = torch.optim.Adam(groups)
    n_trainable = sum(p.numel() for g in groups for p in g["params"])
    print(f"Phase 3 trainable parameters: {n_trainable:,}")

    hparams = {
        **kwargs,
        "phase1_config": {**phase1_model.config, **data_cfg},
        "phase2_config": dict(dyn_model.config),
    }
    _log_hparams_text(writer, hparams)
    _log_hparams_table(writer, hparams, {})

    episode_len = episodes[0][1].shape[0]  # actions per episode
    full_seq_len = episode_len - kwargs["seed_ctx_len"] + 1
    if kwargs["max_seq_len"] > 0:
        full_seq_len = min(full_seq_len, kwargs["max_seq_len"])
    seq_len = min(kwargs["seq_len_start"], full_seq_len)
    ema_loss = None
    ema_cl = None   # EMA of the PIXEL closed-loop loss — gates seq_len curriculum
    best_loss = float("inf")
    converge_streak = 0

    print("\n=== Phase 3: end-to-end finetuning ===")
    for epoch in tqdm(range(kwargs["epochs"]), desc="Phase 3"):
        metrics = _train_epoch_phase3(
            world_model=world_model,
            loader=episode_loader,
            optimizer=optimizer,
            grad_clip=kwargs["grad_clip"],
            device=device,
            seq_len=seq_len,
            seed_ctx_len=kwargs["seed_ctx_len"],
            logdet_weight=kwargs["logdet_weight"],
            recon_weight=kwargs["recon_weight"],
            pixel_cl_weight=kwargs["pixel_cl_weight"],
            h_tf_weight=kwargs["h_tf_weight"],
            h_cl_weight=kwargs["h_cl_weight"],
            closed_loop_gamma=kwargs["closed_loop_gamma"],
            energy_balance_weight=kwargs["energy_balance_weight"],
            huber_delta=kwargs["huber_delta"],
            decode_stride=kwargs["decode_stride"],
        )

        alpha = kwargs["ema_alpha"]
        prev_ema = ema_loss
        ema_loss = (
            metrics["phase3/loss"]
            if ema_loss is None
            else alpha * ema_loss + (1.0 - alpha) * metrics["phase3/loss"]
        )
        ema_cl = (
            metrics["phase3/cl_pixel"]
            if ema_cl is None
            else alpha * ema_cl + (1.0 - alpha) * metrics["phase3/cl_pixel"]
        )

        if prev_ema is not None and kwargs["convergence_patience"] > 0:
            rel_change = abs(ema_loss - prev_ema) / (abs(prev_ema) + 1e-8)
            if rel_change < kwargs["convergence_threshold"]:
                converge_streak += 1
                if converge_streak >= kwargs["convergence_patience"]:
                    tqdm.write(
                        f"  Phase 3 converged at epoch {epoch + 1}"
                        f" (EMA Δ={rel_change:.2e}, streak={converge_streak})"
                    )
                    break
            else:
                converge_streak = 0

        # Gate seq_len curriculum on the pixel closed-loop EMA — the loss whose
        # horizon actually grows with seq_len.
        if ema_cl < kwargs["seq_len_advance_threshold"] and seq_len < full_seq_len:
            seq_len += 1

        if (epoch + 1) % kwargs["log_every"] == 0:
            for k, v in metrics.items():
                writer.add_scalar(k, v, epoch)
            writer.add_scalar("phase3/seq_len", seq_len, epoch)
            writer.add_scalar("phase3/ema_loss", ema_loss, epoch)
            writer.add_scalar("phase3/ema_cl_pixel", ema_cl, epoch)
            if dyn_model.learn_structure:
                writer.add_scalar(
                    "phase3/structure/B_norm",
                    dyn_model.get_B().norm().item(),
                    epoch,
                )
            tqdm.write(
                f"  epoch {epoch + 1:4d}"
                f"  seq_len={seq_len:3d}"
                f"  loss={metrics['phase3/loss']:.4f}"
                f"  recon={metrics['phase3/recon']:.4f}"
                f"  cl_pix={metrics['phase3/cl_pixel']:.4f}"
                f"  ema_cl_pix={ema_cl:.4f}"
                f"  tf_h={metrics['phase3/tf_h']:.4f}"
                f"  cl_h={metrics['phase3/cl_h']:.4f}"
                f"  logdet={metrics['phase3/logdet_reg']:.4f}"
            )

        if kwargs["val_every"] > 0 and (epoch + 1) % kwargs["val_every"] == 0:
            # Full Phase-2 diagnostic suite: end-to-end co-adaptation can trade
            # the learned physics for pixel accuracy, and these are how you see
            # it happening (energy-landscape r, dissipation r, R structure).
            _log_structural_matrices_phase2(dyn_model=dyn_model, writer=writer, epoch=epoch)
            energy_fig = _plot_learned_energy_landscape(
                world_model, energy_grid_episodes, device=device,
            )
            writer.add_figure("val/energy_landscape", energy_fig, epoch)
            plt.close(energy_fig)
            if dyn_model._has_dissipation:
                dissipation_fig = _plot_dissipation_landscape(
                    world_model,
                    energy_grid_episodes,
                    damping=data_cfg.get("damping", 0.0),
                    drag=data_cfg.get("drag", _DRAG_COEFF),
                    device=device,
                )
                writer.add_figure("val/dissipation_landscape", dissipation_fig, epoch)
                plt.close(dissipation_fig)
            policy_val_trajs = (
                (val_energy, "energy_pump"),
                (val_random, "random"),
                (val_spin, "spin"),
            )
            for val_trajs, label in policy_val_trajs:
                if not val_trajs:
                    continue
                val_metrics = {
                    **_eval_loss_phase1(phase1_model, val_trajs, device),
                    **_eval_loss_phase2(world_model, val_trajs, device),
                    **_eval_loss_phase3(world_model, val_trajs, device),
                }
                for k, v in val_metrics.items():
                    writer.add_scalar(f"{k}/{label}", v, epoch)
                _log_phase_space_regression_phase2(
                    world_model=world_model,
                    val_trajs=val_trajs,
                    device=device,
                    writer=writer,
                    epoch=epoch,
                    tag=f"val/phase_space_regression/{label}",
                )
                _log_dreamed_video_phase2(
                    world_model=world_model,
                    val_traj=val_trajs[0],
                    writer=writer,
                    epoch=epoch,
                    seq_len=seq_len,
                    context_frames=kwargs["val_context_frames"],
                    tag=f"val/dreamed_phase3/{label}",
                )
            # Representation-drift probes: with the encoder unfrozen these are
            # the direct read on whether Phase 3 is fixing the Markov mismatch
            # (fold scores dropping) or collapsing h (latent R² degrading).
            scatter_sets = [(vt, label) for vt, label in policy_val_trajs if len(vt) >= 2]
            if scatter_sets:
                _log_latent_scatter_phase1(
                    model=phase1_model, val_traj_sets=scatter_sets,
                    device=device, writer=writer, epoch=epoch,
                )
                _log_latent_distribution_phase1(
                    model=phase1_model, val_traj_sets=scatter_sets,
                    device=device, writer=writer, epoch=epoch,
                )
            _log_markov_pairwise_probe_phase1(
                model=phase1_model, val_traj_sets=policy_val_trajs,
                device=device, writer=writer, epoch=epoch,
            )
            for i, train_traj in enumerate(train_sample_trajs):
                _log_dreamed_video_phase2(
                    world_model=world_model,
                    val_traj=train_traj,
                    writer=writer,
                    epoch=epoch,
                    seq_len=seq_len,
                    context_frames=kwargs["val_context_frames"],
                    tag=f"train/dreamed_phase3/sample_{i}",
                )

        if (
            kwargs["checkpoint_every"] > 0
            and (epoch + 1) % kwargs["checkpoint_every"] == 0
            and metrics["phase3/loss"] < best_loss
        ):
            world_model.save(run_dir, "best", hparams, metrics, epoch)
            best_loss = metrics["phase3/loss"]

    world_model.save(run_dir, "final", hparams, metrics, epoch)

    writer.close()
    print("\nDone. Run: tensorboard --logdir runs")
    os._exit(0)


if __name__ == "__main__":
    cli()
