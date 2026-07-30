"""Offline Pendulum world-model training — two-phase regimen.

Phase 1 (phase1 subcommand):
    Train the LSTM autoencoder (encoder + f_psi + decoder) for reconstruction.
    Loss = MSE(decoder(f_psi(z)[:q_dim]), frame) + kl_weight * KL.
    After training, precomputes and saves h_t = encoder_mu(frame_t) for every
    frame of every training episode to h_cache.pt in the run directory.
    Saves a unified world-model checkpoint with dynamics=None.

Phase 2 (phase2 subcommand):
    Load precomputed h_t cache. Train a new HamiltonianFlowModel (Phi + H + J/R/B)
    that maps h_t → (q, p) such that Hamiltonian dynamics hold:

        L_tf  = MSE(phi^{-1}(RK4(phi(h_t), u_t)),  h_{t+1})       [teacher-forced, all t]
        L_cl  = MSE(phi^{-1}(RK4^k(phi(h_seed), u)), h_{seed+k})  [closed-loop, seq_len steps]

    Architecture and data params (latent_dim, img_size, etc.) are loaded
    automatically from the Phase 1 checkpoint — no need to re-specify them.
    Saves a complete world-model checkpoint (autoencoder + dynamics): the one
    file the dashboard needs.

Inference (dreaming) after both phases:
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
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.pendulum import (
    PendulumDataset,
    collect_data,
    collect_random_trajectories,
    collect_spin_trajectories,
    collect_val_trajectories,
    _energy,
    _G,
)
from hamilton_rl.checkpoint import load_world_model, make_run_dir
from hamilton_rl.models import HamiltonianFlowModel, LSTMAutoencoder, WorldModel


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

def _energy_sweep(
    H,
    min_vel=-10,
    max_vel=10,
    min_angle=-torch.pi,
    max_angle=torch.pi,
    resolution=20,
):
    """Compute and log an energy landscape across angle and angular velocity"""
    theta_dot = torch.linspace(min_vel, max_vel, resolution)
    theta = torch.linspace(min_angle, max_angle, resolution)

    output = H(theta[:, None], theta_dot[None, :])

    return output


def _plot_energy_sweep(
    H,
    min_vel=-10,
    max_vel=10,
    min_angle=-torch.pi,
    max_angle=torch.pi,
    resolution=20,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Render the energy landscape from _energy_sweep as a heatmap (θ x-axis, θ̇ y-axis)."""
    output = _energy_sweep(
        H,
        min_vel=min_vel,
        max_vel=max_vel,
        min_angle=min_angle,
        max_angle=max_angle,
        resolution=resolution,
    )
    # output[i, j] = H(theta[i], theta_dot[j]); imshow expects rows=y, cols=x, so transpose.
    grid = output.detach().cpu().numpy().T

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    im = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        extent=[min_angle, max_angle, min_vel, max_vel],
        cmap="viridis",
    )
    ax.set_xlabel("θ (rad)")
    ax.set_ylabel("θ̇ (rad/s)")
    ax.set_title("Energy landscape")
    fig.colorbar(im, ax=ax, label="H(θ, θ̇)")

    return fig


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

    ax.scatter(theta_all, theta_dot_all, s=2, alpha=0.08, linewidths=0, color="tab:blue")
    ax.set_xlabel("θ (rad)")
    ax.set_ylabel("θ̇ (rad/s)")
    ax.set_title(f"Training data phase-space coverage (N={len(theta_all):,})")
    fig.tight_layout()

    return fig


def _pca_top_k(X: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Top-k principal components of (N, D) via SVD.

    Returns (mean (D,), directions (k, D) unit/orthogonal, projections (N, k),
    explained-variance ratios (k,)).
    """
    mean = X.mean(dim=0)
    Xc = X - mean
    _, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    directions = Vh[:k]
    projections = Xc @ directions.T
    explained = S[:k] ** 2 / (S**2).sum()
    return mean, directions, projections, explained


@torch.no_grad()
def _collect_qp_samples(
    model: WorldModel,
    n_episodes: int = 5,
    max_steps: int = 200,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Roll out real trajectories and encode them through the full model.

    Mixes three policies for state-space coverage: random torque (broad but,
    per earlier plots, tends to trace a near-1D energy band), spin-maximising
    (drives high |theta_dot| regardless of angle), and energy-pumping
    (drives toward upright with the classic swing-up energy-shaping law) —
    together these decorrelate position and velocity better than any one
    alone. Encoding the real frame sequence (rather than synthetic per-point
    clips) lets the causal LSTM infer velocity from actual motion, exactly
    as at train time.

    Returns:
        q, p: each (N, q_dim) learned latents, N = 3 * n_episodes * (max_steps + 1)
        H_true: (N,) ground-truth pendulum energy at each sampled timestep
    """
    if device is None:
        device = next(model.autoencoder.parameters()).device
    img_size = model.data_config.get("img_size", 64)

    episodes = (
        collect_random_trajectories(n_episodes=n_episodes, img_size=img_size, max_steps=max_steps)
        + collect_spin_trajectories(n_episodes=n_episodes, img_size=img_size, max_steps=max_steps)
        + collect_val_trajectories(n_episodes=n_episodes, img_size=img_size, max_steps=max_steps)
    )

    qs, ps, energies = [], [], []
    for frames, _actions, states in episodes:
        mu_all, _ = model.autoencoder.encoder.forward_all(frames.unsqueeze(0).to(device))
        h = mu_all.squeeze(0)  # (T+1, latent_dim)
        q, p = model.dynamics.encode(h)
        qs.append(q.cpu())
        ps.append(p.cpu())

        theta = torch.atan2(states[:, 1], states[:, 0])
        theta_dot = states[:, 2]
        energies.append(0.5 * theta_dot**2 + _G * (1.0 + torch.cos(theta)))

    return torch.cat(qs), torch.cat(ps), torch.cat(energies)


@torch.no_grad()
def _learned_energy_landscape(
    model: WorldModel,
    q: torch.Tensor,
    p: torch.Tensor,
    resolution: int = 40,
) -> dict:
    """PCA the full (q, p) latent jointly and sweep learned H over its top-2 slice.

    q and p individually turn out to be nearly collinear in practice (the
    flow leaks information between blocks — see conversation), so PCA-ing
    them separately just rediscovers the same axis twice. Doing PCA on the
    concatenated z = [q, p] and taking its top 2 components instead finds
    whatever 2 directions actually carry the most variance, regardless of
    which block they fall in.

    Grid point (alpha, beta) maps back to the full latent as
    z_mean + alpha * dir0 + beta * dir1 (all other PCs held at their sample
    mean), then split back into (q, p) halves for the Hamiltonian. Outside
    [min, max] of the observed projections it's extrapolation into latent
    space the model never saw.
    """
    device = next(model.autoencoder.parameters()).device
    q_dim = q.shape[-1]
    z = torch.cat([q, p], dim=-1)
    z_mean, directions, proj, explained = _pca_top_k(z, k=2)
    dir0, dir1 = directions[0], directions[1]

    alpha = torch.linspace(proj[:, 0].min().item(), proj[:, 0].max().item(), resolution)
    beta = torch.linspace(proj[:, 1].min().item(), proj[:, 1].max().item(), resolution)

    grid_a, grid_b = torch.meshgrid(alpha, beta, indexing="ij")  # each (res, res)
    z_grid = (
        z_mean[None, None, :]
        + grid_a[:, :, None] * dir0[None, None, :]
        + grid_b[:, :, None] * dir1[None, None, :]
    ).reshape(-1, z_mean.shape[-1]).to(device)  # (res*res, latent_dim)

    q_grid, p_grid = z_grid[:, :q_dim], z_grid[:, q_dim:]
    pred = model.dynamics.hamiltonian(q_grid, p_grid).reshape(resolution, resolution).cpu()

    return {
        "alpha": alpha,
        "beta": beta,
        "pred": pred,
        "proj": proj,
        "explained": explained,
    }


def _plot_learned_energy_landscape(
    model: WorldModel,
    n_episodes: int = 5,
    max_steps: int = 200,
    resolution: int = 40,
    device: torch.device | None = None,
) -> plt.Figure:
    """Sanity-check the learned Hamiltonian's shape against the true pendulum energy.

    q and p are learned latents (q_dim = p_dim = latent_dim // 2), not
    literally (θ, θ̇), and H is only constrained through its gradient, so the
    two panels can only be expected to agree in *shape* — up to an unknown
    affine offset/scale (or even an axis flip) — not in absolute value.

    Left panel:  learned H swept over the top-2-PC slice of the joint
                 [q, p] latent, with the actual sampled points overlaid.
    Right panel: those same sampled points, colored by their true energy.
    """
    q, p, H_true = _collect_qp_samples(model, n_episodes=n_episodes, max_steps=max_steps, device=device)
    land = _learned_energy_landscape(model, q, p, resolution=resolution)
    proj = land["proj"]

    fig, (ax_pred, ax_true) = plt.subplots(1, 2, figsize=(11, 5))

    im = ax_pred.imshow(
        land["pred"].numpy().T,
        origin="lower",
        aspect="auto",
        extent=[
            land["alpha"][0].item(), land["alpha"][-1].item(),
            land["beta"][0].item(), land["beta"][-1].item(),
        ],
        cmap="viridis",
    )
    ax_pred.scatter(proj[:, 0], proj[:, 1], c="white", s=4, alpha=0.12, linewidths=0)
    ax_pred.set_xlabel(f"[q,p] PC1 ({land['explained'][0]:.0%} var)")
    ax_pred.set_ylabel(f"[q,p] PC2 ({land['explained'][1]:.0%} var)")
    ax_pred.set_title("Learned H (PCA slice)")
    fig.colorbar(im, ax=ax_pred, label="learned H")

    sc = ax_true.scatter(proj[:, 0], proj[:, 1], c=H_true, cmap="viridis", s=8, alpha=0.3, linewidths=0)
    ax_true.set_xlabel(f"[q,p] PC1 ({land['explained'][0]:.0%} var)")
    ax_true.set_ylabel(f"[q,p] PC2 ({land['explained'][1]:.0%} var)")
    ax_true.set_title("True energy at same points")
    fig.colorbar(sc, ax=ax_true, label="true H")

    fig.suptitle("Learned vs. true energy landscape (PCA sanity check)")
    fig.tight_layout()

    return fig

# ---------------------------------------------------------------------------
# Phase 1: autoencoder training
# ---------------------------------------------------------------------------


def _train_epoch_phase1(
    model: LSTMAutoencoder,
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
) -> dict[str, float]:
    """Reconstruction-only epoch: encoder + f_psi + decoder, no Hamiltonian.

    Two prediction targets from the causal (forward-only) LSTM:
      - h_t → current frame   (reconstruction signal)
      - h_t → next frame      (predictive signal; h_t has seen only 0..t)
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
    model: LSTMAutoencoder,
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
    model: LSTMAutoencoder,
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
def _log_latent_scatter_phase1(
    model: LSTMAutoencoder,
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
            axes[i].scatter(true_i, pred_i, s=2, alpha=0.12, color=colors[j % len(colors)], label=label, linewidths=0)
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
    model: LSTMAutoencoder,
    val_trajs: list,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    tag: str = "val/h_state_regression_coeffs",
) -> None:
    """Bar charts of bidirectional linear-probe coefficients between raw h_t and state.

    Unlike `_log_latent_scatter_phase1` (which probes s_all = f_psi(h_t)), this
    probes h_t = mu_all directly — the exact quantity Phase 2 consumes from
    h_cache.pt. `_log_latent_scatter_phase1`'s R² only shows whether state is
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


# ---------------------------------------------------------------------------
# Precompute h_t cache (between phases)
# ---------------------------------------------------------------------------


class LatentDataset(Dataset):
    """Dataset of precomputed (h_all, actions) pairs — no images."""

    def __init__(self, cache: list[tuple[torch.Tensor, torch.Tensor]]):
        self.cache = cache

    def __len__(self) -> int:
        return len(self.cache)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cache[i]


def precompute_latents(
    model: LSTMAutoencoder,
    episodes: list,
    device: torch.device,
    chunk_size: int = 8,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Run encoder over all training episodes (chunked batches) and cache h_t = mu_all."""
    model.eval()
    cache = []
    frames_all = torch.stack([e[0] for e in episodes])  # (N, T+1, C, H, W)
    with torch.no_grad():
        for i in tqdm(range(0, len(episodes), chunk_size), desc="Precomputing latents"):
            chunk = frames_all[i:i + chunk_size].to(device)
            mu_all, _ = model.encoder.forward_all(chunk)
            mu_cpu = mu_all.cpu()
            for j in range(mu_cpu.shape[0]):
                cache.append((mu_cpu[j], episodes[i + j][1]))
    return cache


# ---------------------------------------------------------------------------
# Phase 2: dynamics training
# ---------------------------------------------------------------------------


def _train_epoch_phase2(
    dyn_model: HamiltonianFlowModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
    device: torch.device,
    seq_len: int,
    logdet_weight: float,
    l1_weight: float = 0.0,
    max_seed_k: int = 0,
    teacher_force_weight: float = 1.0,
    structural_reg_weight: float = 0.0,
    h_noise_std: float = 0.0,
    h_noise_scale: torch.Tensor | None = None,
) -> dict[str, float]:
    """Dynamics epoch: joint teacher-forced + closed-loop rollout.

    The full h sequence is encoded through phi in one batched call, sharing
    that forward pass between both objectives.

    Teacher-forced: for every consecutive pair (h_t, h_{t+1}), take one RK4
    step from (q_t, p_t) and compare the decoded prediction to h_{t+1}.  All
    T steps are independent so they are batched as (B*T, q_dim) — no Python
    loop, no sequential graph depth.

    Closed-loop: starting from (q_k, p_k) already computed above, roll
    seq_len Hamiltonian steps without re-encoding and compare each decoded
    prediction to the corresponding real h value.  Gradients from the
    closed-loop loss flow back through phi at position k alongside those from
    the teacher-forced objective.

    Logdet regulariser is applied over all T+1 encoded timesteps rather than
    a single seed point, so its strength stays constant as seq_len grows.

    If h_noise_std > 0, zero-mean Gaussian noise is added to the h values fed
    into phi (both teacher-forced and closed-loop seeds), while the prediction
    targets stay clean.  This is denoising-style augmentation: the model learns
    to map jittered latents back onto the dynamics manifold, which improves
    closed-loop stability where rollout error accumulates.  When h_noise_scale
    (per-dim std of the data) is given, h_noise_std is a multiplier on each
    dimension's spread; otherwise it is an absolute std applied uniformly.
    """
    dyn_model.train()
    total_dynamics = total_tf = total_cl = total_logdet_reg = 0.0
    total_q_var = total_p_var = total_hamiltonian_l1 = total_grad_H_norm = total_struct_reg = 0.0
    q_dim = dyn_model.latent_dim // 2

    for h_all, actions in loader:
        h_all = h_all.to(device)      # (B, T+1, latent_dim)
        actions = actions.to(device)  # (B, T)
        B_size, T_seq, D = h_all.shape
        T_full = actions.shape[1]     # = T_seq - 1

        # --- Encode the full sequence through phi in one batched call ---
        # Augmentation: jitter the inputs to phi but keep h_all (the targets)
        # clean, so the model learns to denoise toward the dynamics manifold.
        # h_noise_scale (per-dim std) makes the std relative to each dimension's
        # spread; without it the std is absolute and uniform across dims.
        if h_noise_std > 0:
            std = h_noise_std if h_noise_scale is None else h_noise_std * h_noise_scale
            h_in = h_all + torch.randn_like(h_all) * std
        else:
            h_in = h_all
        h_flat = h_in.reshape(B_size * T_seq, D)
        s_flat, log_det_flat = dyn_model.phi.forward_with_logdet(h_flat)
        q_all = s_flat[:, :q_dim].reshape(B_size, T_seq, q_dim)  # (B, T+1, q_dim)
        p_all = s_flat[:, q_dim:].reshape(B_size, T_seq, q_dim)
        log_det_all = log_det_flat.reshape(B_size, T_seq)         # (B, T+1)
        logdet_metric = log_det_all.detach().pow(2).mean()        # save before backward

        logdet_reg = logdet_weight * log_det_all.pow(2).mean()

        # --- Teacher-forced loss: one batched RK4 step at every t ---
        # All T steps are independent — reshape to (B*T, q_dim) for one forward pass.
        q_tf = q_all[:, :T_full].reshape(B_size * T_full, q_dim)
        p_tf = p_all[:, :T_full].reshape(B_size * T_full, q_dim)
        a_tf = actions.reshape(B_size * T_full, 1)
        q_tf_next, p_tf_next = dyn_model.controlled_step(q_tf, p_tf, a_tf)
        h_tf_pred = dyn_model.decode(q_tf_next, p_tf_next)
        h_tf_target = h_all[:, 1:].reshape(B_size * T_full, D)
        tf_loss = F.mse_loss(h_tf_pred, h_tf_target)

        # --- Closed-loop rollout from seed k ---
        # Seed (q_k, p_k) is taken from the already-encoded sequence so the
        # phi forward pass is shared with the teacher-forced objective.
        if max_seed_k >= 2:
            k = int(torch.randint(1, min(max_seed_k, T_full - 1) + 1, (1,)).item())
        else:
            k = 1
        T = min(seq_len, T_full - k)

        q, p = q_all[:, k], p_all[:, k]
        q_k_log, p_k_log = q.detach(), p.detach()  # save before graph is freed
        qs_steps, ps_steps = [], []
        for t in range(T):
            q, p = dyn_model.controlled_step(q, p, actions[:, k + t:k + t + 1])
            qs_steps.append(q)
            ps_steps.append(p)
        # The decode never feeds back into the rollout, so all T states go
        # through phi^{-1} in one batched call instead of T tiny ones.
        q_traj = torch.stack(qs_steps, dim=1)  # (B, T, q_dim)
        p_traj = torch.stack(ps_steps, dim=1)
        h_cl_pred = dyn_model.decode(
            q_traj.reshape(B_size * T, q_dim), p_traj.reshape(B_size * T, q_dim)
        )
        cl_loss = F.mse_loss(h_cl_pred, h_all[:, k + 1:k + 1 + T].reshape(B_size * T, D))

        loss = logdet_reg + teacher_force_weight * tf_loss + cl_loss

        if l1_weight > 0:
            l1_loss = sum(param.abs().sum() for param in dyn_model.hamiltonian.parameters())
            loss = loss + l1_weight * l1_loss
            total_hamiltonian_l1 = total_hamiltonian_l1 + l1_loss.detach()

        if structural_reg_weight > 0 and dyn_model.learn_structure:
            # J is a fixed buffer in HamiltonianFlowModel — only R is learned,
            # so penalizing ‖J‖² would just add a constant 2·q_dim to the loss.
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
    }


def _encode_val_h(
    phase1_model: LSTMAutoencoder,
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
    J = dyn_model.get_J().cpu()
    R = dyn_model.get_R().cpu()
    writer.add_scalar("phase2/structure/J_frob", J.pow(2).sum().sqrt().item(), epoch)
    writer.add_scalar("phase2/structure/R_frob", R.pow(2).sum().sqrt().item(), epoch)
    writer.add_histogram("phase2/structure/R_eigenvalues", torch.linalg.eigvalsh(R), epoch)
    for name, mat in (("J", J), ("R", R)):
        fig, ax = plt.subplots(figsize=(4, 4))
        m = mat.numpy()
        vmax = max(abs(m.max()), abs(m.min()), 1e-6)
        im = ax.imshow(m, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"{name} (epoch {epoch + 1})")
        fig.tight_layout()
        writer.add_figure(f"phase2/structure/{name}", fig, epoch)
        plt.close(fig)


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
        sc = axes[i].scatter(true_i, pred_i, c=val_idx, cmap="viridis", s=2, alpha=0.15, linewidths=0)
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
# CLI
# ---------------------------------------------------------------------------


@click.group()
def cli():
    """Offline Pendulum world-model training (two phases)."""
    pass


@cli.command("phase1")
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
# training
@click.option("--epochs", type=int, default=3000, show_default=True)
@click.option("--batch-size", type=int, default=8, show_default=True)
@click.option("--lr", type=float, default=1e-4, show_default=True)
@click.option("--kl-weight", type=float, default=1e-3, show_default=True)
@click.option("--free-bits", type=float, default=0.5, show_default=True)
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
    # band and under-covers the phase space (see _collect_qp_samples).
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

    model = LSTMAutoencoder(
        latent_dim=kwargs["latent_dim"],
        feat_dim=kwargs["feat_dim"],
        pos_ch=kwargs["pos_ch"],
        img_size=kwargs["img_size"],
        control_dim=1,
    ).to(device)
    print(f"Phase 1 model parameters: {sum(p.numel() for p in model.parameters()):,}")

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

    # Precompute and save h_t cache
    print(f"\nPrecomputing h_t cache for {len(episodes)} episodes...")
    cache = precompute_latents(model, episodes, device)
    h_cache_path = run_dir / "h_cache.pt"
    torch.save(cache, h_cache_path)
    print(f"Saved h_cache to {h_cache_path}")
    print(f"\nTo run Phase 2:\n  uv run python experiments/pendulum_offline.py phase2 --phase1-run {run_dir}")

    writer.close()
    print("\nDone. Run: tensorboard --logdir runs")
    os._exit(0)


@cli.command("phase2")
# input — architecture + data params are loaded from the Phase 1 checkpoint
@click.option("--phase1-run", type=str, required=True,
              help="Path to a Phase 1 run directory; loads best.pt (arch/data config + "
                   "autoencoder weights) and h_cache.pt for latents")
@click.option("--phase1-checkpoint", type=str, default=None,
              help="Override the Phase 1 checkpoint (default: {phase1-run}/best.pt, "
                   "falling back to final.pt)")
@click.option("--h-cache", type=str, default=None,
              help="Override the h_t cache path (default: {phase1-run}/h_cache.pt)")
# dynamics model
@click.option("--dt", type=float, default=0.05, show_default=True,
              help="Integration step size (should match the env frame interval)")
@click.option("--separable/--no-separable", default=True, show_default=True,
              help="Use a separable Hamiltonian H = T(p) + V(q); required for --integrator leapfrog")
@click.option("--learn-structure/--no-learn-structure", default=True, show_default=True,
              help="Learn R/B matrices; --no-learn-structure fixes R from the data damping, B=1")
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
@click.option("--h-noise-std", type=float, default=0.0, show_default=True,
              help="Zero-mean Gaussian noise added to h inputs (augmentation; targets "
                   "stay clean), as a multiplier on each h-dim's spread across the cache. "
                   "e.g. 0.05 = 5% of each dim's std. 0 disables.")
@click.option("--max-seed-k", type=int, default=0, show_default=True,
              help="Max closed-loop seed timestep (0 = always start from h_1); "
                   "randomises seed in [1, max-seed-k] to match varying encoder context lengths")
@click.option("--seq-len-start", type=int, default=5, show_default=True,
              help="Initial closed-loop rollout length for curriculum")
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
    h_cache_path = kwargs["h_cache"] or str(run1 / "h_cache.pt")

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

    # Load h_cache
    if not Path(h_cache_path).exists():
        raise click.UsageError(
            f"h_cache not found at {h_cache_path}. "
            "Re-run Phase 1 to regenerate it, or pass --h-cache explicitly."
        )
    print(f"Loading h_cache from {h_cache_path}...")
    cache = torch.load(h_cache_path, weights_only=False)

    # Collect val episodes (only if dreaming logs are enabled)
    train_sample_trajs = []
    val_energy, val_random, val_spin = [], [], []
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

    # Infer latent_dim from cache (authoritative; overrides any stale default)
    latent_dim = cache[0][0].shape[-1]
    print(f"Latent dim from cache: {latent_dim}")

    # Per-dim std of h across the whole cache — used to scale augmentation noise
    # so --h-noise-std acts as a multiplier on each dimension's spread.
    h_noise_scale = None
    if kwargs["h_noise_std"] > 0:
        h_noise_scale = torch.cat([h for h, _ in cache], dim=0).std(dim=0).to(device)
        print(
            f"h noise: std={kwargs['h_noise_std']} × per-dim spread "
            f"(mean dim std={h_noise_scale.mean().item():.4f})"
        )

    latent_dataset = LatentDataset(cache)
    latent_loader = DataLoader(
        latent_dataset, batch_size=kwargs["batch_size"], shuffle=True, num_workers=0,
    )
    print(f"Latent dataset: {len(latent_dataset)} episodes")

    dyn_model = HamiltonianFlowModel(
        latent_dim=latent_dim,
        control_dim=1,
        separable=kwargs["separable"],
        learn_structure=kwargs["learn_structure"],
        dt=kwargs["dt"],
        damping=data_cfg.get("damping", 0.0),
        integrator=kwargs["integrator"],
        quadratic_t=kwargs["quadratic_t"],
    ).to(device)
    print(f"Phase 2 model parameters: {sum(p.numel() for p in dyn_model.parameters()):,}")
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
                "params": [dyn_model.L_param, dyn_model.B],
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

    full_seq_len = cache[0][1].shape[0] - 1  # max steps starting from h_1
    seq_len = kwargs["seq_len_start"]
    ema_loss = None
    ema_cl = None   # separate EMA for closed-loop loss — gates seq_len curriculum
    best_loss = float("inf")
    converge_streak = 0

    print("\n=== Phase 2: dynamics flow training ===")
    for epoch in tqdm(range(kwargs["epochs"]), desc="Phase 2"):
        metrics = _train_epoch_phase2(
            dyn_model=dyn_model,
            loader=latent_loader,
            optimizer=optimizer,
            grad_clip=kwargs["grad_clip"],
            device=device,
            seq_len=seq_len,
            logdet_weight=kwargs["logdet_weight"],
            l1_weight=kwargs["l1_weight"],
            max_seed_k=kwargs["max_seed_k"],
            teacher_force_weight=kwargs["teacher_force_weight"],
            structural_reg_weight=kwargs["structural_reg_weight"],
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

    writer.close()
    print("\nDone. Run: tensorboard --logdir runs")
    os._exit(0)


if __name__ == "__main__":
    cli()
