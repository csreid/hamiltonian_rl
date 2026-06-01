"""Offline Pendulum world-model training — two-phase regimen.

Phase 1 (--phase 1):
    Train the HGN autoencoder (encoder + f_psi + decoder) for reconstruction.
    Loss = MSE(decoder(f_psi(z)[:q_dim]), frame) + kl_weight * KL.
    After training, precomputes and saves h_t = encoder_mu(frame_t) for every
    frame of every training episode to h_cache.pt in the run directory.

Phase 2 (--phase 2):
    Load precomputed h_t cache. Train a new HamiltonianFlowModel (Phi + H + J/R/B)
    that maps h_t → (q, p) such that Hamiltonian dynamics hold:

        L = MSE(HamiltonianStep(Phi(h_t), u_t),  Phi(h_{t+1}).detach())

    No images needed during training — very fast.

Inference (dreaming) after both phases:
    h_0  = encoder(frame_0)
    q, p = Phi(h_0)                                  [Phase 2 forward]
    q_k, p_k = HamiltonianRollout(q, p, actions)     [Phase 2 dynamics]
    h_k  = Phi^{-1}(q_k, p_k)                       [Phase 2 inverse]
    q_dec= f_psi(h_k)[:q_dim]                        [Phase 1 f_psi]
    frame= decoder(q_dec)                            [Phase 1 decoder]
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
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from checkpoint_common import make_run_dir, save_checkpoint
from data.pendulum import (
    PendulumDataset,
    collect_data,
    collect_random_trajectories,
    collect_spin_trajectories,
    collect_val_trajectories,
    _G,
)
from phgn_lstm import ControlledDHGN_LSTM, HamiltonianFlowModel


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _log_latent_variance(
    qs: torch.Tensor,
    ps: torch.Tensor,
) -> tuple[float, float]:
    q_dim = qs.shape[-1]
    q_var = qs.detach().reshape(-1, q_dim).var(dim=0).mean().item()
    p_var = ps.detach().reshape(-1, q_dim).var(dim=0).mean().item()
    return q_var, p_var


def _true_hamiltonian(states: torch.Tensor) -> np.ndarray:
    cos_theta = states[:, 0].numpy()
    theta_dot = states[:, 2].numpy()
    return 0.5 * theta_dot**2 + _G * (1.0 + cos_theta)


def _annotate_frame(frame: torch.Tensor, text: str) -> torch.Tensor:
    img = Image.fromarray((frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    draw.text((2, 2), text, fill=(255, 255, 0))
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0


# ---------------------------------------------------------------------------
# Phase 1: autoencoder training
# ---------------------------------------------------------------------------


def _train_epoch_phase1(
    model: ControlledDHGN_LSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    kl_weight: float,
    free_bits: float,
    grad_clip: float,
    device: torch.device,
    temporal_reg_weight: float = 0.0,
    temporal_scale: float = 0.01,
    max_context_len: int = 0,
) -> dict[str, float]:
    """Reconstruction-only epoch: encoder + f_psi + decoder, no Hamiltonian.

    Two prediction targets from the causal (forward-only) LSTM:
      - h_t → current frame   (reconstruction signal)
      - h_t → next frame      (predictive signal; h_t has seen only 0..t)
    """
    model.train()
    total_recon = total_recon_next = total_kl = total_temporal = total_loss = 0.0

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
            total_temporal += temporal_reg.item()

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_recon += recon.item()
        total_recon_next += recon_next.item()
        total_kl += kl.item()
        total_loss += loss.item()

    n = len(loader)
    return {
        "phase1/loss": total_loss / n,
        "phase1/recon": total_recon / n,
        "phase1/recon_next": total_recon_next / n,
        "phase1/kl": total_kl / n,
        "phase1/temporal_reg": total_temporal / n,
    }


@torch.no_grad()
def _eval_loss_phase1(
    model: ControlledDHGN_LSTM,
    val_trajs: list,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    q_dim = model.latent_dim // 2
    total_perframe = 0.0
    for frames, actions, _ in val_trajs:
        frames = frames.unsqueeze(0).to(device)
        mu_all, _ = model.encoder.forward_all(frames)
        s_all = model.f_psi(mu_all.squeeze(0))
        z_dec = s_all[:, :q_dim]
        pred = model.decoder(z_dec)
        total_perframe += F.mse_loss(pred, frames.squeeze(0)).item()
    return {"phase1/val_recon": total_perframe / len(val_trajs)}


@torch.no_grad()
def _log_reconstruction_lstm_video(
    model: ControlledDHGN_LSTM,
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
    model: ControlledDHGN_LSTM,
    val_trajs: list,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    tag: str = "val/latent_regression",
) -> None:
    model.eval()
    all_s, all_st = [], []
    for frames, actions, states in val_trajs:
        ctx = frames.unsqueeze(0).to(device)
        mu_all, _ = model.encoder.forward_all(ctx)
        s_all = model.f_psi(mu_all.squeeze(0)).cpu()
        all_s.append(s_all)
        all_st.append(states.float())

    s_cat = torch.cat(all_s, dim=0)
    st_cat = torch.cat(all_st, dim=0)
    mid = len(s_cat) // 2
    A = torch.linalg.lstsq(s_cat[:mid], st_cat[:mid]).solution
    st_pred = (s_cat[mid:] @ A).numpy()
    st_true = st_cat[mid:].numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, name in enumerate(["cos(θ)", "sin(θ)", "θ̇ (rad/s)"]):
        true_i, pred_i = st_true[:, i], st_pred[:, i]
        axes[i].scatter(true_i, pred_i, s=2, alpha=0.3)
        lo, hi = min(true_i.min(), pred_i.min()), max(true_i.max(), pred_i.max())
        axes[i].plot([lo, hi], [lo, hi], "r--", linewidth=0.8)
        axes[i].set_xlabel(f"True {name}")
        axes[i].set_ylabel(f"Predicted {name}")
        ss_res = ((true_i - pred_i) ** 2).sum()
        ss_tot = ((true_i - true_i.mean()) ** 2).sum()
        axes[i].set_title(f"{name}  R²={1 - ss_res / (ss_tot + 1e-8):.3f}")
    fig.suptitle(f"Latent → state regression, held-out half (epoch {epoch + 1})")
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
    model: ControlledDHGN_LSTM,
    episodes: list,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Run encoder over all training episodes and cache h_t = mu_all."""
    model.eval()
    cache = []
    with torch.no_grad():
        for frames, actions, _ in tqdm(episodes, desc="Precomputing latents"):
            mu_all, _ = model.encoder.forward_all(frames.unsqueeze(0).to(device))
            cache.append((mu_all.squeeze(0).cpu(), actions))
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
) -> dict[str, float]:
    """Dynamics-only epoch: closed-loop seq-to-seq rollout from h_1.

    Encodes h_1 (two frames seen by LSTM, carrying momentum info) to (q, p)
    once, then rolls Hamiltonian dynamics forward for seq_len steps without
    re-encoding, comparing phi^{-1}(q_k, p_k) against real h_{k+2}.

    Teacher-forcing (re-encoding at every step) was the old approach; it
    trained only 1-step predictions from real h values, so phi^{-1} drifted
    out of the h-manifold during open-loop inference.

    A log-det regulariser on the single phi(h_1) call keeps the flow
    near-volume-preserving.
    """
    dyn_model.train()
    total_dynamics = total_logdet_reg = total_q_var = total_p_var = total_hamiltonian_l1 = total_grad_H_norm = 0.0

    for h_all, actions in loader:
        h_all = h_all.to(device)      # (B, T+1, latent_dim)
        actions = actions.to(device)  # (B, T)
        T_full = actions.shape[1]

        # Sample a random start timestep so phi sees h values encoded from
        # varying context lengths — matching what the encoder produces at inference.
        # max_seed_k is independent of seq_len; T then adjusts to stay in bounds.
        if max_seed_k >= 2:
            k = int(torch.randint(1, min(max_seed_k, T_full - 1) + 1, (1,)).item())
        else:
            k = 1
        T = min(seq_len, T_full - k)

        q, p, log_det = dyn_model.encode_with_logdet(h_all[:, k])
        loss = logdet_weight * log_det.pow(2).mean()
        qs_log, ps_log = [q.detach()], [p.detach()]

        for t in range(T):
            q, p = dyn_model.controlled_step(q, p, actions[:, k + t:k + t + 1])
            h_pred = dyn_model.decode(q, p)
            loss = loss + F.mse_loss(h_pred, h_all[:, k + 1 + t])
            qs_log.append(q.detach())
            ps_log.append(p.detach())
        loss = loss / T

        if l1_weight > 0:
            l1_loss = sum(p.abs().sum() for p in dyn_model.hamiltonian.parameters())
            loss = loss + l1_weight * l1_loss
            total_hamiltonian_l1 += l1_loss.item()

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(dyn_model.parameters(), grad_clip)
        optimizer.step()

        with torch.no_grad():
            logdet_reg = dyn_model.phi.forward_with_logdet(h_all[:, k])[1].pow(2).mean().item()
            total_logdet_reg += logdet_reg
        with torch.enable_grad():
            q_dim = dyn_model.latent_dim // 2
            q_eval, p_eval = dyn_model.encode(h_all[:, k].detach())
            z_eval = torch.cat([q_eval, p_eval], dim=-1).requires_grad_(True)
            H_eval = dyn_model.hamiltonian(z_eval[:, :q_dim], z_eval[:, q_dim:]).sum()
            grad_eval = torch.autograd.grad(H_eval, z_eval)[0]
            total_grad_H_norm += grad_eval.norm(dim=-1).mean().item()
        total_dynamics += loss.item()
        with torch.no_grad():
            q_var, p_var = _log_latent_variance(
                torch.stack(qs_log, dim=1), torch.stack(ps_log, dim=1)
            )
            total_q_var += q_var
            total_p_var += p_var

    n = len(loader)
    return {
        "phase2/dynamics": total_dynamics / n,
        "phase2/logdet_reg": total_logdet_reg / n,
        "phase2/q_var": total_q_var / n,
        "phase2/p_var": total_p_var / n,
        "phase2/hamiltonian_l1": total_hamiltonian_l1 / n,
        "phase2/grad_H_norm": total_grad_H_norm / n,
    }


@torch.no_grad()
def _log_structural_matrices_phase2(
    dyn_model: HamiltonianFlowModel,
    writer: SummaryWriter,
    epoch: int,
) -> None:
    J = dyn_model.get_J().cpu()
    R = dyn_model.get_R().cpu()
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


# ---------------------------------------------------------------------------
# Phase 2 dreaming video
# ---------------------------------------------------------------------------


@torch.no_grad()
def _log_dreamed_video_phase2(
    phase1_model: ControlledDHGN_LSTM,
    dyn_model: HamiltonianFlowModel,
    val_traj: tuple,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    seq_len: int,
    tag: str = "val/dreamed_phase2",
    fps: int = 10,
) -> None:
    """Log a dreamed rollout alongside ground truth.

    Context: 5 frames fed through the Phase 1 bidirectional LSTM encoder → h.
    Rollout: seq_len Hamiltonian steps in phase space, decoded back to pixels
             via phi^{-1} → f_psi → decoder.
    """
    phase1_model.eval()
    dyn_model.eval()
    frames, actions, _ = val_traj
    q_dim = phase1_model.latent_dim // 2
    context_frames = 5

    # Seed: encode context frames with Phase 1 bidirectional LSTM
    ctx = frames[:context_frames].unsqueeze(0).to(device)                  # (1, context_frames, C, H, W)
    ctx_actions = actions[:context_frames - 1].unsqueeze(0).to(device)     # (1, context_frames-1)
    mu_ctx, _ = phase1_model.encoder.forward_all(ctx)                       # (1, context_frames, latent_dim)
    h = mu_ctx[:, -1]                                                       # (1, latent_dim)

    # Map to phase space via Phase 2 phi
    q, p = dyn_model.encode(h)  # (1, q_dim) each

    # Roll out Hamiltonian dynamics, decode each step
    n_steps = min(seq_len, len(actions) - (context_frames - 1))
    dreamed_frames = []
    for k in range(n_steps):
        u = actions[context_frames - 1 + k].view(1, 1).to(device)  # (1, 1)
        q, p = dyn_model.controlled_step(q, p, u)
        h_pred = dyn_model.decode(q, p)                         # (1, latent_dim)
        s_pred = phase1_model.f_psi(h_pred)                     # (1, latent_dim)
        frame_pred = phase1_model.decoder(s_pred[:, :q_dim])    # (1, C, H, W)
        dreamed_frames.append(frame_pred.squeeze(0).cpu())

    if not dreamed_frames:
        return

    dreamed = torch.stack(dreamed_frames)                # (n_steps, C, H, W)
    gt = frames[context_frames:context_frames + n_steps] # (n_steps, C, H, W)

    gt_ann = torch.stack([
        _annotate_frame(gt[i], f"gt {context_frames + i}") for i in range(len(gt))
    ])
    dream_ann = torch.stack([
        _annotate_frame(dreamed[i].clamp(0, 1), f"dr {context_frames + i}") for i in range(len(dreamed))
    ])
    side_by_side = torch.cat([gt_ann, dream_ann], dim=3).unsqueeze(0)
    writer.add_video(tag, (side_by_side.clamp(0, 1) * 255).byte(), epoch, fps=fps)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
# phase control
@click.option("--phase", type=click.Choice(["1", "2"]), default="1", show_default=True,
              help="Training phase: 1=autoencoder, 2=dynamics flow")
@click.option("--phase1-run", type=str, default=None,
              help="Path to a Phase 1 run directory (e.g. models/pendulum_offline_phase1/2026-05-11_10-00-00); "
                   "loads best.pt as the checkpoint and h_cache.pt from the same directory")
@click.option("--phase1-checkpoint", type=str, default=None,
              help="Path to Phase 1 .pt checkpoint; overrides --phase1-run if both given")
@click.option("--h-cache", type=str, default=None,
              help="Path to precomputed h_t cache .pt; overrides --phase1-run if both given")
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
# model
@click.option("--pos-ch", type=int, default=8, show_default=True)
@click.option("--feat-dim", type=int, default=256, show_default=True)
@click.option("--latent-dim", type=int, default=32, show_default=True,
              help="Encoder output / phase-space dimension (must match Phase 1 when running Phase 2)")
@click.option("--dt", type=float, default=0.05, show_default=True)
@click.option("--no-separable", "separable", default=True, flag_value=False)
@click.option("--learn-structure/--no-learn-structure", default=True, show_default=True,
              help="Learn J/R/B matrices; --no-learn-structure fixes J to canonical symplectic, R=0, B=1")
# training
@click.option("--epochs", type=int, default=3000, show_default=True)
@click.option("--batch-size", type=int, default=8, show_default=True,
              help="Episodes per batch")
@click.option("--lr", type=float, default=1e-4, show_default=True)
@click.option("--structural-lr", type=float, default=1e-2, show_default=True)
@click.option("--kl-weight", type=float, default=1e-3, show_default=True)
@click.option("--free-bits", type=float, default=0.5, show_default=True)
@click.option("--grad-clip", type=float, default=1.0, show_default=True)
@click.option("--logdet-weight", type=float, default=1e-3, show_default=True,
              help="Weight on log|det J_Phi|^2 regulariser (Phase 2 only); keeps flow near-volume-preserving")
@click.option("--l1-weight", type=float, default=0.0, show_default=True,
              help="L1 penalty on Hamiltonian network weights (Phase 2 only); encourages simpler dynamics")
@click.option("--max-context-len", type=int, default=0, show_default=True,
              help="Max frames fed to LSTM per Phase 1 batch step (0 = full sequence). "
                   "Context length is sampled uniformly from [2, max-context-len] each step.")
@click.option("--temporal-reg-weight", type=float, default=0.1, show_default=True,
              help="Phase 1 temporal metric regulariser weight (0 to disable)")
@click.option("--temporal-scale", type=float, default=0.01, show_default=True,
              help="Expected h-space distance per timestep; pairs closer than scale*dt are penalised")
@click.option("--ema-alpha", type=float, default=0.99, show_default=True,
              help="EMA smoothing for loss-gated curriculum")
@click.option("--convergence-patience", type=int, default=0, show_default=True,
              help="Epochs of stable EMA before stopping; 0 disables convergence stopping")
@click.option("--convergence-threshold", type=float, default=1e-4, show_default=True,
              help="Relative per-epoch EMA change below which an epoch counts as stable")
@click.option("--seq-len-start", type=int, default=5, show_default=True,
              help="Initial rollout length for curriculum")
@click.option("--seq-len-advance-threshold", type=float, default=0.005, show_default=True,
              help="EMA loss below which rollout length advances by 1")
# logging
@click.option("--log-every", type=int, default=5, show_default=True)
@click.option("--val-every", type=int, default=10, show_default=True,
              help="Epochs between validation plots (0 to disable)")
@click.option("--n-val-episodes", type=int, default=-1, show_default=True,
              help="Val episodes per type (-1 = n_episodes // 2)")
@click.option("--val-max-steps", type=int, default=0, show_default=True,
              help="Steps per val episode (0 = 2x --max-steps)")
@click.option("--checkpoint-every", type=int, default=10, show_default=True)
def main(**kwargs):
    assert kwargs["img_size"] % 8 == 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    phase = kwargs["phase"]

    if kwargs.get("phase1_run") and phase == "2":
        from pathlib import Path as _Path
        _run = _Path(kwargs["phase1_run"])
        if kwargs.get("phase1_checkpoint") is None:
            kwargs["phase1_checkpoint"] = str(_run / "best.pt")
        if kwargs.get("h_cache") is None:
            kwargs["h_cache"] = str(_run / "h_cache.pt")

    writer = SummaryWriter(comment=f"_pendulum_offline_phase{phase}")
    run_dir = make_run_dir(f"pendulum_offline_phase{phase}")

    n_val_episodes = kwargs["n_val_episodes"]
    if n_val_episodes < 0:
        n_val_episodes = kwargs["n_episodes"] // 2
    n_val = n_val_episodes if kwargs["val_every"] > 0 else 0
    val_steps = kwargs["val_max_steps"] or kwargs["max_steps"] * 2

    print(f"\nCollecting {kwargs['n_episodes']} train episodes...")
    episodes = collect_data(
        n_episodes=kwargs["n_episodes"],
        img_size=kwargs["img_size"],
        epsilon=kwargs["epsilon"],
        energy_k=kwargs["energy_k"],
        max_steps=kwargs["max_steps"],
        damping=kwargs["damping"],
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

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    if phase == "1":
        dataset = PendulumDataset(episodes)
        loader = DataLoader(
            dataset, batch_size=kwargs["batch_size"], shuffle=True,
            num_workers=0, pin_memory=device.type == "cuda",
        )
        print(f"Dataset: {len(dataset)} episodes")

        model = ControlledDHGN_LSTM(
            pos_ch=kwargs["pos_ch"],
            img_ch=3,
            dt=kwargs["dt"],
            feat_dim=kwargs["feat_dim"],
            latent_dim=kwargs["latent_dim"],
            img_size=kwargs["img_size"],
            control_dim=1,
            separable=kwargs["separable"],
            learn_structure=kwargs["learn_structure"],
            damping=kwargs["damping"],
        ).to(device)
        print(f"Phase 1 model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Optimize only the reconstruction components — Hamiltonian/structure unused in Phase 1
        optimizer = torch.optim.Adam(
            list(model.encoder.parameters())
            + list(model.f_psi.parameters())
            + list(model.decoder.parameters())
            + list(model.next_frame_decoder.parameters()),
            lr=kwargs["lr"],
        )

        hparams = {k: v for k, v in kwargs.items()}
        best_loss = float("inf")
        full_seq_len = episodes[0][1].shape[0]
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
                for val_trajs, label in (
                    (val_energy, "energy_pump"),
                    (val_random, "random"),
                    (val_spin, "spin"),
                ):
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
                if val_energy:
                    _log_latent_scatter_phase1(
                        model=model, val_trajs=val_energy,
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
                save_checkpoint(run_dir, epoch, model, hparams, metrics, stem="best")
                best_loss = metrics["phase1/loss"]

        # Always save final checkpoint
        save_checkpoint(run_dir, kwargs["epochs"] - 1, model, hparams, metrics, stem="final")

        # Precompute and save h_t cache
        print(f"\nPrecomputing h_t cache for {len(episodes)} episodes...")
        cache = precompute_latents(model, episodes, device)
        h_cache_path = run_dir / "h_cache.pt"
        torch.save(cache, h_cache_path)
        print(f"Saved h_cache to {h_cache_path}")
        print(f"\nTo run Phase 2:\n  uv run python experiments/pendulum_offline.py --phase 2 --phase1-run {run_dir}")

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    else:
        h_cache_path = kwargs["h_cache"]
        phase1_ckpt = kwargs["phase1_checkpoint"]

        # Load Phase 1 model whenever a checkpoint is given — kept alive for
        # video logging throughout Phase 2 training.
        phase1_model = None
        if phase1_ckpt is not None:
            print(f"Loading Phase 1 model from {phase1_ckpt}...")
            phase1_model = ControlledDHGN_LSTM(
                pos_ch=kwargs["pos_ch"],
                img_ch=3,
                dt=kwargs["dt"],
                feat_dim=kwargs["feat_dim"],
                latent_dim=kwargs["latent_dim"],
                img_size=kwargs["img_size"],
                control_dim=1,
                separable=kwargs["separable"],
                learn_structure=kwargs["learn_structure"],
                damping=kwargs["damping"],
            ).to(device)
            phase1_model.load_state_dict(
                torch.load(phase1_ckpt, map_location=device, weights_only=True)
            )
            phase1_model.eval()

        if h_cache_path is not None:
            print(f"Loading h_cache from {h_cache_path}...")
            cache = torch.load(h_cache_path, weights_only=False)
        elif phase1_model is not None:
            print("Precomputing latents from Phase 1 model...")
            cache = precompute_latents(phase1_model, episodes, device)
            h_cache_save = run_dir / "h_cache.pt"
            torch.save(cache, h_cache_save)
            print(f"Saved h_cache to {h_cache_save}")
        else:
            raise click.UsageError("--phase 2 requires either --h-cache or --phase1-checkpoint")

        if phase1_model is None:
            print("Note: no --phase1-checkpoint provided; dreaming video logs will be skipped.")

        # Infer latent_dim from cache
        latent_dim = cache[0][0].shape[-1]
        print(f"Latent dim from cache: {latent_dim}")

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
            damping=kwargs["damping"],
        ).to(device)
        print(f"Phase 2 model parameters: {sum(p.numel() for p in dyn_model.parameters()):,}")

        hparams = {k: v for k, v in kwargs.items()}
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
                    "params": [dyn_model.A, dyn_model.L_param, dyn_model.B],
                    "lr": kwargs["structural_lr"],
                },
            ])
        else:
            optimizer = torch.optim.Adam(dyn_model.parameters(), lr=kwargs["lr"])

        full_seq_len = cache[0][1].shape[0] - 1  # -1: start from h_1, predict up to h_{T}
        seq_len = kwargs["seq_len_start"]
        ema_loss = None
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
                max_seed_k=kwargs["max_context_len"],
            )

            alpha = kwargs["ema_alpha"]
            prev_ema = ema_loss
            ema_loss = (
                metrics["phase2/dynamics"]
                if ema_loss is None
                else alpha * ema_loss + (1.0 - alpha) * metrics["phase2/dynamics"]
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

            if ema_loss < kwargs["seq_len_advance_threshold"] and seq_len < full_seq_len:
                seq_len += 1

            if (epoch + 1) % kwargs["log_every"] == 0:
                for k, v in metrics.items():
                    writer.add_scalar(k, v, epoch)
                writer.add_scalar("phase2/seq_len", seq_len, epoch)
                writer.add_scalar("phase2/ema_loss", ema_loss, epoch)
                if kwargs["learn_structure"]:
                    writer.add_scalar(
                        "phase2/structure/B_norm",
                        dyn_model.get_B().norm().item(),
                        epoch,
                    )
                tqdm.write(
                    f"  epoch {epoch + 1:4d}"
                    f"  seq_len={seq_len:3d}"
                    f"  dynamics={metrics['phase2/dynamics']:.4f}"
                    f"  ema={ema_loss:.4f}"
                    f"  logdet={metrics['phase2/logdet_reg']:.4f}"
                    f"  q_var={metrics['phase2/q_var']:.4f}"
                    f"  p_var={metrics['phase2/p_var']:.4f}"
                )

            if kwargs["val_every"] > 0 and (epoch + 1) % kwargs["val_every"] == 0:
                _log_structural_matrices_phase2(dyn_model=dyn_model, writer=writer, epoch=epoch)
                if phase1_model is not None:
                    for val_trajs, label in (
                        (val_energy, "energy_pump"),
                        (val_random, "random"),
                        (val_spin, "spin"),
                    ):
                        if val_trajs:
                            _log_dreamed_video_phase2(
                                phase1_model=phase1_model,
                                dyn_model=dyn_model,
                                val_traj=val_trajs[0],
                                device=device,
                                writer=writer,
                                epoch=epoch,
                                seq_len=seq_len,
                                tag=f"val/dreamed_phase2/{label}",
                            )

            if (
                kwargs["checkpoint_every"] > 0
                and (epoch + 1) % kwargs["checkpoint_every"] == 0
                and metrics["phase2/dynamics"] < best_loss
            ):
                save_checkpoint(run_dir, epoch, dyn_model, hparams, metrics, stem="best")
                best_loss = metrics["phase2/dynamics"]

        save_checkpoint(run_dir, kwargs["epochs"] - 1, dyn_model, hparams, metrics, stem="final")

    writer.close()
    print("\nDone. Run: tensorboard --logdir runs")
    os._exit(0)


if __name__ == "__main__":
    main()
