"""Offline Pendulum world-model training — two-phase regimen.

Phase 1 (phase1 subcommand):
    Train the HGN autoencoder (encoder + f_psi + decoder) for reconstruction.
    Loss = MSE(decoder(f_psi(z)[:q_dim]), frame) + kl_weight * KL.
    After training, precomputes and saves h_t = encoder_mu(frame_t) for every
    frame of every training episode to h_cache.pt in the run directory.

Phase 2 (phase2 subcommand):
    Load precomputed h_t cache. Train a new HamiltonianFlowModel (Phi + H + J/R/B)
    that maps h_t → (q, p) such that Hamiltonian dynamics hold:

        L_tf  = MSE(phi^{-1}(RK4(phi(h_t), u_t)),  h_{t+1})       [teacher-forced, all t]
        L_cl  = MSE(phi^{-1}(RK4^k(phi(h_seed), u)), h_{seed+k})  [closed-loop, seq_len steps]

    Architecture params (latent_dim, img_size, etc.) are loaded automatically
    from the Phase 1 checkpoint YAML — no need to re-specify them.

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
import yaml
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
        logdet_metric = log_det_all.pow(2).mean().item()          # save before backward

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
        qs_log, ps_log = [q_k_log], [p_k_log]
        cl_loss = torch.zeros((), device=device)
        for t in range(T):
            q, p = dyn_model.controlled_step(q, p, actions[:, k + t:k + t + 1])
            h_pred = dyn_model.decode(q, p)
            cl_loss = cl_loss + F.mse_loss(h_pred, h_all[:, k + 1 + t])
            qs_log.append(q.detach())
            ps_log.append(p.detach())
        cl_loss = cl_loss / T

        loss = logdet_reg + teacher_force_weight * tf_loss + cl_loss

        if l1_weight > 0:
            l1_loss = sum(param.abs().sum() for param in dyn_model.hamiltonian.parameters())
            loss = loss + l1_weight * l1_loss
            total_hamiltonian_l1 += l1_loss.item()

        if structural_reg_weight > 0 and dyn_model.learn_structure:
            struct_reg = dyn_model.get_J().pow(2).sum() + dyn_model.get_R().pow(2).sum()
            loss = loss + structural_reg_weight * struct_reg
            total_struct_reg += struct_reg.item()

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
            total_grad_H_norm += grad_eval.norm(dim=-1).mean().item()

        total_logdet_reg += logdet_metric
        total_tf += tf_loss.item()
        total_cl += cl_loss.item()
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
        "phase2/tf_loss": total_tf / n,
        "phase2/cl_loss": total_cl / n,
        "phase2/logdet_reg": total_logdet_reg / n,
        "phase2/q_var": total_q_var / n,
        "phase2/p_var": total_p_var / n,
        "phase2/hamiltonian_l1": total_hamiltonian_l1 / n,
        "phase2/grad_H_norm": total_grad_H_norm / n,
        "phase2/struct_reg": total_struct_reg / n,
    }


@torch.no_grad()
def _eval_loss_phase2(
    phase1_model: ControlledDHGN_LSTM,
    dyn_model: HamiltonianFlowModel,
    val_trajs: list,
    device: torch.device,
    seq_len: int,
) -> dict[str, float]:
    phase1_model.eval()
    dyn_model.eval()
    total_teacher_forced = total_closed_loop = 0.0
    q_dim = dyn_model.latent_dim // 2

    for frames, actions, _ in val_trajs:
        frames_b = frames.unsqueeze(0).to(device)    # (1, T+1, C, H, W)
        actions_b = actions.unsqueeze(0).to(device)  # (1, T)
        mu_all, _ = phase1_model.encoder.forward_all(frames_b)
        h_all = mu_all  # (1, T+1, latent_dim)
        B, T_seq, D = h_all.shape
        T_full = actions_b.shape[1]

        h_flat = h_all.reshape(B * T_seq, D)
        q_flat, p_flat = dyn_model.encode(h_flat)
        q_all = q_flat.reshape(B, T_seq, q_dim)
        p_all = p_flat.reshape(B, T_seq, q_dim)

        q_teacher = q_all[:, :T_full].reshape(B * T_full, q_dim)
        p_teacher = p_all[:, :T_full].reshape(B * T_full, q_dim)
        actions_teacher = actions_b.float().reshape(B * T_full, 1)
        q_next, p_next = dyn_model.controlled_step(q_teacher, p_teacher, actions_teacher)
        h_teacher_pred = dyn_model.decode(q_next, p_next)
        h_teacher_target = h_all[:, 1:].reshape(B * T_full, D)
        total_teacher_forced += F.mse_loss(h_teacher_pred, h_teacher_target).item()

        n_rollout_steps = min(seq_len, T_full - 1)
        q, p = q_all[:, 1], p_all[:, 1]
        closed_loop_sum = 0.0
        for t in range(n_rollout_steps):
            u = actions_b[:, 1 + t: 2 + t].float()
            q, p = dyn_model.controlled_step(q, p, u)
            h_pred = dyn_model.decode(q, p)
            closed_loop_sum += F.mse_loss(h_pred, h_all[:, 2 + t]).item()
        total_closed_loop += closed_loop_sum / n_rollout_steps if n_rollout_steps > 0 else 0.0

    n = len(val_trajs)
    return {
        "phase2/val_tf_loss": total_teacher_forced / n,
        "phase2/val_cl_loss": total_closed_loop / n,
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
    phase1_model: ControlledDHGN_LSTM,
    dyn_model: HamiltonianFlowModel,
    val_traj: tuple,
    device: torch.device,
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
    phase1_model.eval()
    dyn_model.eval()
    frames, actions, _ = val_traj
    q_dim = phase1_model.latent_dim // 2

    # Seed: encode context frames with Phase 1 LSTM encoder
    ctx = frames[:context_frames].unsqueeze(0).to(device)   # (1, context_frames, C, H, W)
    mu_ctx, _ = phase1_model.encoder.forward_all(ctx)        # (1, context_frames, latent_dim)
    h = mu_ctx[:, -1]                                        # (1, latent_dim)

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
# CLI helpers
# ---------------------------------------------------------------------------


def _load_phase1_hparams(run_dir: Path) -> dict:
    """Load hparams from Phase 1 YAML (tries best.yaml, falls back to final.yaml)."""
    for stem in ("best", "final"):
        p = run_dir / f"{stem}.yaml"
        if p.exists():
            return yaml.safe_load(p.read_text())["hparams"]
    raise click.UsageError(
        f"No checkpoint YAML found in {run_dir}. "
        "Expected best.yaml or final.yaml from a completed Phase 1 run."
    )


def _make_phase1_model(hp: dict, device: torch.device) -> ControlledDHGN_LSTM:
    return ControlledDHGN_LSTM(
        pos_ch=hp["pos_ch"],
        img_ch=3,
        dt=hp["dt"],
        feat_dim=hp["feat_dim"],
        latent_dim=hp["latent_dim"],
        img_size=hp["img_size"],
        control_dim=1,
        separable=hp["separable"],
        learn_structure=hp["learn_structure"],
        damping=hp["damping"],
    ).to(device)


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
@click.option("--dt", type=float, default=0.05, show_default=True)
@click.option("--no-separable", "separable", default=True, flag_value=False)
@click.option("--learn-structure/--no-learn-structure", default=True, show_default=True,
              help="Learn J/R/B matrices; --no-learn-structure fixes J to canonical symplectic, R=0, B=1")
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
    """Phase 1: train HGN autoencoder (encoder + f_psi + decoder)."""
    assert kwargs["img_size"] % 8 == 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    writer = SummaryWriter(comment="_pendulum_offline_phase1")
    run_dir = make_run_dir("pendulum_offline_phase1")

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

    # Only train reconstruction components — Hamiltonian/structure unused in Phase 1
    optimizer = torch.optim.Adam(
        list(model.encoder.parameters())
        + list(model.f_psi.parameters())
        + list(model.decoder.parameters())
        + list(model.next_frame_decoder.parameters()),
        lr=kwargs["lr"],
    )

    hparams = {k: v for k, v in kwargs.items()}
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
    save_checkpoint(run_dir, epoch, model, hparams, metrics, stem="final")

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
# input — architecture + data params are loaded from the Phase 1 YAML
@click.option("--phase1-run", type=str, required=True,
              help="Path to a Phase 1 run directory; loads best.yaml for arch/data params, "
                   "best.pt for the model, and h_cache.pt for latents")
@click.option("--phase1-checkpoint", type=str, default=None,
              help="Override the Phase 1 model checkpoint (default: {phase1-run}/best.pt)")
@click.option("--h-cache", type=str, default=None,
              help="Override the h_t cache path (default: {phase1-run}/h_cache.pt)")
# training
@click.option("--epochs", type=int, default=3000, show_default=True)
@click.option("--batch-size", type=int, default=8, show_default=True)
@click.option("--lr", type=float, default=1e-4, show_default=True)
@click.option("--structural-lr", type=float, default=1e-2, show_default=True,
              help="Learning rate for J/R/B structure matrices")
@click.option("--grad-clip", type=float, default=1.0, show_default=True)
@click.option("--logdet-weight", type=float, default=1e-3, show_default=True,
              help="Weight on log|det J_Phi|^2 regulariser; keeps flow near-volume-preserving")
@click.option("--l1-weight", type=float, default=0.0, show_default=True,
              help="L1 penalty on Hamiltonian network weights; encourages simpler dynamics")
@click.option("--structural-reg-weight", type=float, default=0.0, show_default=True,
              help="Frobenius norm penalty on J and R (Phase 2, learn-structure only); prevents structural matrices from growing unbounded")
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
    print(f"Device: {device}")

    run1 = Path(kwargs["phase1_run"])
    hp1 = _load_phase1_hparams(run1)
    print("Loaded Phase 1 hparams: " + ", ".join(
        f"{k}={hp1[k]}" for k in
        ("latent_dim", "img_size", "feat_dim", "pos_ch", "dt", "separable", "learn_structure", "damping")
    ))

    phase1_ckpt = kwargs["phase1_checkpoint"] or str(run1 / "best.pt")
    h_cache_path = kwargs["h_cache"] or str(run1 / "h_cache.pt")

    writer = SummaryWriter(comment="_pendulum_offline_phase2")
    run_dir = make_run_dir("pendulum_offline_phase2")

    # Load Phase 1 model (kept alive for dreaming video logs)
    phase1_model = None
    if Path(phase1_ckpt).exists():
        print(f"Loading Phase 1 model from {phase1_ckpt}...")
        phase1_model = _make_phase1_model(hp1, device)
        phase1_model.load_state_dict(
            torch.load(phase1_ckpt, map_location=device, weights_only=True)
        )
        phase1_model.eval()
    else:
        print(f"Warning: {phase1_ckpt} not found — dreaming video logs will be skipped.")

    # Load h_cache
    if Path(h_cache_path).exists():
        print(f"Loading h_cache from {h_cache_path}...")
        cache = torch.load(h_cache_path, weights_only=False)
    elif phase1_model is not None:
        raise click.UsageError(
            f"h_cache not found at {h_cache_path}. "
            "Re-run Phase 1 to regenerate it, or pass --h-cache explicitly."
        )
    else:
        raise click.UsageError(
            "--phase1-run must point to a directory containing h_cache.pt and best.pt."
        )

    # Collect val episodes (only if dreaming logs are enabled and Phase 1 model is available)
    train_sample_trajs = []
    val_energy, val_random, val_spin = [], [], []
    if kwargs["val_every"] > 0 and phase1_model is not None:
        n_val = kwargs["n_val_episodes"]
        if n_val < 0:
            n_val = hp1.get("n_episodes", 200) // 2
        val_steps = kwargs["val_max_steps"] or hp1.get("max_steps", 200) * 2
        print(f"Collecting {n_val} val episodes per type ({val_steps} steps each)...")
        val_energy = collect_val_trajectories(
            n_episodes=n_val, img_size=hp1["img_size"],
            max_steps=val_steps, energy_k=hp1.get("energy_k", 1.0),
            damping=hp1.get("damping", 0.0),
        )
        val_random = collect_random_trajectories(
            n_episodes=n_val, img_size=hp1["img_size"],
            max_steps=val_steps, damping=hp1.get("damping", 0.0),
        )
        val_spin = collect_spin_trajectories(
            n_episodes=n_val, img_size=hp1["img_size"],
            max_steps=val_steps, damping=hp1.get("damping", 0.0),
        )
        print("Collecting 3 training-distribution episodes for video logging...")
        train_sample_trajs = collect_data(
            n_episodes=3,
            img_size=hp1["img_size"],
            epsilon=hp1.get("epsilon", 0.1),
            energy_k=hp1.get("energy_k", 1.0),
            max_steps=hp1.get("max_steps", 200),
            damping=hp1.get("damping", 0.0),
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
        separable=hp1["separable"],
        learn_structure=hp1["learn_structure"],
        dt=hp1["dt"],
        damping=hp1["damping"],
    ).to(device)
    print(f"Phase 2 model parameters: {sum(p.numel() for p in dyn_model.parameters()):,}")

    if hp1["learn_structure"]:
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

    # Store both phase2 CLI kwargs and the phase1 arch params that govern the run
    hparams = {
        **kwargs,
        "phase1_hparams": {k: hp1[k] for k in (
            "latent_dim", "pos_ch", "feat_dim", "img_size", "dt",
            "separable", "learn_structure", "damping",
        )},
    }

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
            if hp1["learn_structure"]:
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
            if phase1_model is not None:
                for val_trajs, label in (
                    (val_energy, "energy_pump"),
                    (val_random, "random"),
                    (val_spin, "spin"),
                ):
                    if val_trajs:
                        val_loss_metrics = _eval_loss_phase2(
                            phase1_model=phase1_model,
                            dyn_model=dyn_model,
                            val_trajs=val_trajs,
                            device=device,
                            seq_len=seq_len,
                        )
                        for k, v in val_loss_metrics.items():
                            writer.add_scalar(f"{k}/{label}", v, epoch)
                        _log_dreamed_video_phase2(
                            phase1_model=phase1_model,
                            dyn_model=dyn_model,
                            val_traj=val_trajs[0],
                            device=device,
                            writer=writer,
                            epoch=epoch,
                            seq_len=seq_len,
                            context_frames=kwargs["val_context_frames"],
                            tag=f"val/dreamed_phase2/{label}",
                        )
                for i, train_traj in enumerate(train_sample_trajs):
                    _log_dreamed_video_phase2(
                        phase1_model=phase1_model,
                        dyn_model=dyn_model,
                        val_traj=train_traj,
                        device=device,
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
            save_checkpoint(run_dir, epoch, dyn_model, hparams, metrics, stem="best")
            best_loss = metrics["phase2/dynamics"]

    save_checkpoint(run_dir, epoch, dyn_model, hparams, metrics, stem="final")

    writer.close()
    print("\nDone. Run: tensorboard --logdir runs")
    os._exit(0)


if __name__ == "__main__":
    cli()
