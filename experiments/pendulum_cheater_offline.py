"""Cheater experiment: supervise Phase 2's normalizing-flow output against ground truth.

Phase 2 (``experiments/pendulum_offline.py``) trains ``HamiltonianFlowModel``:
an invertible normalizing flow phi mapping the frozen Phase 1 encoder's h_t to
a phase space (q, p), fit purely from prediction error (teacher-forced +
closed-loop rollout, compared back in h-space via phi's inverse). This
experiment keeps that exact setup — phi, H, R, B, all of Phase 2's ordinary
losses — and bolts on a second head, ``PhaseSpaceBridge``, that reads phi's
(q, p) output and is regressed directly against the environment's true
(cos θ, sin θ, θ̇) at every timestep.

The bridge is a diagnostic tap, not a step in the dynamics graph: its output
never feeds back into ``controlled_step``/``decode``, so it cannot change
what the ordinary phase2 losses teach phi/H/R/B — it only adds a ground-truth
"cheat" signal shaping phi on top of them (through phi's own parameters,
since the bridge's loss backpropagates through phi to get there).

Diagnostic purpose: if phase2's dynamics loss alone struggles to recover a
phase space that lines up with the true (theta, theta_dot), handing phi an
extra, direct supervision toward that target isolates whether the bottleneck
is "phi can't find the right coordinates from prediction error alone" (this
should measurably help) or something further downstream in H/R/B (this
shouldn't).

--skip-foreplay drops the ordinary phase2 losses (teacher-forced, closed-loop,
structural, energy-balance) entirely, training only phi + the bridge against
ground truth — a cheap probe of whether the structured embedding is even
linearly-ish readable as ground truth, without the cost of also fitting
Hamiltonian dynamics via rollout.

Usage:
    uv run python experiments/pendulum_cheater_offline.py --phase1-run models/pendulum_offline_phase1/<run>
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.pendulum import (
    PendulumMultiRolloutDataset,
    collect_random_trajectories,
    collect_spin_trajectories,
    collect_val_trajectories,
    _DRAG_COEFF,
)
from hamilton_rl.checkpoint import load_world_model, make_run_dir
from hamilton_rl.models import HamiltonianFlowModel, WorldModel
from experiments.pendulum_offline import (
    _collect_energy_grid_episodes,
    _encode_val_h,
    _energy_balance_loss,
    _eval_loss_phase2,
    _log_dreamed_video_phase2,
    _log_hparams_table,
    _log_hparams_text,
    _log_phase_space_regression_phase2,
    _log_structural_matrices_phase2,
    _log_training_rollout,
    _plot_dissipation_landscape,
    _plot_gradient_magnitude_landscape,
    _plot_learned_energy_landscape,
)


# ---------------------------------------------------------------------------
# The cheater layer
# ---------------------------------------------------------------------------


class PhaseSpaceBridge(nn.Module):
    """Small MLP: learned (q, p) -> ground-truth-supervised (cos θ, sin θ, θ̇).

    Inserted as a diagnostic tap on phi's output, not a step in the dynamics
    graph — controlled_step/decode never see its prediction, so it can only
    add supervision on top of phi, never let the Hamiltonian dynamics
    themselves "cheat" by routing through ground truth at inference time.
    """

    def __init__(self, q_dim: int, p_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(q_dim + p_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([q, p], dim=-1))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _train_epoch_cheater(
    dyn_model: HamiltonianFlowModel,
    bridge: PhaseSpaceBridge,
    encoder: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
    device: torch.device,
    seq_len: int,
    seed_ctx_len: int,
    logdet_weight: float,
    teacher_force_weight: float,
    closed_loop_weight: float,
    closed_loop_gamma: float,
    huber_delta: float,
    structural_reg_weight: float,
    energy_balance_weight: float,
    cheat_weight: float,
    skip_foreplay: bool,
) -> dict[str, float]:
    """Phase2-style dynamics epoch plus a ground-truth-supervised bridge loss.

    Mirrors ``pendulum_offline._train_epoch_phase2`` (teacher-forced +
    closed-loop rollout in h-space, same random-window sampling). The
    addition: every encoded (q, p) in the window is fed through ``bridge``
    and compared by MSE against the window's true (cosθ, sinθ, θ̇) (states are
    already stored in that representation — see
    ``data.pendulum.PendulumMultiRolloutDataset``).

    If skip_foreplay, the ordinary phase2 losses (teacher-forced, closed-loop,
    logdet, structural, energy-balance) are skipped entirely and no rollout is
    computed — the loss is just cheat_weight * cheat_loss, a cheap probe of
    whether phi's embedding is directly readable as ground truth without also
    paying for fitting the dynamics.
    """
    dyn_model.train()
    bridge.train()
    encoder.eval()
    q_dim = dyn_model.latent_dim // 2
    ctx = seed_ctx_len
    totals = {
        "dynamics": 0.0, "tf": 0.0, "cl": 0.0, "logdet": 0.0,
        "struct_reg": 0.0, "energy_balance": 0.0, "cheat": 0.0,
        "q_var": 0.0, "p_var": 0.0,
    }

    for frames, actions, states in loader:
        actions = actions.to(device)  # (B, T_full)
        states = states.to(device)    # (B, T_full+1, 3)
        B_size = frames.shape[0]
        T_full = actions.shape[1]

        # --- Sample one random window [s, s+ctx+T), shared across the batch ---
        T = max(min(seq_len, T_full - ctx + 1), 1)
        W = ctx + T
        max_s = T_full + 1 - W
        s = int(torch.randint(0, max_s + 1, (1,)).item()) if max_s > 0 else 0

        frames_win = frames[:, s:s + W].to(device)   # (B, W, C, H, W)
        actions_win = actions[:, s:s + W - 1]         # (B, W-1)
        states_win = states[:, s:s + W]                # (B, W, 3)

        with torch.no_grad():
            h_all, _ = encoder.forward_all(frames_win)  # (B, W, latent_dim)
        D = h_all.shape[-1]

        h_flat = h_all.reshape(B_size * W, D)
        q_flat, p_flat, log_det_flat = dyn_model.encode_with_logdet(h_flat)
        q_all = q_flat.reshape(B_size, W, q_dim)
        p_all = p_flat.reshape(B_size, W, q_dim)
        log_det_all = log_det_flat.reshape(B_size, W)
        logdet_metric = log_det_all.detach().pow(2).mean()

        # --- The cheat: supervise every encoded point against ground truth ---
        cheat_pred = bridge(q_flat, p_flat).reshape(B_size, W, 3)
        cheat_loss = F.mse_loss(cheat_pred, states_win)

        if skip_foreplay:
            loss = cheat_weight * cheat_loss
            tf_loss = cl_loss = struct_reg = eb_loss = torch.zeros((), device=device)
            q_traj, p_traj = q_all[:, ctx - 1:], p_all[:, ctx - 1:]
        else:
            logdet_reg = logdet_weight * log_det_all.pow(2).mean()

            # Teacher-forced: every consecutive pair, batched as one step.
            T_tf = W - 1
            q_tf = q_all[:, :T_tf].reshape(B_size * T_tf, q_dim)
            p_tf = p_all[:, :T_tf].reshape(B_size * T_tf, q_dim)
            a_tf = actions_win.reshape(B_size * T_tf, 1)
            q_tf_next, p_tf_next = dyn_model.controlled_step(q_tf, p_tf, a_tf)
            h_tf_pred = dyn_model.decode(q_tf_next, p_tf_next)
            h_tf_target = h_all[:, 1:].reshape(B_size * T_tf, D)
            tf_loss = F.mse_loss(h_tf_pred, h_tf_target)

            # Closed-loop rollout from the end of context (local index ctx-1).
            k = ctx - 1
            q, p = q_all[:, k], p_all[:, k]
            qs_steps, ps_steps = [], []
            for t in range(T):
                q, p = dyn_model.controlled_step(q, p, actions_win[:, k + t:k + t + 1])
                qs_steps.append(q)
                ps_steps.append(p)
            q_traj = torch.stack(qs_steps, dim=1)  # (B, T, q_dim)
            p_traj = torch.stack(ps_steps, dim=1)
            h_cl_pred = dyn_model.decode(
                q_traj.reshape(B_size * T, q_dim), p_traj.reshape(B_size * T, q_dim)
            ).reshape(B_size, T, D)
            h_cl_target = h_all[:, k + 1:k + 1 + T]
            if huber_delta > 0:
                elem_err = 2.0 * F.huber_loss(
                    h_cl_pred, h_cl_target, reduction="none", delta=huber_delta
                )
            else:
                elem_err = (h_cl_pred - h_cl_target).pow(2)
            per_step_loss = elem_err.mean(dim=(0, 2))
            step_weights = closed_loop_gamma ** torch.arange(T, device=device, dtype=per_step_loss.dtype)
            cl_loss = (per_step_loss * step_weights).sum() / step_weights.sum()

            loss = (
                logdet_reg
                + teacher_force_weight * tf_loss
                + closed_loop_weight * cl_loss
                + cheat_weight * cheat_loss
            )

            struct_reg = torch.zeros((), device=device)
            if structural_reg_weight > 0 and dyn_model.r_parameters():
                if dyn_model.state_dep_r:
                    z_r = torch.cat([q_all, p_all], dim=-1).reshape(-1, D).detach()
                    struct_reg = dyn_model.get_R_pp(z_r).pow(2).sum(dim=(-2, -1)).mean()
                else:
                    struct_reg = dyn_model.get_R_pp().pow(2).sum()
                loss = loss + structural_reg_weight * struct_reg

            eb_loss = torch.zeros((), device=device)
            if energy_balance_weight > 0:
                eb_loss = _energy_balance_loss(dyn_model, q_all, p_all, actions_win)
                loss = loss + energy_balance_weight * eb_loss

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(dyn_model.parameters()) + list(bridge.parameters()), grad_clip
            )
        optimizer.step()

        totals["dynamics"] += float(loss.detach())
        totals["tf"] += float(tf_loss.detach())
        totals["cl"] += float(cl_loss.detach())
        totals["logdet"] += float(logdet_metric)
        totals["struct_reg"] += float(struct_reg.detach())
        totals["energy_balance"] += float(eb_loss.detach())
        totals["cheat"] += float(cheat_loss.detach())
        with torch.no_grad():
            totals["q_var"] += q_traj.detach().reshape(-1, q_dim).var(dim=0).mean().item()
            totals["p_var"] += p_traj.detach().reshape(-1, q_dim).var(dim=0).mean().item()

    n = len(loader)
    return {f"cheater/{k}": v / n for k, v in totals.items()}


@torch.no_grad()
def _eval_cheat_loss(
    dyn_model: HamiltonianFlowModel,
    bridge: PhaseSpaceBridge,
    phase1_model,
    val_trajs: list,
    device: torch.device,
) -> float:
    """Teacher-forced (no rollout) bridge MSE against ground truth, held-out."""
    dyn_model.eval()
    bridge.eval()
    h_all, _ = _encode_val_h(phase1_model, val_trajs, device)
    N, T_seq, D = h_all.shape
    q, p = dyn_model.encode(h_all.reshape(N * T_seq, D))
    pred = bridge(q, p)
    states_all = torch.stack([t[2] for t in val_trajs]).to(device).reshape(N * T_seq, 3)
    return F.mse_loss(pred, states_all).item()


@torch.no_grad()
def _log_bridge_regression(
    world_model: WorldModel,
    bridge: PhaseSpaceBridge,
    val_trajs: list,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    tag: str = "val/bridge_regression",
) -> None:
    """What the trained bridge actually predicts over a closed-loop rollout.

    Unlike ``pendulum_offline._log_phase_space_regression_phase2`` (which fits
    a fresh linear probe each call to ask what the dynamics *could* recover),
    this plots the cheater bridge's own predictions — showing how well the
    ground-truth supervision holds up as rollout error compounds.
    """
    phase1_model, dyn_model = world_model.autoencoder, world_model.dynamics
    h_all, actions_all = _encode_val_h(phase1_model, val_trajs, device)
    N = h_all.shape[0]
    T_full = actions_all.shape[1]
    n_steps = T_full - 1
    if n_steps <= 0 or N < 2:
        return

    q, p = dyn_model.encode(h_all[:, 1])
    preds = [bridge(q, p)]
    for t in range(n_steps):
        q, p = dyn_model.controlled_step(q, p, actions_all[:, 1 + t: 2 + t])
        preds.append(bridge(q, p))
    st_pred = torch.stack(preds, dim=1).cpu().numpy()  # (N, n_steps+1, 3)
    st_true = torch.stack([t[2] for t in val_trajs]).float()[:, 1:2 + n_steps].numpy()

    n_pts = st_pred.shape[1]
    step_idx = np.repeat(np.arange(n_pts)[None, :], N, axis=0).reshape(-1)
    st_pred_flat = st_pred.reshape(-1, 3)
    st_true_flat = st_true.reshape(-1, 3)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    sc = None
    for i, name in enumerate(["cos(θ)", "sin(θ)", "θ̇ (rad/s)"]):
        true_i, pred_i = st_true_flat[:, i], st_pred_flat[:, i]
        sc = axes[i].scatter(true_i, pred_i, c=step_idx, cmap="viridis", s=2, alpha=0.4, linewidths=0)
        lo, hi = min(true_i.min(), pred_i.min()), max(true_i.max(), pred_i.max())
        axes[i].plot([lo, hi], [lo, hi], "r--", linewidth=0.8)
        axes[i].set_xlabel(f"True {name}")
        axes[i].set_ylabel(f"Bridge-predicted {name}")
        ss_res = ((true_i - pred_i) ** 2).sum()
        ss_tot = ((true_i - true_i.mean()) ** 2).sum()
        axes[i].set_title(f"{name}  R²={1 - ss_res / (ss_tot + 1e-8):.3f}")
    fig.colorbar(sc, ax=axes, label="rollout step", fraction=0.03, pad=0.02)
    fig.suptitle(f"Cheater bridge: closed-loop (q,p) → ground truth (epoch {epoch + 1})")
    writer.add_figure(tag, fig, epoch)
    plt.close(fig)


def _save_bridge(run_dir: Path, stem: str, bridge: PhaseSpaceBridge, hidden_dim: int) -> None:
    torch.save(
        {"state_dict": bridge.state_dict(), "hidden_dim": hidden_dim},
        Path(run_dir) / f"{stem}_bridge.pt",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option("--phase1-run", type=str, required=True,
              help="Path to a Phase 1 run directory (models/pendulum_offline_phase1/<run>); "
                   "loads best.pt (falling back to final.pt) and rollout_cache.pt")
@click.option("--phase1-checkpoint", type=str, default=None,
              help="Override the Phase 1 checkpoint (default: {phase1-run}/best.pt, "
                   "falling back to final.pt)")
@click.option("--rollout-cache", type=str, default=None,
              help="Override the rollout cache path (default: {phase1-run}/rollout_cache.pt)")
@click.option("--resume-from", type=str, default=None,
              help="Path to a Phase 2 or cheater checkpoint (.pt) whose dynamics weights to "
                   "warm-start from; training still writes to a fresh run dir")
# dynamics model (HamiltonianFlowModel — same as Phase 2)
@click.option("--dt", type=float, default=0.05, show_default=True)
@click.option("--separable/--no-separable", default=True, show_default=True)
@click.option("--h-source", type=click.Choice(["learned", "canonical"]), default="learned",
              show_default=True)
@click.option("--r-source", type=click.Choice(["learned", "fixed_damping", "canonical"]),
              default="learned", show_default=True)
@click.option("--b-source", type=click.Choice(["learned", "fixed_ones", "canonical"]),
              default="learned", show_default=True)
@click.option("--drag", type=float, default=_DRAG_COEFF, show_default=True)
@click.option("--state-dep-r/--no-state-dep-r", default=False, show_default=True)
@click.option("--integrator", type=click.Choice(["rk4", "leapfrog"]), default="leapfrog",
              show_default=True)
@click.option("--quadratic-t/--no-quadratic-t", default=True, show_default=True)
# the cheat
@click.option("--bridge-hidden-dim", type=int, default=64, show_default=True,
              help="Width of the cheater bridge MLP's hidden layers")
@click.option("--cheat-weight", type=float, default=1.0, show_default=True,
              help="Weight on the bridge's supervised MSE loss against ground-truth "
                   "(cosθ, sinθ, θ̇)")
@click.option("--skip-foreplay", is_flag=True, default=False, show_default=True,
              help="Skip the ordinary phase2 dynamics losses (teacher-forced, closed-loop, "
                   "structural, energy-balance) entirely and train phi + the bridge with only "
                   "direct ground-truth supervision — a fast probe of whether the structured "
                   "embedding is readable as ground truth at all, without also fitting the "
                   "Hamiltonian dynamics.")
# training
@click.option("--epochs", type=int, default=3000, show_default=True)
@click.option("--batch-size", type=int, default=8, show_default=True)
@click.option("--lr", type=float, default=1e-4, show_default=True)
@click.option("--bridge-lr", type=float, default=1e-3, show_default=True)
@click.option("--structural-lr", type=float, default=1e-2, show_default=True)
@click.option("--grad-clip", type=float, default=1.0, show_default=True)
@click.option("--logdet-weight", type=float, default=1e-3, show_default=True)
@click.option("--teacher-force-weight", type=float, default=1.0, show_default=True)
@click.option("--closed-loop-weight", type=float, default=1.0, show_default=True)
@click.option("--closed-loop-gamma", type=float, default=1.0, show_default=True)
@click.option("--huber-delta", type=float, default=0.2, show_default=True)
@click.option("--structural-reg-weight", type=float, default=0.0, show_default=True)
@click.option("--energy-balance-weight", type=float, default=1.0, show_default=True)
@click.option("--seed-ctx-len", type=int, default=3, show_default=True)
@click.option("--seq-len-start", type=int, default=5, show_default=True)
@click.option("--max-seq-len", type=int, default=0, show_default=True)
@click.option("--seq-len-advance-threshold", type=float, default=0.005, show_default=True)
@click.option("--ema-alpha", type=float, default=0.99, show_default=True)
# logging
@click.option("--log-every", type=int, default=5, show_default=True)
@click.option("--val-every", type=int, default=10, show_default=True)
@click.option("--n-val-episodes", type=int, default=-1, show_default=True,
              help="Val episodes per type (-1 = phase1 n_windows // 2)")
@click.option("--val-max-steps", type=int, default=0, show_default=True,
              help="Steps per val episode (0 = 2x phase1 max_steps)")
@click.option("--val-context-frames", type=int, default=5, show_default=True)
@click.option("--checkpoint-every", type=int, default=10, show_default=True)
def main(**kwargs):
    """Phase 2 dynamics training with a ground-truth-supervised bridge on phi's output."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
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
    rollout_cache_path = kwargs["rollout_cache"] or str(run1 / "rollout_cache.pt")

    writer = SummaryWriter(comment="_pendulum_cheater_offline")
    run_dir = make_run_dir("pendulum_cheater_offline")

    print(f"Loading Phase 1 checkpoint from {phase1_ckpt}...")
    world_model = load_world_model(phase1_ckpt, device)
    phase1_model = world_model.autoencoder
    data_cfg = world_model.data_config
    print("Phase 1 config: " + ", ".join(
        f"{k}={v}" for k, v in {**phase1_model.config, **data_cfg}.items()
    ))

    if not Path(rollout_cache_path).exists():
        raise click.UsageError(
            f"Rollout cache not found at {rollout_cache_path}. "
            "Re-run Phase 1 to regenerate it, or pass --rollout-cache explicitly."
        )
    print(f"Loading rollout cache from {rollout_cache_path}...")
    rollouts = torch.load(rollout_cache_path, weights_only=False)
    _log_training_rollout(rollouts, writer)

    val_energy, val_random, val_spin = [], [], []
    energy_grid_episodes = []
    if kwargs["val_every"] > 0:
        n_val = kwargs["n_val_episodes"]
        if n_val < 0:
            n_val = data_cfg.get("n_windows", 200) // 2
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
        print("Collecting grid episodes for energy-landscape logging...")
        energy_grid_episodes = _collect_energy_grid_episodes(
            img_size=img_size, damping=damping,
            drag=data_cfg.get("drag", _DRAG_COEFF),
            context_frames=kwargs["val_context_frames"],
        )

    latent_dim = phase1_model.config["latent_dim"]
    print(f"Latent dim from Phase 1 config: {latent_dim}")

    rollout_dataset = PendulumMultiRolloutDataset(
        rollouts,
        window_len=data_cfg.get("max_steps", 200),
        n_windows=data_cfg.get("n_windows", 200),
    )
    rollout_loader = DataLoader(
        rollout_dataset, batch_size=kwargs["batch_size"], shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    print(
        f"Rollout dataset: {len(rollout_dataset)} windows/epoch "
        f"of {rollout_dataset.window_len} steps"
    )

    dyn_model = HamiltonianFlowModel(
        latent_dim=latent_dim,
        control_dim=1,
        separable=kwargs["separable"],
        h_source=kwargs["h_source"],
        r_source=kwargs["r_source"],
        b_source=kwargs["b_source"],
        dt=kwargs["dt"],
        damping=data_cfg.get("damping", 0.0),
        drag=kwargs["drag"],
        integrator=kwargs["integrator"],
        quadratic_t=kwargs["quadratic_t"],
        state_dep_r=kwargs["state_dep_r"],
    ).to(device)
    q_dim = dyn_model.latent_dim // 2
    bridge = PhaseSpaceBridge(q_dim, latent_dim - q_dim, hidden_dim=kwargs["bridge_hidden_dim"]).to(device)
    print(f"Dynamics parameters: {sum(p.numel() for p in dyn_model.parameters()):,}")
    print(f"Bridge parameters: {sum(p.numel() for p in bridge.parameters()):,}")

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

    opt_groups = [
        {
            "params": (
                list(dyn_model.phi_q.parameters())
                + list(dyn_model.phi_p.parameters())
                + list(dyn_model.hamiltonian.parameters())
            ),
            "lr": kwargs["lr"],
        },
        {"params": bridge.parameters(), "lr": kwargs["bridge_lr"]},
    ]
    structural_params = dyn_model.structural_parameters()
    if structural_params:
        opt_groups.append({"params": structural_params, "lr": kwargs["structural_lr"]})
    optimizer = torch.optim.Adam(opt_groups)

    hparams = {
        **kwargs,
        "phase1_config": {**phase1_model.config, **data_cfg},
    }
    _log_hparams_text(writer, hparams)
    _log_hparams_table(writer, hparams, {})

    window_len = rollout_dataset.window_len
    full_seq_len = window_len - kwargs["seed_ctx_len"] + 1
    if kwargs["max_seq_len"] > 0:
        full_seq_len = min(full_seq_len, kwargs["max_seq_len"])
    seq_len = min(kwargs["seq_len_start"], full_seq_len)
    ema_cl = None
    best_loss = float("inf")

    mode = "skip-foreplay (bridge-only)" if kwargs["skip_foreplay"] else "joint (dynamics + bridge)"
    print(f"\n=== Cheater experiment: {mode} ===")
    for epoch in tqdm(range(kwargs["epochs"]), desc="Cheater", dynamic_ncols=True):
        metrics = _train_epoch_cheater(
            dyn_model=dyn_model,
            bridge=bridge,
            encoder=phase1_model.encoder,
            loader=rollout_loader,
            optimizer=optimizer,
            grad_clip=kwargs["grad_clip"],
            device=device,
            seq_len=seq_len,
            seed_ctx_len=kwargs["seed_ctx_len"],
            logdet_weight=kwargs["logdet_weight"],
            teacher_force_weight=kwargs["teacher_force_weight"],
            closed_loop_weight=kwargs["closed_loop_weight"],
            closed_loop_gamma=kwargs["closed_loop_gamma"],
            huber_delta=kwargs["huber_delta"],
            structural_reg_weight=kwargs["structural_reg_weight"],
            energy_balance_weight=kwargs["energy_balance_weight"],
            cheat_weight=kwargs["cheat_weight"],
            skip_foreplay=kwargs["skip_foreplay"],
        )

        alpha = kwargs["ema_alpha"]
        ema_cl = (
            metrics["cheater/cl"] if ema_cl is None
            else alpha * ema_cl + (1.0 - alpha) * metrics["cheater/cl"]
        )
        if ema_cl < kwargs["seq_len_advance_threshold"] and seq_len < full_seq_len:
            seq_len += 1

        if (epoch + 1) % kwargs["log_every"] == 0:
            for k, v in metrics.items():
                writer.add_scalar(k, v, epoch)
            writer.add_scalar("cheater/seq_len", seq_len, epoch)
            tqdm.write(
                f"  epoch {epoch + 1:4d}  seq_len={seq_len:3d}"
                f"  loss={metrics['cheater/dynamics']:.6f}"
                f"  cheat={metrics['cheater/cheat']:.6f}"
                f"  tf={metrics['cheater/tf']:.6f}  cl={metrics['cheater/cl']:.6f}"
            )

        if kwargs["val_every"] > 0 and (epoch + 1) % kwargs["val_every"] == 0:
            _log_structural_matrices_phase2(dyn_model=dyn_model, writer=writer, epoch=epoch)
            # Energy/gradient/dissipation landscapes only need the frozen
            # phase1 encoder + phi (H/R evaluated on grid episodes' encoded
            # points) — no dependency on the cheat/rollout losses, so these
            # stay on even under --skip-foreplay.
            energy_fig = _plot_learned_energy_landscape(
                world_model, energy_grid_episodes, device=device,
            )
            writer.add_figure("val/energy_landscape", energy_fig, epoch)
            plt.close(energy_fig)
            grad_mag_fig = _plot_gradient_magnitude_landscape(
                world_model, energy_grid_episodes, device=device,
            )
            writer.add_figure("val/gradient_magnitude_landscape", grad_mag_fig, epoch)
            plt.close(grad_mag_fig)
            if dyn_model._has_dissipation:
                dissipation_fig = _plot_dissipation_landscape(
                    world_model, energy_grid_episodes,
                    damping=data_cfg.get("damping", 0.0),
                    drag=data_cfg.get("drag", _DRAG_COEFF),
                    device=device,
                )
                writer.add_figure("val/dissipation_landscape", dissipation_fig, epoch)
                plt.close(dissipation_fig)
            for val_trajs, label in (
                (val_energy, "energy_pump"), (val_random, "random"), (val_spin, "spin"),
            ):
                if not val_trajs:
                    continue
                writer.add_scalar(
                    f"val/cheat_loss/{label}",
                    _eval_cheat_loss(dyn_model, bridge, phase1_model, val_trajs, device),
                    epoch,
                )
                _log_bridge_regression(
                    world_model=world_model, bridge=bridge, val_trajs=val_trajs,
                    device=device, writer=writer, epoch=epoch,
                    tag=f"val/bridge_regression/{label}",
                )
                val_loss_metrics = _eval_loss_phase2(
                    world_model=world_model, val_trajs=val_trajs, device=device,
                )
                for k, v in val_loss_metrics.items():
                    writer.add_scalar(f"{k}/{label}", v, epoch)
                _log_phase_space_regression_phase2(
                    world_model=world_model, val_trajs=val_trajs, device=device,
                    writer=writer, epoch=epoch, tag=f"val/phase_space_regression/{label}",
                )
                _log_dreamed_video_phase2(
                    world_model=world_model, val_traj=val_trajs[0], writer=writer,
                    epoch=epoch, seq_len=full_seq_len,
                    context_frames=kwargs["val_context_frames"],
                    tag=f"val/dreamed_phase2/{label}",
                )

        if (
            kwargs["checkpoint_every"] > 0
            and (epoch + 1) % kwargs["checkpoint_every"] == 0
            and metrics["cheater/dynamics"] < best_loss
        ):
            world_model.save(run_dir, "best", hparams, metrics, epoch)
            _save_bridge(run_dir, "best", bridge, kwargs["bridge_hidden_dim"])
            best_loss = metrics["cheater/dynamics"]

    world_model.save(run_dir, "final", hparams, metrics, epoch)
    _save_bridge(run_dir, "final", bridge, kwargs["bridge_hidden_dim"])

    writer.close()
    print("\nDone. Run: tensorboard --logdir runs")
    os._exit(0)


if __name__ == "__main__":
    main()
