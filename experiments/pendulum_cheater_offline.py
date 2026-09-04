"""Cheater experiment: replace Phase 2's normalizing flow with a regression head.

Phase 2 (``experiments/pendulum_offline.py``) learns phi — an invertible
normalizing flow mapping the frozen Phase 1 encoder's h_t to a phase space
(q, p) — purely from prediction error. This experiment replaces phi with a
plain (non-invertible) MLP, ``CheaterEncoder``, trained by direct MSE against
the environment's true (cos θ, sin θ, θ̇) at every timestep. There is no
decoder and no h-space loss: once phi is "the known phase space," the
dynamics (``hamilton_rl.models.StatePHGN`` — the same ground-truth-phase-space
model ``pendulum_statebased_offline.py`` trains) can be fit with ordinary
teacher-forced/closed-loop rollout losses evaluated directly against ground
truth, exactly as normal state-based training — the only difference is the
initial condition and every teacher-forced input comes from the vision
pipeline (phase1 encoder -> cheater head) instead of the simulator's exact
state.

Diagnostic purpose: isolates whether phase2's bottleneck is "phi can't find
coordinates that line up with true phase space from prediction error alone"
(handing it the coordinates directly should fix downstream dynamics fitting)
versus something in H/R/B themselves (this shouldn't help those).

--skip-foreplay drops the dynamics rollout entirely and trains only the
cheater head against ground truth — a cheap probe of whether Phase 1's
structured embedding is even linearly-ish readable as ground truth.

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
from hamilton_rl.checkpoint import load_world_model, make_run_dir, save_state_model
from hamilton_rl.models import StatePHGN
from experiments.pendulum_offline import (
    _encode_val_h,
    _log_hparams_table,
    _log_hparams_text,
    _log_training_rollout,
)
from experiments.pendulum_statebased_offline import (
    _log_structural_matrices,
    _plot_state_dissipation_landscape,
    _plot_state_energy_landscape,
    _plot_state_gradient_magnitude_landscape,
)

_LANDSCAPE_VEL_CLIP = 7.0


# ---------------------------------------------------------------------------
# The cheater layer
# ---------------------------------------------------------------------------


class CheaterEncoder(nn.Module):
    """MLP replacing phi: h_t -> ground-truth-supervised (cos θ̂, sin θ̂, θ̇̂).

    Regressing onto (cos θ, sin θ) rather than raw θ sidesteps the angle's
    branch cut entirely — both in the loss (no wraparound discontinuity to
    fight) and downstream, where ``theta()`` recovers a usable scalar via
    atan2 for StatePHGN's raw-angle phase space.
    """

    def __init__(self, h_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h (..., h_dim) -> (..., 3) raw (cos θ̂, sin θ̂, θ̇̂), NOT renormalized."""
        return self.net(h)

    @staticmethod
    def theta(cheat_out: torch.Tensor) -> torch.Tensor:
        """(..., 3) -> (..., 1) angle recovered via atan2(sin, cos)."""
        return torch.atan2(cheat_out[..., 1:2], cheat_out[..., 0:1])

    @staticmethod
    def theta_dot(cheat_out: torch.Tensor) -> torch.Tensor:
        return cheat_out[..., 2:3]


def _to_repr(theta: torch.Tensor, theta_dot: torch.Tensor) -> torch.Tensor:
    """(θ, θ̇) -> (cos θ, sin θ, θ̇), matching the dataset's ground-truth representation."""
    return torch.cat([torch.cos(theta), torch.sin(theta), theta_dot], dim=-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _train_epoch_cheater(
    dyn_model: StatePHGN,
    cheater: CheaterEncoder,
    encoder: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
    device: torch.device,
    seq_len: int,
    seed_ctx_len: int,
    teacher_force_weight: float,
    closed_loop_weight: float,
    closed_loop_gamma: float,
    huber_delta: float,
    structural_reg_weight: float,
    cheat_weight: float,
    skip_foreplay: bool,
) -> dict[str, float]:
    """One epoch of cheater-encoder + (optionally) StatePHGN dynamics fitting.

    Random-window sampling and fresh-LSTM-state re-encoding mirror
    ``pendulum_offline._train_epoch_phase2``. Every encoded (q̂, p̂) =
    CheaterEncoder(h_t) is compared to the window's true (cosθ, sinθ, θ̇) —
    the cheat. When not skip_foreplay, StatePHGN.step is additionally rolled
    forward teacher-forced and closed-loop, with predictions compared
    directly against ground truth (via ``_to_repr``, sidestepping θ's branch
    cut) rather than against a re-encoded target — there is no decoder here,
    so ground truth doubles as the only well-posed target.
    """
    dyn_model.train()
    cheater.train()
    encoder.eval()
    ctx = seed_ctx_len
    totals = {
        "dynamics": 0.0, "tf": 0.0, "cl": 0.0,
        "struct_reg": 0.0, "cheat": 0.0, "q_var": 0.0, "p_var": 0.0,
    }

    for frames, actions, states in loader:
        actions = actions.to(device)  # (B, T_full)
        states = states.to(device)    # (B, T_full+1, 3)
        B_size = frames.shape[0]
        T_full = actions.shape[1]

        T = max(min(seq_len, T_full - ctx + 1), 1)
        W = ctx + T
        max_s = T_full + 1 - W
        s = int(torch.randint(0, max_s + 1, (1,)).item()) if max_s > 0 else 0

        frames_win = frames[:, s:s + W].to(device)
        actions_win = actions[:, s:s + W - 1]
        states_win = states[:, s:s + W]  # (B, W, 3)

        with torch.no_grad():
            h_all, _ = encoder.forward_all(frames_win)  # (B, W, latent_dim)
        D = h_all.shape[-1]

        cheat_out = cheater(h_all.reshape(B_size * W, D)).reshape(B_size, W, 3)
        cheat_loss = F.mse_loss(cheat_out, states_win)

        q_all = CheaterEncoder.theta(cheat_out)       # (B, W, 1)
        p_all = CheaterEncoder.theta_dot(cheat_out)   # (B, W, 1)

        if skip_foreplay:
            loss = cheat_weight * cheat_loss
            tf_loss = cl_loss = struct_reg = torch.zeros((), device=device)
            q_traj, p_traj = q_all[:, ctx - 1:], p_all[:, ctx - 1:]
        else:
            # Teacher-forced: every consecutive pair, batched as one step.
            T_tf = W - 1
            q_tf = q_all[:, :T_tf].reshape(B_size * T_tf, 1)
            p_tf = p_all[:, :T_tf].reshape(B_size * T_tf, 1)
            a_tf = actions_win.reshape(B_size * T_tf, 1)
            q_tf_next, p_tf_next = dyn_model.step(q_tf, p_tf, a_tf)
            pred_tf = _to_repr(q_tf_next, p_tf_next)
            target_tf = states_win[:, 1:].reshape(B_size * T_tf, 3)
            tf_loss = F.mse_loss(pred_tf, target_tf)

            # Closed-loop rollout from the end of context (local index ctx-1).
            k = ctx - 1
            q, p = q_all[:, k], p_all[:, k]
            qs_steps, ps_steps, preds = [], [], []
            for t in range(T):
                q, p = dyn_model.step(q, p, actions_win[:, k + t:k + t + 1])
                qs_steps.append(q)
                ps_steps.append(p)
                preds.append(_to_repr(q, p))
            q_traj = torch.stack(qs_steps, dim=1)  # (B, T, 1)
            p_traj = torch.stack(ps_steps, dim=1)
            pred_cl = torch.stack(preds, dim=1)     # (B, T, 3)
            target_cl = states_win[:, k + 1:k + 1 + T]
            if huber_delta > 0:
                elem_err = 2.0 * F.huber_loss(
                    pred_cl, target_cl, reduction="none", delta=huber_delta
                )
            else:
                elem_err = (pred_cl - target_cl).pow(2)
            per_step_loss = elem_err.mean(dim=(0, 2))
            step_weights = closed_loop_gamma ** torch.arange(T, device=device, dtype=per_step_loss.dtype)
            cl_loss = (per_step_loss * step_weights).sum() / step_weights.sum()

            loss = (
                teacher_force_weight * tf_loss
                + closed_loop_weight * cl_loss
                + cheat_weight * cheat_loss
            )

            struct_reg = torch.zeros((), device=device)
            if structural_reg_weight > 0 and dyn_model.r_source == "learned":
                if dyn_model.state_dep_r:
                    z_r = dyn_model._r_input(
                        q_all.reshape(-1, 1).detach(), p_all.reshape(-1, 1).detach()
                    )
                    struct_reg = dyn_model.get_R_pp(z_r).pow(2).sum(dim=(-2, -1)).mean()
                else:
                    struct_reg = dyn_model.get_R_pp().pow(2).sum()
                loss = loss + structural_reg_weight * struct_reg

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(dyn_model.parameters()) + list(cheater.parameters()), grad_clip
            )
        optimizer.step()

        totals["dynamics"] += float(loss.detach())
        totals["tf"] += float(tf_loss.detach())
        totals["cl"] += float(cl_loss.detach())
        totals["struct_reg"] += float(struct_reg.detach())
        totals["cheat"] += float(cheat_loss.detach())
        with torch.no_grad():
            totals["q_var"] += q_traj.detach().reshape(-1, 1).var(dim=0).mean().item()
            totals["p_var"] += p_traj.detach().reshape(-1, 1).var(dim=0).mean().item()

    n = len(loader)
    return {f"cheater/{k}": v / n for k, v in totals.items()}


@torch.no_grad()
def _eval_cheat_loss(
    cheater: CheaterEncoder,
    phase1_model,
    val_trajs: list,
    device: torch.device,
) -> float:
    """Teacher-forced (no rollout) cheater MSE against ground truth, held-out."""
    cheater.eval()
    h_all, _ = _encode_val_h(phase1_model, val_trajs, device)
    N, T_seq, D = h_all.shape
    cheat_out = cheater(h_all.reshape(N * T_seq, D))
    states_all = torch.stack([t[2] for t in val_trajs]).to(device).reshape(N * T_seq, 3)
    return F.mse_loss(cheat_out, states_all).item()


@torch.no_grad()
def _eval_dynamics_loss(
    dyn_model: StatePHGN,
    cheater: CheaterEncoder,
    phase1_model,
    val_trajs: list,
    device: torch.device,
) -> dict[str, float]:
    """Teacher-forced + closed-loop MSE (state-space) over the full val horizon."""
    dyn_model.eval()
    cheater.eval()
    h_all, actions_all = _encode_val_h(phase1_model, val_trajs, device)
    N, T_seq, D = h_all.shape
    T_full = actions_all.shape[1]
    states_all = torch.stack([t[2] for t in val_trajs]).to(device)  # (N, T_seq, 3)

    cheat_out = cheater(h_all.reshape(N * T_seq, D)).reshape(N, T_seq, 3)
    q_all = CheaterEncoder.theta(cheat_out)
    p_all = CheaterEncoder.theta_dot(cheat_out)

    q_tf = q_all[:, :T_full].reshape(N * T_full, 1)
    p_tf = p_all[:, :T_full].reshape(N * T_full, 1)
    q_next, p_next = dyn_model.step(q_tf, p_tf, actions_all.reshape(N * T_full, 1))
    pred_tf = _to_repr(q_next, p_next)
    target_tf = states_all[:, 1:].reshape(N * T_full, 3)
    teacher_forced = F.mse_loss(pred_tf, target_tf).item()

    n_rollout_steps = T_full - 1
    closed_loop = 0.0
    if n_rollout_steps > 0:
        q, p = q_all[:, 1], p_all[:, 1]
        preds = []
        for t in range(n_rollout_steps):
            q, p = dyn_model.step(q, p, actions_all[:, 1 + t:2 + t])
            preds.append(_to_repr(q, p))
        pred_cl = torch.stack(preds, dim=1)
        target_cl = states_all[:, 2:2 + n_rollout_steps]
        closed_loop = F.mse_loss(pred_cl, target_cl).item()

    return {"val/tf_loss": teacher_forced, "val/cl_loss": closed_loop}


@torch.no_grad()
def _log_cheater_regression(
    dyn_model: StatePHGN,
    cheater: CheaterEncoder,
    phase1_model,
    val_trajs: list,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    tag: str = "val/cheater_regression",
) -> None:
    """What the cheater + dynamics actually predict over a closed-loop rollout."""
    h_all, actions_all = _encode_val_h(phase1_model, val_trajs, device)
    N, T_seq, D = h_all.shape
    T_full = actions_all.shape[1]
    n_steps = T_full - 1
    if n_steps <= 0 or N < 2:
        return

    cheat_out = cheater(h_all[:, 1].reshape(N, D))
    q, p = CheaterEncoder.theta(cheat_out), CheaterEncoder.theta_dot(cheat_out)
    preds = [_to_repr(q, p)]
    for t in range(n_steps):
        q, p = dyn_model.step(q, p, actions_all[:, 1 + t:2 + t])
        preds.append(_to_repr(q, p))
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
        axes[i].set_ylabel(f"Predicted {name}")
        ss_res = ((true_i - pred_i) ** 2).sum()
        ss_tot = ((true_i - true_i.mean()) ** 2).sum()
        axes[i].set_title(f"{name}  R²={1 - ss_res / (ss_tot + 1e-8):.3f}")
    fig.colorbar(sc, ax=axes, label="rollout step", fraction=0.03, pad=0.02)
    fig.suptitle(f"Cheater: closed-loop dynamics → ground truth (epoch {epoch + 1})")
    writer.add_figure(tag, fig, epoch)
    plt.close(fig)


def _save_cheater(run_dir: Path, stem: str, cheater: CheaterEncoder, hidden_dim: int) -> None:
    torch.save(
        {"state_dict": cheater.state_dict(), "hidden_dim": hidden_dim},
        Path(run_dir) / f"{stem}_cheater.pt",
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
              help="Path to a cheater checkpoint (.pt, StatePHGN via save_state_model) whose "
                   "dynamics weights to warm-start from")
# dynamics model (StatePHGN — same ground-truth phase space as pendulum_statebased_offline.py)
@click.option("--hidden-dim", type=int, default=256, show_default=True,
              help="Width of Hamiltonian MLP hidden layers")
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
@click.option("--integrator", type=click.Choice(["auto", "rk4", "leapfrog"]), default="auto",
              show_default=True)
@click.option("--quadratic-t/--no-quadratic-t", default=True, show_default=True)
# the cheat
@click.option("--cheater-hidden-dim", type=int, default=64, show_default=True,
              help="Width of the cheater encoder MLP's hidden layers")
@click.option("--cheat-weight", type=float, default=1.0, show_default=True,
              help="Weight on the cheater's supervised MSE loss against ground-truth "
                   "(cosθ, sinθ, θ̇)")
@click.option("--skip-foreplay", is_flag=True, default=False, show_default=True,
              help="Skip StatePHGN dynamics fitting entirely and train only the cheater "
                   "encoder against ground truth — a fast probe of whether the structured "
                   "embedding is readable as ground truth at all.")
# training
@click.option("--epochs", type=int, default=3000, show_default=True)
@click.option("--batch-size", type=int, default=8, show_default=True)
@click.option("--lr", type=float, default=1e-4, show_default=True)
@click.option("--cheater-lr", type=float, default=1e-3, show_default=True)
@click.option("--structural-lr", type=float, default=1e-2, show_default=True)
@click.option("--grad-clip", type=float, default=1.0, show_default=True)
@click.option("--teacher-force-weight", type=float, default=1.0, show_default=True)
@click.option("--closed-loop-weight", type=float, default=1.0, show_default=True)
@click.option("--closed-loop-gamma", type=float, default=1.0, show_default=True)
@click.option("--huber-delta", type=float, default=0.2, show_default=True)
@click.option("--structural-reg-weight", type=float, default=0.0, show_default=True)
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
@click.option("--checkpoint-every", type=int, default=10, show_default=True)
def main(**kwargs):
    """Replace Phase 2's normalizing flow with a ground-truth-regressed cheater MLP."""
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

    dyn_model = StatePHGN(
        hidden_dim=kwargs["hidden_dim"],
        dt=kwargs["dt"],
        control_dim=1,
        separable=kwargs["separable"],
        h_source=kwargs["h_source"],
        r_source=kwargs["r_source"],
        b_source=kwargs["b_source"],
        damping=data_cfg.get("damping", 0.0),
        drag=kwargs["drag"],
        quadratic_t=kwargs["quadratic_t"],
        state_dep_r=kwargs["state_dep_r"],
        integrator=kwargs["integrator"],
    ).to(device)
    cheater = CheaterEncoder(latent_dim, hidden_dim=kwargs["cheater_hidden_dim"]).to(device)
    print(f"Dynamics parameters: {sum(p.numel() for p in dyn_model.parameters()):,}")
    print(f"Cheater parameters: {sum(p.numel() for p in cheater.parameters()):,}")

    if kwargs["resume_from"]:
        print(f"Resuming dynamics weights from {kwargs['resume_from']}...")
        ckpt = torch.load(kwargs["resume_from"], map_location=device, weights_only=True)
        if ckpt.get("kind") != "state_model":
            raise click.UsageError(
                f"{kwargs['resume_from']} is a {ckpt.get('kind')!r} checkpoint, not 'state_model'."
            )
        dyn_model.load_state_dict(ckpt["model"])

    opt_groups = [
        {"params": dyn_model.hamiltonian.parameters(), "lr": kwargs["lr"]},
        {"params": cheater.parameters(), "lr": kwargs["cheater_lr"]},
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

    mode = "skip-foreplay (cheater-only)" if kwargs["skip_foreplay"] else "joint (dynamics + cheater)"
    print(f"\n=== Cheater experiment: {mode} ===")
    for epoch in tqdm(range(kwargs["epochs"]), desc="Cheater", dynamic_ncols=True):
        metrics = _train_epoch_cheater(
            dyn_model=dyn_model,
            cheater=cheater,
            encoder=phase1_model.encoder,
            loader=rollout_loader,
            optimizer=optimizer,
            grad_clip=kwargs["grad_clip"],
            device=device,
            seq_len=seq_len,
            seed_ctx_len=kwargs["seed_ctx_len"],
            teacher_force_weight=kwargs["teacher_force_weight"],
            closed_loop_weight=kwargs["closed_loop_weight"],
            closed_loop_gamma=kwargs["closed_loop_gamma"],
            huber_delta=kwargs["huber_delta"],
            structural_reg_weight=kwargs["structural_reg_weight"],
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
            _log_structural_matrices(dyn_model, writer, epoch)
            for val_trajs, label in (
                (val_energy, "energy_pump"), (val_random, "random"), (val_spin, "spin"),
            ):
                if not val_trajs:
                    continue
                writer.add_scalar(
                    f"val/cheat_loss/{label}",
                    _eval_cheat_loss(cheater, phase1_model, val_trajs, device),
                    epoch,
                )
                _log_cheater_regression(
                    dyn_model=dyn_model, cheater=cheater, phase1_model=phase1_model,
                    val_trajs=val_trajs, device=device, writer=writer, epoch=epoch,
                    tag=f"val/cheater_regression/{label}",
                )
                if not kwargs["skip_foreplay"]:
                    for k, v in _eval_dynamics_loss(
                        dyn_model, cheater, phase1_model, val_trajs, device
                    ).items():
                        writer.add_scalar(f"{k}/{label}", v, epoch)
            if not kwargs["skip_foreplay"]:
                state_episodes = [
                    (
                        torch.cat(
                            [torch.atan2(s[:, 1:2], s[:, 0:1]), s[:, 2:3]], dim=-1
                        ),
                        a,
                    )
                    for (_frames, a, s) in (val_energy + val_random + val_spin)
                ]
                if state_episodes:
                    energy_fig = _plot_state_energy_landscape(
                        dyn_model, state_episodes, -_LANDSCAPE_VEL_CLIP, _LANDSCAPE_VEL_CLIP,
                        device=device,
                    )
                    writer.add_figure("val/energy_landscape", energy_fig, epoch)
                    plt.close(energy_fig)
                    grad_fig = _plot_state_gradient_magnitude_landscape(
                        dyn_model, -_LANDSCAPE_VEL_CLIP, _LANDSCAPE_VEL_CLIP, device=device,
                    )
                    writer.add_figure("val/gradient_magnitude_landscape", grad_fig, epoch)
                    plt.close(grad_fig)
                    if dyn_model._has_dissipation:
                        dissipation_fig = _plot_state_dissipation_landscape(
                            dyn_model, damping=data_cfg.get("damping", 0.0),
                            drag=data_cfg.get("drag", _DRAG_COEFF),
                            min_vel=-_LANDSCAPE_VEL_CLIP, max_vel=_LANDSCAPE_VEL_CLIP,
                            device=device,
                        )
                        writer.add_figure("val/dissipation_landscape", dissipation_fig, epoch)
                        plt.close(dissipation_fig)

        if (
            kwargs["checkpoint_every"] > 0
            and (epoch + 1) % kwargs["checkpoint_every"] == 0
            and metrics["cheater/dynamics"] < best_loss
        ):
            save_state_model(run_dir, "best", dyn_model, hparams, metrics, epoch, data_config=data_cfg)
            _save_cheater(run_dir, "best", cheater, kwargs["cheater_hidden_dim"])
            best_loss = metrics["cheater/dynamics"]

    save_state_model(run_dir, "final", dyn_model, hparams, metrics, epoch, data_config=data_cfg)
    _save_cheater(run_dir, "final", cheater, kwargs["cheater_hidden_dim"])

    writer.close()
    print("\nDone. Run: tensorboard --logdir runs")
    os._exit(0)


if __name__ == "__main__":
    main()
