"""Offline Pendulum dynamics training on a synthetic noisy high-dim latent.

Diagnostic ablation: isolates "the latent is high-dimensional and noisy" from
every other way the pixel autoencoder could be a bottleneck (nonlinear
entanglement of q/p, temporal misalignment, decoder loss shaping the
gradient, etc). Instead of encoding pixel frames, ground-truth (θ, θ̇) is
pushed through a fixed random linear map into a --projection-dim latent and
perturbed with Gaussian noise:

    h = W [θ, θ̇]ᵀ + ε,   ε ~ N(0, σ²)

W is a generic (D, 2) random matrix — it does NOT preserve a q/p block
structure, so nothing here hands the model a shortcut back to the canonical
split; whatever structure it recovers has to come from the dynamics loss
alone. ``hamilton_rl.models.HamiltonianFlowModel`` (the same dynamics model
the pixel pipeline's Phase 2 trains) is then fit directly on this synthetic
h_t, exactly as if it were a frozen pixel-encoder output.

If the model still learns clean port-Hamiltonian structure here, the pixel
pipeline's bottleneck is NOT dimension/noise — it's something structural in
the real encoder (see ``experiments/pendulum_offline.py`` phase1/phase2). If
it struggles here too, dimension/noise alone can reproduce the bottleneck.
"""

from __future__ import annotations

import os
import sys

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hamilton_rl.checkpoint import make_run_dir, save_projected_model
from hamilton_rl.models import HamiltonianFlowModel
from data.pendulum import (
    _DRAG_COEFF,
    _G,
    collect_state_data,
    collect_state_random_trajectories,
    collect_state_spin_trajectories,
    collect_state_val_trajectories,
    PendulumStateDataset,
)
from experiments.pendulum_offline import _energy_balance_loss
from experiments.pendulum_statebased_offline import (
    _LANDSCAPE_VEL_CLIP,
    _plot_state_phase_space_coverage,
    _true_hamiltonian,
)


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def make_projection(dim: int, seed: int) -> torch.Tensor:
    """Fixed (dim, 2) random linear map — generic, not q/p-block-preserving.

    Columns scaled by 1/sqrt(2) so Var[h_i] ~= Var[theta] + Var[theta_dot]
    roughly matches the scale of the raw ground-truth state, keeping
    --proj-noise-std interpretable as an absolute latent-space noise level
    comparable across --projection-dim choices.
    """
    g = torch.Generator().manual_seed(seed)
    return torch.randn(dim, 2, generator=g) / (2.0 ** 0.5)


def project(states: torch.Tensor, W: torch.Tensor, noise_std: float) -> torch.Tensor:
    """states (..., 2) -> h (..., D) = states @ W.T + noise."""
    h = states @ W.T.to(states.device)
    if noise_std > 0:
        h = h + torch.randn_like(h) * noise_std
    return h


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _train_epoch(
    model: HamiltonianFlowModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
    device: torch.device,
    seq_len: int,
    W: torch.Tensor,
    noise_std: float,
    logdet_weight: float,
    teacher_force_weight: float,
    closed_loop_weight: float,
    closed_loop_gamma: float,
    huber_delta: float,
    structural_reg_weight: float,
    energy_balance_weight: float,
) -> dict[str, float]:
    """Teacher-forced + closed-loop rollout on synthetic h_t, mirroring
    ``pendulum_offline._train_epoch_phase2`` but over full episodes (h_t is a
    stateless function of the ground-truth state, so there's no LSTM context
    to re-window)."""
    model.train()
    q_dim = model.latent_dim // 2
    totals = {
        "dynamics": 0.0, "tf": 0.0, "cl": 0.0, "logdet": 0.0,
        "struct_reg": 0.0, "energy_balance": 0.0, "q_var": 0.0, "p_var": 0.0,
    }

    for states, actions in loader:
        states = states.to(device)   # (B, T+1, 2)
        actions = actions.to(device)  # (B, T)
        B, Tp1, _ = states.shape
        T_full = Tp1 - 1

        h_all = project(states, W, noise_std)  # (B, T+1, D)
        D = h_all.shape[-1]

        s_flat, logdet_flat = model.phi.forward_with_logdet(h_all.reshape(B * Tp1, D))
        q_all = s_flat[:, :q_dim].reshape(B, Tp1, q_dim)
        p_all = s_flat[:, q_dim:].reshape(B, Tp1, q_dim)
        logdet_all = logdet_flat.reshape(B, Tp1)
        logdet_metric = logdet_all.detach().pow(2).mean()
        logdet_reg = logdet_weight * logdet_all.pow(2).mean()

        # Teacher-forced: every consecutive pair, batched as one step.
        q_tf = q_all[:, :T_full].reshape(B * T_full, q_dim)
        p_tf = p_all[:, :T_full].reshape(B * T_full, q_dim)
        a_tf = actions.reshape(B * T_full, 1)
        q_tf_next, p_tf_next = model.controlled_step(q_tf, p_tf, a_tf)
        h_tf_pred = model.decode(q_tf_next, p_tf_next)
        h_tf_target = h_all[:, 1:].reshape(B * T_full, D)
        tf_loss = F.mse_loss(h_tf_pred, h_tf_target)

        # Closed-loop rollout from t=0, curriculum-limited length.
        T = max(min(seq_len, T_full), 1)
        q, p = q_all[:, 0], p_all[:, 0]
        qs, ps = [], []
        for t in range(T):
            q, p = model.controlled_step(q, p, actions[:, t:t + 1])
            qs.append(q)
            ps.append(p)
        q_traj = torch.stack(qs, dim=1)  # (B, T, q_dim)
        p_traj = torch.stack(ps, dim=1)
        h_cl_pred = model.decode(
            q_traj.reshape(B * T, q_dim), p_traj.reshape(B * T, q_dim)
        ).reshape(B, T, D)
        h_cl_target = h_all[:, 1:1 + T]
        if huber_delta > 0:
            elem_err = 2.0 * F.huber_loss(
                h_cl_pred, h_cl_target, reduction="none", delta=huber_delta
            )
        else:
            elem_err = (h_cl_pred - h_cl_target).pow(2)
        per_step_loss = elem_err.mean(dim=(0, 2))
        step_weights = closed_loop_gamma ** torch.arange(T, device=device, dtype=per_step_loss.dtype)
        cl_loss = (per_step_loss * step_weights).sum() / step_weights.sum()

        loss = logdet_reg + teacher_force_weight * tf_loss + closed_loop_weight * cl_loss

        struct_reg = torch.zeros((), device=device)
        if structural_reg_weight > 0 and model.r_parameters():
            if model.state_dep_r:
                z_r = torch.cat([q_all, p_all], dim=-1).reshape(-1, D).detach()
                struct_reg = model.get_R_pp(z_r).pow(2).sum(dim=(-2, -1)).mean()
            else:
                struct_reg = model.get_R_pp().pow(2).sum()
            loss = loss + structural_reg_weight * struct_reg

        eb_loss = torch.zeros((), device=device)
        if energy_balance_weight > 0:
            eb_loss = _energy_balance_loss(model, q_all, p_all, actions)
            loss = loss + energy_balance_weight * eb_loss

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        totals["dynamics"] += loss.item()
        totals["tf"] += tf_loss.item()
        totals["cl"] += cl_loss.item()
        totals["logdet"] += logdet_metric.item()
        totals["struct_reg"] += float(struct_reg.detach())
        totals["energy_balance"] += float(eb_loss.detach())
        with torch.no_grad():
            totals["q_var"] += q_traj.detach().reshape(-1, q_dim).var(dim=0).mean().item()
            totals["p_var"] += p_traj.detach().reshape(-1, q_dim).var(dim=0).mean().item()

    n = len(loader)
    return {
        "train/loss": totals["dynamics"] / n,
        "train/tf_loss": totals["tf"] / n,
        "train/cl_loss": totals["cl"] / n,
        "train/logdet_reg": totals["logdet"] / n,
        "train/struct_reg": totals["struct_reg"] / n,
        "train/energy_balance": totals["energy_balance"] / n,
        "train/q_var": totals["q_var"] / n,
        "train/p_var": totals["p_var"] / n,
    }


# ---------------------------------------------------------------------------
# Validation / logging helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def _eval_h_loss(
    model: HamiltonianFlowModel,
    val_trajs: list,
    W: torch.Tensor,
    noise_std: float,
    device: torch.device,
) -> float:
    """Closed-loop rollout MSE in h-space over held-out trajectories."""
    model.eval()
    total = 0.0
    for states, actions in val_trajs:
        states = states.to(device)
        actions = actions.to(device)
        T = len(actions)
        h_all = project(states, W, noise_std)
        q, p = model.encode(h_all[0:1])
        loss = 0.0
        for t in range(T):
            u = actions[t].reshape(1, 1)
            q, p = model.controlled_step(q, p, u)
            h_pred = model.decode(q, p)
            loss += F.mse_loss(h_pred, h_all[t + 1:t + 2]).item()
        total += loss / T
    return total / len(val_trajs)


@torch.no_grad()
def _log_phase_space_regression(
    model: HamiltonianFlowModel,
    val_trajs: list,
    W: torch.Tensor,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    tag: str = "val/phase_space_regression",
) -> float | None:
    """Probe whether closed-loop rolled-out (q, p) linearly encodes true state.

    Adapted from ``pendulum_offline._log_phase_space_regression_phase2``:
    rolls the learned dynamics forward in closed loop from each held-out
    trajectory's true initial state (projected noise-free — the probe should
    ask what the dynamics recovered, not re-introduce noise into the seed),
    then fits (q, p) -> [cos θ, sin θ, θ̇] by OLS on half the trajectories and
    reports held-out R². Returns the mean R² across the three targets (for
    the scalar curve), or None if there isn't enough data.
    """
    model.eval()
    q_dim = model.latent_dim // 2
    N = len(val_trajs)
    if N < 2:
        return None
    T_full = val_trajs[0][1].shape[0]
    n_steps = T_full
    if n_steps <= 0:
        return None

    states_all = torch.stack([s for s, _ in val_trajs]).to(device)   # (N, T+1, 2)
    actions_all = torch.stack([a for _, a in val_trajs]).to(device)  # (N, T)
    h0 = project(states_all[:, 0], W, 0.0)
    q, p = model.encode(h0)
    qp_steps = []
    for t in range(n_steps):
        q, p = model.controlled_step(q, p, actions_all[:, t:t + 1])
        qp_steps.append(torch.cat([q, p], dim=-1))
    all_qp = torch.stack(qp_steps, dim=1).cpu()  # (N, n_steps, latent_dim)

    theta = states_all[:, 1:1 + n_steps, 0].cpu()
    theta_dot = states_all[:, 1:1 + n_steps, 1].cpu()
    all_st = torch.stack([torch.cos(theta), torch.sin(theta), theta_dot], dim=-1)  # (N, n_steps, 3)

    D = all_qp.shape[-1]
    train_qp = all_qp[0::2].reshape(-1, D)
    train_st = all_st[0::2].reshape(-1, 3)
    val_qp = all_qp[1::2].reshape(-1, D)
    val_st = all_st[1::2].reshape(-1, 3)
    val_idx = torch.arange(n_steps, dtype=torch.float32).repeat(all_qp[1::2].shape[0]).numpy()

    A = torch.linalg.lstsq(train_qp, train_st).solution
    st_pred = (val_qp @ A).numpy()
    st_true = val_st.numpy()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    sc = None
    r2s = []
    for i, name in enumerate(["cos(θ)", "sin(θ)", "θ̇ (rad/s)"]):
        true_i, pred_i = st_true[:, i], st_pred[:, i]
        sc = axes[i].scatter(true_i, pred_i, c=val_idx, cmap="viridis", s=2, alpha=0.4, linewidths=0)
        lo, hi = min(true_i.min(), pred_i.min()), max(true_i.max(), pred_i.max())
        axes[i].plot([lo, hi], [lo, hi], "r--", linewidth=0.8)
        axes[i].set_xlabel(f"True {name}")
        axes[i].set_ylabel(f"Predicted {name}")
        ss_res = ((true_i - pred_i) ** 2).sum()
        ss_tot = ((true_i - true_i.mean()) ** 2).sum()
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        r2s.append(r2)
        axes[i].set_title(f"{name}  R²={r2:.3f}")
    fig.colorbar(sc, ax=axes, label="rollout step", fraction=0.03, pad=0.02)
    fig.suptitle(f"Closed-loop (q,p) → state regression, held-out trajectories (epoch {epoch + 1})")
    writer.add_figure(tag, fig, epoch)
    plt.close(fig)
    return float(np.mean(r2s))


@torch.no_grad()
def _plot_energy_landscape(
    model: HamiltonianFlowModel,
    W: torch.Tensor,
    min_vel: float,
    max_vel: float,
    landscape_resolution: int = 200,
    device: torch.device | None = None,
) -> plt.Figure:
    """True energy vs. learned H over ground-truth phase space.

    Grid points are projected through the same fixed, noise-free W (the
    landscape needs a stable input to be legible), encoded through phi, and
    H is evaluated there — mirrors
    ``pendulum_statebased_offline._plot_state_energy_landscape``.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    q_dim = model.latent_dim // 2

    theta_dense = torch.linspace(-torch.pi, torch.pi, landscape_resolution)
    theta_dot_dense = torch.linspace(min_vel, max_vel, landscape_resolution)
    grid_theta, grid_theta_dot = torch.meshgrid(theta_dense, theta_dot_dense, indexing="xy")
    H_true_dense = (0.5 * grid_theta_dot**2 + 1.5 * _G * (1.0 + torch.cos(grid_theta))).numpy()

    states_flat = torch.stack([grid_theta.reshape(-1), grid_theta_dot.reshape(-1)], dim=-1).to(device)
    h_flat = project(states_flat, W, 0.0)
    q, p = model.encode(h_flat)
    H_learned_dense = model.hamiltonian(q, p).reshape(grid_theta.shape).cpu().numpy()

    extent = [-np.pi, np.pi, min_vel, max_vel]
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))

    im0 = axes[0].imshow(H_true_dense, origin="lower", aspect="auto", extent=extent, cmap="viridis")
    axes[0].set_title("Ground truth")
    fig.colorbar(im0, ax=axes[0], label="H_true", pad=0.02)

    im1 = axes[1].imshow(H_learned_dense, origin="lower", aspect="auto", extent=extent, cmap="viridis")
    axes[1].set_title("Learned H (via W + φ)")
    fig.colorbar(im1, ax=axes[1], label="H_learned", pad=0.02)

    for ax in axes[:2]:
        ax.set_xlabel("θ (rad)")
        ax.set_ylabel("θ̇ (rad/s)")
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(min_vel, max_vel)

    H_true_flat = H_true_dense.ravel()
    H_learned_flat = H_learned_dense.ravel()
    r = np.corrcoef(H_true_flat, H_learned_flat)[0, 1]

    rng = np.random.default_rng(0)
    n_scatter = min(3000, H_true_flat.size)
    scatter_idx = rng.choice(H_true_flat.size, size=n_scatter, replace=False)
    slope, intercept = np.polyfit(H_true_flat, H_learned_flat, 1)
    fit_x = np.array([H_true_flat.min(), H_true_flat.max()])
    ax2 = axes[2]
    ax2.scatter(H_true_flat[scatter_idx], H_learned_flat[scatter_idx], s=4, alpha=0.2)
    ax2.plot(
        fit_x, slope * fit_x + intercept, color="crimson",
        label=f"fit: y = {slope:.2f}x + {intercept:.2f}\nR² = {r**2:.3f}",
    )
    ax2.set_xlabel("H_true")
    ax2.set_ylabel("H_learned")
    ax2.set_title("True vs. learned energy")
    ax2.legend(loc="best", fontsize=9)

    fig.suptitle(f"True energy vs. learned H (projection_dim={model.latent_dim}), Pearson r={r:.3f}")
    fig.tight_layout()
    return fig


@torch.no_grad()
def _plot_dissipation_landscape(
    model: HamiltonianFlowModel,
    W: torch.Tensor,
    damping: float,
    drag: float,
    min_vel: float,
    max_vel: float,
    landscape_resolution: int = 200,
    device: torch.device | None = None,
) -> plt.Figure:
    """Learned vs. true energy-dissipation rate over ground-truth phase space.

    Mirrors ``pendulum_statebased_offline._plot_state_dissipation_landscape``.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    theta_dense = torch.linspace(-torch.pi, torch.pi, landscape_resolution)
    theta_dot_dense = torch.linspace(min_vel, max_vel, landscape_resolution)
    grid_theta, grid_theta_dot = torch.meshgrid(theta_dense, theta_dot_dense, indexing="xy")

    states_flat = torch.stack([grid_theta.reshape(-1), grid_theta_dot.reshape(-1)], dim=-1).to(device)
    h_flat = project(states_flat, W, 0.0)
    with torch.enable_grad():
        q, p = model.encode(h_flat)
        q = q.detach().requires_grad_(True)
        p = p.detach().requires_grad_(True)
        H_val = model.hamiltonian(q, p).sum()
        grad_q, grad_p = torch.autograd.grad(H_val, [q, p])
    with torch.no_grad():
        z = torch.cat([q, p], dim=-1)
        Rg = model._apply_R_pp(z, grad_p)
        rate_learned = (grad_p * Rg).sum(dim=-1).reshape(grid_theta.shape).cpu().numpy()

    rate_true = (damping * grid_theta_dot**2 + drag * grid_theta_dot.abs() ** 3).numpy()

    extent = [-np.pi, np.pi, min_vel, max_vel]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, rate, label in (
        (axes[0], rate_true, "true rate: damping·θ̇² + drag·|θ̇|³"),
        (axes[1], rate_learned, "learned rate: ∇ₚHᵀ R_pp ∇ₚH"),
    ):
        im = ax.imshow(rate, origin="lower", aspect="auto", extent=extent, cmap="magma")
        fig.colorbar(im, ax=ax, label="−Ḣ", pad=0.02)
        ax.set_xlabel("θ (rad)")
        ax.set_ylabel("θ̇ (rad/s)")
        ax.set_title(label)

    eps = 1e-8
    log_true = np.log10(rate_true + eps)
    log_learned = np.log10(np.maximum(rate_learned, 0.0) + eps)
    r = np.corrcoef(log_true.ravel(), log_learned.ravel())[0, 1] if np.ptp(rate_true) > 0 else float("nan")
    axes[2].scatter(log_true.ravel(), log_learned.ravel(), s=3, alpha=0.2)
    axes[2].set_xlabel("log₁₀ true rate")
    axes[2].set_ylabel("log₁₀ learned rate")
    axes[2].set_title(f"log-rate agreement, Pearson r={r:.3f}")

    fig.suptitle("Energy-dissipation rate: true vs. learned R")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
# data
@click.option("--n-episodes", type=int, default=200, show_default=True)
@click.option("--epsilon", type=float, default=0.1, show_default=True,
              help="Fraction of steps with random uniform action")
@click.option("--energy-k", type=float, default=1.0, show_default=True,
              help="Gain for energy-pumping controller")
@click.option("--max-steps", type=int, default=200, show_default=True,
              help="Steps per episode")
@click.option("--damping", type=float, default=0.0, show_default=True,
              help="Linear viscous damping (theta_dot *= exp(-b*dt) per step)")
# projection
@click.option("--projection-dim", type=int, default=32, show_default=True,
              help="Dimension D of the synthetic noisy latent h = W[theta,theta_dot] + noise")
@click.option("--proj-noise-std", type=float, default=0.0, show_default=True,
              help="Std of Gaussian noise added to h (absolute, in the ~unit scale set by W)")
@click.option("--proj-seed", type=int, default=0, show_default=True,
              help="Seed for the fixed random projection matrix W")
# model
@click.option("--hidden-dim", type=int, default=256, show_default=True,
              help="Width of Hamiltonian MLP hidden layers (via MLPHamiltonianNet default)")
@click.option("--dt", type=float, default=0.05, show_default=True)
@click.option("--separable/--no-separable", default=True, show_default=True)
@click.option("--h-source", type=click.Choice(["learned", "canonical"]), default="learned",
              show_default=True)
@click.option("--r-source", type=click.Choice(["learned", "fixed_damping", "canonical"]),
              default="learned", show_default=True)
@click.option("--b-source", type=click.Choice(["learned", "fixed_ones", "canonical"]),
              default="learned", show_default=True)
@click.option("--drag", type=float, default=_DRAG_COEFF, show_default=True)
@click.option("--quadratic-t/--no-quadratic-t", default=True, show_default=True)
@click.option("--state-dep-r", is_flag=True, default=False, show_default=True)
@click.option("--integrator", type=click.Choice(["rk4", "leapfrog"]), default="leapfrog",
              show_default=True)
# training
@click.option("--epochs", type=int, default=3000, show_default=True)
@click.option("--batch-size", type=int, default=16, show_default=True)
@click.option("--h-lr", type=float, default=1e-4, show_default=True)
@click.option("--structural-lr", type=float, default=1e-2, show_default=True)
@click.option("--grad-clip", type=float, default=1.0, show_default=True)
@click.option("--logdet-weight", type=float, default=1e-3, show_default=True)
@click.option("--teacher-force-weight", type=float, default=1.0, show_default=True)
@click.option("--closed-loop-weight", type=float, default=1.0, show_default=True)
@click.option("--closed-loop-gamma", type=float, default=1.0, show_default=True)
@click.option("--huber-delta", type=float, default=0.2, show_default=True)
@click.option("--structural-reg-weight", type=float, default=0.0, show_default=True)
@click.option("--energy-balance-weight", type=float, default=1.0, show_default=True)
@click.option("--ema-alpha", type=float, default=0.99, show_default=True)
@click.option("--seq-len-start", type=int, default=5, show_default=True)
@click.option("--seq-len-advance-threshold", type=float, default=0.005, show_default=True)
# logging
@click.option("--log-every", type=int, default=5, show_default=True)
@click.option("--val-every", type=int, default=10, show_default=True)
@click.option("--n-val-episodes", type=int, default=-1, show_default=True)
@click.option("--val-max-steps", type=int, default=0, show_default=True)
@click.option("--checkpoint-every", type=int, default=10, show_default=True)
def main(**kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if kwargs["projection_dim"] < 2 or kwargs["projection_dim"] % 2 != 0:
        raise ValueError("--projection-dim must be a positive even integer")

    writer = SummaryWriter(comment="_pendulum_projected_offline")
    run_dir = make_run_dir("pendulum_projected_offline")

    n_val_episodes = kwargs["n_val_episodes"]
    if n_val_episodes < 0:
        n_val_episodes = kwargs["n_episodes"] // 2
    n_val = n_val_episodes if kwargs["val_every"] > 0 else 0
    val_steps = kwargs["val_max_steps"] or kwargs["max_steps"] * 2

    print(f"\nCollecting {kwargs['n_episodes']} train episodes...")
    train_episodes = collect_state_data(
        n_episodes=kwargs["n_episodes"],
        epsilon=kwargs["epsilon"],
        energy_k=kwargs["energy_k"],
        max_steps=kwargs["max_steps"],
        damping=kwargs["damping"],
    )

    coverage_fig = _plot_state_phase_space_coverage(train_episodes)
    writer.add_figure("data/phase_space_coverage", coverage_fig, 0)
    plt.close(coverage_fig)

    min_vel, max_vel = -_LANDSCAPE_VEL_CLIP, _LANDSCAPE_VEL_CLIP

    val_energy, val_random, val_spin = [], [], []
    if n_val > 0:
        print(f"Collecting {n_val} val episodes per type ({val_steps} steps each)...")
        val_energy = collect_state_val_trajectories(
            n_episodes=n_val, max_steps=val_steps, energy_k=kwargs["energy_k"], damping=kwargs["damping"],
        )
        val_random = collect_state_random_trajectories(
            n_episodes=n_val, max_steps=val_steps, damping=kwargs["damping"],
        )
        val_spin = collect_state_spin_trajectories(
            n_episodes=n_val, max_steps=val_steps, damping=kwargs["damping"],
        )

    dataset = PendulumStateDataset(train_episodes)
    loader = DataLoader(
        dataset, batch_size=kwargs["batch_size"], shuffle=True, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    print(f"Dataset: {len(dataset)} episodes")

    W = make_projection(kwargs["projection_dim"], kwargs["proj_seed"]).to(device)
    print(f"Projection: R^2 -> R^{kwargs['projection_dim']} (seed={kwargs['proj_seed']}, "
          f"noise_std={kwargs['proj_noise_std']})")

    model = HamiltonianFlowModel(
        latent_dim=kwargs["projection_dim"],
        control_dim=1,
        separable=kwargs["separable"],
        h_source=kwargs["h_source"],
        r_source=kwargs["r_source"],
        b_source=kwargs["b_source"],
        dt=kwargs["dt"],
        damping=kwargs["damping"],
        drag=kwargs["drag"],
        integrator=kwargs["integrator"],
        quadratic_t=kwargs["quadratic_t"],
        state_dep_r=kwargs["state_dep_r"],
    ).to(device)
    print(f"Integrator: {model.integrator}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    hparams = dict(kwargs)
    data_config = {
        "n_episodes": kwargs["n_episodes"],
        "epsilon": kwargs["epsilon"],
        "energy_k": kwargs["energy_k"],
        "max_steps": kwargs["max_steps"],
        "damping": kwargs["damping"],
        "projection_dim": kwargs["projection_dim"],
        "proj_noise_std": kwargs["proj_noise_std"],
        "proj_seed": kwargs["proj_seed"],
    }

    groups = [{"params": model.phi.parameters(), "lr": kwargs["h_lr"]}]
    h_params = list(model.hamiltonian.parameters())
    if h_params:
        groups.append({"params": h_params, "lr": kwargs["h_lr"]})
    struct_params = model.structural_parameters()
    if struct_params:
        groups.append({"params": struct_params, "lr": kwargs["structural_lr"]})
    optimizer = torch.optim.Adam(groups)
    best_loss = float("inf")

    full_seq_len = train_episodes[0][1].shape[0]
    seq_len = kwargs["seq_len_start"]
    ema_loss = None

    print("\n=== Training ===")
    for epoch in tqdm(range(kwargs["epochs"]), desc="Training", dynamic_ncols=True):
        metrics = _train_epoch(
            model=model, loader=loader, optimizer=optimizer, grad_clip=kwargs["grad_clip"],
            device=device, seq_len=seq_len, W=W, noise_std=kwargs["proj_noise_std"],
            logdet_weight=kwargs["logdet_weight"],
            teacher_force_weight=kwargs["teacher_force_weight"],
            closed_loop_weight=kwargs["closed_loop_weight"],
            closed_loop_gamma=kwargs["closed_loop_gamma"],
            huber_delta=kwargs["huber_delta"],
            structural_reg_weight=kwargs["structural_reg_weight"],
            energy_balance_weight=kwargs["energy_balance_weight"],
        )

        alpha = kwargs["ema_alpha"]
        ema_loss = (
            metrics["train/loss"] if ema_loss is None
            else alpha * ema_loss + (1.0 - alpha) * metrics["train/loss"]
        )
        if ema_loss < kwargs["seq_len_advance_threshold"] and seq_len < full_seq_len:
            seq_len += 1

        if (epoch + 1) % kwargs["log_every"] == 0:
            for k, v in metrics.items():
                writer.add_scalar(k, v, epoch)
            writer.add_scalar("train/seq_len", seq_len, epoch)
            writer.add_scalar("train/ema_loss", ema_loss, epoch)
            tqdm.write(
                f"  epoch {epoch + 1:4d}  seq_len={seq_len:3d}"
                f"  loss={metrics['train/loss']:.6f}  ema={ema_loss:.6f}"
                f"  tf={metrics['train/tf_loss']:.6f}  cl={metrics['train/cl_loss']:.6f}"
            )

        if kwargs["val_every"] > 0 and (epoch + 1) % kwargs["val_every"] == 0:
            energy_fig = _plot_energy_landscape(model, W, min_vel, max_vel, device=device)
            writer.add_figure("val/energy_landscape", energy_fig, epoch)
            plt.close(energy_fig)
            if model._has_dissipation:
                dissipation_fig = _plot_dissipation_landscape(
                    model, W, damping=kwargs["damping"], drag=_DRAG_COEFF,
                    min_vel=min_vel, max_vel=max_vel, device=device,
                )
                writer.add_figure("val/dissipation_landscape", dissipation_fig, epoch)
                plt.close(dissipation_fig)
            for val_trajs, label in (
                (val_energy, "energy_pump"), (val_random, "random"), (val_spin, "spin"),
            ):
                if not val_trajs:
                    continue
                writer.add_scalar(
                    f"val/h_loss/{label}", _eval_h_loss(model, val_trajs, W, kwargs["proj_noise_std"], device), epoch,
                )
            if val_energy:
                r2 = _log_phase_space_regression(
                    model, val_energy, W, device, writer, epoch, tag="val/phase_space_regression/energy_pump",
                )
                if r2 is not None:
                    writer.add_scalar("val/phase_space_regression_r2/energy_pump", r2, epoch)
            if val_random:
                r2 = _log_phase_space_regression(
                    model, val_random, W, device, writer, epoch, tag="val/phase_space_regression/random",
                )
                if r2 is not None:
                    writer.add_scalar("val/phase_space_regression_r2/random", r2, epoch)

        if (
            kwargs["checkpoint_every"] > 0
            and (epoch + 1) % kwargs["checkpoint_every"] == 0
            and metrics["train/loss"] < best_loss
        ):
            save_projected_model(run_dir, "best", model, W.cpu(), hparams, metrics, epoch, data_config=data_config)
            best_loss = metrics["train/loss"]

    save_projected_model(run_dir, "final", model, W.cpu(), hparams, metrics, kwargs["epochs"] - 1, data_config=data_config)
    writer.close()
    print("\nDone. Run: tensorboard --logdir runs")
    os._exit(0)


if __name__ == "__main__":
    main()
