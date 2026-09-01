"""Offline Pendulum world-model training directly in ground-truth phase space.

Skips pixel encoding/decoding entirely.  The model (``hamilton_rl.models.StatePHGN``)
operates on the ground-truth state (θ, θ̇) ∈ ℝ² treated as the Hamiltonian phase space:

    q = θ    ∈ ℝ¹   (angle)
    p = θ̇   ∈ ℝ¹   (angular velocity)

The controlled port-Hamiltonian ODE

    dz/dt = (J − R(z)) ∇H(z) + [0, b] u

is integrated with RK4 or leapfrog for T steps.  Training loss is MSE between
the predicted and ground-truth next states.  Checkpoints are saved with
``save_state_model`` — a single self-describing ``.pt`` (see
``hamilton_rl.checkpoint``).
"""

from __future__ import annotations

import os
import sys

import click
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hamilton_rl.checkpoint import make_run_dir, save_state_model
from hamilton_rl.models import StatePHGN
from data.pendulum import (
    _DRAG_COEFF,
    _G,
    collect_state_data,
    collect_state_random_trajectories,
    collect_state_spin_trajectories,
    collect_state_val_trajectories,
    PendulumStateDataset,
)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _train_epoch(
    model: StatePHGN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
    device: torch.device,
    seq_len: int,
) -> dict[str, float]:
    model.train()
    total_loss = total_q_var = total_p_var = 0.0

    for states, actions in loader:
        states = states.to(device)  # (B, T+1, 3)
        actions = actions.to(device)  # (B, T)

        q, p = model.split(states[:, 0])

        loss = torch.zeros(1, device=device)
        qs, ps = [q], [p]
        for t in range(seq_len):
            u = actions[:, t].unsqueeze(-1)
            q, p = model.step(q, p, u)
            pred = torch.cat([q, p], dim=-1)
            loss = loss + F.mse_loss(pred, states[:, t + 1])
            qs.append(q)
            ps.append(p)
        loss = loss / seq_len

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        with torch.no_grad():
            qs_t = torch.stack([x.detach() for x in qs], dim=1)
            ps_t = torch.stack([x.detach() for x in ps], dim=1)
            total_q_var += (
                qs_t.reshape(-1, model.Q_DIM).var(dim=0).mean().item()
            )
            total_p_var += (
                ps_t.reshape(-1, model.P_DIM).var(dim=0).mean().item()
            )

    n = len(loader)
    return {
        "train/loss": total_loss / n,
        "train/q_var": total_q_var / n,
        "train/p_var": total_p_var / n,
    }


# ---------------------------------------------------------------------------
# Validation / logging helpers
# ---------------------------------------------------------------------------


def _true_hamiltonian(states: torch.Tensor) -> np.ndarray:
    """H = 0.5 θ̇² + 1.5 g (1 + cos θ) from (T, 2) states.

    The 1.5·g potential matches the env's EOM ṗ = 1.5·g·sin θ under the
    canonical convention T = θ̇²/2 (see data/pendulum.py).
    """
    theta = states[:, 0].numpy()
    theta_dot = states[:, 1].numpy()
    return 0.5 * theta_dot**2 + 1.5 * _G * (1.0 + np.cos(theta))


def _plot_state_phase_space_coverage(
    episodes: list,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Scatter every visited (θ, θ̇) state in ``episodes`` to eyeball state-space coverage.

    Mirrors ``_plot_phase_space_coverage`` in pendulum_offline.py, but reads
    states directly as (θ, θ̇) rather than recovering θ from (cosθ, sinθ).
    """
    theta_all = torch.cat([states[:, 0] for states, _ in episodes]).numpy()
    theta_dot_all = torch.cat([states[:, 1] for states, _ in episodes]).numpy()

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


def _state_vel_range(episodes: list, margin: float = 1.1) -> tuple[float, float]:
    """Symmetric θ̇ plotting range derived from collected episode data."""
    max_vel = max(states[:, 1].abs().max().item() for states, _ in episodes) * margin
    return -max_vel, max_vel


@torch.no_grad()
def _plot_state_energy_landscape(
    model: StatePHGN,
    episodes: list,
    min_vel: float,
    max_vel: float,
    landscape_resolution: int = 200,
    device: torch.device | None = None,
) -> plt.Figure:
    """Compare learned H(q, p) against true pendulum energy over phase space.

    Unlike the pixel pipeline's ``_plot_learned_energy_landscape``, no
    grid-episode collection, encoder pass, or griddata interpolation is
    needed here: (θ, θ̇) *is* the phase space, so H can be evaluated
    directly at every point of a dense grid.

    Three panels: ground-truth H from its closed form, learned H evaluated
    on that same grid (own color scale — H is only fit through its
    gradient, so it's identified up to an unknown affine offset/scale), and
    the learned panel again with the training data's (θ, θ̇) coverage
    overlaid, since agreement only means anything where the model actually
    saw data.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    theta_dense = torch.linspace(-torch.pi, torch.pi, landscape_resolution)
    theta_dot_dense = torch.linspace(min_vel, max_vel, landscape_resolution)
    grid_theta, grid_theta_dot = torch.meshgrid(theta_dense, theta_dot_dense, indexing="xy")
    H_true_dense = (0.5 * grid_theta_dot**2 + 1.5 * _G * (1.0 + torch.cos(grid_theta))).numpy()

    q_flat = grid_theta.reshape(-1, 1).to(device)
    p_flat = grid_theta_dot.reshape(-1, 1).to(device)
    H_learned_dense = model.H(q_flat, p_flat).reshape(grid_theta.shape).cpu().numpy()

    theta_data = torch.cat([states[:, 0] for states, _ in episodes]).numpy()
    theta_dot_data = torch.cat([states[:, 1] for states, _ in episodes]).numpy()

    extent = [-np.pi, np.pi, min_vel, max_vel]
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharex=True, sharey=True)

    im0 = axes[0].imshow(H_true_dense, origin="lower", aspect="auto", extent=extent, cmap="viridis")
    axes[0].set_title("Ground truth")
    fig.colorbar(im0, ax=axes[0], label="H_true", pad=0.02)

    im1 = axes[1].imshow(H_learned_dense, origin="lower", aspect="auto", extent=extent, cmap="viridis")
    axes[1].set_title("Learned H")
    fig.colorbar(im1, ax=axes[1], label="H_learned", pad=0.02)

    im2 = axes[2].imshow(H_learned_dense, origin="lower", aspect="auto", extent=extent, cmap="viridis")
    axes[2].scatter(theta_data, theta_dot_data, s=1, alpha=0.15, color="white")
    axes[2].set_title("Learned H + training data coverage")
    fig.colorbar(im2, ax=axes[2], label="H_learned", pad=0.02)

    for ax in axes:
        ax.set_xlabel("θ (rad)")
    axes[0].set_ylabel("θ̇ (rad/s)")

    r = np.corrcoef(H_true_dense.ravel(), H_learned_dense.ravel())[0, 1]
    fig.suptitle(f"True energy vs. learned H, Pearson r={r:.3f}")
    fig.tight_layout()

    return fig


def _plot_state_gradient_magnitude_landscape(
    model: StatePHGN,
    min_vel: float,
    max_vel: float,
    landscape_resolution: int = 200,
    device: torch.device | None = None,
) -> plt.Figure:
    """Compare ‖∇H_true‖ against ‖∇H_learned‖ over phase space.

    See ``_plot_state_energy_landscape`` for why no grid-episode collection
    or interpolation is needed here, unlike the pixel version's
    ``_plot_gradient_magnitude_landscape``. ∇H is what the dynamics actually
    consume, so a flat/small learned gradient where the true one is large
    indicates real underfitting rather than just an unidentified offset on H.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    theta_dense = torch.linspace(-torch.pi, torch.pi, landscape_resolution)
    theta_dot_dense = torch.linspace(min_vel, max_vel, landscape_resolution)
    grid_theta, grid_theta_dot = torch.meshgrid(theta_dense, theta_dot_dense, indexing="xy")

    with torch.enable_grad():
        q_flat = grid_theta.reshape(-1, 1).to(device).requires_grad_(True)
        p_flat = grid_theta_dot.reshape(-1, 1).to(device).requires_grad_(True)
        H_val = model.H(q_flat, p_flat).sum()
        grad_q, grad_p = torch.autograd.grad(H_val, [q_flat, p_flat])
    grad_mag_learned = (
        torch.sqrt(grad_q**2 + grad_p**2).reshape(grid_theta.shape).detach().cpu().numpy()
    )

    grad_mag_true = torch.sqrt(
        (1.5 * _G * torch.sin(grid_theta)) ** 2 + grid_theta_dot**2
    ).numpy()

    extent = [-np.pi, np.pi, min_vel, max_vel]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    im0 = axes[0].imshow(grad_mag_true, origin="lower", aspect="auto", extent=extent, cmap="viridis")
    axes[0].set_title("Ground truth")
    fig.colorbar(im0, ax=axes[0], label="‖∇H_true‖", pad=0.02)

    im1 = axes[1].imshow(grad_mag_learned, origin="lower", aspect="auto", extent=extent, cmap="viridis")
    axes[1].set_title("Learned ‖∇H‖")
    fig.colorbar(im1, ax=axes[1], label="‖∇H_learned‖", pad=0.02)

    for ax in axes:
        ax.set_xlabel("θ (rad)")
    axes[0].set_ylabel("θ̇ (rad/s)")

    r = np.corrcoef(grad_mag_true.ravel(), grad_mag_learned.ravel())[0, 1]
    fig.suptitle(f"‖∇H_true‖ vs. ‖∇H_learned‖, Pearson r={r:.3f}")
    fig.tight_layout()

    return fig


def _plot_state_dissipation_landscape(
    model: StatePHGN,
    damping: float,
    drag: float,
    min_vel: float,
    max_vel: float,
    landscape_resolution: int = 200,
    device: torch.device | None = None,
) -> plt.Figure:
    """Compare the learned energy-dissipation rate against the true one over phase space.

    See ``_plot_dissipation_landscape`` (pixel version) for the underlying
    formulas — −Ḣ = ∇ₚHᵀ R_pp(z) ∇ₚH for the learned rate, vs. the env's true
    damping·θ̇² + drag·|θ̇|³. No grid-episode collection is needed: R_pp(z)
    and ∇H are evaluated directly on the dense (θ, θ̇) grid.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    theta_dense = torch.linspace(-torch.pi, torch.pi, landscape_resolution)
    theta_dot_dense = torch.linspace(min_vel, max_vel, landscape_resolution)
    grid_theta, grid_theta_dot = torch.meshgrid(theta_dense, theta_dot_dense, indexing="xy")

    with torch.enable_grad():
        q_flat = grid_theta.reshape(-1, 1).to(device).requires_grad_(True)
        p_flat = grid_theta_dot.reshape(-1, 1).to(device).requires_grad_(True)
        H_val = model.H(q_flat, p_flat).sum()
        grad_p = torch.autograd.grad(H_val, p_flat)[0]
    with torch.no_grad():
        z = model._r_input(q_flat.detach(), p_flat.detach())
        R_pp = model.get_R_pp(z)
        Rg = model._apply_R_pp(R_pp, grad_p)
        rate_learned = (grad_p * Rg).sum(dim=-1).reshape(grid_theta.shape).cpu().numpy()

    rate_true = (damping * grid_theta_dot**2 + drag * grid_theta_dot.abs() ** 3).numpy()

    extent = [-np.pi, np.pi, min_vel, max_vel]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, rate, label in (
        (axes[0], rate_true, "true rate: damping·θ̇² + drag·|θ̇|³"),
        (axes[1], rate_learned, "learned rate: ∇ₚHᵀ R_pp(z) ∇ₚH"),
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


@torch.no_grad()
def _eval_loss(
    model: StatePHGN,
    val_trajs: list,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    for states, actions in val_trajs:
        states = states.to(device)
        actions = actions.to(device)
        T = len(actions)
        q, p = model.split(states[0:1])
        loss = 0.0
        for t in range(T):
            u = actions[t].reshape(1, 1)
            q, p = model.step(q, p, u)
            pred = torch.cat([q, p], dim=-1)
            loss += F.mse_loss(pred, states[t + 1 : t + 2]).item()
        total += loss / T
    return total / len(val_trajs)


@torch.no_grad()
def _log_hamiltonian_comparison(
    model: StatePHGN,
    val_traj: tuple,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    tag: str = "val/hamiltonian",
) -> None:
    """Log H values and dH breakdown for one validation trajectory."""
    model.eval()
    states, actions = val_traj  # (T+1, 2), (T,)

    states_dev = states.to(device)
    q_all, p_all = model.split(
        states_dev
    )  # each step is indexed manually below

    # Model H along ground-truth trajectory
    model_H = [
        model.H(
            states_dev[t : t + 1, : model.Q_DIM],
            states_dev[t : t + 1, model.Q_DIM :],
        ).item()
        for t in range(len(states))
    ]
    true_H = _true_hamiltonian(states)

    R = model.get_R()
    dH_model, dH_pred, dH_true = [], [], []

    for t in range(len(actions)):
        u = actions[t].reshape(1, 1).to(device)
        z = states_dev[t : t + 1].detach().requires_grad_(True)
        with torch.enable_grad():
            H_val = model.H(z[:, : model.Q_DIM], z[:, model.Q_DIM :]).sum()
            grad_H = torch.autograd.grad(H_val, z)[0]

        Bu_full = torch.cat(
            [
                torch.zeros(1, model.Q_DIM, device=device),
                u @ model.get_b().T,
            ],
            dim=-1,
        )

        dH_step = (
            -model.dt * (grad_H @ R * grad_H).sum(-1).item()
            + model.dt * (grad_H * Bu_full).sum(-1).item()
        )
        dH_pred.append(dH_step)
        dH_model.append(model_H[t + 1] - model_H[t])
        dH_true.append(float(true_H[t + 1] - true_H[t]))

    t_axis = np.arange(len(true_H))
    dh_axis = np.arange(1, len(true_H))

    fig_h, ax_h = plt.subplots(figsize=(8, 3))
    ax_h.plot(
        t_axis, true_H, label="Ground-truth H", linewidth=1.5, color="tab:blue"
    )
    ax_h.plot(
        t_axis,
        model_H,
        label="Learned H",
        linewidth=1.5,
        linestyle="--",
        color="tab:orange",
    )
    ax_h.axhline(
        _G * 2, color="grey", linestyle=":", linewidth=1, label="H*=20"
    )
    ax_h.set_xlabel("Step")
    ax_h.set_ylabel("H")
    ax_h.legend(fontsize=8)
    ax_h.set_title(f"H comparison (epoch {epoch + 1})")
    fig_h.tight_layout()
    writer.add_figure(tag + "/H_values", fig_h, epoch)
    plt.close(fig_h)

    fig_dh, ax_dh = plt.subplots(figsize=(8, 3))
    ax_dh.plot(
        dh_axis,
        dH_true,
        label="ΔH (ground-truth)",
        linewidth=1.0,
        color="tab:blue",
    )
    ax_dh.plot(
        dh_axis,
        dH_model,
        label="ΔH (empirical)",
        linewidth=1.0,
        color="tab:green",
    )
    ax_dh.plot(
        dh_axis,
        dH_pred,
        label="ΔH (analytic)",
        linewidth=1.0,
        linestyle="--",
        color="tab:red",
    )
    ax_dh.axhline(0, color="lightgrey", linestyle="-", linewidth=0.5)
    ax_dh.set_xlabel("Step")
    ax_dh.set_ylabel("dH")
    ax_dh.legend(fontsize=8)
    ax_dh.set_title(f"dH comparison (epoch {epoch + 1})")
    fig_dh.tight_layout()
    writer.add_figure(tag + "/dH", fig_dh, epoch)
    plt.close(fig_dh)


@torch.no_grad()
def _log_state_rollout(
    model: StatePHGN,
    val_trajs: list,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    tag: str = "val/rollout",
) -> None:
    """Roll out from s0 and compare predicted vs true state trajectory."""
    model.eval()
    state_names = ["θ (rad)", "θ̇ (rad/s)"]

    all_true, all_pred = [], []
    for states, actions in val_trajs:
        q, p = model.split(states[0:1].to(device))
        pred = [torch.cat([q, p], dim=-1).squeeze(0).cpu()]
        for t in range(len(actions)):
            u = actions[t].reshape(1, 1).to(device)
            q, p = model.step(q, p, u)
            pred.append(torch.cat([q, p], dim=-1).squeeze(0).cpu())
        all_pred.append(torch.stack(pred).numpy())
        all_true.append(states.numpy())

    pred_all = np.concatenate(all_pred, axis=0)
    true_all = np.concatenate(all_true, axis=0)

    # Scatter: predicted vs true, pooled across all trajectories
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for i, name in enumerate(state_names):
        ti, pi = true_all[:, i], pred_all[:, i]
        axes[i].scatter(ti, pi, s=2, alpha=0.3)
        lo, hi = min(ti.min(), pi.min()), max(ti.max(), pi.max())
        axes[i].plot([lo, hi], [lo, hi], "r--", linewidth=0.8)
        ss_res = ((ti - pi) ** 2).sum()
        ss_tot = ((ti - ti.mean()) ** 2).sum()
        r2 = 1.0 - ss_res / (ss_tot + 1e-8)
        axes[i].set_xlabel(f"True {name}")
        axes[i].set_ylabel(f"Predicted {name}")
        axes[i].set_title(f"{name}  R²={r2:.3f}")
    fig.suptitle(
        f"Rollout prediction ({len(val_trajs)} trajectories, epoch {epoch + 1})"
    )
    fig.tight_layout()
    writer.add_figure(tag + "/scatter", fig, epoch)
    plt.close(fig)

    # Time-series from first trajectory only
    true_np, pred_np = all_true[0], all_pred[0]
    T = len(true_np)
    fig2, axes2 = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    t_axis = np.arange(T)
    for i, name in enumerate(state_names):
        axes2[i].plot(t_axis, true_np[:, i], label="true", linewidth=1.5)
        axes2[i].plot(
            t_axis, pred_np[:, i], label="pred", linewidth=1.5, linestyle="--"
        )
        axes2[i].set_ylabel(name)
        axes2[i].legend(fontsize=7, loc="upper right")
    axes2[-1].set_xlabel("Step")
    fig2.suptitle(f"Rollout trajectory (epoch {epoch + 1})")
    fig2.tight_layout()
    writer.add_figure(tag + "/trajectory", fig2, epoch)
    plt.close(fig2)


@torch.no_grad()
def _log_structural_matrices(
    model: StatePHGN,
    writer: SummaryWriter,
    epoch: int,
) -> None:
    J = model.get_J().cpu()
    R = model.get_R().cpu()

    writer.add_histogram("structure/R_eigenvalues", torch.linalg.eigvalsh(R), epoch)

    for name, mat in (("J", J), ("R", R)):
        fig, ax = plt.subplots(figsize=(3, 3))
        m = mat.numpy()
        vmax = max(abs(m.max()), abs(m.min()), 1e-6)
        im = ax.imshow(m, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{m[i, j]:.3f}", ha="center", va="center", fontsize=9)
        ax.set_title(f"{name} (epoch {epoch + 1})")
        fig.tight_layout()
        writer.add_figure(f"structure/{name}", fig, epoch)
        plt.close(fig)


def _annotate_frame(frame: np.ndarray, label: str) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.text((4, 4), label, fill=(255, 255, 0))
    return np.array(img)


@torch.no_grad()
def _log_rollout_videos(
    model: StatePHGN,
    val_traj: tuple,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    tag: str = "val/video",
    fps: int = 20,
) -> None:
    """Log side-by-side ground-truth and Hamiltonian-rollout videos to TensorBoard."""
    model.eval()
    states, actions = val_traj  # (T+1, 2), (T,)
    T = len(actions)

    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    env.reset()

    def _render_at(
        theta: float, theta_dot: float, u: float | None = None
    ) -> np.ndarray:
        if not (np.isfinite(theta) and np.isfinite(theta_dot)):
            theta, theta_dot = 0.0, 0.0
        env.unwrapped.state = np.array([theta, theta_dot], dtype=np.float64)
        env.unwrapped.last_u = np.float32(u) if u is not None else None
        return env.render()  # (H, W, 3) uint8

    gt_frames = [_render_at(states[0, 0].item(), states[0, 1].item())]
    for t in range(T):
        gt_frames.append(
            _render_at(states[t + 1, 0].item(), states[t + 1, 1].item(), u=actions[t].item())
        )

    q = states[0:1, : model.Q_DIM].to(device)
    p = states[0:1, model.Q_DIM :].to(device)
    hgn_frames = [_render_at(q.item(), p.item())]
    for t in range(T):
        u = actions[t].reshape(1, 1).to(device)
        q, p = model.step(q, p, u)
        hgn_frames.append(_render_at(q.item(), p.item(), u=actions[t].item()))

    env.close()

    combined = []
    for t, (gt_f, hgn_f) in enumerate(zip(gt_frames, hgn_frames)):
        frame = np.concatenate([gt_f, hgn_f], axis=1)
        combined.append(_annotate_frame(frame, f"t={t}"))

    arr = np.stack(combined, axis=0).transpose(0, 3, 1, 2)
    video = torch.from_numpy(arr).unsqueeze(0)

    writer.add_video(tag + "/gt_vs_hamiltonian_rollout", video, epoch, fps=fps)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
# data
@click.option("--n-episodes", type=int, default=200, show_default=True)
@click.option(
    "--epsilon",
    type=float,
    default=0.1,
    show_default=True,
    help="Fraction of steps with random uniform action",
)
@click.option(
    "--energy-k",
    type=float,
    default=1.0,
    show_default=True,
    help="Gain for energy-pumping controller",
)
@click.option(
    "--max-steps",
    type=int,
    default=200,
    show_default=True,
    help="Steps per episode",
)
@click.option(
    "--damping",
    type=float,
    default=0.0,
    show_default=True,
    help="Linear viscous damping (theta_dot *= exp(-b*dt) per step)",
)
# model
@click.option(
    "--hidden-dim",
    type=int,
    default=256,
    show_default=True,
    help="Width of Hamiltonian MLP hidden layers",
)
@click.option("--dt", type=float, default=0.05, show_default=True)
@click.option("--separable/--no-separable", default=True, show_default=True)
@click.option(
    "--learn-structure/--no-learn-structure",
    default=True,
    show_default=True,
    help="Learn R/b (J is always canonical); --no-learn-structure fixes R=[[0,0],[0,damping]], b=3",
)
@click.option(
    "--quadratic-t/--no-quadratic-t",
    default=True,
    show_default=True,
    help="T(p) = 1/2 p^T M^-1 p with learned constant mass, instead of a free MLP (requires separable)",
)
@click.option(
    "--state-dep-r",
    is_flag=True,
    default=False,
    show_default=True,
    help="R_pp(z) from a small MLP over the current state, instead of a constant matrix (requires --learn-structure)",
)
@click.option(
    "--integrator",
    type=click.Choice(["auto", "rk4", "leapfrog"]),
    default="auto",
    show_default=True,
    help="'auto' = leapfrog if separable else rk4",
)
# training
@click.option("--epochs", type=int, default=3000, show_default=True)
@click.option("--batch-size", type=int, default=16, show_default=True)
@click.option("--h-lr", type=float, default=1e-4, show_default=True)
@click.option("--structural-lr", type=float, default=1e-2, show_default=True)
@click.option("--grad-clip", type=float, default=1.0, show_default=True)
@click.option(
    "--ema-alpha",
    type=float,
    default=0.99,
    show_default=True,
    help="EMA smoothing factor for loss-gated curriculum (higher = smoother)",
)
@click.option(
    "--seq-len-start",
    type=int,
    default=5,
    show_default=True,
    help="Initial rollout length for curriculum",
)
@click.option(
    "--seq-len-advance-threshold",
    type=float,
    default=0.005,
    show_default=True,
    help="EMA loss below which rollout length advances by 1",
)
# logging
@click.option("--log-every", type=int, default=5, show_default=True)
@click.option(
    "--val-every",
    type=int,
    default=10,
    show_default=True,
    help="Epochs between val plots (0 to disable)",
)
@click.option(
    "--n-val-episodes",
    type=int,
    default=-1,
    show_default=True,
    help="Val episodes per type (-1 = n_episodes // 2)",
)
@click.option(
    "--val-max-steps",
    type=int,
    default=0,
    show_default=True,
    help="Steps per val episode (0 = 2x --max-steps)",
)
@click.option("--checkpoint-every", type=int, default=10, show_default=True)
def main(**kwargs):
    # Force SDL software rendering to avoid CUDA/OpenGL context conflict when
    # calling env.render() while a CUDA context is active on the same GPU.
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    writer = SummaryWriter(comment="_pendulum_statebased_offline")
    run_dir = make_run_dir("pendulum_statebased_offline")

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

    min_vel = max_vel = None
    if kwargs["val_every"] > 0:
        min_vel, max_vel = _state_vel_range(train_episodes)

    val_energy, val_random, val_spin = [], [], []
    if n_val > 0:
        print(
            f"Collecting {n_val} val episodes per type ({val_steps} steps each)..."
        )
        val_energy = collect_state_val_trajectories(
            n_episodes=n_val,
            max_steps=val_steps,
            energy_k=kwargs["energy_k"],
            damping=kwargs["damping"],
        )
        val_random = collect_state_random_trajectories(
            n_episodes=n_val,
            max_steps=val_steps,
            damping=kwargs["damping"],
        )
        val_spin = collect_state_spin_trajectories(
            n_episodes=n_val,
            max_steps=val_steps,
            damping=kwargs["damping"],
        )

    dataset = PendulumStateDataset(train_episodes)
    loader = DataLoader(
        dataset,
        batch_size=kwargs["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    print(f"Dataset: {len(dataset)} episodes")

    model = StatePHGN(
        hidden_dim=kwargs["hidden_dim"],
        dt=kwargs["dt"],
        control_dim=1,
        separable=kwargs["separable"],
        learn_structure=kwargs["learn_structure"],
        damping=kwargs["damping"],
        quadratic_t=kwargs["quadratic_t"],
        state_dep_r=kwargs["state_dep_r"],
        integrator=kwargs["integrator"],
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
    }
    if kwargs["learn_structure"]:
        optimizer = torch.optim.Adam(
            [
                {"params": model.hamiltonian.parameters(), "lr": kwargs["h_lr"]},
                {
                    "params": model.structural_parameters(),
                    "lr": kwargs["structural_lr"],
                },
            ]
        )
    else:
        optimizer = torch.optim.Adam(
            model.hamiltonian.parameters(), lr=kwargs["h_lr"]
        )
    best_loss = float("inf")

    full_seq_len = train_episodes[0][1].shape[0]
    seq_len = kwargs["seq_len_start"]
    ema_loss = None

    print("\n=== Training ===")
    for epoch in tqdm(range(kwargs["epochs"]), desc="Training", dynamic_ncols=True):
        metrics = _train_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            grad_clip=kwargs["grad_clip"],
            device=device,
            seq_len=seq_len,
        )

        alpha = kwargs["ema_alpha"]
        ema_loss = (
            metrics["train/loss"]
            if ema_loss is None
            else alpha * ema_loss + (1.0 - alpha) * metrics["train/loss"]
        )
        if ema_loss < kwargs["seq_len_advance_threshold"] and seq_len < full_seq_len:
            seq_len += 1

        if (epoch + 1) % kwargs["log_every"] == 0:
            for k, v in metrics.items():
                writer.add_scalar(k, v, epoch)
            writer.add_scalar("train/seq_len", seq_len, epoch)
            writer.add_scalar("train/ema_loss", ema_loss, epoch)
            writer.add_scalar("structure/b", model.get_b().item(), epoch)
            _log_structural_matrices(model=model, writer=writer, epoch=epoch)
            tqdm.write(
                f"  epoch {epoch + 1:4d}"
                f"  seq_len={seq_len:3d}"
                f"  loss={metrics['train/loss']:.6f}"
                f"  ema={ema_loss:.6f}"
                f"  q_var={metrics['train/q_var']:.4f}"
                f"  p_var={metrics['train/p_var']:.4f}"
            )

        if kwargs["val_every"] > 0 and (epoch + 1) % kwargs["val_every"] == 0:
            energy_fig = _plot_state_energy_landscape(
                model, train_episodes, min_vel=min_vel, max_vel=max_vel, device=device,
            )
            writer.add_figure("val/energy_landscape", energy_fig, epoch)
            plt.close(energy_fig)
            grad_mag_fig = _plot_state_gradient_magnitude_landscape(
                model, min_vel=min_vel, max_vel=max_vel, device=device,
            )
            writer.add_figure("val/gradient_magnitude_landscape", grad_mag_fig, epoch)
            plt.close(grad_mag_fig)
            if model._has_dissipation:
                dissipation_fig = _plot_state_dissipation_landscape(
                    model,
                    damping=kwargs["damping"],
                    drag=_DRAG_COEFF,
                    min_vel=min_vel,
                    max_vel=max_vel,
                    device=device,
                )
                writer.add_figure("val/dissipation_landscape", dissipation_fig, epoch)
                plt.close(dissipation_fig)
            for val_trajs, label in (
                (val_energy, "energy_pump"),
                (val_random, "random"),
                (val_spin, "spin"),
            ):
                if not val_trajs:
                    continue
                writer.add_scalar(
                    f"val/loss/{label}",
                    _eval_loss(model, val_trajs, device),
                    epoch,
                )
                _log_state_rollout(
                    model=model,
                    val_trajs=val_trajs,
                    device=device,
                    writer=writer,
                    epoch=epoch,
                    tag=f"val/rollout/{label}",
                )
            if val_energy:
                _log_hamiltonian_comparison(
                    model=model,
                    val_traj=val_energy[0],
                    device=device,
                    writer=writer,
                    epoch=epoch,
                    tag="val/hamiltonian/energy_pump",
                )
                _log_rollout_videos(
                    model=model,
                    val_traj=val_energy[0],
                    device=device,
                    writer=writer,
                    epoch=epoch,
                    tag="val/video/energy_pump",
                )
            train_sample = train_episodes[: max(1, n_val)]
            _log_state_rollout(
                model=model,
                val_trajs=train_sample,
                device=device,
                writer=writer,
                epoch=epoch,
                tag="train/rollout",
            )
            _log_hamiltonian_comparison(
                model=model,
                val_traj=train_sample[0],
                device=device,
                writer=writer,
                epoch=epoch,
                tag="train/hamiltonian",
            )

        if (
            kwargs["checkpoint_every"] > 0
            and (epoch + 1) % kwargs["checkpoint_every"] == 0
            and metrics["train/loss"] < best_loss
        ):
            save_state_model(run_dir, "best", model, hparams, metrics, epoch, data_config=data_config)
            best_loss = metrics["train/loss"]

    save_state_model(run_dir, "final", model, hparams, metrics, kwargs["epochs"] - 1, data_config=data_config)
    writer.close()
    print("\nDone. Run: tensorboard --logdir runs")
    os._exit(0)


if __name__ == "__main__":
    main()
