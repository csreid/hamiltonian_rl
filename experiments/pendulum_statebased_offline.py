"""Offline Pendulum world-model training directly in ground-truth phase space.

Skips pixel encoding/decoding entirely.  The model operates on the ground-truth
state (θ, θ̇) ∈ ℝ² treated as the Hamiltonian phase space:

    q = θ    ∈ ℝ¹   (angle)
    p = θ̇   ∈ ℝ¹   (angular velocity)

The controlled port-Hamiltonian ODE

    dz/dt = (J − R) ∇H(z) + [0, b] u

is integrated with RK4 for T steps.  Training loss is MSE between the
predicted and ground-truth next states.
"""

from __future__ import annotations

import os
import sys

import click
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from checkpoint_common import make_run_dir, save_checkpoint
from data.pendulum import (
    _G,
    collect_state_data,
    collect_state_random_trajectories,
    collect_state_spin_trajectories,
    collect_state_val_trajectories,
    PendulumStateDataset,
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class _HamiltonianMLP(nn.Module):
    """H(q, p) for asymmetric q_dim / p_dim.

    Separable mode: H = T(q, p) + V(q) — matches physical structure where
    kinetic energy depends on both q and p but potential only on q.
    """

    def __init__(
        self,
        q_dim: int,
        p_dim: int,
        hidden: int = 256,
        separable: bool = True,
    ):
        super().__init__()
        self.separable = separable
        if separable:
            self.kinetic = nn.Sequential(
                nn.Linear(q_dim + p_dim, hidden),
                nn.Softplus(),
                nn.Linear(hidden, hidden),
                nn.Softplus(),
                nn.Linear(hidden, 1),
            )
            self.potential = nn.Sequential(
                nn.Linear(q_dim, hidden),
                nn.Softplus(),
                nn.Linear(hidden, hidden),
                nn.Softplus(),
                nn.Linear(hidden, 1),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(q_dim + p_dim, hidden),
                nn.Softplus(),
                nn.Linear(hidden, hidden),
                nn.Softplus(),
                nn.Linear(hidden, 1),
            )

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        if self.separable:
            T = self.kinetic(torch.cat([q, p], dim=-1)).squeeze(-1)
            V = self.potential(q).squeeze(-1)
            return T + V
        return self.net(torch.cat([q, p], dim=-1)).squeeze(-1)


class StatePHGN(nn.Module):
    """Controlled port-Hamiltonian model operating on ground-truth pendulum state.

    Phase space: z = (θ, θ̇) with q = z[:1], p = z[1:]
    ODE: dz/dt = (J − R) ∇H(z) + [0, b] u
    Integrated with RK4.

    Args:
        hidden_dim:  width of Hamiltonian MLP hidden layers
        dt:          RK4 step size (should match env timestep, 0.05 for Pendulum-v1)
        control_dim: dimension of control input u (1 for Pendulum-v1)
        separable:   use T(q,p) + V(q) Hamiltonian decomposition
    """

    Q_DIM = 1
    P_DIM = 1
    STATE_DIM = 2  # Q_DIM + P_DIM
    Q_ENC_DIM = 2  # sin(θ), cos(θ)

    def __init__(
        self,
        hidden_dim: int = 256,
        dt: float = 0.05,
        control_dim: int = 1,
        separable: bool = True,
        learn_structure: bool = True,
        damping: float = 0.0,
    ):
        super().__init__()
        self.dt = dt
        self.control_dim = control_dim
        self.learn_structure = learn_structure
        D = self.STATE_DIM

        self.hamiltonian = _HamiltonianMLP(
            q_dim=self.Q_ENC_DIM,
            p_dim=self.P_DIM,
            hidden=hidden_dim,
            separable=separable,
        )

        if learn_structure:
            # J = A − Aᵀ (skew-symmetric), R = L Lᵀ (PSD)
            self.A = nn.Parameter(torch.zeros(D, D))
            self.L_param = nn.Parameter(torch.zeros(D, D))
            nn.init.normal_(self.A, std=1e-2)
            nn.init.normal_(self.L_param, std=1e-2)
            # Control: b maps scalar torque to dp (1-D momentum update)
            self.b = nn.Parameter(torch.zeros(self.P_DIM, control_dim))
            nn.init.normal_(self.b, std=1e-2)
        else:
            self.register_buffer("J_fixed", torch.tensor([[0.0, 1.0], [-1.0, 0.0]]))
            self.register_buffer("R_fixed", torch.tensor([[0.0, 0.0], [0.0, damping]]))
            self.register_buffer("b_fixed", torch.full((self.P_DIM, control_dim), 3.0))

    # ── Structure matrix helpers ────────────────────────────────────────────

    def get_J(self) -> torch.Tensor:
        if not self.learn_structure:
            return self.J_fixed
        return self.A - self.A.T

    def get_L(self) -> torch.Tensor:
        L_lower = self.L_param.tril(-1)
        diag_pos = F.softplus(self.L_param.diagonal())
        return L_lower + torch.diag(diag_pos)

    def get_R(self) -> torch.Tensor:
        if not self.learn_structure:
            return self.R_fixed
        L = self.get_L()
        return L @ L.T

    def get_b(self) -> torch.Tensor:
        if not self.learn_structure:
            return self.b_fixed
        return self.b

    # ── Phase-space helpers ─────────────────────────────────────────────────

    def split(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split flat state s (B, 3) → q (B, 2), p (B, 1)."""
        return s[:, : self.Q_DIM], s[:, self.Q_DIM :]

    @staticmethod
    def encode_q(q: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.sin(q), torch.cos(q)], dim=-1)

    def H(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return self.hamiltonian(self.encode_q(q), p)

    # ── Dynamics ────────────────────────────────────────────────────────────

    @torch.enable_grad()
    def _dynamics(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        u: torch.Tensor,
        M: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """dz/dt = (J − R) ∇H(z) + [0, 0, b] u."""
        z_ = torch.cat([q, p], dim=-1).detach().requires_grad_(True)
        H_val = self.hamiltonian(
            self.encode_q(z_[:, : self.Q_DIM]), z_[:, self.Q_DIM :]
        ).sum()
        grad_H = torch.autograd.grad(H_val, z_, create_graph=self.training)[0]

        dz = torch.einsum("ij,bj->bi", M, grad_H)

        # Control acts on the momentum component only
        Bu = u @ self.get_b().T  # (B, P_DIM)
        dz = dz + torch.cat([torch.zeros_like(q), Bu], dim=-1)

        return dz[:, : self.Q_DIM], dz[:, self.Q_DIM :]

    @torch.enable_grad()
    def step(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        u: torch.Tensor,
        dt: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One RK4 step of dz/dt = (J − R) ∇H + control."""
        if dt is None:
            dt = self.dt
        M = self.get_J() - self.get_R()

        dq1, dp1 = self._dynamics(q, p, u, M)
        dq2, dp2 = self._dynamics(q + 0.5 * dt * dq1, p + 0.5 * dt * dp1, u, M)
        dq3, dp3 = self._dynamics(q + 0.5 * dt * dq2, p + 0.5 * dt * dp2, u, M)
        dq4, dp4 = self._dynamics(q + dt * dq3, p + dt * dp3, u, M)

        q_next = q + (dt / 6.0) * (dq1 + 2 * dq2 + 2 * dq3 + dq4)
        p_next = p + (dt / 6.0) * (dp1 + 2 * dp2 + 2 * dp3 + dp4)
        return q_next, p_next


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
    """H = 0.5 θ̇² + g (1 + cos θ) from (T, 2) states."""
    theta = states[:, 0].numpy()
    theta_dot = states[:, 1].numpy()
    return 0.5 * theta_dot**2 + _G * (1.0 + np.cos(theta))


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
                u @ model.b.T,
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
@click.option("--no-separable", "separable", default=True, flag_value=False)
@click.option(
    "--learn-structure/--no-learn-structure",
    default=True,
    show_default=True,
    help="Learn J/R/B matrices; --no-learn-structure fixes J=[[0,1],[-1,0]], R=[[0,0],[0,damping]], B=3",
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
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    hparams = dict(kwargs)
    if kwargs["learn_structure"]:
        optimizer = torch.optim.Adam(
            [
                {"params": model.hamiltonian.parameters(), "lr": kwargs["h_lr"]},
                {
                    "params": [model.L_param, model.A, model.b],
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
    for epoch in tqdm(range(kwargs["epochs"]), desc="Training"):
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
            save_checkpoint(run_dir, epoch, model, hparams, metrics)
            best_loss = metrics["train/loss"]

    writer.close()
    print("\nDone. Run: tensorboard --logdir runs")
    os._exit(0)


if __name__ == "__main__":
    main()
