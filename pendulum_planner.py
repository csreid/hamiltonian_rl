"""Streamlit visualiser: MPPI planning on ground-truth vs learned pendulum dynamics.

Usage:
    streamlit run pendulum_planner.py

Loads a unified world-model checkpoint (LSTM autoencoder + Hamiltonian flow
dynamics), then runs two independent receding-horizon MPPI controllers, both
trying to swing up and balance the pendulum from the same initial condition:

  - Ground truth: MPPI samples candidate action sequences and rolls them
    forward through an analytic replica of Pendulum-v1's own dynamics
    (`data.pendulum.analytic_pendulum_step`), cost = angle^2 + damping term +
    action penalty. The chosen action is applied to a real `PendulumPixelEnv`
    each control step, so the rendered frames are real renders.

  - Learned: MPPI samples candidate action sequences and rolls them forward
    through the checkpoint's `HamiltonianFlowModel.controlled_step` in
    latent (q, p) space, cost = pixel MSE (decoded frame vs. a fixed upright
    target frame) + action penalty. The model drives itself — no real env
    involved on this side.

Both closed-loop rollouts are rendered as a side-by-side pixel GIF, plus an
animated phase-space (theta, theta_dot) comparison: the ground-truth
trajectory vs. the latent trajectory mapped to physical coordinates via a
linear probe h -> (cos theta, sin theta, theta_dot), fit once per checkpoint.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from data.pendulum import (
    PendulumPixelEnv,
    _MAX_SPEED,
    analytic_pendulum_step,
    angle_normalize,
    collect_random_trajectories,
    collect_spin_trajectories,
    collect_val_trajectories,
    collect_zero_trajectory_from,
)
from hamilton_rl.mppi import MPPIConfig, mppi_plan, shift_mean
from hamilton_rl.models import WorldModel
from hamilton_rl.streamlit_common import (
    build_sidebyside_frames,
    frames_to_gif,
    pick_checkpoint,
)


# ── Model / regression loading ──────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading world model…")
def load_model(pt_path_str: str) -> WorldModel:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return WorldModel.load(Path(pt_path_str), device)


@st.cache_resource(show_spinner="Fitting h → state linear probe…")
def fit_h_state_regression(
    pt_path_str: str, n_episodes: int = 6, max_steps: int = 60
) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear probe h -> (cos theta, sin theta, theta_dot), fit on fresh held-out episodes.

    Mirrors `_log_h_state_regression_coeffs_phase1` in
    experiments/pendulum_offline.py: pool h_t across policies via
    `torch.linalg.lstsq`, report per-target R^2 for transparency.
    """
    world_model = load_model(pt_path_str)
    device = next(world_model.autoencoder.parameters()).device
    data_cfg = world_model.data_config
    img_size = data_cfg.get("img_size", 64)
    damping = data_cfg.get("damping", 0.0)

    episodes = (
        collect_val_trajectories(n_episodes=n_episodes, img_size=img_size, max_steps=max_steps, damping=damping)
        + collect_random_trajectories(n_episodes=n_episodes, img_size=img_size, max_steps=max_steps, damping=damping)
        + collect_spin_trajectories(n_episodes=n_episodes, img_size=img_size, max_steps=max_steps, damping=damping)
    )

    all_h, all_st = [], []
    with torch.no_grad():
        for frames, _actions, states in episodes:
            ctx = frames.unsqueeze(0).to(device)
            mu_all, _ = world_model.autoencoder.encoder.forward_all(ctx)
            all_h.append(mu_all.squeeze(0).cpu())
            all_st.append(states.float())

    h_pool = torch.cat(all_h, dim=0)
    st_pool = torch.cat(all_st, dim=0)
    A = torch.linalg.lstsq(h_pool, st_pool).solution  # (latent_dim, 3)

    pred = h_pool @ A
    ss_res = ((st_pool - pred) ** 2).sum(dim=0)
    ss_tot = ((st_pool - st_pool.mean(dim=0)) ** 2).sum(dim=0)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    return A, r2


# ── Ground-truth MPPI ────────────────────────────────────────────────────────


def _gt_rollout_cost_fn(theta0, theta_dot0, damping, action_weight):
    def cost_fn(candidates: torch.Tensor) -> torch.Tensor:
        K, H, _ = candidates.shape
        theta = theta0.expand(K).clone()
        theta_dot = theta_dot0.expand(K).clone()
        total = torch.zeros(K, device=candidates.device, dtype=candidates.dtype)
        for t in range(H):
            u = candidates[:, t, 0]
            theta, theta_dot = analytic_pendulum_step(theta, theta_dot, u, damping=damping)
            total = total + angle_normalize(theta) ** 2 + 0.1 * theta_dot**2 + action_weight * u**2
        return total

    return cost_fn


@torch.no_grad()
def run_ground_truth_mppi(
    theta0: float,
    theta_dot0: float,
    img_size: int,
    damping: float,
    total_steps: int,
    cfg: MPPIConfig,
    action_weight: float,
    device: torch.device,
    progress_cb=None,
) -> dict:
    """Closed-loop receding-horizon MPPI on the real Pendulum-v1 dynamics."""
    env = PendulumPixelEnv(img_size=img_size, damping=damping)
    env.reset()
    env.set_state(theta0, theta_dot0)

    theta_cur = torch.tensor(theta0, device=device)
    theta_dot_cur = torch.tensor(theta_dot0, device=device)
    mean = torch.zeros(cfg.horizon, 1, device=device)

    frames, thetas, theta_dots = [], [theta0], [theta_dot0]
    try:
        for step in range(total_steps):
            cost_fn = _gt_rollout_cost_fn(theta_cur, theta_dot_cur, damping, action_weight)
            mean = mppi_plan(mean, cost_fn, cfg)
            u0 = float(mean[0, 0].clamp(cfg.action_low, cfg.action_high))

            obs, _, _, _, _ = env.step(np.array([u0], dtype=np.float32))
            theta_new, theta_dot_new = env.unwrapped.state  # post-damping
            theta_cur = torch.tensor(float(theta_new), device=device)
            theta_dot_cur = torch.tensor(float(theta_dot_new), device=device)

            frames.append(torch.from_numpy(obs).float() / 255.0)
            thetas.append(float(theta_new))
            theta_dots.append(float(theta_dot_new))

            mean = shift_mean(mean)
            if progress_cb is not None:
                progress_cb(step + 1, total_steps)
    finally:
        env.close()

    return {
        "frames": frames,           # list of (C, H, W) float tensors, len total_steps
        "theta": thetas,            # len total_steps + 1 (includes t=0)
        "theta_dot": theta_dots,
    }


# ── Latent MPPI ──────────────────────────────────────────────────────────────


def _latent_rollout_cost_fn(dynamics, autoencoder, q0, p0, target_frame, action_weight):
    def cost_fn(candidates: torch.Tensor) -> torch.Tensor:
        K, H, _ = candidates.shape
        q = q0.expand(K, -1).clone()
        p = p0.expand(K, -1).clone()
        qs, ps, us = [], [], []
        for t in range(H):
            u = candidates[:, t, :]
            q, p = dynamics.controlled_step(q, p, u)
            qs.append(q)
            ps.append(p)
            us.append(u)
        q_traj = torch.stack(qs, dim=1)  # (K, H, q_dim)
        p_traj = torch.stack(ps, dim=1)
        u_traj = torch.stack(us, dim=1)  # (K, H, 1)
        q_dim = q_traj.shape[-1]

        h_traj = dynamics.decode(q_traj.reshape(K * H, q_dim), p_traj.reshape(K * H, q_dim))
        pred_frames = autoencoder.decode_latent(h_traj)  # (K*H, C, Himg, Wimg)
        target = target_frame.unsqueeze(0).expand_as(pred_frames)
        pixel_mse = (pred_frames - target).pow(2).mean(dim=(1, 2, 3)).reshape(K, H)

        action_cost = u_traj.squeeze(-1).pow(2)  # (K, H)
        return (pixel_mse + action_weight * action_cost).sum(dim=1)

    return cost_fn


@torch.no_grad()
def run_latent_mppi(
    world_model: WorldModel,
    theta0: float,
    theta_dot0: float,
    n_context: int,
    total_steps: int,
    cfg: MPPIConfig,
    action_weight: float,
    regression: torch.Tensor,
    progress_cb=None,
) -> dict:
    """Closed-loop receding-horizon MPPI driven entirely by the model's own dynamics."""
    device = next(world_model.autoencoder.parameters()).device
    data_cfg = world_model.data_config
    img_size = data_cfg.get("img_size", 64)
    damping = data_cfg.get("damping", 0.0)
    autoencoder, dynamics = world_model.autoencoder, world_model.dynamics

    # Real zero-action motion context so the causal LSTM encoder can infer velocity.
    ctx_frames, _ctx_actions, _ctx_states = collect_zero_trajectory_from(
        theta0, theta_dot0, img_size=img_size, max_steps=max(n_context - 1, 0), damping=damping,
    )
    ctx = ctx_frames.unsqueeze(0).to(device)
    mu_all, _ = autoencoder.encoder.forward_all(ctx)
    h0 = mu_all[:, -1]
    q, p = dynamics.encode(h0)

    target_env = PendulumPixelEnv(img_size=img_size, damping=damping)
    target_env.reset()
    target_obs = target_env.set_state(0.0, 0.0)
    target_env.close()
    target_frame = (torch.from_numpy(target_obs).float() / 255.0).to(device)

    mean = torch.zeros(cfg.horizon, 1, device=device)
    frames = []
    theta_hats, theta_dot_hats = [], []

    A = regression.to(device)
    h_init = dynamics.decode(q, p)
    st0 = (h_init @ A).squeeze(0)
    theta_hats.append(float(torch.atan2(st0[1], st0[0])))
    theta_dot_hats.append(float(st0[2]))

    for step in range(total_steps):
        cost_fn = _latent_rollout_cost_fn(dynamics, autoencoder, q, p, target_frame, action_weight)
        mean = mppi_plan(mean, cost_fn, cfg)
        u0 = mean[0:1].clamp(cfg.action_low, cfg.action_high)

        q, p = dynamics.controlled_step(q, p, u0)
        h_exec = dynamics.decode(q, p)
        frame = autoencoder.decode_latent(h_exec).squeeze(0).cpu()
        frames.append(frame)

        st_pred = (h_exec @ A).squeeze(0)
        theta_hats.append(float(torch.atan2(st_pred[1], st_pred[0])))
        theta_dot_hats.append(float(st_pred[2]))

        mean = shift_mean(mean)
        if progress_cb is not None:
            progress_cb(step + 1, total_steps)

    return {
        "frames": frames,
        "theta_hat": theta_hats,
        "theta_dot_hat": theta_dot_hats,
    }


# ── Phase-space animation ────────────────────────────────────────────────────


def build_phase_space_frames(
    theta_actual: list, theta_dot_actual: list, theta_planned: list, theta_dot_planned: list,
) -> list[Image.Image]:
    T = min(len(theta_actual), len(theta_planned))
    frames = []
    for i in range(1, T + 1):
        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
        ax.plot(theta_actual[:i], theta_dot_actual[:i], color="tab:blue", linewidth=1.3, label="Actual (GT MPPI)")
        ax.scatter(theta_actual[i - 1:i], theta_dot_actual[i - 1:i], color="tab:blue", s=30, zorder=3)
        ax.plot(theta_planned[:i], theta_dot_planned[:i], color="tab:orange", linewidth=1.3, label="Planned (latent MPPI)")
        ax.scatter(theta_planned[i - 1:i], theta_dot_planned[i - 1:i], color="tab:orange", s=30, zorder=3)
        ax.axvline(0.0, color="gray", linewidth=0.6, linestyle="--")
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-_MAX_SPEED, _MAX_SPEED)
        ax.set_xlabel("θ (rad)")
        ax.set_ylabel("θ̇ (rad/s)")
        ax.set_title(f"Phase space  (t={i})")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))
    return frames


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Pendulum MPPI Planner", layout="wide")
st.title("Pendulum MPPI Planner: Ground Truth vs. Learned Dynamics")

with st.sidebar:
    st.header("Checkpoint")
    models_root = Path("models")
    ckpt_path = pick_checkpoint(models_root, "World model", "wm")

    st.divider()
    st.header("Initial condition")
    theta0 = st.slider("θ₀ (rad)", -float(np.pi), float(np.pi), float(np.pi), step=0.05,
                        help="π = hanging straight down (swing-up); 0 = upright")
    theta_dot0 = st.slider("θ̇₀ (rad/s)", -8.0, 8.0, 0.0, step=0.1)
    n_context = st.slider("Encoder context frames", min_value=2, max_value=20, value=5, step=1)

    st.divider()
    st.header("MPPI")
    horizon = st.slider("Horizon H", min_value=5, max_value=40, value=20, step=1)
    n_samples = st.slider("Samples K", min_value=32, max_value=1024, value=256, step=32)
    n_iterations = st.slider("Iterations / control step", min_value=1, max_value=5, value=1, step=1)
    noise_std = st.slider("Noise std", 0.1, 3.0, 1.0, step=0.1)
    temperature = st.slider("Temperature λ", 0.01, 5.0, 1.0, step=0.01)

    st.divider()
    st.header("Cost weights")
    w_u_gt = st.number_input("Action weight (ground truth)", min_value=0.0, value=0.001, step=0.001, format="%.4f")
    w_u_latent = st.number_input("Action weight (pixel MSE)", min_value=0.0, value=0.01, step=0.01, format="%.3f")

    st.divider()
    st.header("Rollout")
    total_steps = st.slider("Control steps T", min_value=5, max_value=300, value=60, step=5)

    st.divider()
    generate_btn = st.button("▶ Plan & Generate", type="primary", use_container_width=True)


# ── Load model ────────────────────────────────────────────────────────────────

try:
    world_model = load_model(str(ckpt_path))
    device = next(world_model.autoencoder.parameters()).device
except Exception as exc:
    st.error(f"Failed to load checkpoint:\n\n```\n{exc}\n```")
    st.stop()

if world_model.dynamics is None:
    st.warning(f"`{ckpt_path}` is a Phase-1-only checkpoint (no dynamics). Pick a Phase 2 checkpoint to plan.")
    st.stop()

data_cfg = world_model.data_config
img_size = data_cfg.get("img_size", 64)
damping = data_cfg.get("damping", 0.0)

with st.sidebar:
    st.caption(f"img_size={img_size}  damping={damping}  device={device}")


# ── Generation ────────────────────────────────────────────────────────────────

if generate_btn:
    cfg = MPPIConfig(
        horizon=horizon, n_samples=n_samples, n_iterations=n_iterations,
        noise_std=noise_std, temperature=temperature, action_low=-2.0, action_high=2.0,
    )

    with st.spinner("Fitting h → state regression on fresh held-out episodes…"):
        try:
            A, r2 = fit_h_state_regression(str(ckpt_path))
        except Exception as exc:
            st.error(f"Regression fit failed:\n\n```\n{exc}\n```")
            st.stop()

    gt_progress = st.progress(0.0, text="Ground-truth MPPI…")
    try:
        gt_result = run_ground_truth_mppi(
            theta0=theta0, theta_dot0=theta_dot0, img_size=img_size, damping=damping,
            total_steps=total_steps, cfg=cfg, action_weight=w_u_gt, device=device,
            progress_cb=lambda i, n: gt_progress.progress(i / n, text=f"Ground-truth MPPI… {i}/{n}"),
        )
    except Exception as exc:
        st.error(f"Ground-truth MPPI failed:\n\n```\n{exc}\n```")
        st.stop()
    gt_progress.empty()

    latent_progress = st.progress(0.0, text="Learned-dynamics MPPI…")
    try:
        latent_result = run_latent_mppi(
            world_model=world_model, theta0=theta0, theta_dot0=theta_dot0, n_context=n_context,
            total_steps=total_steps, cfg=cfg, action_weight=w_u_latent, regression=A,
            progress_cb=lambda i, n: latent_progress.progress(i / n, text=f"Learned-dynamics MPPI… {i}/{n}"),
        )
    except Exception as exc:
        st.error(f"Learned-dynamics MPPI failed:\n\n```\n{exc}\n```")
        st.stop()
    latent_progress.empty()

    st.session_state.update(
        gt_result=gt_result,
        latent_result=latent_result,
        r2=r2,
        ckpt_path=str(ckpt_path),
        total_steps=total_steps,
    )


# ── Display ───────────────────────────────────────────────────────────────────

gt_result = st.session_state.get("gt_result")
latent_result = st.session_state.get("latent_result")

if gt_result is None or latent_result is None:
    st.info("Configure settings in the sidebar and press **▶ Plan & Generate**.")
    st.stop()

r2 = st.session_state["r2"]
st.success(
    f"Planned **{st.session_state['total_steps']}** control steps  |  "
    f"Checkpoint: `{st.session_state['ckpt_path']}`  |  "
    f"h→state R² — cosθ: {r2[0]:.3f}  sinθ: {r2[1]:.3f}  θ̇: {r2[2]:.3f}"
)

col_fps, col_size = st.columns(2)
with col_fps:
    fps = st.slider("Playback FPS", min_value=1, max_value=60, value=10, step=1)
with col_size:
    display_size = st.select_slider("Frame size (px)", options=[64, 128, 192, 256, 384], value=128)

with st.spinner("Rendering pixel-rollout GIF…"):
    gt_frames_u8 = [
        (f.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8) for f in gt_result["frames"]
    ]
    latent_frames_u8 = [
        (f.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8) for f in latent_result["frames"]
    ]
    composite = build_sidebyside_frames(
        left_frames=gt_frames_u8, right_frames=latent_frames_u8,
        display_size=display_size, left_label="GT MPPI", right_label="Learned MPPI",
    )
    pixel_gif = frames_to_gif(composite, fps)

st.subheader("Ground-truth MPPI  (left)  |  Learned-dynamics MPPI  (right)")
st.image(pixel_gif, use_container_width=False)

with st.spinner("Rendering phase-space animation…"):
    phase_frames = build_phase_space_frames(
        theta_actual=gt_result["theta"][1:], theta_dot_actual=gt_result["theta_dot"][1:],
        theta_planned=latent_result["theta_hat"][1:], theta_dot_planned=latent_result["theta_dot_hat"][1:],
    )
    phase_gif = frames_to_gif(phase_frames, fps)

st.subheader("Phase space: actual (ground truth) vs. planned (latent, via h→state regression)")
st.image(phase_gif, use_container_width=False)
