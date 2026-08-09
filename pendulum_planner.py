"""Streamlit visualiser: MPPI planning on ground-truth vs learned pendulum dynamics.

Usage:
    streamlit run pendulum_planner.py

Loads a unified world-model checkpoint (LSTM autoencoder + Hamiltonian flow
dynamics), then runs two independent receding-horizon MPPI controllers, both
trying to swing up and balance the pendulum from the same initial condition.
Both are real closed-loop MPC: at every control step, MPPI samples candidate
action sequences, *imagines* them forward through a dynamics model to score
them, picks the best one by the usual MPPI soft-min weighting, and applies
only its first action to a real `PendulumPixelEnv` — then repeats from the
resulting real observation. Neither side ever free-runs open-loop.

  - Ground truth: imagination = an analytic replica of Pendulum-v1's own
    dynamics (`data.pendulum.analytic_pendulum_step`), cost = angle^2 +
    velocity + action penalty.

  - Learned: imagination = the checkpoint's `HamiltonianFlowModel` in latent
    (q, p) space. Cost is the *same* angle^2 + velocity^2 + action-penalty
    shape as ground truth, evaluated by applying the already-fitted h->state
    linear probe to each imagined step's h — not pixel MSE against a target
    frame, which stayed nearly flat until the pendulum was already most of
    the way upright and gave MPPI little gradient to plan against early on.
    Critically, the latent state used to *start* each planning step is
    re-grounded every step by re-encoding a rolling window of the real
    rendered frames through the frozen Phase-1 LSTM encoder (exactly how
    Phase 2 training itself re-encodes fresh windows) — the learned dynamics
    are only ever used inside the horizon-H imagination, never to carry real
    state forward in time. Both panels therefore render real environment
    frames.

Both closed-loop rollouts are rendered as a side-by-side pixel GIF, plus an
animated phase-space (theta, theta_dot) comparison: the ground-truth
trajectory vs. the *actual* physical trajectory of the pendulum under the
learned controller's actions (both read directly from the real simulator
state, `env.unwrapped.state`) — the dashed preview lines are the only place
each side's own model of dynamics (analytic vs. learned, the latter mapped
to physical coordinates via a linear probe h -> (cos theta, sin theta,
theta_dot), fit once per checkpoint) shows up.
"""

from __future__ import annotations

import io
import sys
from collections import deque
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
    analytic_pendulum_step,
    angle_normalize,
    collect_random_trajectories,
    collect_spin_trajectories,
    collect_val_trajectories,
)
from hamilton_rl.mppi import MPPIConfig, mppi_plan, shift_mean
from hamilton_rl.models import WorldModel
from hamilton_rl.streamlit_common import (
    build_sidebyside_frames,
    frames_to_gif,
    pick_checkpoint,
    to_uint8,
)

# Display-sanity bound for theta_dot: not a physics guarantee (the env's old
# hard clip is gone, replaced by quadratic drag that only *softly* bounds
# speed — see data.pendulum._DRAG_COEFF), just a generous ceiling past the
# drag's calibrated worst-case range (~10-12 rad/s) so a wild linear-probe
# extrapolation can't spuriously dominate the MPPI cost or blow out the plots.
_DISPLAY_VEL_LIMIT = 15.0


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
    n_context: int,
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

    # Same zero-action warm-up duration the learned controller needs to fill
    # its encoder context, purely so both panels' t=0 is the same real time.
    for _ in range(max(n_context - 1, 0)):
        env.step(np.array([0.0], dtype=np.float32))
    theta_b, theta_dot_b = env.unwrapped.state

    theta_cur = torch.tensor(float(theta_b), device=device)
    theta_dot_cur = torch.tensor(float(theta_dot_b), device=device)
    mean = torch.zeros(cfg.horizon, 1, device=device)

    frames = []
    thetas = [float(angle_normalize(torch.tensor(float(theta_b))))]
    theta_dots = [float(theta_dot_b)]
    plan_thetas, plan_theta_dots = [], []
    try:
        for step in range(total_steps):
            cost_fn = _gt_rollout_cost_fn(theta_cur, theta_dot_cur, damping, action_weight)
            mean = mppi_plan(mean, cost_fn, cfg)

            # Roll the chosen (noise-free) plan forward once, purely to show
            # "what does the controller currently intend to do" — never executed.
            pt, ptd = theta_cur.unsqueeze(0), theta_dot_cur.unsqueeze(0)
            plan_t, plan_td = [], []
            for t in range(cfg.horizon):
                pt, ptd = analytic_pendulum_step(pt, ptd, mean[t : t + 1, 0], damping=damping)
                plan_t.append(float(angle_normalize(pt)))
                plan_td.append(float(ptd))
            plan_thetas.append(plan_t)
            plan_theta_dots.append(plan_td)

            u0 = float(mean[0, 0].clamp(cfg.action_low, cfg.action_high))

            env.step(np.array([u0], dtype=np.float32))
            # gymnasium's internal state theta is NOT wrapped to [-pi, pi] — it
            # just accumulates theta += theta_dot*dt across a full rotation, so
            # it can drift arbitrarily far outside the plotted range even though
            # the pendulum's physical orientation is periodic. sin/cos (and thus
            # the dynamics) only ever depend on theta through trig functions, so
            # feeding the unwrapped value back into the next planning step is
            # still exactly correct — only display needs the wrapped value.
            theta_new, theta_dot_new = env.unwrapped.state  # post-damping
            theta_cur = torch.tensor(float(theta_new), device=device)
            theta_dot_cur = torch.tensor(float(theta_dot_new), device=device)

            # No encoder involved on this side, so the display frame can carry
            # the torque-arrow overlay for the applied action freely.
            display_obs = env.render_with_action(u0)
            frames.append(torch.from_numpy(display_obs).float() / 255.0)
            thetas.append(float(angle_normalize(torch.tensor(float(theta_new)))))
            theta_dots.append(float(theta_dot_new))

            mean = shift_mean(mean)
            if progress_cb is not None:
                progress_cb(step + 1, total_steps)
    finally:
        env.close()

    return {
        "frames": frames,                     # list of (C, H, W) float tensors, len total_steps
        "theta": thetas,                      # len total_steps + 1 (includes t=0)
        "theta_dot": theta_dots,
        "plan_theta": plan_thetas,            # len total_steps; plan_theta[i] = H-step preview from theta[i]
        "plan_theta_dot": plan_theta_dots,
    }


# ── Latent MPPI ──────────────────────────────────────────────────────────────


def _latent_rollout_cost_fn(dynamics, q0, p0, regression, action_weight):
    """Imagined-rollout cost in angle-space, via the h->state linear probe.

    Pixel-MSE against a fixed target frame made the cost nearly flat until
    the pendulum was already most of the way upright (the decoded image only
    starts to resemble "vertical" quite late), which gives MPPI very little
    gradient signal to plan against early on. Reusing the already-fitted
    linear regression turns every imagined step directly into an estimated
    (theta, theta_dot) and lets the cost be the same angle^2 + velocity^2
    shape as the ground-truth cost — smooth from step 1 — and skips a CNN
    decoder forward pass per candidate step entirely.
    """
    def cost_fn(candidates: torch.Tensor) -> torch.Tensor:
        K, H, _ = candidates.shape
        q = q0.expand(K, -1).clone()
        p = p0.expand(K, -1).clone()
        total = torch.zeros(K, device=candidates.device, dtype=candidates.dtype)
        for t in range(H):
            u = candidates[:, t, :]
            q, p = dynamics.controlled_step(q, p, u)
            h = dynamics.decode(q, p)
            st = h @ regression  # (K, 3) -> (cos theta_hat, sin theta_hat, theta_dot_hat)
            theta_hat = torch.atan2(st[:, 1], st[:, 0])
            # The linear probe is unconstrained and can predict velocities
            # well outside the simulator's typical range — clamp to a
            # generous display-sanity bound so a wild out-of-domain
            # extrapolation can't spuriously dominate the cost.
            theta_dot_hat = st[:, 2].clamp(-_DISPLAY_VEL_LIMIT, _DISPLAY_VEL_LIMIT)
            total = total + angle_normalize(theta_hat) ** 2 + 0.1 * theta_dot_hat**2 + action_weight * u.squeeze(-1) ** 2
        return total

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
    """Closed-loop receding-horizon MPPI: plan with the learned dynamics, act on the real pendulum.

    At every control step the latent state used to *start* planning is
    re-grounded from a rolling window of the real rendered frames (re-encoded
    through the frozen Phase-1 LSTM encoder from scratch each time — exactly
    how Phase 2 training re-encodes fresh windows, see `_train_epoch_phase2`
    in experiments/pendulum_offline.py). The learned `HamiltonianFlowModel`
    is only ever used *inside* MPPI's horizon-H imagination to score candidate
    action sequences; only the resulting first action is ever applied, and
    only to the real `PendulumPixelEnv`. So the rendered frames are real, and
    latent state can never silently drift off-manifold across the rollout —
    each step starts fresh from what the encoder actually sees.
    """
    device = next(world_model.autoencoder.parameters()).device
    data_cfg = world_model.data_config
    img_size = data_cfg.get("img_size", 64)
    damping = data_cfg.get("damping", 0.0)
    autoencoder, dynamics = world_model.autoencoder, world_model.dynamics
    A = regression.to(device)

    env = PendulumPixelEnv(img_size=img_size, damping=damping)
    env.reset()
    obs0 = env.set_state(theta0, theta_dot0)
    context = deque([torch.from_numpy(obs0).float() / 255.0], maxlen=n_context)
    for _ in range(max(n_context - 1, 0)):
        obs, _, _, _, _ = env.step(np.array([0.0], dtype=np.float32))
        context.append(torch.from_numpy(obs).float() / 255.0)

    def ground() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-encode the current real-frame context window -> grounded (h, q, p)."""
        ctx = torch.stack(list(context)).unsqueeze(0).to(device)  # (1, n_context, C, H, W)
        mu_all, _ = autoencoder.encoder.forward_all(ctx)
        h_t = mu_all[:, -1]
        q_t, p_t = dynamics.encode(h_t)
        return h_t, q_t, p_t

    mean = torch.zeros(cfg.horizon, 1, device=device)
    frames = []
    thetas, theta_dots = [], []
    plan_thetas, plan_theta_dots = [], []
    plan_q0, plan_p0, plan_actions = [], [], []

    h_t, q, p = ground()
    theta_b, theta_dot_b = env.unwrapped.state
    thetas.append(float(angle_normalize(torch.tensor(float(theta_b)))))
    theta_dots.append(float(theta_dot_b))

    try:
        for step in range(total_steps):
            cost_fn = _latent_rollout_cost_fn(dynamics, q, p, A, action_weight)
            mean = mppi_plan(mean, cost_fn, cfg)

            # Snapshot the grounded latent state this step planned from, plus
            # the chosen action sequence, so a single step's plan can later be
            # decoded to pixels on demand (see decode_plan_frames).
            plan_q0.append(q.clone().cpu())
            plan_p0.append(p.clone().cpu())
            plan_actions.append(mean.clone().cpu())

            # Roll the chosen (noise-free) plan forward once, purely to show
            # "what does the controller currently intend to do" — never executed.
            pq, pp = q.clone(), p.clone()
            plan_t, plan_td = [], []
            for t in range(cfg.horizon):
                pq, pp = dynamics.controlled_step(pq, pp, mean[t : t + 1, :])
                h_p = dynamics.decode(pq, pp)
                st_p = (h_p @ A).squeeze(0)
                plan_t.append(float(angle_normalize(torch.atan2(st_p[1], st_p[0]))))
                # atan2 is scale-invariant, so theta stays plausible-looking
                # even if the linear probe's raw output has blown up under
                # extrapolation this far into pure imagination — theta_dot has
                # no such protection, so clamp it to the same display-sanity
                # bound used by the MPPI cost function.
                plan_td.append(float(st_p[2].clamp(-_DISPLAY_VEL_LIMIT, _DISPLAY_VEL_LIMIT)))
            plan_thetas.append(plan_t)
            plan_theta_dots.append(plan_td)

            u0 = float(mean[0, 0].clamp(cfg.action_low, cfg.action_high))

            obs, _, _, _, _ = env.step(np.array([u0], dtype=np.float32))
            # Arrow-free frame grounds the encoder (context); a separately
            # rendered arrow frame is display-only and never touches it.
            context.append(torch.from_numpy(obs).float() / 255.0)
            display_obs = env.render_with_action(u0)
            frames.append(torch.from_numpy(display_obs).float() / 255.0)

            h_t, q, p = ground()
            theta_new, theta_dot_new = env.unwrapped.state  # post-step, actual physical state
            thetas.append(float(angle_normalize(torch.tensor(float(theta_new)))))
            theta_dots.append(float(theta_dot_new))

            mean = shift_mean(mean)
            if progress_cb is not None:
                progress_cb(step + 1, total_steps)
    finally:
        env.close()

    return {
        "frames": frames,
        "theta": thetas,                  # len total_steps + 1 (includes t=0); actual physical state
        "theta_dot": theta_dots,
        "plan_theta": plan_thetas,            # len total_steps; plan_theta[i] = H-step preview from theta[i]
        "plan_theta_dot": plan_theta_dots,
        "plan_q0": plan_q0,                # len total_steps; grounded (q, p) each step planned from
        "plan_p0": plan_p0,
        "plan_actions": plan_actions,      # len total_steps; the chosen (noise-free) H-step action sequence
    }


# ── Decoded plan (single control step) ──────────────────────────────────────


@torch.no_grad()
def decode_plan_frames(
    world_model: WorldModel, q0: torch.Tensor, p0: torch.Tensor, actions: torch.Tensor
) -> list:
    """Decode one control step's imagined H-step plan to pixels.

    Same per-step pipeline as `WorldModel.dream` (controlled_step -> decode ->
    decode_latent), just starting from an already-grounded (q0, p0) instead of
    an encoded context window, and following the fixed `actions` sequence MPPI
    committed to for that step rather than a held-out action tape.
    """
    device = next(world_model.autoencoder.parameters()).device
    dynamics, autoencoder = world_model.dynamics, world_model.autoencoder
    q, p = q0.to(device), p0.to(device)
    horizon = actions.shape[0]
    decoded = []
    for t in range(horizon):
        q, p = dynamics.controlled_step(q, p, actions[t : t + 1].to(device))
        h_pred = dynamics.decode(q, p)
        decoded.append(autoencoder.decode_latent(h_pred).squeeze(0).cpu())
    return to_uint8(torch.stack(decoded)) if decoded else []


# ── Phase-space animation ────────────────────────────────────────────────────


def _break_wrapped(theta: list, theta_dot: list) -> tuple[list, list]:
    """Insert a NaN separator at every angle-wrap boundary (|delta theta| > pi).

    theta lives in [-pi, pi], so a genuine wrap (e.g. +3.1 -> -3.1 across one
    real timestep) is a tiny physical motion but a huge jump in the plotted
    coordinate. Without a break, a line plot draws a spurious edge-to-edge
    segment straight across the axes; matplotlib treats NaN as a gap, so
    inserting one at each such jump splits the line there without dropping
    either real point.
    """
    if len(theta) < 2:
        return list(theta), list(theta_dot)
    out_t, out_td = [theta[0]], [theta_dot[0]]
    for i in range(1, len(theta)):
        if abs(theta[i] - theta[i - 1]) > np.pi:
            out_t.append(float("nan"))
            out_td.append(float("nan"))
        out_t.append(theta[i])
        out_td.append(theta_dot[i])
    return out_t, out_td


def build_phase_space_frames(
    gt_theta: list, gt_theta_dot: list, gt_plan_theta: list, gt_plan_theta_dot: list,
    learned_theta: list, learned_theta_dot: list, learned_plan_theta: list, learned_plan_theta_dot: list,
) -> list[Image.Image]:
    """One frame per control step: solid = trajectory so far, dashed = the
    *current* planned horizon only (not a history of past plans) — so a
    controller that keeps replanning very differently from step to step
    shows up as a dashed line that visibly changes shape frame to frame,
    while one that tracks its own plan closely stays nearly stationary.
    """
    total_steps = len(gt_plan_theta)
    frames = []
    for i in range(total_steps + 1):
        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)

        gt_t, gt_td = _break_wrapped(gt_theta[: i + 1], gt_theta_dot[: i + 1])
        ax.plot(gt_t, gt_td, color="tab:blue", linewidth=1.3, label="GT MPPI (actual)")
        ax.scatter(gt_theta[i : i + 1], gt_theta_dot[i : i + 1], color="tab:blue", s=30, zorder=3)

        lr_t, lr_td = _break_wrapped(learned_theta[: i + 1], learned_theta_dot[: i + 1])
        ax.plot(lr_t, lr_td, color="tab:orange", linewidth=1.3, label="Learned MPPI (actual)")
        ax.scatter(learned_theta[i : i + 1], learned_theta_dot[i : i + 1], color="tab:orange", s=30, zorder=3)

        if i < total_steps:
            gp_t, gp_td = _break_wrapped(
                [gt_theta[i], *gt_plan_theta[i]], [gt_theta_dot[i], *gt_plan_theta_dot[i]]
            )
            ax.plot(gp_t, gp_td, color="tab:blue", linewidth=1.1, linestyle="--", alpha=0.7, label="GT plan")

            lp_t, lp_td = _break_wrapped(
                [learned_theta[i], *learned_plan_theta[i]], [learned_theta_dot[i], *learned_plan_theta_dot[i]]
            )
            ax.plot(lp_t, lp_td, color="tab:orange", linewidth=1.1, linestyle="--", alpha=0.7, label="Learned plan")

        ax.axvline(0.0, color="gray", linewidth=0.6, linestyle="--")
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-_DISPLAY_VEL_LIMIT, _DISPLAY_VEL_LIMIT)
        ax.set_xlabel("θ (rad)")
        ax.set_ylabel("θ̇ (rad/s)")

        title = f"Phase space  (t={i})"
        if i < total_steps:
            title += (
                f"\nLearned plan's estimated final state: "
                f"θ={learned_plan_theta[i][-1]:+.2f}  θ̇={learned_plan_theta_dot[i][-1]:+.2f}"
            )
        ax.set_title(title, fontsize=9)
        ax.legend(loc="upper right", fontsize=7)
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
    w_u_latent = st.number_input("Action weight (learned)", min_value=0.0, value=0.001, step=0.001, format="%.4f")

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
            n_context=n_context, total_steps=total_steps, cfg=cfg, action_weight=w_u_gt, device=device,
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
st.markdown(
    "Both panels render real `PendulumPixelEnv` frames — the learned side plans with the "
    "checkpoint's dynamics but only ever executes on the real pendulum, re-grounding its "
    "latent state from real frames every step."
)
st.image(pixel_gif, use_container_width=False)

with st.spinner("Rendering phase-space animation…"):
    phase_frames = build_phase_space_frames(
        gt_theta=gt_result["theta"], gt_theta_dot=gt_result["theta_dot"],
        gt_plan_theta=gt_result["plan_theta"], gt_plan_theta_dot=gt_result["plan_theta_dot"],
        learned_theta=latent_result["theta"], learned_theta_dot=latent_result["theta_dot"],
        learned_plan_theta=latent_result["plan_theta"], learned_plan_theta_dot=latent_result["plan_theta_dot"],
    )
    phase_gif = frames_to_gif(phase_frames, fps)

st.subheader("Phase space: trajectory so far (solid) vs. current planned horizon (dashed)")
st.markdown(
    "A well-calibrated planner's dashed preview should barely move step to step, since the "
    "trajectory ends up following it closely; a planner replanning very differently each step "
    "will show a dashed line that visibly reshapes every frame."
)
st.image(phase_gif, use_container_width=False)

st.subheader("Decoded imagined plan at one control step")
st.markdown(
    "MPPI replans from scratch every control step, so the imagined horizon can look "
    "completely different step to step — averaging or animating across steps would just "
    "blur that together. Instead, pick one step below to decode *that step's* full "
    "H-frame plan straight from the learned dynamics (`controlled_step` → `decode` → "
    "`decode_latent`, the same pipeline `WorldModel.dream` uses), and compare it against "
    "the real frame the pendulum was actually in when that plan was made."
)
n_control_steps = len(latent_result["plan_actions"])
plan_step = st.slider(
    "Control step to inspect", min_value=0, max_value=n_control_steps - 1, value=0
)
with st.spinner("Decoding imagined plan…"):
    plan_frames_u8 = decode_plan_frames(
        world_model=world_model,
        q0=latent_result["plan_q0"][plan_step],
        p0=latent_result["plan_p0"][plan_step],
        actions=latent_result["plan_actions"][plan_step],
    )
    plan_pil = [
        Image.fromarray(f).resize((display_size, display_size), Image.BILINEAR)
        for f in plan_frames_u8
    ]
    plan_gif = frames_to_gif(plan_pil, fps)

col_real, col_plan = st.columns(2)
with col_real:
    st.caption(f"Real frame at t={plan_step} (start of this step's plan)")
    real_frame = Image.fromarray(latent_frames_u8[max(plan_step - 1, 0)]).resize(
        (display_size, display_size), Image.BILINEAR
    )
    st.image(real_frame, use_container_width=False)
with col_plan:
    horizon_len = len(latent_result["plan_actions"][plan_step])
    st.caption(f"Imagined {horizon_len}-step plan from t={plan_step}")
    st.image(plan_gif, use_container_width=False)
