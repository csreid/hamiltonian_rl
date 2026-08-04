"""Streamlit visualiser: ground-truth vs port-Hamiltonian dreamed rollout.

Usage:
    streamlit run pendulum_dreamer.py

Loads a unified world-model checkpoint (LSTM autoencoder + Hamiltonian flow
dynamics in one .pt), collects one Pendulum-v1 episode under a selectable
policy, encodes N context frames via the LSTM, then rolls out port-Hamiltonian
dynamics in phase space and decodes back to pixels for comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).parent))

from data.pendulum import (
    collect_random_trajectories,
    collect_spin_trajectories,
    collect_val_trajectories,
    collect_zero_trajectories,
)
from hamilton_rl.models import WorldModel
from hamilton_rl.streamlit_common import (
    build_sidebyside_frames,
    frames_to_gif,
    pick_checkpoint,
    to_uint8,
)


# ── Model loading ─────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading world model…")
def load_model(pt_path_str: str) -> WorldModel:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return WorldModel.load(Path(pt_path_str), device)


# ── Episode collection ────────────────────────────────────────────────────────

_POLICY_LABELS = {
    "energy_pump": "Energy pump",
    "random": "Random",
    "spin": "Spin",
    "zero": "Zero action (uncontrolled)",
}


def collect_pendulum_episode(
    policy: str,
    img_size: int,
    max_steps: int,
    energy_k: float = 1.0,
    damping: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect a single Pendulum-v1 episode under the given policy.

    Returns (frames, actions, states):
        frames  : (T+1, 3, img_size, img_size) float32 [0, 1]
        actions : (T,) float32
        states  : (T+1, 3) float32 — (cos θ, sin θ, θ̇)
    """
    common = dict(n_episodes=1, img_size=img_size, max_steps=max_steps, damping=damping)
    if policy == "energy_pump":
        eps = collect_val_trajectories(**common, energy_k=energy_k)
    elif policy == "random":
        eps = collect_random_trajectories(**common)
    elif policy == "spin":
        eps = collect_spin_trajectories(**common)
    elif policy == "zero":
        eps = collect_zero_trajectories(**common)
    else:
        raise ValueError(f"Unknown policy: {policy!r}")
    return eps[0]


# ── Dreamed rollout ───────────────────────────────────────────────────────────


@torch.no_grad()
def run_dreamed_rollout(
    world_model: WorldModel,
    frames: torch.Tensor,
    actions: torch.Tensor,
    n_context: int,
    rollout_length: int,
) -> dict:
    """Encode n_context frames, dream rollout_length steps, return numpy arrays.

    Returns a dict with:
        gt_frames    : list of (H, W, 3) uint8 arrays  (n_steps frames starting at n_context)
        dream_frames : list of (H, W, 3) uint8 arrays  (same length)
        n_steps      : actual number of dreamed steps
    """
    dreamed = world_model.dream(
        frames, actions, n_context=n_context, n_steps=rollout_length
    )
    n_steps = len(dreamed)
    gt_slice = frames[n_context : n_context + n_steps]  # (n_steps, C, H, W)
    return {
        "gt_frames": to_uint8(gt_slice),
        "dream_frames": to_uint8(dreamed),
        "n_steps": n_steps,
    }


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Pendulum Dreamer", layout="wide")
st.title("Pendulum Port-Hamiltonian Dreamer")

with st.sidebar:
    st.header("Checkpoint")

    models_root = Path("models")
    ckpt_path = pick_checkpoint(models_root, "World model", "wm")

    st.divider()
    st.header("Episode")
    policy = st.selectbox(
        "Policy",
        list(_POLICY_LABELS.keys()),
        format_func=_POLICY_LABELS.__getitem__,
    )
    max_steps = st.number_input(
        "Max episode steps", min_value=10, max_value=1000, value=100, step=10
    )
    energy_k = st.slider(
        "Energy-pump gain k  (ignored for random/spin)",
        0.1, 5.0, 1.0, step=0.1,
    )

    st.divider()
    st.header("Rollout")
    n_context = st.slider("Context frames", min_value=1, max_value=20, value=2, step=1)
    rollout_length = st.slider(
        "Rollout length (steps)", min_value=5, max_value=300, value=60, step=5
    )

    st.divider()
    generate_btn = st.button("▶ Generate", type="primary", use_container_width=True)


# ── Load model ────────────────────────────────────────────────────────────────

try:
    world_model = load_model(str(ckpt_path))
    device = next(world_model.autoencoder.parameters()).device
except Exception as exc:
    st.error(f"Failed to load checkpoint:\n\n```\n{exc}\n```")
    st.stop()

if world_model.dynamics is None:
    st.warning(
        f"`{ckpt_path}` is a Phase-1-only checkpoint (no dynamics). "
        "Pick a Phase 2 checkpoint to dream."
    )
    st.stop()

data_cfg = world_model.data_config
latent_dim = world_model.latent_dim
img_size = data_cfg.get("img_size", 64)
dt = world_model.dynamics.dt
damping = data_cfg.get("damping", 0.0)

with st.sidebar:
    st.caption(
        f"latent_dim={latent_dim}  img_size={img_size}  "
        f"dt={dt}  damping={damping}  "
        f"integrator={world_model.dynamics.integrator}  device={device}"
    )


# ── Generation ────────────────────────────────────────────────────────────────

if generate_btn:
    with st.spinner("Collecting Pendulum episode…"):
        try:
            frames, actions, states = collect_pendulum_episode(
                policy=policy,
                img_size=img_size,
                max_steps=int(max_steps),
                energy_k=float(energy_k),
                damping=damping,
            )
        except Exception as exc:
            st.error(f"Episode collection failed:\n\n```\n{exc}\n```")
            st.stop()

    with st.spinner("Running dreamed rollout…"):
        try:
            rollout = run_dreamed_rollout(
                world_model=world_model,
                frames=frames,
                actions=actions,
                n_context=int(n_context),
                rollout_length=int(rollout_length),
            )
        except Exception as exc:
            st.error(f"Rollout failed:\n\n```\n{exc}\n```")
            st.stop()

    st.session_state.update(
        rollout=rollout,
        ckpt_path=str(ckpt_path),
        policy=policy,
        n_context=int(n_context),
    )


# ── Display ───────────────────────────────────────────────────────────────────

rollout = st.session_state.get("rollout")

if rollout is None:
    st.info("Configure settings in the sidebar and press **▶ Generate**.")
    st.stop()

n_steps = rollout["n_steps"]
st.success(
    f"Dreamed **{n_steps}** steps after **{st.session_state['n_context']}** context frames  |  "
    f"Policy: `{_POLICY_LABELS[st.session_state['policy']]}`  |  "
    f"Checkpoint: `{st.session_state['ckpt_path']}`"
)

col_fps, col_size = st.columns(2)
with col_fps:
    fps = st.slider("Playback FPS", min_value=1, max_value=60, value=10, step=1)
with col_size:
    display_size = st.select_slider(
        "Frame size (px)", options=[64, 128, 192, 256, 384], value=128
    )

with st.spinner("Rendering GIF…"):
    composite = build_sidebyside_frames(
        left_frames=rollout["gt_frames"],
        right_frames=rollout["dream_frames"],
        display_size=display_size,
        left_label="GT",
        right_label="PHn",
    )
    gif_bytes = frames_to_gif(composite, fps)

st.subheader("Ground truth  (left)  |  Port-Hamiltonian dream  (right)")
st.markdown(
    "Blue label = GT frame · Orange label = dreamed frame · "
    "Both indexed from t=0 after context window."
)
st.image(gif_bytes, use_container_width=False)
