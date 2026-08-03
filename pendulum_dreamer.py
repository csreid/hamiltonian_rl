"""Streamlit visualiser: ground-truth vs port-Hamiltonian dreamed rollout.

Usage:
    streamlit run pendulum_dreamer.py

Loads a unified world-model checkpoint (LSTM autoencoder + Hamiltonian flow
dynamics in one .pt), collects one Pendulum-v1 episode under a selectable
policy, encodes N context frames via the LSTM, then rolls out port-Hamiltonian
dynamics in phase space and decodes back to pixels for comparison.
"""

from __future__ import annotations

import io
import re
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))

from data.pendulum import (
    collect_random_trajectories,
    collect_spin_trajectories,
    collect_val_trajectories,
    collect_zero_trajectories,
)
from hamilton_rl.models import WorldModel


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


def _to_uint8(frames: torch.Tensor) -> list[np.ndarray]:
    """(N, C, H, W) float [0,1] → list of (H, W, C) uint8 arrays."""
    return [
        (f.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        for f in frames
    ]


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
        "gt_frames": _to_uint8(gt_slice),
        "dream_frames": _to_uint8(dreamed),
        "n_steps": n_steps,
    }


# ── Frame compositing ─────────────────────────────────────────────────────────


def _label_frame(img: Image.Image, text: str, color: tuple) -> Image.Image:
    img = img.copy()
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (img.width - 1, 14)], fill=(0, 0, 0, 180))
    draw.text((3, 2), text, fill=color)
    return img


def build_sidebyside_frames(
    gt_frames: list[np.ndarray],
    dream_frames: list[np.ndarray],
    display_size: int,
    gap: int = 4,
) -> list[Image.Image]:
    """Combine GT (left) and dreamed (right) into one wide frame."""
    out: list[Image.Image] = []
    total_w = display_size * 2 + gap
    for i, (gt_arr, dr_arr) in enumerate(zip(gt_frames, dream_frames)):
        gt_pil = Image.fromarray(gt_arr).resize((display_size, display_size), Image.BILINEAR)
        dr_pil = Image.fromarray(dr_arr).resize((display_size, display_size), Image.BILINEAR)
        gt_pil = _label_frame(gt_pil, f"GT  t={i}", color=(100, 200, 255))
        dr_pil = _label_frame(dr_pil, f"PHn t={i}", color=(255, 160, 60))
        canvas = Image.new("RGB", (total_w, display_size), (40, 40, 40))
        canvas.paste(gt_pil, (0, 0))
        canvas.paste(dr_pil, (display_size + gap, 0))
        out.append(canvas)
    return out


def frames_to_gif(frames: list[Image.Image], fps: float) -> bytes:
    duration_ms = max(20, int(1000 / fps))
    buf = io.BytesIO()
    frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return buf.getvalue()


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Pendulum Dreamer", layout="wide")
st.title("Pendulum Port-Hamiltonian Dreamer")

with st.sidebar:
    st.header("Checkpoint")

    models_root = Path("models")
    _NON_CHECKPOINT_STEMS = {"h_cache", "episodes_cache"}
    pt_files = sorted(
        f for f in models_root.rglob("*.pt") if f.stem not in _NON_CHECKPOINT_STEMS
    ) if models_root.exists() else []

    if not pt_files:
        st.warning("No `.pt` checkpoints found under `models/` (excluding data caches).")
        st.stop()

    # Parse each .pt file into (identifier, date, time_str, file).
    # Expected layout: models/<identifier>/<YYYY-MM-DD>_<HH-MM-SS>/<stem>.pt
    _TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})$")

    # nested dict: identifier → date → time_str → [Path, ...]
    _tree: dict[str, dict[date, dict[str, list[Path]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    _unstructured: list[Path] = []

    for f in pt_files:
        rel = f.relative_to(models_root)
        parts = rel.parts  # e.g. ("pendulum_offline", "2024-04-29_12-36-49", "checkpoint_5.pt")
        if len(parts) >= 3:
            *id_parts, ts_part, _ = parts
            m = _TIMESTAMP_RE.match(ts_part)
            if m:
                identifier = "/".join(id_parts)
                run_date = datetime.strptime(m.group(1), "%Y-%m-%d").date()
                run_time = m.group(2)
                _tree[identifier][run_date][run_time].append(f)
                continue
        _unstructured.append(f)

    def _pick_checkpoint(label: str, key_prefix: str) -> Path:
        """Nested date-aware selector: identifier → date → time → checkpoint."""
        identifiers = sorted(_tree.keys())
        identifier = st.selectbox(
            f"{label} — model",
            identifiers,
            key=f"{key_prefix}_id",
        )
        dates = sorted(_tree[identifier].keys(), reverse=True)
        chosen_date = st.selectbox(
            f"{label} — date",
            dates,
            format_func=lambda d: d.strftime("%A, %B %-d %Y"),
            key=f"{key_prefix}_date",
        )
        times = sorted(_tree[identifier][chosen_date].keys(), reverse=True)
        chosen_time = st.selectbox(
            f"{label} — run",
            times,
            format_func=lambda t: t.replace("-", ":"),
            key=f"{key_prefix}_time",
        )
        files = _tree[identifier][chosen_date][chosen_time]
        file_names = [f.name for f in files]
        chosen_name = st.selectbox(
            f"{label} — checkpoint",
            file_names,
            index=len(file_names) - 1,  # default to last (highest epoch)
            key=f"{key_prefix}_file",
        )
        return models_root / identifier / f"{chosen_date.strftime('%Y-%m-%d')}_{chosen_time}" / chosen_name

    ckpt_path = _pick_checkpoint("World model", "wm")

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
        gt_frames=rollout["gt_frames"],
        dream_frames=rollout["dream_frames"],
        display_size=display_size,
    )
    gif_bytes = frames_to_gif(composite, fps)

st.subheader("Ground truth  (left)  |  Port-Hamiltonian dream  (right)")
st.markdown(
    "Blue label = GT frame · Orange label = dreamed frame · "
    "Both indexed from t=0 after context window."
)
st.image(gif_bytes, use_container_width=False)
