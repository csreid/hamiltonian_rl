"""Streamlit visualiser: ground-truth vs port-Hamiltonian dreamed rollout.

Usage:
    streamlit run pendulum_dreamer.py

Loads a Phase 1 (ControlledDHGN_LSTM) and Phase 2 (HamiltonianFlowModel)
checkpoint, collects one Pendulum-v1 episode under a selectable policy, encodes
N context frames via the LSTM, then rolls out port-Hamiltonian dynamics in phase
space and decodes back to pixels for comparison.
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
import yaml
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))

from data.pendulum import (
    collect_random_trajectories,
    collect_spin_trajectories,
    collect_val_trajectories,
)
from phgn_lstm import ControlledDHGN_LSTM, HamiltonianFlowModel


# ── Model loading ─────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading Phase 1 model…")
def load_phase1_model(pt_path_str: str) -> tuple[ControlledDHGN_LSTM, dict]:
    pt_path = Path(pt_path_str)
    hparams: dict = {}
    yaml_path = pt_path.with_suffix(".yaml")
    if yaml_path.exists():
        with open(yaml_path) as fh:
            doc = yaml.safe_load(fh)
            hparams = doc.get("hparams", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ControlledDHGN_LSTM(
        pos_ch=hparams.get("pos_ch", 8),
        img_ch=3,
        dt=hparams.get("dt", 0.05),
        feat_dim=hparams.get("feat_dim", 256),
        img_size=hparams.get("img_size", 64),
        latent_dim=hparams.get("latent_dim", 32),
        control_dim=1,
        separable=hparams.get("separable", True),
        learn_structure=hparams.get("learn_structure", True),
        damping=hparams.get("damping", 0.0),
    ).to(device)
    sd = torch.load(pt_path, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    return model, hparams


@st.cache_resource(show_spinner="Loading Phase 2 model…")
def load_phase2_model(
    pt_path_str: str, latent_dim: int, dt: float
) -> HamiltonianFlowModel:
    pt_path = Path(pt_path_str)
    hparams: dict = {}
    yaml_path = pt_path.with_suffix(".yaml")
    if yaml_path.exists():
        with open(yaml_path) as fh:
            doc = yaml.safe_load(fh)
            hparams = doc.get("hparams", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HamiltonianFlowModel(
        latent_dim=latent_dim,
        control_dim=1,
        separable=hparams.get("separable", True),
        learn_structure=hparams.get("learn_structure", True),
        dt=hparams.get("dt", dt),
        damping=hparams.get("damping", 0.0),
    ).to(device)
    sd = torch.load(pt_path, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    return model


# ── Episode collection ────────────────────────────────────────────────────────

_POLICY_LABELS = {
    "energy_pump": "Energy pump",
    "random": "Random",
    "spin": "Spin",
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
    else:
        raise ValueError(f"Unknown policy: {policy!r}")
    return eps[0]


# ── Dreamed rollout ───────────────────────────────────────────────────────────


@torch.no_grad()
def run_dreamed_rollout(
    phase1_model: ControlledDHGN_LSTM,
    dyn_model: HamiltonianFlowModel,
    frames: torch.Tensor,
    actions: torch.Tensor,
    n_context: int,
    rollout_length: int,
    device: torch.device,
) -> dict:
    """Encode n_context frames, dream rollout_length steps, return numpy arrays.

    Pipeline per dreamed step k (k = 0 … n_steps-1):
        1. u = actions[n_context - 1 + k]
        2. (q, p) = dyn_model.controlled_step(q, p, u)       [phase-2 dynamics]
        3. h_pred = dyn_model.decode(q, p)                     [phi^{-1}]
        4. s_pred = phase1_model.f_psi(h_pred)                 [phase-1 NF]
        5. frame  = phase1_model.decoder(s_pred[:, :q_dim])    [CNN decoder]

    Returns a dict with:
        gt_frames    : list of (H, W, 3) uint8 arrays  (n_steps frames starting at n_context)
        dream_frames : list of (H, W, 3) uint8 arrays  (same length)
        n_steps      : actual number of dreamed steps
    """
    q_dim = phase1_model.latent_dim // 2

    ctx = frames[:n_context].unsqueeze(0).to(device)       # (1, n_context, C, H, W)
    mu_ctx, _ = phase1_model.encoder.forward_all(ctx)       # (1, n_context, latent_dim)
    h = mu_ctx[:, -1]                                        # (1, latent_dim)

    q, p = dyn_model.encode(h)                              # (1, q_dim) each

    n_steps = min(rollout_length, len(actions) - (n_context - 1))

    dream_frames: list[np.ndarray] = []
    for k in range(n_steps):
        u = actions[n_context - 1 + k].view(1, 1).to(device)
        q, p = dyn_model.controlled_step(q, p, u)
        h_pred = dyn_model.decode(q, p)                     # (1, latent_dim)
        s_pred = phase1_model.f_psi(h_pred)                 # (1, latent_dim)
        frame_pred = phase1_model.decoder(s_pred[:, :q_dim])  # (1, C, H, W)
        arr = frame_pred.squeeze(0).detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        dream_frames.append((arr * 255).astype(np.uint8))

    gt_slice = frames[n_context : n_context + n_steps]      # (n_steps, C, H, W)
    gt_frames: list[np.ndarray] = [
        (gt_slice[i].clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        for i in range(len(gt_slice))
    ]

    return {"gt_frames": gt_frames, "dream_frames": dream_frames, "n_steps": n_steps}


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
    st.header("Checkpoints")

    models_root = Path("models")
    pt_files = sorted(
        f for f in models_root.rglob("*.pt") if f.stem != "h_cache"
    ) if models_root.exists() else []

    if not pt_files:
        st.warning("No `.pt` checkpoints found under `models/` (excluding h_cache).")
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

    st.subheader("Phase 1  (encoder + decoder)")
    p1_path = _pick_checkpoint("Phase 1", "p1")

    st.subheader("Phase 2  (dynamics flow)")
    p2_path = _pick_checkpoint("Phase 2", "p2")

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


# ── Load models ───────────────────────────────────────────────────────────────

try:
    phase1_model, p1_hparams = load_phase1_model(str(p1_path))
    device = next(phase1_model.parameters()).device
except Exception as exc:
    st.error(f"Failed to load Phase 1 checkpoint:\n\n```\n{exc}\n```")
    st.stop()

latent_dim = p1_hparams.get("latent_dim", 32)
img_size = p1_hparams.get("img_size", 64)
dt = p1_hparams.get("dt", 0.05)
damping = p1_hparams.get("damping", 0.0)

try:
    dyn_model = load_phase2_model(str(p2_path), latent_dim=latent_dim, dt=dt)
    dyn_model = dyn_model.to(device)
except Exception as exc:
    st.error(f"Failed to load Phase 2 checkpoint:\n\n```\n{exc}\n```")
    st.stop()

with st.sidebar:
    st.caption(
        f"latent_dim={latent_dim}  img_size={img_size}  "
        f"dt={dt}  damping={damping}  device={device}"
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
                phase1_model=phase1_model,
                dyn_model=dyn_model,
                frames=frames,
                actions=actions,
                n_context=int(n_context),
                rollout_length=int(rollout_length),
                device=device,
            )
        except Exception as exc:
            st.error(f"Rollout failed:\n\n```\n{exc}\n```")
            st.stop()

    st.session_state.update(
        rollout=rollout,
        p1_path=str(p1_path),
        p2_path=str(p2_path),
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
    f"Phase 1: `{st.session_state['p1_path']}`  |  "
    f"Phase 2: `{st.session_state['p2_path']}`"
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
