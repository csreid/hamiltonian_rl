"""Streamlit visualiser: interactively sweep a Phase-1 autoencoder's latent space.

Usage:
    streamlit run pendulum_feature_explorer.py

Loads a Phase-1 (or later) checkpoint, detects which latent dims the L0
hard-concrete gate has actually kept open, and gives you sliders to move
around in that space — either one slider per active dim, or 3 sliders along
the top PCA components fit over active dims from real encoded rollouts.
Every slider move decodes straight back to a pixel frame via f_psi + decoder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).parent))

from data.pendulum import collect_seeded_random_rollouts
from hamilton_rl.models import WorldModel
from hamilton_rl.streamlit_common import pick_checkpoint


# ── Model loading ────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading world model…")
def load_model(pt_path_str: str) -> WorldModel:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return WorldModel.load(Path(pt_path_str), device)


# ── Active-dim detection ─────────────────────────────────────────────────────


@torch.no_grad()
def active_dim_mask(model: WorldModel) -> torch.Tensor:
    """Boolean (latent_dim,) mask of dims the L0 gate keeps open.

    Uses the same per-dim "probability this dim is used" that
    HardConcreteGate.l0_penalty sums to get effective_dim, thresholded at
    0.5 — the standard MAP criterion for a Hard Concrete gate, and the one
    consistent with the effective_dim number already logged during training.
    Falls back to "every dim active" when the checkpoint has no gate.
    """
    gate = model.autoencoder.encoder.gate
    if gate is None:
        return torch.ones(model.latent_dim, dtype=torch.bool)
    per_dim_prob = torch.sigmoid(
        gate.log_alpha - gate.temperature * torch.log(torch.tensor(-gate.gamma / gate.zeta))
    )
    return per_dim_prob > 0.5


# ── Sample encoding (baseline + PCA basis) ───────────────────────────────────


@st.cache_data(show_spinner="Encoding sample rollouts…")
def encode_sample_latents(
    _model: WorldModel,
    ckpt_path_str: str,
    n_samples: int,
    rollout_len: int,
    img_size: int,
    damping: float,
) -> torch.Tensor:
    """Encode a batch of seeded random rollouts through the frozen encoder.

    Returns (N, latent_dim) h vectors on CPU, pooled across all timesteps of
    all rollouts — the same data distribution Phase 1 trained on.
    ``ckpt_path_str`` is the real cache key; ``_model`` is excluded from
    hashing (leading underscore) since nn.Modules aren't hashable.
    """
    device = next(_model.autoencoder.parameters()).device
    rollouts = collect_seeded_random_rollouts(
        n_samples=n_samples, rollout_len=rollout_len, img_size=img_size, damping=damping,
    )
    all_h = []
    with torch.no_grad():
        for frames, _actions, _states in rollouts:
            ctx = frames.unsqueeze(0).to(device)
            mu_all, _ = _model.autoencoder.encoder.forward_all(ctx)
            all_h.append(mu_all.squeeze(0).cpu())
    return torch.cat(all_h, dim=0)


def fit_pca(h_active: torch.Tensor, n_components: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PCA over (N, d_active). Returns (mean (d,), components (d, k), std (k,))."""
    mean = h_active.mean(dim=0)
    centered = h_active - mean
    k = min(n_components, h_active.shape[1], h_active.shape[0] - 1)
    _u, s, v = torch.pca_lowrank(centered, q=k)
    std = s / max(h_active.shape[0] - 1, 1) ** 0.5
    return mean, v[:, :k], std[:k]


# ── Decoding ──────────────────────────────────────────────────────────────────


@torch.no_grad()
def decode_h(model: WorldModel, h: torch.Tensor) -> np.ndarray:
    device = next(model.autoencoder.parameters()).device
    frame = model.autoencoder.decode_latent(h.unsqueeze(0).to(device)).squeeze(0).cpu()
    return (frame.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Pendulum Feature Explorer", layout="wide")
st.title("Pendulum Latent Feature Explorer")
st.caption(
    "Detects the latent dims the L0 gate kept open and lets you sweep them "
    "by hand or along the top-3 PCA axes of that active subspace."
)

with st.sidebar:
    st.header("Checkpoint")
    models_root = Path("models")
    ckpt_path = pick_checkpoint(models_root, "Phase 1 model", "p1")

    st.divider()
    st.header("Sample rollouts (for baseline + PCA)")
    n_samples = st.number_input("Total sampled steps", min_value=200, max_value=20000, value=2000, step=200)
    rollout_len = st.number_input("Steps per seeded rollout", min_value=5, max_value=200, value=30, step=5)

try:
    world_model = load_model(str(ckpt_path))
    device = next(world_model.autoencoder.parameters()).device
except Exception as exc:
    st.error(f"Failed to load checkpoint:\n\n```\n{exc}\n```")
    st.stop()

data_cfg = world_model.data_config
latent_dim = world_model.latent_dim
img_size = data_cfg.get("img_size", 64)
damping = data_cfg.get("damping", 0.0)

mask = active_dim_mask(world_model)
active_idx = torch.nonzero(mask).flatten().tolist()
n_active = len(active_idx)

with st.sidebar:
    st.caption(
        f"latent_dim={latent_dim}  active_dims={n_active}  "
        f"img_size={img_size}  damping={damping}  device={device}"
    )

if world_model.autoencoder.encoder.gate is None:
    st.warning(
        "This checkpoint has no L0 gate (`use_gate=False` at train time) — "
        "treating every latent dim as active."
    )

if n_active == 0:
    st.error("The L0 gate has closed every dim — nothing to sweep.")
    st.stop()

h_samples = encode_sample_latents(
    world_model, str(ckpt_path), int(n_samples), int(rollout_len), img_size, damping,
)
baseline = h_samples.mean(dim=0)  # (latent_dim,) — inactive dims are already ~0 (gated at encode time)
dim_std = h_samples.std(dim=0)    # (latent_dim,) — per-dim scale, for sizing sliders

tab_dims, tab_pca = st.tabs(["Per-dimension sliders", "PCA (3D)"])

with tab_dims:
    st.subheader(f"{n_active} active latent dim(s)")
    st.caption("Each slider spans ±3σ of that dim's value over the sampled rollouts.")
    cols = st.columns(2)
    h = baseline.clone()
    for i, dim in enumerate(active_idx):
        with cols[i % 2]:
            span = max(3.0 * float(dim_std[dim]), 1e-3)
            lo, hi = float(baseline[dim]) - span, float(baseline[dim]) + span
            val = st.slider(
                f"h[{dim}]  (σ={dim_std[dim]:.3f})", min_value=lo, max_value=hi,
                value=float(baseline[dim]), step=span / 40,
                key=f"dim_slider_{dim}",
            )
            h[dim] = val

    col_img, _ = st.columns([1, 2])
    with col_img:
        st.image(decode_h(world_model, h), caption="Reconstructed frame", width=img_size * 3)

with tab_pca:
    if n_active < 2:
        st.info("Need at least 2 active dims to fit a PCA basis.")
    else:
        h_active_samples = h_samples[:, active_idx]
        n_components = min(3, n_active)
        mean_active, components, std = fit_pca(h_active_samples, n_components)

        st.subheader(f"Top {n_components} PCA component(s) over the {n_active} active dim(s)")
        pc_coeffs = []
        for i in range(n_components):
            coeff = st.slider(
                f"PC{i + 1}  (σ={std[i]:.3f})", min_value=-3.0, max_value=3.0,
                value=0.0, step=0.1, key=f"pca_slider_{i}",
            )
            pc_coeffs.append(coeff)

        offset_active = sum(
            coeff * std[i] * components[:, i] for i, coeff in enumerate(pc_coeffs)
        )
        h_active = mean_active + offset_active

        h = baseline.clone()
        h[active_idx] = h_active

        col_img, _ = st.columns([1, 2])
        with col_img:
            st.image(decode_h(world_model, h), caption="Reconstructed frame", width=img_size * 3)
