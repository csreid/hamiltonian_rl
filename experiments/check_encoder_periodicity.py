"""Diagnose whether the learned latent q has the wrong angular periodicity
relative to true theta — the suspected cause of the striping/banding seen in
the phase3 ||grad H_true|| vs ||grad H_learned|| scatter plot.

Usage in a Jupyter cell (on the workstation, with a GPU):

    %run experiments/check_encoder_periodicity.py --phase2-checkpoint <path/to/best.pt>

or import and call `run(...)` directly with a path.

What it does:
  1. Loads the world model checkpoint (autoencoder + dynamics).
  2. Rolls out the same zero-action grid episodes used by the phase3
     landscape plots, and encodes them to (q, p) exactly like
     `_collect_grid_qp_samples` does.
  3. Picks out the coordinate(s) to check for periodicity:
       - If the dynamics are in block_mode (h/r/b-source != "learned",
         e.g. canonical H/R/B) -- the current setup -- `dyn.n_phys` says
         how many of the leading q/p dims are the true physical pair (1 for
         the pendulum's single DOF). We use q[..., 0] / p[..., 0] directly
         -- no PCA needed, since it's already the one scalar the canonical
         Hamiltonian is being evaluated on, i.e. the actual quantity whose
         periodicity determines grad_H_true vs. grad_H_learned agreement.
       - Otherwise (fully learned H, no fixed physical sub-block) falls
         back to a top-2 PCA projection of the full q (or p), since there's
         no single latent dim to point to a priori.
  4. Plots:
       a) the (q_phys, p_phys) plane (or 2D PCA plane in the fallback
          case), colored by true theta -- a correct embedding should trace
          a single non-self-intersecting loop as theta goes around once.
          Multiple loops / self-intersections means the encoder is using a
          higher angular frequency than the true pendulum angle.
       b) unwrapped angle(q_phys, p_phys) vs. unwrapped true theta, with a
          linear fit -- the fitted slope is the effective frequency ratio.
          Slope ~1 means correct periodicity; slope ~2, ~3, etc. means the
          learned embedding is aliased/harmonic relative to true theta.
     Also, when n_phys < q_dim (there are nuisance dims beyond the physical
     block), the same two plots are made for the *nuisance* block via PCA,
     since a periodicity mismatch hiding there wouldn't show up in the
     q_phys/p_phys pair at all.
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pendulum_offline import _DRAG_COEFF, _collect_energy_grid_episodes, _collect_grid_qp_samples
from hamilton_rl.checkpoint import load_world_model


def _pca_2d(x: np.ndarray) -> np.ndarray:
    """Project (N, D) onto its top-2 principal components -> (N, 2)."""
    x = x - x.mean(axis=0, keepdims=True)
    if x.shape[1] == 2:
        return x
    if x.shape[1] == 1:
        return np.concatenate([x, np.zeros_like(x)], axis=1)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def _unwrap_sorted(theta: np.ndarray, angle: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort by theta, unwrap both signals along that order, for a clean
    angle-vs-angle fit (raw sample order isn't a walk around the circle)."""
    order = np.argsort(theta)
    theta_sorted = np.unwrap(theta[order])
    angle_sorted = np.unwrap(angle[order])
    return theta_sorted, angle_sorted


def _plot_periodicity(
    theta: np.ndarray, xy: np.ndarray, name: str, xlabel: str, ylabel: str,
) -> plt.Figure:
    """xy: (N, 2) plane whose polar angle should track true theta 1:1 if the
    latent's periodicity matches the physical angle."""
    pc_angle = np.arctan2(xy[:, 1], xy[:, 0])

    theta_u, angle_u = _unwrap_sorted(theta, pc_angle)
    slope, intercept = np.polyfit(theta_u, angle_u, 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sc = axes[0].scatter(xy[:, 0], xy[:, 1], c=theta, cmap="twilight", s=12)
    axes[0].set_title(f"{name}: plane, colored by true theta")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].set_aspect("equal", adjustable="datalim")
    fig.colorbar(sc, ax=axes[0], label="theta (rad)")

    axes[1].scatter(theta_u, angle_u, s=10, alpha=0.4)
    fit_x = np.array([theta_u.min(), theta_u.max()])
    axes[1].plot(
        fit_x, slope * fit_x + intercept, color="crimson",
        label=f"slope = {slope:.2f}  (1.0 = correct periodicity)",
    )
    axes[1].set_xlabel("true theta, unwrapped (rad)")
    axes[1].set_ylabel(f"angle({name}), unwrapped (rad)")
    axes[1].set_title(f"{name}: angular frequency vs. true theta")
    axes[1].legend(loc="best", fontsize=9)

    fig.suptitle(f"{name} periodicity check -- fitted frequency ratio: {slope:.2f}")
    fig.tight_layout()
    return fig


def run(
    phase2_checkpoint: str,
    resolution: int = 20,
    context_frames: int = 5,
    device: str | None = None,
):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Loading checkpoint from {phase2_checkpoint}...")
    model = load_world_model(phase2_checkpoint, device)
    data_cfg = model.data_config or {}

    print("Collecting zero-action grid episodes...")
    episodes = _collect_energy_grid_episodes(
        resolution=resolution,
        img_size=data_cfg.get("img_size", 64),
        damping=data_cfg.get("damping", 0.0),
        drag=data_cfg.get("drag", _DRAG_COEFF),
        context_frames=context_frames,
    )

    print("Encoding to (q, p)...")
    samples = _collect_grid_qp_samples(model, episodes, device=device, vel_clip=None)
    theta = samples["theta"].numpy()
    q = samples["q"].numpy()
    p = samples["p"].numpy()

    dyn = model.dynamics
    block_mode = getattr(dyn, "block_mode", False)
    n_phys = getattr(dyn, "n_phys", q.shape[1]) if block_mode else q.shape[1]
    print(f"block_mode={block_mode}, n_phys={n_phys}, q_dim={q.shape[1]}")

    figs = {}
    if block_mode and n_phys == 1:
        # The actual scalar the canonical H is evaluated on -- exactly the
        # coordinate whose periodicity determines grad_H_true vs.
        # grad_H_learned agreement, no PCA needed.
        phys_plane = np.stack([q[:, 0], p[:, 0]], axis=-1)
        figs["phys"] = _plot_periodicity(
            theta, phys_plane, "physical (q_phys, p_phys)", "q_phys", "p_phys",
        )
    else:
        # No single physical scalar to point to (fully learned H, or
        # n_phys > 1) -- fall back to a PCA plane of the physical block.
        phys_q = q[:, :n_phys]
        phys_p = p[:, :n_phys]
        figs["phys_q"] = _plot_periodicity(theta, _pca_2d(phys_q), "q_phys (PCA)", "PC1", "PC2")
        figs["phys_p"] = _plot_periodicity(theta, _pca_2d(phys_p), "p_phys (PCA)", "PC1", "PC2")

    n_nuis = q.shape[1] - n_phys
    if block_mode and n_nuis > 0:
        # A periodicity mismatch hiding in the nuisance dims wouldn't show
        # up in the physical pair above at all -- check it separately.
        nuis_q = q[:, n_phys:]
        nuis_p = p[:, n_phys:]
        figs["nuis_q"] = _plot_periodicity(theta, _pca_2d(nuis_q), "q_nuisance (PCA)", "PC1", "PC2")
        figs["nuis_p"] = _plot_periodicity(theta, _pca_2d(nuis_p), "p_nuisance (PCA)", "PC1", "PC2")

    plt.show()
    return figs, samples


@click.command()
@click.option("--phase2-checkpoint", required=True, type=str,
              help="Path to a Phase 2/3 world-model checkpoint (.pt) with dynamics filled in.")
@click.option("--resolution", default=20, show_default=True,
              help="Grid resolution (resolution^2 episodes) for the (theta, theta_dot) sweep.")
@click.option("--context-frames", default=5, show_default=True,
              help="Context frames per episode, matching how the checkpoint was validated.")
@click.option("--device", default=None, help="Override device (default: cuda if available).")
def main(phase2_checkpoint, resolution, context_frames, device):
    run(phase2_checkpoint, resolution=resolution, context_frames=context_frames, device=device)


if __name__ == "__main__":
    main()
