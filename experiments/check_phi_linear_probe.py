"""Linear probes on Phase 2/3's post-phi phase space, mirroring
`_log_latent_scatter_phase1` (which probes Phase 1's `f_psi(h)`) but for
`dynamics.phi(h)` instead -- the flow actually used for the Hamiltonian
rollout, and the one `h_source="canonical"` depends on being physically
meaningful.

Two probes are fit and plotted side by side:

  1. "full"  -- linear regression from the *entire* (q, p) vector (all
     latent_dim components phi outputs) to (cos theta, sin theta, theta_dot).
     Tells you whether phi as a whole preserves state information linearly
     recoverable from *somewhere* in the vector.

  2. "phys"  -- linear regression from *just* the physical sub-block
     (q[..., :n_phys], p[..., :n_phys]) -- the only numbers the fixed
     canonical Hamiltonian ever looks at. Tells you whether the specific
     quantity h_source="canonical" depends on actually carries state
     information linearly, as opposed to it being smeared elsewhere in the
     vector (nuisance dims) while q_phys/p_phys itself is uninformative.

A high "full" R^2 next to a low "phys" R^2 would confirm the physical block
specifically failed to capture theta/theta_dot, independent of whatever the
rest of phi's output is doing.

Usage in a Jupyter cell (on the workstation, with a GPU):

    %run experiments/check_phi_linear_probe.py --phase2-checkpoint <path/to/best.pt>

or import and call `run(...)` directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hamilton_rl.checkpoint import load_world_model
from data.pendulum import (
    collect_random_trajectories,
    collect_spin_trajectories,
    collect_val_trajectories,
)


@torch.no_grad()
def _encode_trajectories(
    model, val_trajs: list, device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Each trajectory -> s_all = phi(h_all) (T, latent_dim), plus its states (T, 3)."""
    model.eval()
    all_s, all_st = [], []
    for frames, _actions, states in val_trajs:
        ctx = frames.unsqueeze(0).to(device)
        mu_all, _ = model.autoencoder.encoder.forward_all(ctx)
        h_all = mu_all.squeeze(0)
        s_all = model.dynamics.phi(h_all).cpu()
        all_s.append(s_all)
        all_st.append(states.float())
    return all_s, all_st


def _plot_probe(
    per_policy_x: dict[str, list[torch.Tensor]],
    per_policy_st: dict[str, list[torch.Tensor]],
    title: str,
) -> plt.Figure:
    train_x = torch.cat([x for xs in per_policy_x.values() for x in xs[0::2]], dim=0)
    train_st = torch.cat([st for sts in per_policy_st.values() for st in sts[0::2]], dim=0)
    A = torch.linalg.lstsq(train_x, train_st).solution

    val_pred, val_true = {}, {}
    for label in per_policy_x:
        val_x = torch.cat(per_policy_x[label][1::2], dim=0)
        val_st = torch.cat(per_policy_st[label][1::2], dim=0)
        val_pred[label] = (val_x @ A).numpy()
        val_true[label] = val_st.numpy()

    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, name in enumerate(["cos(θ)", "sin(θ)", "θ̇ (rad/s)"]):
        all_true_i, all_pred_i = [], []
        for j, label in enumerate(per_policy_x):
            true_i, pred_i = val_true[label][:, i], val_pred[label][:, i]
            axes[i].scatter(
                true_i, pred_i, s=2, alpha=0.3, color=colors[j % len(colors)],
                label=label, linewidths=0,
            )
            all_true_i.append(true_i)
            all_pred_i.append(pred_i)
        true_i = np.concatenate(all_true_i)
        pred_i = np.concatenate(all_pred_i)
        lo, hi = min(true_i.min(), pred_i.min()), max(true_i.max(), pred_i.max())
        axes[i].plot([lo, hi], [lo, hi], "r--", linewidth=0.8)
        axes[i].set_xlabel(f"True {name}")
        axes[i].set_ylabel(f"Predicted {name}")
        ss_res = ((true_i - pred_i) ** 2).sum()
        ss_tot = ((true_i - true_i.mean()) ** 2).sum()
        axes[i].set_title(f"{name}  R²={1 - ss_res / (ss_tot + 1e-8):.3f}")
    axes[0].legend(markerscale=4, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def run(
    phase2_checkpoint: str,
    n_val: int = 20,
    val_steps: int | None = None,
    device: str | None = None,
):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Loading checkpoint from {phase2_checkpoint}...")
    model = load_world_model(phase2_checkpoint, device)
    data_cfg = model.data_config or {}
    img_size = data_cfg.get("img_size", 64)
    damping = data_cfg.get("damping", 0.0)
    steps = val_steps or data_cfg.get("max_steps", 200) * 2

    print(f"Collecting {n_val} val episodes per policy ({steps} steps each)...")
    val_traj_sets = [
        (collect_val_trajectories(
            n_episodes=n_val, img_size=img_size, max_steps=steps,
            energy_k=data_cfg.get("energy_k", 1.0), damping=damping,
        ), "energy-pump"),
        (collect_random_trajectories(
            n_episodes=n_val, img_size=img_size, max_steps=steps, damping=damping,
        ), "random"),
        (collect_spin_trajectories(
            n_episodes=n_val, img_size=img_size, max_steps=steps, damping=damping,
        ), "spin"),
    ]

    print("Encoding trajectories through encoder + phi...")
    per_policy_s, per_policy_st = {}, {}
    for val_trajs, label in val_traj_sets:
        all_s, all_st = _encode_trajectories(model, val_trajs, device)
        per_policy_s[label] = all_s
        per_policy_st[label] = all_st

    dyn = model.dynamics
    q_dim = dyn.latent_dim // 2
    n_phys = getattr(dyn, "n_phys", q_dim)
    print(f"latent_dim={dyn.latent_dim}, q_dim={q_dim}, n_phys={n_phys}")

    fig_full = _plot_probe(
        per_policy_s, per_policy_st,
        "Full (q, p) vector -> state, post-phi (held-out trajectories)",
    )

    per_policy_s_phys = {
        label: [torch.cat([s[:, :n_phys], s[:, q_dim:q_dim + n_phys]], dim=-1) for s in ss]
        for label, ss in per_policy_s.items()
    }
    fig_phys = _plot_probe(
        per_policy_s_phys, per_policy_st,
        f"Physical block only (q[:{n_phys}], p[:{n_phys}]) -> state, post-phi",
    )

    plt.show()
    return fig_full, fig_phys, per_policy_s, per_policy_st


@click.command()
@click.option("--phase2-checkpoint", required=True, type=str,
              help="Path to a Phase 2/3 world-model checkpoint (.pt) with dynamics filled in.")
@click.option("--n-val", default=20, show_default=True, help="Episodes per policy type.")
@click.option("--val-steps", default=None, type=int, help="Steps per episode (default: 2x training max_steps).")
@click.option("--device", default=None, help="Override device (default: cuda if available).")
def main(phase2_checkpoint, n_val, val_steps, device):
    run(phase2_checkpoint, n_val=n_val, val_steps=val_steps, device=device)


if __name__ == "__main__":
    main()
