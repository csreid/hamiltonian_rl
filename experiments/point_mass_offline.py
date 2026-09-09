"""Offline point-mass world-model training from pixels.

The point-mass analog of ``pendulum_offline.py``, with the pendulum's periodic
angle removed entirely: state is q = (x, y), p = (vx, vy), with no wraparound
anywhere, so none of the pendulum pipeline's angle-discontinuity handling
(sin/cos encoding, fold/tear consistency probes) has an analog here — see
``data/point_mass.py``'s module docstring for the physics.

Same 3-phase regimen as ``pendulum_offline.py``:

    phase1  Train ``TemporalAutoencoder`` (encoder + f_psi + decoder) by
            reconstruction + next-frame prediction + KL, on windows drawn
            from ``data.point_mass.collect_seeded_random_rollouts``. Saves a
            ``WorldModel`` checkpoint with ``dynamics=None``, plus the raw
            rollouts to ``rollout_cache.pt``.
    phase2  Train ``HamiltonianFlowModel`` (physics="point_mass") on frozen
            Phase-1 encoder outputs, with teacher-forced + closed-loop
            rollout losses.
    phase3  End-to-end finetune of encoder+f_psi+decoder+dynamics through the
            full pixel "dreaming" pipeline, anchored by a plain reconstruction
            loss to guard against representation collapse.

Diagnostics that ``pendulum_offline.py`` needs for its 2D (θ, θ̇) phase space
(covering-grid coverage plots, energy-landscape grids, cross-trajectory
fold/tear consistency probes) either don't apply here (no periodicity to
break) or are replaced with simpler position/velocity scatter plots, since
the point mass's phase space has no topology to violate.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.point_mass import (  # noqa: E402
    _DRAG_COEFF,
    _K_WALL,
    _L_WALL,
    _U_MAX,
)
from data.point_mass import MultiRolloutDataset  # noqa: E402
from data.point_mass import collect_seeded_random_rollouts  # noqa: E402
from experiments.pendulum_offline import (  # noqa: E402
    _eval_loss_phase1,
    _log_hparams_table,
    _log_hparams_text,
)
from hamilton_rl.checkpoint import load_world_model, make_run_dir  # noqa: E402
from hamilton_rl.cli_config import config_option  # noqa: E402
from hamilton_rl.models import HamiltonianFlowModel, TemporalAutoencoder, WorldModel  # noqa: E402


# ── Plotting / logging helpers ────────────────────────────────────────────────


def _plot_phase_space_coverage(rollouts) -> plt.Figure:
    """Scatter of every visited (x, y) position across all training rollouts.

    Unlike the pendulum's covering-grid + scipy.griddata energy-landscape
    plots (needed because (theta, theta_dot) has a periodic axis and a
    nontrivial energy landscape to interpolate), the point mass's position
    space is flat and directly plottable — no interpolation needed.
    """
    xs = np.concatenate([r[2][:, 0].numpy() for r in rollouts])
    ys = np.concatenate([r[2][:, 1].numpy() for r in rollouts])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(xs, ys, s=2, alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Position coverage")
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


@torch.no_grad()
def _log_reconstruction_video(
    model: TemporalAutoencoder,
    val_traj: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    tag: str,
    fps: int = 10,
) -> None:
    """Real frames | reconstructed frames, side by side, as a TensorBoard video."""
    frames, _, _ = val_traj
    frames = frames.unsqueeze(0).to(device)  # (1, T+1, C, H, W)
    q_dim = model.latent_dim // 2
    T1 = frames.shape[1]
    mu_all, _ = model.encoder.forward_all(frames)
    s = model.f_psi(mu_all.reshape(T1, -1)[:, :q_dim])
    pred = model.decoder(s).reshape(1, T1, *frames.shape[2:])
    combined = torch.cat([frames, pred], dim=-1)  # side by side along width
    writer.add_video(tag, combined.clamp(0, 1).cpu(), epoch, fps=fps)


@torch.no_grad()
def _log_dreamed_video(
    world_model: WorldModel,
    val_traj: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    seed_ctx_len: int,
    tag: str,
    fps: int = 10,
) -> None:
    """Real frames | closed-loop dreamed frames from the end of the seed context."""
    dyn_model = world_model.dynamics
    autoencoder = world_model.autoencoder
    frames, actions, _ = val_traj
    frames = frames.unsqueeze(0).to(device)
    actions = actions.unsqueeze(0).to(device=device, dtype=frames.dtype)
    W1 = frames.shape[1]
    W = W1 - 1
    q_dim = dyn_model.latent_dim // 2

    h_all, _ = autoencoder.encoder.forward_all(frames)
    q_all, p_all = dyn_model.encode(h_all.reshape(W1, -1))
    q_all = q_all.reshape(1, W1, q_dim)
    p_all = p_all.reshape(1, W1, -1)

    ctx = min(seed_ctx_len, W)
    k = ctx - 1
    q_roll, p_roll = q_all[:, k], p_all[:, k]
    dreamed = [autoencoder.decode_latent(dyn_model.decode(q_roll, p_roll))]
    for t in range(k + 1, W):
        q_roll, p_roll = dyn_model.controlled_step(q_roll, p_roll, actions[:, t])
        dreamed.append(autoencoder.decode_latent(dyn_model.decode(q_roll, p_roll)))
    dreamed = torch.stack(dreamed, dim=1)  # (1, W-k, C, H, W)
    real = frames[:, k + 1 :]
    combined = torch.cat([real, dreamed], dim=-1)
    writer.add_video(tag, combined.clamp(0, 1).cpu(), epoch, fps=fps)


def _ema_convergence_check(
    ema: float | None, new_value: float, alpha: float, patience: int, threshold: float,
    streak: int, epoch: int, label: str,
) -> tuple[float, int, bool]:
    """Update an EMA and check convergence-patience; returns (new_ema, new_streak, should_stop)."""
    prev = ema
    ema = new_value if ema is None else alpha * ema + (1.0 - alpha) * new_value
    if prev is None or patience <= 0:
        return ema, 0, False
    rel_change = abs(ema - prev) / (abs(prev) + 1e-8)
    if rel_change < threshold:
        streak += 1
        if streak >= patience:
            tqdm.write(f"  {label} converged at epoch {epoch + 1} (EMA Δ={rel_change:.2e}, streak={streak})")
            return ema, streak, True
        return ema, streak, False
    return ema, 0, False


# ── Phase 1: reconstruction ────────────────────────────────────────────────────


def _train_epoch_phase1(
    model: TemporalAutoencoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    kl_weight: float,
    free_bits: float,
    grad_clip: float,
    device: torch.device,
    temporal_reg_weight: float = 0.0,
    temporal_scale: float = 0.01,
    sparsity_weight: float = 0.0,
    max_context_len: int = 0,
    deterministic: bool = False,
) -> dict[str, float]:
    """Reconstruction-only epoch: encoder + f_psi + decoder, no Hamiltonian.

    Adapted from ``pendulum_offline._train_epoch_phase1`` with one
    substantive fix: the action fed to ``next_frame_decoder`` is reshaped to
    ``actions.shape[-1]`` rather than hardcoded to 1, since the point mass has
    a 2D action (horizontal/vertical force) where the pendulum has a 1D one.
    """
    model.train()
    total_recon = total_recon_next = total_kl = total_temporal = total_sparsity = total_loss = 0.0

    for frames, actions, _ in loader:
        frames = frames.to(device)
        actions = actions.to(device=device, dtype=frames.dtype)
        B_size = frames.shape[0]
        if max_context_len >= 2:
            max_L = min(max_context_len, frames.shape[1])
            L = int(torch.randint(2, max_L + 1, (1,)).item())
            frames = frames[:, :L]
            actions = actions[:, : L - 1]
        T_full = frames.shape[1] - 1
        q_dim = model.latent_dim // 2
        action_dim = actions.shape[-1]

        mu_all, logvar_all = model.encoder.forward_all(frames)

        if deterministic:
            z_all = mu_all
        else:
            logvar_all = logvar_all.clamp(-10, 2)
            z_all = mu_all + torch.randn_like(mu_all) * (0.5 * logvar_all).exp()

        def _decode(z: torch.Tensor, B: int, T: int):
            s = model.f_psi(z.reshape(B * T, -1)[:, :q_dim])
            return model.decoder(s).reshape(B, T, *frames.shape[2:])

        pred_curr = _decode(z_all, B_size, T_full + 1)
        recon = F.mse_loss(pred_curr, frames)

        h_curr = z_all[:, :-1].reshape(B_size * T_full, -1)
        a_curr = actions[:, :T_full].reshape(B_size * T_full, action_dim)
        pred_next = model.next_frame_decoder(h_curr, a_curr).reshape(B_size, T_full, *frames.shape[2:])
        recon_next = F.mse_loss(pred_next, frames[:, 1:])

        if deterministic:
            kl = torch.zeros((), device=device)
        else:

            def _kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
                return (
                    (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()))
                    .clamp(min=free_bits)
                    .sum(dim=-1)
                    .mean()
                )

            kl = _kl(mu_all, logvar_all)

        loss = recon + recon_next + kl_weight * kl

        if sparsity_weight > 0:
            sparsity = mu_all.abs().sum(dim=-1).mean()
            loss = loss + sparsity_weight * sparsity
            total_sparsity = total_sparsity + sparsity.detach()

        if temporal_reg_weight > 0:
            T_seq = mu_all.shape[1]
            t1 = torch.randint(T_seq, (T_seq,), device=device)
            t2 = torch.randint(T_seq, (T_seq,), device=device)
            dt = (t1 - t2).abs().float()
            h1 = mu_all[:, t1]
            h2 = mu_all[:, t2]
            dist = torch.norm(h1 - h2, dim=-1)
            temporal_reg = F.relu(temporal_scale * dt - dist).mean()
            loss = loss + temporal_reg_weight * temporal_reg
            total_temporal = total_temporal + temporal_reg.detach()

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_recon = total_recon + recon.detach()
        total_recon_next = total_recon_next + recon_next.detach()
        total_kl = total_kl + kl.detach()
        total_loss = total_loss + loss.detach()

    n = len(loader)
    return {
        "phase1/loss": float(total_loss) / n,
        "phase1/recon": float(total_recon) / n,
        "phase1/recon_next": float(total_recon_next) / n,
        "phase1/kl": float(total_kl) / n,
        "phase1/temporal_reg": float(total_temporal) / n,
        "phase1/sparsity": float(total_sparsity) / n,
    }


@click.group()
def cli():
    pass


@cli.command("phase1")
@config_option
@click.option("--resume-from", type=str, default=None,
              help="Path to a checkpoint (.pt) whose autoencoder weights to warm-start from")
# data
@click.option("--n-windows", type=int, default=200, show_default=True)
@click.option("--img-size", type=int, default=64, show_default=True)
@click.option("--max-steps", type=int, default=100, show_default=True,
              help="Steps per training window (val episodes default to 2x this)")
@click.option("--n-samples", type=int, default=2000, show_default=True,
              help="Total env-step budget for training data collection")
@click.option("--rollout-len", type=int, default=0, show_default=True,
              help="Steps per seeded rollout (0 = 2x --max-steps)")
@click.option("--damping", type=float, default=0.0, show_default=True)
@click.option("--drag", type=float, default=_DRAG_COEFF, show_default=True)
@click.option("--wall-stiffness", type=float, default=_K_WALL, show_default=True)
@click.option("--wall-distance", type=float, default=_L_WALL, show_default=True)
@click.option("--u-max", type=float, default=_U_MAX, show_default=True)
# model architecture
@click.option("--pos-ch", type=int, default=8, show_default=True)
@click.option("--feat-dim", type=int, default=256, show_default=True)
@click.option("--latent-dim", type=int, default=32, show_default=True)
@click.option("--lstm-layers", type=int, default=1, show_default=True)
@click.option("--encoder-type", type=click.Choice(["lstm", "framestack"]), default="lstm", show_default=True)
# training
@click.option("--epochs", type=int, default=3000, show_default=True)
@click.option("--batch-size", type=int, default=8, show_default=True)
@click.option("--lr", type=float, default=1e-4, show_default=True)
@click.option("--kl-weight", type=float, default=1e-3, show_default=True)
@click.option("--free-bits", type=float, default=0.5, show_default=True)
@click.option("--deterministic", is_flag=True, default=False, show_default=True)
@click.option("--grad-clip", type=float, default=1.0, show_default=True)
@click.option("--max-context-len", type=int, default=0, show_default=True)
@click.option("--temporal-reg-weight", type=float, default=0.1, show_default=True)
@click.option("--temporal-scale", type=float, default=0.01, show_default=True)
@click.option("--sparsity-weight", type=float, default=0.0, show_default=True)
@click.option("--ema-alpha", type=float, default=0.99, show_default=True)
@click.option("--convergence-patience", type=int, default=0, show_default=True)
@click.option("--convergence-threshold", type=float, default=1e-4, show_default=True)
# logging
@click.option("--log-every", type=int, default=5, show_default=True)
@click.option("--val-every", type=int, default=10, show_default=True)
@click.option("--n-val-episodes", type=int, default=8, show_default=True)
@click.option("--val-max-steps", type=int, default=0, show_default=True)
@click.option("--checkpoint-every", type=int, default=10, show_default=True)
def phase1_cmd(**kwargs):
    """Phase 1: train the LSTM autoencoder (encoder + f_psi + decoder)."""
    assert kwargs["img_size"] % 8 == 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")

    writer = SummaryWriter(comment="_point_mass_offline_phase1")
    run_dir = make_run_dir("point_mass_offline_phase1")

    rollout_len = kwargs["rollout_len"] or kwargs["max_steps"] * 2
    val_steps = kwargs["val_max_steps"] or kwargs["max_steps"] * 2
    n_val = kwargs["n_val_episodes"] if kwargs["val_every"] > 0 else 0

    n_seeds = max(1, kwargs["n_samples"] // rollout_len)
    print(f"\nCollecting {n_seeds} seeded random rollouts of {rollout_len} steps each...")
    rollouts = collect_seeded_random_rollouts(
        n_samples=kwargs["n_samples"], rollout_len=rollout_len, img_size=kwargs["img_size"],
        damping=kwargs["damping"], drag=kwargs["drag"], k=kwargs["wall_stiffness"],
        L=kwargs["wall_distance"], u_max=kwargs["u_max"],
    )

    rollout_cache_path = run_dir / "rollout_cache.pt"
    torch.save(rollouts, rollout_cache_path)
    print(f"Saved rollout cache ({len(rollouts)} rollouts x {rollout_len} steps) to {rollout_cache_path}")

    coverage_fig = _plot_phase_space_coverage(rollouts)
    writer.add_figure("data/phase_space_coverage", coverage_fig, 0)
    plt.close(coverage_fig)

    val_trajs = []
    if n_val > 0:
        print(f"Collecting {n_val} val rollouts ({val_steps} steps each)...")
        val_trajs = collect_seeded_random_rollouts(
            n_samples=n_val * val_steps, rollout_len=val_steps, img_size=kwargs["img_size"],
            damping=kwargs["damping"], drag=kwargs["drag"], k=kwargs["wall_stiffness"],
            L=kwargs["wall_distance"], u_max=kwargs["u_max"],
        )

    dataset = MultiRolloutDataset(rollouts, window_len=kwargs["max_steps"], n_windows=kwargs["n_windows"])
    loader = DataLoader(
        dataset, batch_size=kwargs["batch_size"], shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    print(f"Dataset: {len(dataset)} windows/epoch of {dataset.window_len} steps")

    model = TemporalAutoencoder(
        latent_dim=kwargs["latent_dim"], feat_dim=kwargs["feat_dim"], pos_ch=kwargs["pos_ch"],
        img_size=kwargs["img_size"], control_dim=2, num_layers=kwargs["lstm_layers"],
        encoder_type=kwargs["encoder_type"],
    ).to(device)
    print(f"Phase 1 model parameters: {sum(p.numel() for p in model.parameters()):,}")

    if kwargs["resume_from"]:
        print(f"Resuming autoencoder weights from {kwargs['resume_from']}...")
        resume_model = load_world_model(kwargs["resume_from"], device)
        model.load_state_dict(resume_model.autoencoder.state_dict())
        del resume_model

    optimizer = torch.optim.Adam(model.parameters(), lr=kwargs["lr"])

    data_config = {k: kwargs[k] for k in (
        "n_windows", "n_samples", "img_size", "max_steps", "damping", "drag",
        "wall_stiffness", "wall_distance", "u_max",
    )}
    data_config["rollout_len"] = rollout_len
    world_model = WorldModel(model, dynamics=None, data_config=data_config)

    hparams = dict(kwargs)
    _log_hparams_text(writer, hparams)
    _log_hparams_table(writer, hparams, {})
    best_loss = float("inf")
    ema_loss = None
    converge_streak = 0

    print("\n=== Phase 1: reconstruction training ===")
    for epoch in tqdm(range(kwargs["epochs"]), desc="Phase 1", dynamic_ncols=True):
        metrics = _train_epoch_phase1(
            model=model, loader=loader, optimizer=optimizer,
            kl_weight=kwargs["kl_weight"], free_bits=kwargs["free_bits"],
            grad_clip=kwargs["grad_clip"], device=device,
            temporal_reg_weight=kwargs["temporal_reg_weight"], temporal_scale=kwargs["temporal_scale"],
            sparsity_weight=kwargs["sparsity_weight"], max_context_len=kwargs["max_context_len"],
            deterministic=kwargs["deterministic"],
        )

        ema_loss, converge_streak, should_stop = _ema_convergence_check(
            ema_loss, metrics["phase1/loss"], kwargs["ema_alpha"],
            kwargs["convergence_patience"], kwargs["convergence_threshold"],
            converge_streak, epoch, "Phase 1",
        )
        if should_stop:
            break

        if (epoch + 1) % kwargs["log_every"] == 0:
            for k, v in metrics.items():
                writer.add_scalar(k, v, epoch)
            writer.add_scalar("phase1/ema_loss", ema_loss, epoch)
            tqdm.write(
                f"  epoch {epoch + 1:4d}  loss={metrics['phase1/loss']:.4f}  ema={ema_loss:.4f}"
                f"  recon={metrics['phase1/recon']:.4f}  next={metrics['phase1/recon_next']:.4f}"
                f"  kl={metrics['phase1/kl']:.4f}"
            )

        if kwargs["val_every"] > 0 and (epoch + 1) % kwargs["val_every"] == 0 and val_trajs:
            val_metrics = _eval_loss_phase1(model, val_trajs, device)
            for k, v in val_metrics.items():
                writer.add_scalar(k, v, epoch)
            _log_reconstruction_video(model, val_trajs[0], device, writer, epoch, "val/reconstruction")
            _log_reconstruction_video(model, dataset[0], device, writer, epoch, "train/reconstruction")

        if (
            kwargs["checkpoint_every"] > 0
            and (epoch + 1) % kwargs["checkpoint_every"] == 0
            and metrics["phase1/loss"] < best_loss
        ):
            world_model.save(run_dir, "best", hparams, metrics, epoch)
            best_loss = metrics["phase1/loss"]

    world_model.save(run_dir, "final", hparams, metrics, epoch)
    print(f"\nTo run Phase 2:\n  uv run python experiments/point_mass_offline.py phase2 --phase1-run {run_dir}")

    writer.close()
    print("\nDone. Run: tensorboard --logdir runs")
    os._exit(0)


# ── Phase 2: dynamics ──────────────────────────────────────────────────────────


def _train_epoch_phase2(
    dyn_model: HamiltonianFlowModel,
    encoder: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
    device: torch.device,
    seq_len: int,
    seed_ctx_len: int,
    logdet_weight: float,
    teacher_force_weight: float = 1.0,
    closed_loop_weight: float = 1.0,
    closed_loop_gamma: float = 1.0,
    l1_weight: float = 0.0,
    structural_reg_weight: float = 0.0,
) -> dict[str, float]:
    """Teacher-forced one-step loss + closed-loop rollout loss, frozen encoder.

    ``encoder`` (Phase 1's, frozen) turns each batch's window of frames into
    h_t under ``torch.no_grad()`` — its parameters are never in ``optimizer``.
    Teacher-forced: one ``controlled_step`` per adjacent (h_t, h_{t+1}) pair
    in the window. Closed-loop: starting from the state at the end of the
    seed context, roll ``dyn_model`` forward ``seq_len`` steps with no
    re-encoding, comparing the decoded latents against the encoder's own
    (ground-truth) h at each step — a curriculum on ``seq_len`` should ramp
    this from short to long as training progresses (see ``phase2_cmd``).
    """
    dyn_model.train()
    encoder.eval()
    total_tf = total_cl = total_logdet = total_loss = 0.0

    for frames, actions, _ in loader:
        frames = frames.to(device)
        actions = actions.to(device=device, dtype=frames.dtype)
        B, W1 = frames.shape[:2]
        W = W1 - 1
        ctx = min(seed_ctx_len, W)
        T = min(seq_len, W - ctx + 1)
        action_dim = actions.shape[-1]

        with torch.no_grad():
            h_all, _ = encoder.forward_all(frames)  # (B, W+1, latent_dim)

        q_dim = dyn_model.latent_dim // 2
        p_dim = dyn_model.latent_dim - q_dim
        h_flat = h_all.reshape(B * W1, -1)
        q_all, p_all, log_det = dyn_model.encode_with_logdet(h_flat)
        logdet_reg = -log_det.mean()
        q_all = q_all.reshape(B, W1, q_dim)
        p_all = p_all.reshape(B, W1, p_dim)

        # Teacher-forced one-step loss over the whole window.
        q_t = q_all[:, :-1].reshape(B * W, q_dim)
        p_t = p_all[:, :-1].reshape(B * W, p_dim)
        a_t = actions[:, :W].reshape(B * W, action_dim)
        q_tf, p_tf = dyn_model.controlled_step(q_t, p_t, a_t)
        h_tf = dyn_model.decode(q_tf, p_tf).reshape(B, W, -1)
        tf_loss = F.mse_loss(h_tf, h_all[:, 1:])

        # Closed-loop rollout from the end of the seed context.
        k = ctx - 1
        q_roll, p_roll = q_all[:, k], p_all[:, k]
        preds = []
        for t in range(T):
            q_roll, p_roll = dyn_model.controlled_step(q_roll, p_roll, actions[:, k + t])
            preds.append(dyn_model.decode(q_roll, p_roll))
        h_pred = torch.stack(preds, dim=1)  # (B, T, latent_dim)
        h_target = h_all[:, k + 1 : k + 1 + T]
        weights = closed_loop_gamma ** torch.arange(T, device=device, dtype=h_pred.dtype)
        weights = weights / weights.sum()
        cl_loss = (weights.view(1, T, 1) * (h_pred - h_target) ** 2).sum(dim=1).mean()

        loss = logdet_weight * logdet_reg + teacher_force_weight * tf_loss + closed_loop_weight * cl_loss

        if l1_weight > 0:
            l1 = sum(p.abs().sum() for p in dyn_model.hamiltonian.parameters())
            loss = loss + l1_weight * l1

        if structural_reg_weight > 0:
            structural_reg = (dyn_model.get_R_pp() ** 2).mean()
            loss = loss + structural_reg_weight * structural_reg

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(dyn_model.parameters(), grad_clip)
        optimizer.step()

        total_tf = total_tf + tf_loss.detach()
        total_cl = total_cl + cl_loss.detach()
        total_logdet = total_logdet + logdet_reg.detach()
        total_loss = total_loss + loss.detach()

    n = len(loader)
    return {
        "phase2/loss": float(total_loss) / n,
        "phase2/tf_loss": float(total_tf) / n,
        "phase2/cl_loss": float(total_cl) / n,
        "phase2/logdet_reg": float(total_logdet) / n,
    }


@torch.no_grad()
def _eval_loss_phase2(
    world_model: WorldModel,
    val_trajs: list,
    device: torch.device,
    seed_ctx_len: int,
) -> dict[str, float]:
    dyn_model = world_model.dynamics
    encoder = world_model.autoencoder.encoder
    dyn_model.eval()
    frames_all = torch.stack([t[0] for t in val_trajs]).to(device)
    actions_all = torch.stack([t[1] for t in val_trajs]).to(device=device, dtype=frames_all.dtype)
    N, W1 = frames_all.shape[:2]
    W = W1 - 1
    q_dim = dyn_model.latent_dim // 2
    action_dim = actions_all.shape[-1]

    h_all, _ = encoder.forward_all(frames_all)
    h_flat = h_all.reshape(N * W1, -1)
    q_all, p_all = dyn_model.encode(h_flat)
    q_all = q_all.reshape(N, W1, q_dim)
    p_all = p_all.reshape(N, W1, -1)

    q_t = q_all[:, :-1].reshape(N * W, q_dim)
    p_t = p_all[:, :-1].reshape(N * W, -1)
    a_t = actions_all[:, :W].reshape(N * W, action_dim)
    q_tf, p_tf = dyn_model.controlled_step(q_t, p_t, a_t)
    h_tf = dyn_model.decode(q_tf, p_tf).reshape(N, W, -1)
    tf_loss = F.mse_loss(h_tf, h_all[:, 1:])

    ctx = min(seed_ctx_len, W)
    k = ctx - 1
    q_roll, p_roll = q_all[:, k], p_all[:, k]
    preds = []
    for t in range(k, W):
        q_roll, p_roll = dyn_model.controlled_step(q_roll, p_roll, actions_all[:, t])
        preds.append(dyn_model.decode(q_roll, p_roll))
    h_pred = torch.stack(preds, dim=1)
    h_target = h_all[:, k + 1 :]
    cl_loss = F.mse_loss(h_pred, h_target)

    return {"phase2/val_tf_loss": tf_loss.item(), "phase2/val_cl_loss": cl_loss.item()}


@cli.command("phase2")
@config_option
@click.option("--phase1-run", type=str, required=True,
              help="Run dir from phase1 (models/point_mass_offline_phase1/<timestamp>)")
@click.option("--separable/--no-separable", default=True, show_default=True)
@click.option("--h-source", type=click.Choice(["learned", "canonical"]), default="learned", show_default=True)
@click.option("--r-source", type=click.Choice(["learned", "fixed_damping", "canonical"]), default="learned", show_default=True)
@click.option("--b-source", type=click.Choice(["learned", "fixed_ones", "canonical"]), default="learned", show_default=True)
@click.option("--integrator", type=click.Choice(["rk4", "leapfrog"]), default="leapfrog", show_default=True)
@click.option("--quadratic-t/--no-quadratic-t", default=True, show_default=True)
@click.option("--state-dep-r/--no-state-dep-r", default=False, show_default=True)
@click.option("--epochs", type=int, default=3000, show_default=True)
@click.option("--batch-size", type=int, default=8, show_default=True)
@click.option("--lr", type=float, default=1e-3, show_default=True)
@click.option("--structural-lr", type=float, default=1e-3, show_default=True)
@click.option("--grad-clip", type=float, default=1.0, show_default=True)
@click.option("--logdet-weight", type=float, default=1e-3, show_default=True)
@click.option("--l1-weight", type=float, default=0.0, show_default=True)
@click.option("--teacher-force-weight", type=float, default=1.0, show_default=True)
@click.option("--closed-loop-weight", type=float, default=1.0, show_default=True)
@click.option("--closed-loop-gamma", type=float, default=1.0, show_default=True)
@click.option("--structural-reg-weight", type=float, default=0.0, show_default=True)
@click.option("--seed-ctx-len", type=int, default=5, show_default=True)
@click.option("--seq-len-start", type=int, default=4, show_default=True)
@click.option("--max-seq-len", type=int, default=0, show_default=True, help="0 = full window")
@click.option("--seq-len-advance-threshold", type=float, default=1e-3, show_default=True)
@click.option("--ema-alpha", type=float, default=0.99, show_default=True)
@click.option("--convergence-patience", type=int, default=0, show_default=True)
@click.option("--convergence-threshold", type=float, default=1e-4, show_default=True)
@click.option("--log-every", type=int, default=5, show_default=True)
@click.option("--val-every", type=int, default=10, show_default=True)
@click.option("--n-val-episodes", type=int, default=8, show_default=True)
@click.option("--checkpoint-every", type=int, default=10, show_default=True)
def phase2_cmd(**kwargs):
    """Phase 2: train HamiltonianFlowModel (physics='point_mass') on frozen Phase-1 h_t."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    phase1_run = Path(kwargs["phase1_run"])
    ckpt_path = phase1_run / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = phase1_run / "final.pt"
    print(f"Loading phase 1 checkpoint from {ckpt_path}...")
    world_model = load_world_model(ckpt_path, device)
    phase1_model = world_model.autoencoder
    data_cfg = world_model.data_config

    rollout_cache_path = phase1_run / "rollout_cache.pt"
    rollouts = torch.load(rollout_cache_path, weights_only=False)
    print(f"Loaded {len(rollouts)} cached rollouts from {rollout_cache_path}")

    window_len = data_cfg.get("max_steps", 100)
    n_windows = data_cfg.get("n_windows", 200)
    dataset = MultiRolloutDataset(rollouts, window_len=window_len, n_windows=n_windows)
    loader = DataLoader(
        dataset, batch_size=kwargs["batch_size"], shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda",
    )

    n_val = kwargs["n_val_episodes"] if kwargs["val_every"] > 0 else 0
    val_trajs = rollouts[:n_val] if n_val > 0 else []

    writer = SummaryWriter(comment="_point_mass_offline_phase2")
    run_dir = make_run_dir("point_mass_offline_phase2")

    dyn_model = HamiltonianFlowModel(
        latent_dim=phase1_model.config["latent_dim"], control_dim=2,
        separable=kwargs["separable"], h_source=kwargs["h_source"], r_source=kwargs["r_source"],
        b_source=kwargs["b_source"], dt=data_cfg.get("dt", 0.05), damping=data_cfg.get("damping", 0.0),
        drag=data_cfg.get("drag", _DRAG_COEFF), integrator=kwargs["integrator"],
        quadratic_t=kwargs["quadratic_t"], state_dep_r=kwargs["state_dep_r"], physics="point_mass",
    ).to(device)
    print(f"Phase 2 dynamics parameters: {sum(p.numel() for p in dyn_model.parameters()):,}")

    opt_groups = [{
        "params": (
            list(dyn_model.phi_q.parameters())
            + list(dyn_model.phi_p.parameters())
            + list(dyn_model.hamiltonian.parameters())
        ),
        "lr": kwargs["lr"],
    }]
    struct_params = dyn_model.structural_parameters()
    if struct_params:
        opt_groups.append({"params": struct_params, "lr": kwargs["structural_lr"]})
    optimizer = torch.optim.Adam(opt_groups)

    world_model.dynamics = dyn_model
    world_model.to(device)

    full_seq_len = window_len - kwargs["seed_ctx_len"] + 1
    if kwargs["max_seq_len"] > 0:
        full_seq_len = min(full_seq_len, kwargs["max_seq_len"])
    seq_len = min(kwargs["seq_len_start"], full_seq_len)

    hparams = {**kwargs, "phase1_config": {**phase1_model.config, **data_cfg}}
    _log_hparams_text(writer, hparams)
    _log_hparams_table(writer, hparams, {})
    best_loss = float("inf")
    ema_loss = None
    ema_cl = None
    converge_streak = 0

    print("\n=== Phase 2: dynamics training ===")
    for epoch in tqdm(range(kwargs["epochs"]), desc="Phase 2", dynamic_ncols=True):
        metrics = _train_epoch_phase2(
            dyn_model=dyn_model, encoder=phase1_model.encoder, loader=loader, optimizer=optimizer,
            grad_clip=kwargs["grad_clip"], device=device, seq_len=seq_len, seed_ctx_len=kwargs["seed_ctx_len"],
            logdet_weight=kwargs["logdet_weight"], teacher_force_weight=kwargs["teacher_force_weight"],
            closed_loop_weight=kwargs["closed_loop_weight"], closed_loop_gamma=kwargs["closed_loop_gamma"],
            l1_weight=kwargs["l1_weight"], structural_reg_weight=kwargs["structural_reg_weight"],
        )
        metrics["phase2/seq_len"] = seq_len

        ema_cl = metrics["phase2/cl_loss"] if ema_cl is None else 0.9 * ema_cl + 0.1 * metrics["phase2/cl_loss"]
        if ema_cl < kwargs["seq_len_advance_threshold"] and seq_len < full_seq_len:
            seq_len += 1

        ema_loss, converge_streak, should_stop = _ema_convergence_check(
            ema_loss, metrics["phase2/loss"], kwargs["ema_alpha"],
            kwargs["convergence_patience"], kwargs["convergence_threshold"],
            converge_streak, epoch, "Phase 2",
        )
        if should_stop:
            break

        if (epoch + 1) % kwargs["log_every"] == 0:
            for k, v in metrics.items():
                writer.add_scalar(k, v, epoch)
            writer.add_scalar("phase2/ema_loss", ema_loss, epoch)
            tqdm.write(
                f"  epoch {epoch + 1:4d}  loss={metrics['phase2/loss']:.4f}  tf={metrics['phase2/tf_loss']:.4f}"
                f"  cl={metrics['phase2/cl_loss']:.4f}  seq_len={seq_len}"
            )

        if kwargs["val_every"] > 0 and (epoch + 1) % kwargs["val_every"] == 0 and val_trajs:
            val_metrics = _eval_loss_phase2(world_model, val_trajs, device, kwargs["seed_ctx_len"])
            for k, v in val_metrics.items():
                writer.add_scalar(k, v, epoch)
            _log_dreamed_video(world_model, val_trajs[0], device, writer, epoch, kwargs["seed_ctx_len"], "val/dreamed")

        if (
            kwargs["checkpoint_every"] > 0
            and (epoch + 1) % kwargs["checkpoint_every"] == 0
            and metrics["phase2/loss"] < best_loss
        ):
            world_model.save(run_dir, "best", hparams, metrics, epoch)
            best_loss = metrics["phase2/loss"]

    world_model.save(run_dir, "final", hparams, metrics, epoch)
    print(f"\nTo run Phase 3:\n  uv run python experiments/point_mass_offline.py phase3 "
          f"--phase2-run {run_dir} --phase1-run {phase1_run}")

    writer.close()
    print("\nDone. Run: tensorboard --logdir runs")
    os._exit(0)


# ── Phase 3: end-to-end finetune ───────────────────────────────────────────────


def _train_epoch_phase3(
    world_model: WorldModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
    device: torch.device,
    seed_ctx_len: int,
    seq_len: int,
    anchor_weight: float = 0.1,
) -> dict[str, float]:
    """Pixel-space dreaming loss, encoder+decoder+dynamics all trainable.

    ``anchor_weight`` keeps a plain reconstruction loss on the seed context
    active throughout — the dreaming loss alone has no term that constrains
    the encoder/decoder pair away from a degenerate shortcut, since it only
    ever sees frames generated by feeding decoded latents back through the
    dynamics, never a real frame decoded directly.
    """
    world_model.train()
    autoencoder = world_model.autoencoder
    dyn_model = world_model.dynamics
    total_dream = total_anchor = total_loss = 0.0

    for frames, actions, _ in loader:
        frames = frames.to(device)
        actions = actions.to(device=device, dtype=frames.dtype)
        B, W1 = frames.shape[:2]
        W = W1 - 1
        ctx = min(seed_ctx_len, W)
        T = min(seq_len, W - ctx)
        q_dim = dyn_model.latent_dim // 2

        mu_ctx, _ = autoencoder.encoder.forward_all(frames[:, :ctx])
        q, p = dyn_model.encode(mu_ctx[:, -1])

        dreamed = []
        for t in range(T):
            q, p = dyn_model.controlled_step(q, p, actions[:, ctx - 1 + t])
            h = dyn_model.decode(q, p)
            dreamed.append(autoencoder.decode_latent(h))
        dreamed = torch.stack(dreamed, dim=1)
        target = frames[:, ctx : ctx + T]
        dream_loss = F.mse_loss(dreamed, target)

        s = autoencoder.f_psi(mu_ctx.reshape(-1, mu_ctx.shape[-1])[:, :q_dim])
        recon = autoencoder.decoder(s).reshape(B, ctx, *frames.shape[2:])
        anchor_loss = F.mse_loss(recon, frames[:, :ctx])

        loss = dream_loss + anchor_weight * anchor_loss

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(world_model.parameters(), grad_clip)
        optimizer.step()

        total_dream = total_dream + dream_loss.detach()
        total_anchor = total_anchor + anchor_loss.detach()
        total_loss = total_loss + loss.detach()

    n = len(loader)
    return {
        "phase3/loss": float(total_loss) / n,
        "phase3/dream_loss": float(total_dream) / n,
        "phase3/anchor_loss": float(total_anchor) / n,
    }


@cli.command("phase3")
@config_option
@click.option("--phase2-run", type=str, required=True)
@click.option("--phase1-run", type=str, required=True, help="Run dir holding rollout_cache.pt")
@click.option("--epochs", type=int, default=1000, show_default=True)
@click.option("--batch-size", type=int, default=8, show_default=True)
@click.option("--lr", type=float, default=1e-5, show_default=True)
@click.option("--grad-clip", type=float, default=1.0, show_default=True)
@click.option("--seed-ctx-len", type=int, default=5, show_default=True)
@click.option("--seq-len", type=int, default=10, show_default=True)
@click.option("--anchor-weight", type=float, default=0.1, show_default=True)
@click.option("--ema-alpha", type=float, default=0.99, show_default=True)
@click.option("--convergence-patience", type=int, default=0, show_default=True)
@click.option("--convergence-threshold", type=float, default=1e-4, show_default=True)
@click.option("--log-every", type=int, default=5, show_default=True)
@click.option("--val-every", type=int, default=10, show_default=True)
@click.option("--n-val-episodes", type=int, default=8, show_default=True)
@click.option("--checkpoint-every", type=int, default=10, show_default=True)
def phase3_cmd(**kwargs):
    """Phase 3: end-to-end finetune of encoder+f_psi+decoder+dynamics through pixel dreaming."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    phase2_run = Path(kwargs["phase2_run"])
    ckpt_path = phase2_run / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = phase2_run / "final.pt"
    print(f"Loading phase 2 checkpoint from {ckpt_path}...")
    world_model = load_world_model(ckpt_path, device)
    data_cfg = world_model.data_config

    phase1_run = Path(kwargs["phase1_run"])
    rollouts = torch.load(phase1_run / "rollout_cache.pt", weights_only=False)
    print(f"Loaded {len(rollouts)} cached rollouts")

    window_len = data_cfg.get("max_steps", 100)
    n_windows = data_cfg.get("n_windows", 200)
    dataset = MultiRolloutDataset(rollouts, window_len=window_len, n_windows=n_windows)
    loader = DataLoader(
        dataset, batch_size=kwargs["batch_size"], shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda",
    )

    n_val = kwargs["n_val_episodes"] if kwargs["val_every"] > 0 else 0
    val_trajs = rollouts[:n_val] if n_val > 0 else []

    writer = SummaryWriter(comment="_point_mass_offline_phase3")
    run_dir = make_run_dir("point_mass_offline_phase3")

    optimizer = torch.optim.Adam(world_model.parameters(), lr=kwargs["lr"])

    hparams = dict(kwargs)
    _log_hparams_text(writer, hparams)
    _log_hparams_table(writer, hparams, {})
    best_loss = float("inf")
    ema_loss = None
    converge_streak = 0

    print("\n=== Phase 3: end-to-end finetune ===")
    for epoch in tqdm(range(kwargs["epochs"]), desc="Phase 3", dynamic_ncols=True):
        metrics = _train_epoch_phase3(
            world_model=world_model, loader=loader, optimizer=optimizer, grad_clip=kwargs["grad_clip"],
            device=device, seed_ctx_len=kwargs["seed_ctx_len"], seq_len=kwargs["seq_len"],
            anchor_weight=kwargs["anchor_weight"],
        )

        ema_loss, converge_streak, should_stop = _ema_convergence_check(
            ema_loss, metrics["phase3/loss"], kwargs["ema_alpha"],
            kwargs["convergence_patience"], kwargs["convergence_threshold"],
            converge_streak, epoch, "Phase 3",
        )
        if should_stop:
            break

        if (epoch + 1) % kwargs["log_every"] == 0:
            for k, v in metrics.items():
                writer.add_scalar(k, v, epoch)
            writer.add_scalar("phase3/ema_loss", ema_loss, epoch)
            tqdm.write(
                f"  epoch {epoch + 1:4d}  loss={metrics['phase3/loss']:.4f}"
                f"  dream={metrics['phase3/dream_loss']:.4f}  anchor={metrics['phase3/anchor_loss']:.4f}"
            )

        if kwargs["val_every"] > 0 and (epoch + 1) % kwargs["val_every"] == 0 and val_trajs:
            _log_dreamed_video(
                world_model, val_trajs[0], device, writer, epoch,
                kwargs["seed_ctx_len"], "val/dreamed_phase3",
            )

        if (
            kwargs["checkpoint_every"] > 0
            and (epoch + 1) % kwargs["checkpoint_every"] == 0
            and metrics["phase3/loss"] < best_loss
        ):
            world_model.save(run_dir, "best", hparams, metrics, epoch)
            best_loss = metrics["phase3/loss"]

    world_model.save(run_dir, "final", hparams, metrics, epoch)
    writer.close()
    print("\nDone. Run: tensorboard --logdir runs")
    os._exit(0)


if __name__ == "__main__":
    cli()
