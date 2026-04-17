"""Offline Pendulum world-model training for ControlledDHGN_LSTM.

Collects episodes with an energy-pumping + PD controller (with random gains),
then trains the model as a multi-step world model:

    encode all episode frames → (q0, p0)  via bidirectional LSTM encoder
    roll out T RK4 steps      → (q1..qT)  with recorded actions
    decode each qi            → pred_frame_i
    loss = mean MSE(pred_frame_i, frame_{i+1}) + kl_weight * KL
"""

from __future__ import annotations

import os

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from checkpoint_common import make_run_dir, save_checkpoint
from data.pendulum import (
    PendulumDataset,
    collect_data,
    collect_random_trajectories,
    collect_spin_trajectories,
    collect_val_trajectories,
    _G,
)
from phgn_lstm import ControlledDHGN_LSTM



def _log_latent_variance(
    qs_enc: torch.Tensor,
    ps_enc: torch.Tensor,
) -> tuple[float, float]:
    """Mean per-dim variance of q and p across batch×time."""
    q_dim = qs_enc.shape[-1]
    q_var = qs_enc.detach().reshape(-1, q_dim).var(dim=0).mean().item()
    p_var = ps_enc.detach().reshape(-1, q_dim).var(dim=0).mean().item()
    return q_var, p_var


def _batch_r2(z_flat: torch.Tensor, st_flat: torch.Tensor) -> float:
    """Fit linear regression z → state and return mean R² across output dims."""
    A = torch.linalg.lstsq(z_flat, st_flat).solution
    st_pred = z_flat @ A
    ss_res = ((st_flat - st_pred) ** 2).sum(0)
    ss_tot = ((st_flat - st_flat.mean(0)) ** 2).sum(0)
    return (1 - ss_res / (ss_tot + 1e-8)).clamp(max=1.0).mean().item()


def _train_epoch_phase1(
    model: ControlledDHGN_LSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    kl_weight: float,
    free_bits: float,
    grad_clip: float,
    fd_weight: float,
    device: torch.device,
) -> dict[str, float]:
    """Phase 1: train encoder → f_psi → decoder with recon + KL + fd_p supervision."""
    model.train()
    total_recon = total_kl = total_loss = total_fd = total_r2 = 0.0
    total_q_var = total_p_var = 0.0

    for frames, actions, states in loader:
        frames = frames.to(device)
        actions = actions.to(device)
        T = actions.shape[1]
        B_size = frames.shape[0]
        q_dim = model.latent_dim // 2

        mu_all, logvar_all = model.encoder.forward_all(frames)
        logvar_all = logvar_all.clamp(-10, 2)
        z_all = mu_all + torch.randn_like(mu_all) * (0.5 * logvar_all).exp()

        B_T1 = B_size * (T + 1)
        s_all = model.f_psi(z_all.reshape(B_T1, -1)).reshape(B_size, T + 1, -1)
        qs_enc = s_all[:, :, :q_dim]
        ps_enc = s_all[:, :, q_dim:]

        pred_all = model.decoder(qs_enc.reshape(B_T1, q_dim))
        pred_all = pred_all.reshape(B_size, T + 1, *frames.shape[2:])
        recon = F.mse_loss(pred_all, frames)

        kl = (
            (-0.5 * (1 + logvar_all - mu_all.pow(2) - logvar_all.exp()))
            .clamp(min=free_bits)
            .sum(dim=-1)
            .mean()
        )

        # Supervise p[t] to predict the finite difference q[t+1] - q[t].
        fd_targets = (qs_enc[:, 1:] - qs_enc[:, :-1]).detach()
        fd_loss = F.mse_loss(ps_enc[:, :-1], fd_targets)

        loss = recon + kl_weight * kl + fd_weight * fd_loss

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_recon += recon.item()
        total_kl += kl.item()
        total_fd += fd_loss.item()
        total_loss += loss.item()
        q_var, p_var = _log_latent_variance(qs_enc, ps_enc)
        total_q_var += q_var
        total_p_var += p_var

        with torch.no_grad():
            z_flat = s_all.detach().cpu().reshape(-1, model.latent_dim)
            st_flat = states.reshape(-1, 2).float()
            total_r2 += _batch_r2(z_flat, st_flat)

    n = len(loader)
    return {
        "train/loss": total_loss / n,
        "train/recon": total_recon / n,
        "train/kl": total_kl / n,
        "train/fd_loss": total_fd / n,
        "train/r2": total_r2 / n,
        "train/latent": 0.0,
        "train/q_var": total_q_var / n,
        "train/p_var": total_p_var / n,
    }


def _train_epoch_phase2(
    model: ControlledDHGN_LSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
    device: torch.device,
) -> dict[str, float]:
    """Phase 2: freeze encoder/f_psi/decoder, train H/R/B with latent loss."""
    model.train()
    total_latent = 0.0
    total_q_var = total_p_var = 0.0

    for frames, actions, _ in loader:
        frames = frames.to(device)
        actions = actions.to(device)
        T = actions.shape[1]
        B_size = frames.shape[0]
        q_dim = model.latent_dim // 2

        with torch.no_grad():
            mu_all, _ = model.encoder.forward_all(frames)
            B_T1 = B_size * (T + 1)
            s_all = model.f_psi(mu_all.reshape(B_T1, -1)).reshape(B_size, T + 1, -1)
            qs_enc = s_all[:, :, :q_dim]
            ps_enc = s_all[:, :, q_dim:]

        q, p = qs_enc[:, 0], ps_enc[:, 0]
        latent_loss = torch.zeros(1, device=device)

        for t in range(T):
            u = actions[:, t].unsqueeze(-1)
            q, p = model.controlled_step(q, p, u)
            latent_loss = (
                latent_loss
                + F.mse_loss(q, qs_enc[:, t + 1])
                + F.mse_loss(p, ps_enc[:, t + 1])
            )

        latent_loss = latent_loss / T

        optimizer.zero_grad()
        latent_loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_latent += latent_loss.item()
        q_var, p_var = _log_latent_variance(qs_enc, ps_enc)
        total_q_var += q_var
        total_p_var += p_var

    n = len(loader)
    return {
        "train/loss": total_latent / n,
        "train/recon": 0.0,
        "train/kl": 0.0,
        "train/latent": total_latent / n,
        "train/q_var": total_q_var / n,
        "train/p_var": total_p_var / n,
    }

def _true_hamiltonian(states: torch.Tensor) -> np.ndarray:
    """Compute H = 0.5*theta_dot^2 + g*(1 - cos(theta)) from (T, 2) states."""
    theta = states[:, 0].numpy()
    theta_dot = states[:, 1].numpy()
    return 0.5 * theta_dot**2 + _G * (1.0 + np.cos(theta))


def _annotate_frame(frame: torch.Tensor, text: str) -> torch.Tensor:
    """Draw text in the top-left corner of a (C, H, W) float tensor [0,1]."""
    img = Image.fromarray((frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    draw.text((2, 2), text, fill=(255, 255, 0))
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0


@torch.no_grad()
def _log_reconstruction_video(
    model: ControlledDHGN_LSTM,
    val_traj: tuple,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    tag: str = "val/reconstruction",
    fps: int = 10,
    use_dynamics: bool = False,
) -> None:
    """Log a side-by-side ground-truth / reconstruction video to TensorBoard.

    use_dynamics=False (default): encode each frame via forward_all, matching
    the Phase-1 training objective (per-frame reconstruction).
    use_dynamics=True: encode a single initial state then roll out with the
    learned Hamiltonian dynamics (appropriate after Phase 2).
    """
    model.eval()
    frames, actions, _ = val_traj  # (T+1, C, H, W), (T,)

    ctx = frames.unsqueeze(0).to(device)  # (1, T+1, C, H, W)
    q_dim = model.latent_dim // 2

    if use_dynamics:
        q, p = model.encode_mean(ctx)
        recon_frames = [model.decoder(q).squeeze(0).cpu()]
        for t in range(len(actions)):
            u = actions[t].reshape(1, 1).to(device)
            q, p = model.controlled_step(q, p, u)
            recon_frames.append(model.decoder(q).squeeze(0).cpu())
        recon = torch.stack(recon_frames)
    else:
        mu_all, _ = model.encoder.forward_all(ctx)   # (1, T+1, latent_dim)
        s_all = model.f_psi(mu_all.squeeze(0))        # (T+1, latent_dim)
        qs_enc = s_all[:, :q_dim]                     # (T+1, q_dim)
        recon = model.decoder(qs_enc).cpu()            # (T+1, C, H, W)

    gt = frames  # (T+1, C, H, W)

    # Annotate each frame with its frame number.
    gt_ann = torch.stack([_annotate_frame(gt[i], f"{i}") for i in range(len(gt))])
    recon_ann = torch.stack([_annotate_frame(recon[i], f"{i}") for i in range(len(recon))])

    # Side-by-side along width: (T+1, C, H, 2W) → (1, T+1, C, H, 2W)
    side_by_side = torch.cat([gt_ann, recon_ann], dim=3).unsqueeze(0)
    # TensorBoard add_video expects uint8 (N, T, C, H, W)
    video = (side_by_side.clamp(0, 1) * 255).byte()

    writer.add_video(tag, video, epoch, fps=fps)


@torch.no_grad()
def _log_hamiltonian_comparison(
    model: ControlledDHGN_LSTM,
    val_traj: tuple,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    tag: str = "val/hamiltonian",
) -> None:
    """Log two figures: network H vs true H, and dH predicted vs dH actual."""
    model.eval()
    frames, actions, states = val_traj  # (T+1, C, H, W), (T,), (T+1, 2)

    q_dim = model.latent_dim // 2
    R = model.get_R()

    ctx = frames.unsqueeze(0).to(device)
    mu_all, _ = model.encoder.forward_all(ctx)
    s_all = model.f_psi(mu_all.squeeze(0))
    qs_enc = s_all[:, :q_dim]
    ps_enc = s_all[:, q_dim:]

    net_H = [model.H(qs_enc[t:t+1], ps_enc[t:t+1]).item() for t in range(len(qs_enc))]
    dH_actual = []
    dH_predicted = []

    for t in range(len(actions)):
        u = actions[t].reshape(1, 1).to(device)
        z_curr = torch.cat([qs_enc[t:t+1], ps_enc[t:t+1]], dim=-1)
        z_next = torch.cat([qs_enc[t+1:t+2], ps_enc[t+1:t+2]], dim=-1)

        with torch.enable_grad():
            z_mid = (0.5 * (z_curr + z_next)).detach().requires_grad_(True)
            H_mid = model.hamiltonian(z_mid[:, :q_dim], z_mid[:, q_dim:])
            grad_H = torch.autograd.grad(H_mid.sum(), z_mid)[0]

        Bu_full = torch.cat([
            torch.zeros(1, q_dim, device=device),
            u @ model.B.T,
        ], dim=-1)

        dt = model.dt
        dH_pred = (
            -dt * (grad_H @ R * grad_H).sum(-1)
            + dt * (grad_H * Bu_full).sum(-1)
        )
        dH_predicted.append(dH_pred.item())
        dH_actual.append(net_H[t + 1] - net_H[t])

    true_H = _true_hamiltonian(states)
    t_axis = np.arange(len(true_H))
    dh_axis = np.arange(1, len(true_H))

    fig_h, ax_h = plt.subplots(figsize=(8, 3))
    ax_h.plot(t_axis, true_H, label="True H", linewidth=1.5, color="tab:blue")
    ax_h.plot(t_axis, net_H, label="Network H", linewidth=1.5, linestyle="--", color="tab:orange")
    ax_h.axhline(_G * 2, color="grey", linestyle=":", linewidth=1, label="H*=20")
    ax_h.set_xlabel("Step")
    ax_h.set_ylabel("H")
    ax_h.legend(fontsize=8, loc="upper left")
    ax_h.set_title(f"H comparison (epoch {epoch + 1})")
    fig_h.tight_layout()
    writer.add_figure(tag + "/H_values", fig_h, epoch)
    plt.close(fig_h)

    fig_dh, ax_dh = plt.subplots(figsize=(8, 3))
    ax_dh.plot(dh_axis, dH_actual, label="dH actual", linewidth=1.0, color="tab:green")
    ax_dh.plot(dh_axis, dH_predicted, label="dH predicted", linewidth=1.0, linestyle="--", color="tab:red")
    ax_dh.axhline(0, color="lightgrey", linestyle="-", linewidth=0.5)
    ax_dh.set_xlabel("Step")
    ax_dh.set_ylabel("dH")
    ax_dh.legend(fontsize=8, loc="upper left")
    ax_dh.set_title(f"dH comparison (epoch {epoch + 1})")
    fig_dh.tight_layout()
    writer.add_figure(tag + "/dH", fig_dh, epoch)
    plt.close(fig_dh)


@torch.no_grad()
def _log_R_eigenvalues(
    model: ControlledDHGN_LSTM,
    writer: SummaryWriter,
    epoch: int,
) -> None:
    """Log eigenvalue distribution of the dissipation matrix R = L Lᵀ."""
    model.eval()
    R = model.get_R().cpu()
    eigenvalues = torch.linalg.eigvalsh(R)  # real-valued since R is symmetric PSD
    writer.add_histogram("structure/R_eigenvalues", eigenvalues, epoch)


@torch.no_grad()
def _log_latent_scatter(
    model: ControlledDHGN_LSTM,
    val_traj: tuple,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    tag: str = "val/latent_regression",
) -> None:
    """Fit linear latent→state regression on a val trajectory and log scatter plots."""
    model.eval()
    frames, actions, states = val_traj  # (T+1, C, H, W), (T,), (T+1, 2)

    ctx = frames.unsqueeze(0).to(device)
    mu_all, _ = model.encoder.forward_all(ctx)
    s_all = model.f_psi(mu_all.squeeze(0)).cpu()  # (T+1, latent_dim)

    st = states.float()  # (T+1, 2): [θ, θ̇]
    A = torch.linalg.lstsq(s_all, st).solution  # (latent_dim, 2)
    st_pred = (s_all @ A).numpy()
    st_true = st.numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for i, name in enumerate(["θ (rad)", "θ̇ (rad/s)"]):
        true_i = st_true[:, i]
        pred_i = st_pred[:, i]
        axes[i].scatter(true_i, pred_i, s=3, alpha=0.5)
        lo, hi = min(true_i.min(), pred_i.min()), max(true_i.max(), pred_i.max())
        axes[i].plot([lo, hi], [lo, hi], "r--", linewidth=0.8)
        axes[i].set_xlabel(f"True {name}")
        axes[i].set_ylabel(f"Predicted {name}")
        ss_res = ((true_i - pred_i) ** 2).sum()
        ss_tot = ((true_i - true_i.mean()) ** 2).sum()
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        axes[i].set_title(f"{name}  R²={r2:.3f}")

    fig.suptitle(f"Latent → state regression (epoch {epoch + 1})")
    fig.tight_layout()
    writer.add_figure(tag, fig, epoch)
    plt.close(fig)


@click.command()
# data
@click.option("--n-episodes", type=int, default=200, show_default=True)
@click.option("--img-size", type=int, default=64, show_default=True)
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
    help="Number of steps per episode",
)
@click.option(
    "--damping",
    type=float,
    default=0.0,
    show_default=True,
    help="Linear viscous damping coefficient (theta_dot *= exp(-b*dt) per step)",
)
# model
@click.option("--pos-ch", type=int, default=8, show_default=True)
@click.option("--feat-dim", type=int, default=256, show_default=True)
@click.option("--dt", type=float, default=0.05, show_default=True)
@click.option("--no-separable", "separable", default=True, flag_value=False)
# training
@click.option("--phase1-epochs", type=int, default=1500, show_default=True)
@click.option("--phase2-epochs", type=int, default=1500, show_default=True)
@click.option(
    "--batch-size",
    type=int,
    default=8,
    show_default=True,
    help="Episodes per batch; full rollouts are memory-heavy",
)
@click.option("--lr", type=float, default=1e-4, show_default=True)
@click.option("--kl-weight", type=float, default=1e-3, show_default=True)
@click.option("--free-bits", type=float, default=0.5, show_default=True)
@click.option("--grad-clip", type=float, default=1.0, show_default=True)
@click.option(
    "--fd-weight",
    type=float,
    default=0.1,
    show_default=True,
    help="Weight for finite-difference p supervision loss in phase 1",
)
# logging
@click.option("--log-every", type=int, default=5, show_default=True)
@click.option(
    "--val-every",
    type=int,
    default=10,
    show_default=True,
    help="Epochs between H comparison plots (0 to disable)",
)
@click.option("--n-val-episodes", type=int, default=3, show_default=True)
@click.option(
    "--val-max-steps",
    type=int,
    default=0,
    show_default=True,
    help="Steps per val episode (0 = 2x --max-steps)",
)
@click.option("--checkpoint-every", type=int, default=10, show_default=True)
def main(**kwargs):
    assert kwargs["img_size"] % 8 == 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    writer = SummaryWriter(comment="_pendulum_offline")
    run_dir = make_run_dir("pendulum_offline")

    n_val = kwargs["n_val_episodes"] if kwargs["val_every"] > 0 else 0
    val_steps = kwargs["val_max_steps"] or kwargs["max_steps"] * 2
    print(f"\nCollecting {kwargs['n_episodes']} train episodes...")
    episodes = collect_data(
        n_episodes=kwargs["n_episodes"],
        img_size=kwargs["img_size"],
        epsilon=kwargs["epsilon"],
        energy_k=kwargs["energy_k"],
        max_steps=kwargs["max_steps"],
        damping=kwargs["damping"],
    )

    val_energy = []
    val_random = []
    val_spin = []
    if n_val > 0:
        print(
            f"Collecting {n_val} val episodes per type ({val_steps} steps each)..."
        )
        val_energy = collect_val_trajectories(
            n_episodes=n_val,
            img_size=kwargs["img_size"],
            max_steps=val_steps,
            energy_k=kwargs["energy_k"],
            damping=kwargs["damping"],
        )
        val_random = collect_random_trajectories(
            n_episodes=n_val,
            img_size=kwargs["img_size"],
            max_steps=val_steps,
            damping=kwargs["damping"],
        )
        val_spin = collect_spin_trajectories(
            n_episodes=n_val,
            img_size=kwargs["img_size"],
            max_steps=val_steps,
            damping=kwargs["damping"],
        )

    dataset = PendulumDataset(episodes)

    loader = DataLoader(
        dataset,
        batch_size=kwargs["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    print(f"Dataset: {len(dataset)} episodes")

    model = ControlledDHGN_LSTM(
        pos_ch=kwargs["pos_ch"],
        img_ch=3,
        dt=kwargs["dt"],
        feat_dim=kwargs["feat_dim"],
        img_size=kwargs["img_size"],
        control_dim=1,
        separable=kwargs["separable"],
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    hparams = {k: v for k, v in kwargs.items()}

    perception_params = (
        list(model.encoder.parameters())
        + list(model.f_psi.parameters())
        + list(model.decoder.parameters())
    )
    dynamics_params = (
        list(model.hamiltonian.parameters())
        + [model.A, model.L_param, model.B]
    )
    opt_phase1 = torch.optim.Adam(perception_params, lr=kwargs["lr"])
    opt_phase2 = torch.optim.Adam(dynamics_params, lr=kwargs["lr"])

    best_recon = float("inf")
    best_latent = float("inf")
    global_epoch = 0

    def _log_and_checkpoint(metrics, epoch, phase, best_val, metric_key):
        if (epoch + 1) % kwargs["log_every"] == 0:
            for k, v in metrics.items():
                writer.add_scalar(k, v, global_epoch)
            extra = ""
            if "train/fd_loss" in metrics:
                extra += f"  fd={metrics['train/fd_loss']:.4f}"
            if "train/r2" in metrics:
                extra += f"  r2={metrics['train/r2']:.4f}"
            tqdm.write(
                f"  [{phase}] epoch {epoch + 1:4d}"
                f"  loss={metrics['train/loss']:.4f}"
                f"  recon={metrics['train/recon']:.4f}"
                f"  kl={metrics['train/kl']:.4f}"
                f"  latent={metrics['train/latent']:.4f}"
                f"  q_var={metrics['train/q_var']:.4f}"
                f"  p_var={metrics['train/p_var']:.4f}"
                + extra
            )
        if (
            kwargs["val_every"] > 0
            and (epoch + 1) % kwargs["val_every"] == 0
        ):
            for val_trajs, label in (
                (val_energy, "energy_pump"),
                (val_random, "random"),
                (val_spin, "spin"),
            ):
                if not val_trajs:
                    continue
                _log_reconstruction_video(
                    model=model,
                    val_traj=val_trajs[0],
                    device=device,
                    writer=writer,
                    epoch=global_epoch,
                    tag=f"val/reconstruction/{label}",
                    use_dynamics=(phase == "P2"),
                )
            if val_energy:
                _log_hamiltonian_comparison(
                    model=model,
                    val_traj=val_energy[0],
                    device=device,
                    writer=writer,
                    epoch=global_epoch,
                    tag="val/hamiltonian/energy_pump",
                )
                _log_latent_scatter(
                    model=model,
                    val_traj=val_energy[0],
                    device=device,
                    writer=writer,
                    epoch=global_epoch,
                )
            _log_R_eigenvalues(model=model, writer=writer, epoch=global_epoch)
        if (
            kwargs["checkpoint_every"] > 0
            and (epoch + 1) % kwargs["checkpoint_every"] == 0
            and metrics[metric_key] < best_val
        ):
            save_checkpoint(run_dir, global_epoch, model, hparams, metrics)
            return metrics[metric_key]
        return best_val

    print("\n=== Phase 1: encoder/f_psi/decoder (recon + KL) ===")
    for epoch in tqdm(range(kwargs["phase1_epochs"]), desc="Phase 1"):
        metrics = _train_epoch_phase1(
            model=model,
            loader=loader,
            optimizer=opt_phase1,
            kl_weight=kwargs["kl_weight"],
            free_bits=kwargs["free_bits"],
            grad_clip=kwargs["grad_clip"],
            fd_weight=kwargs["fd_weight"],
            device=device,
        )
        best_recon = _log_and_checkpoint(metrics, epoch, "P1", best_recon, "train/recon")
        global_epoch += 1

    print("\n=== Phase 2: H/R/B (latent loss) ===")
    for epoch in tqdm(range(kwargs["phase2_epochs"]), desc="Phase 2"):
        metrics = _train_epoch_phase2(
            model=model,
            loader=loader,
            optimizer=opt_phase2,
            grad_clip=kwargs["grad_clip"],
            device=device,
        )
        best_latent = _log_and_checkpoint(metrics, epoch, "P2", best_latent, "train/latent")
        global_epoch += 1

    writer.close()
    print("\nDone. Run: tensorboard --logdir runs")
    os._exit(0)


if __name__ == "__main__":
    main()
