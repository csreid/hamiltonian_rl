"""
Benchmark DissipativeHGN on synthetic damped-SHO image sequences.

A 1D underdamped harmonic oscillator is rendered as a Gaussian blob moving
horizontally across a 32×32 canvas.  The oscillator loses energy over time
due to viscous damping:

    m q̈ + γ q̇ + k q = 0

In phase space:
    dq/dt =  p / m
    dp/dt = −k q − γ p / m

The Hamiltonian H = p²/2m + k q²/2 is NOT conserved; it decays as:
    dH/dt = −γ p² / m² ≤ 0

The model must learn J and R such that dz/dt = (J − R) ∇H reproduces this
behaviour.  The ground-truth structure for this system is:
    J = [[0, 1], [−1, 0]]   (up to scaling by k / m)
    R = [[0, 0], [0, γ / m]]

Training signal:
  - Encoder processes N_FRAMES context frames → (q0, p0)
  - (q0, p0) is rolled out using the learned port-Hamiltonian dynamics
  - Each decoded frame is compared to the corresponding GT frame (pixel MSE)
  - KL loss regularises the latent posterior

Validation:
  - Pixel MSE between decoded rollout and ground-truth frames
  - Centroid correlation between decoded frames and true q
  - H(q, p) over rollout (expected to decrease monotonically)
  - Side-by-side GT / predicted video logged to TensorBoard
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import functional as tvf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import click

from cli_common import shared_options
from dhgn import DissipativeHGN
from diag_common import (
    ActivationMonitor,
    image_centroid,
    log_gradient_stats,
    log_hamiltonian_conservation,
    log_hamiltonian_grad_stats,
    log_gt_pred_video,
    log_histograms,
    log_latent_stats,
    log_marker_video,
    log_weight_norms,
    render_frame,
)


# ---- Dataset generation (re-exported from data.sho) ----

from data.sho import generate_damped_dataset  # noqa: F401, E402


# ---- Script-specific logging ----


def log_correlation_plots(writer, pred_frames, q_gt, epoch):
    """Compare x-centroid of decoded frames to ground-truth q."""
    pred_pos = image_centroid(pred_frames)  # (N, T, 2)
    x_pred = pred_pos[..., 0].reshape(-1).cpu().numpy()
    q_gt_np = q_gt.reshape(-1).cpu().numpy()

    q_min, q_max = q_gt_np.min(), q_gt_np.max()
    q_norm = (q_gt_np - q_min) / (q_max - q_min + 1e-8)
    corr = float(np.corrcoef(q_norm, x_pred)[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(q_norm, x_pred, alpha=0.15, s=4)
    ax.set_xlabel("True q (normalised)")
    ax.set_ylabel("Decoded frame centroid x")
    ax.set_title(f"Position correlation  (epoch {epoch + 1})")
    ax.text(0.05, 0.93, f"r = {corr:.3f}", transform=ax.transAxes, fontsize=11)
    plt.tight_layout()
    writer.add_figure("val/q_centroid_correlation", fig, epoch)
    writer.add_scalar("val/corr_q_centroid", abs(corr), epoch)
    plt.close(fig)
    return corr


def log_J_R_structure(writer, model, epoch):
    """Log the learned J and R matrices as images and their Frobenius norms."""
    with torch.no_grad():
        J = model.get_J().cpu()
        R = model.get_R().cpu()

    writer.add_scalar("val/J_frobenius", J.norm(p="fro").item(), epoch)
    writer.add_scalar("val/R_frobenius", R.norm(p="fro").item(), epoch)
    writer.add_scalar("val/R_trace", R.trace().item(), epoch)

    # Skew-symmetry residual for J (should be ~0 by construction, sanity check)
    skew_res = (J + J.T).norm().item()
    writer.add_scalar("diag/J_skew_residual", skew_res, epoch)

    # PSD check: smallest eigenvalue of R (should be ≥ 0)
    eigvals = torch.linalg.eigvalsh(R)
    writer.add_scalar("diag/R_min_eigenvalue", eigvals.min().item(), epoch)


# ---- CLI ----


@click.command()
@shared_options
@click.option(
    "--pos-ch",
    type=int,
    default=16,
    show_default=True,
    help="Position channel depth.",
)
@click.option(
    "--damping",
    type=float,
    default=0.3,
    show_default=True,
    help="Viscous damping coefficient γ. Must give ζ = γ/(2√(km)) < 1.",
)
@click.option(
    "--coord-weight",
    type=float,
    default=0.0,
    show_default=True,
    help="Weight for coordinate head MSE loss.",
)
@click.option(
    "--energy-weight",
    type=float,
    default=0.0,
    show_default=True,
    help="Weight for auxiliary energy conservation penalty (set >0 to debug).",
)
def main(
    img_size,
    blob_sigma,
    seq_len,
    n_frames,
    train_rollout,
    dt,
    n_train,
    n_val,
    batch_size,
    n_epochs,
    lr,
    kl_weight,
    recon_weight,
    free_bits,
    pos_ch,
    grad_clip,
    log_every,
    diag_every,
    coord_weight,
    energy_weight,
    max_amplitude,
    spring_constant,
    mass,
    damping,
):
    assert img_size == 32, (
        "DissipativeHGN decoder outputs 32x32; set --img-size 32"
    )
    assert seq_len >= n_frames, (
        f"seq_len ({seq_len}) must be >= n_frames ({n_frames})"
    )
    assert seq_len >= train_rollout + 1, (
        f"seq_len ({seq_len}) must be >= train_rollout ({train_rollout}) + 1"
    )

    zeta = damping / (2.0 * np.sqrt(spring_constant * mass))
    print(f"Damped SHO: ω₀={np.sqrt(spring_constant / mass):.3f}  ζ={zeta:.3f}")
    assert zeta < 1.0, f"overdamped (ζ={zeta:.3f}); reduce --damping"

    # ---- Dataset generation ----
    print("Generating datasets...")
    p_train, q_train, frames_train = generate_damped_dataset(
        n_train,
        seq_len,
        dt,
        img_size,
        blob_sigma,
        max_amplitude,
        spring_constant,
        mass,
        damping,
        margin=4,
    )
    p_val, q_val, frames_val = generate_damped_dataset(
        n_val,
        seq_len,
        dt,
        img_size,
        blob_sigma,
        max_amplitude,
        spring_constant,
        mass,
        damping,
        margin=4,
    )
    print(f"  train frames: {frames_train.shape}")
    print(f"  val   frames: {frames_val.shape}\n")

    train_loader = DataLoader(
        TensorDataset(frames_train, p_train, q_train),
        batch_size=batch_size,
        shuffle=True,
    )

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # ---- Model, optimiser, writer ----
    model = DissipativeHGN(
        n_frames=n_frames, pos_ch=pos_ch, img_ch=3, dt=dt
    ).to(device)
    frames_val = frames_val.to(device)
    p_val = p_val.to(device)
    q_val = q_val.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    writer = SummaryWriter(comment="_dhgn_damped_sho")

    hparam_text = (
        "| Hyperparameter | Value |\n"
        "|---|---|\n"
        f"| img_size | {img_size} |\n"
        f"| blob_sigma | {blob_sigma} |\n"
        f"| seq_len | {seq_len} |\n"
        f"| n_frames | {n_frames} |\n"
        f"| train_rollout | {train_rollout} |\n"
        f"| dt | {dt} |\n"
        f"| n_train | {n_train} |\n"
        f"| n_val | {n_val} |\n"
        f"| batch_size | {batch_size} |\n"
        f"| n_epochs | {n_epochs} |\n"
        f"| lr | {lr} |\n"
        f"| kl_weight | {kl_weight} |\n"
        f"| recon_weight | {recon_weight} |\n"
        f"| free_bits | {free_bits} |\n"
        f"| grad_clip | {grad_clip} |\n"
        f"| pos_ch | {pos_ch} |\n"
        f"| coord_weight | {coord_weight} |\n"
        f"| energy_weight | {energy_weight} |\n"
        f"| max_amplitude | {max_amplitude} |\n"
        f"| spring_constant | {spring_constant} |\n"
        f"| mass | {mass} |\n"
        f"| damping (γ) | {damping} |\n"
        f"| damping_ratio (ζ) | {zeta:.4f} |\n"
    )
    writer.add_text("hparams", hparam_text, 0)

    act_monitor = ActivationMonitor(model)

    # ---- Training loop ----
    global_step = 0
    rollout_steps = train_rollout

    for epoch in (epoch_bar := tqdm(range(n_epochs), desc="Epochs")):
        model.train()
        epoch_loss = epoch_kl = epoch_recon = epoch_coord = epoch_energy = 0.0

        for frames, _, _ in tqdm(
            train_loader, desc=f"  epoch {epoch}", leave=False
        ):
            frames = frames.to(device)

            q0, p0, kl, mu, log_var = model(frames[:, :n_frames])
            kl_loss = kl.clamp(min=free_bits).mean()

            H0 = model.H(q0.detach(), p0.detach()).detach()

            pred_frames_list, pred_coords_list, qs, ps = model.rollout(
                q0, p0, n_steps=rollout_steps, dt=dt, return_states=True
            )
            pred_frames = torch.stack(pred_frames_list, dim=1)
            pred_coords = torch.stack(pred_coords_list, dim=1)

            target_frames = frames[:, : rollout_steps + 1]

            with torch.no_grad():
                gt_coords = image_centroid(target_frames)

            recon_loss = F.mse_loss(pred_frames, target_frames)
            coord_loss = F.mse_loss(pred_coords, gt_coords)

            # For a damped system, H should decrease; energy_weight > 0 adds a
            # penalty if H increases, which can help regularise early training.
            H_traj = torch.stack([model.H(q, p) for q, p in zip(qs, ps)], dim=1)
            energy_loss = F.relu(H_traj - H0.unsqueeze(1)).mean()

            loss = (
                recon_weight * recon_loss
                + kl_weight * kl_loss
                + coord_weight * coord_loss
                + energy_weight * energy_loss
            )

            optimizer.zero_grad()
            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            if diag_every > 0 and global_step % diag_every == 0:
                log_gradient_stats(writer, model, global_step)
                act_monitor.log(writer, global_step)
                act_monitor.check_flags(global_step)
                log_latent_stats(
                    writer,
                    {"z": (mu.detach(), log_var.detach())},
                    global_step,
                    free_bits,
                )
                log_hamiltonian_grad_stats(
                    writer, model.H, q0.detach(), p0.detach(), global_step
                )
                log_weight_norms(writer, model, global_step)

                # Log J and R norms during training
                with torch.no_grad():
                    writer.add_scalar(
                        "diag/J_norm",
                        model.get_J().norm(p="fro").item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "diag/R_norm",
                        model.get_R().norm(p="fro").item(),
                        global_step,
                    )

            optimizer.step()

            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar(
                "train/recon_loss", recon_loss.item(), global_step
            )
            writer.add_scalar("train/kl_loss", kl_loss.item(), global_step)
            writer.add_scalar(
                "train/coord_loss", coord_loss.item(), global_step
            )
            writer.add_scalar(
                "train/energy_loss", energy_loss.item(), global_step
            )
            global_step += 1

            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
            epoch_coord += coord_loss.item()
            epoch_energy += energy_loss.item()

        n = len(train_loader)
        epoch_bar.set_postfix(
            loss=f"{epoch_loss / n:.4f}",
            recon=f"{epoch_recon / n:.4f}",
            kl=f"{epoch_kl / n:.4f}",
        )

        if (epoch + 1) % log_every == 0:
            log_histograms(writer, model, epoch)

        if (epoch + 1) % log_every == 0:
            model.eval()
            val_rollout_steps = seq_len - 1

            with torch.no_grad():
                q0_val, p0_val, kl_val, _, _ = model(frames_val[:, :n_frames])
                pred_val_list, pred_coords_list, qs_val, ps_val = model.rollout(
                    q0_val,
                    p0_val,
                    n_steps=val_rollout_steps,
                    dt=dt,
                    return_states=True,
                )
                pred_val = torch.stack(pred_val_list, dim=1).clamp(0, 1)
                pred_coords_val = torch.stack(pred_coords_list, dim=1).cpu()

            target_val = frames_val[:, : val_rollout_steps + 1]
            val_mse = F.mse_loss(pred_val, target_val).item()
            writer.add_scalar("val/pixel_mse", val_mse, epoch)

            gt_coords_val = image_centroid(target_val.cpu())
            val_coord_mse = F.mse_loss(pred_coords_val, gt_coords_val).item()
            writer.add_scalar("val/coord_mse", val_coord_mse, epoch)

            q_gt_val = q_val[:, : val_rollout_steps + 1]
            corr = log_correlation_plots(
                writer, pred_val.cpu(), q_gt_val.cpu(), epoch
            )

            # Side-by-side GT / predicted video
            N_VID = min(4, frames_val.shape[0])
            gt_vid = frames_val[:N_VID, : val_rollout_steps + 1].cpu()
            pr_vid = pred_val[:N_VID].cpu()
            log_gt_pred_video(writer, "val/gt_vs_pred", gt_vid, pr_vid, epoch)

            # GT frames with X markers
            gt_pos = image_centroid(gt_vid)
            pred_pos = pred_coords_val[:N_VID]
            N_VID_t, T_v = gt_vid.shape[:2]
            gt_vid_up = tvf.resize(
                gt_vid.reshape(-1, 3, img_size, img_size), (64, 64)
            ).reshape(N_VID, T_v, 3, 64, 64)
            log_marker_video(
                writer,
                "val/coord_markers",
                gt_vid_up,
                gt_pos,
                pred_pos,
                epoch,
                size=4,
            )

            # H(q, p) over time — for a damped system this should decrease.
            # The context-end marker shows where training trajectories ended.
            log_hamiltonian_conservation(
                writer,
                model.H,
                qs_val,
                ps_val,
                N_VID,
                epoch,
                context_len=n_frames,
            )

            # Log structure matrix properties
            log_J_R_structure(writer, model, epoch)

            tqdm.write(
                f"  epoch {epoch + 1:3d}  val_mse={val_mse:.4f}"
                f"  corr_q={abs(corr):.3f}  coord_mse={val_coord_mse:.4f}"
            )

    act_monitor.remove()
    writer.close()
    print("\nDone. Run: tensorboard --logdir runs")


if __name__ == "__main__":
    main()
