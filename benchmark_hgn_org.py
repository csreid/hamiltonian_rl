"""
Benchmark HGN (Toth et al., ICLR 2020) on synthetic SHO image sequences.

A 1D simple harmonic oscillator is rendered as a Gaussian blob moving
horizontally across a 32×32 canvas. The model encodes the first N_FRAMES
frames into an abstract phase-space state and decodes rolled-out states
back to pixel space.

Training signal:
  - The encoder processes N_FRAMES context frames → (q0, p0)
  - (q0, p0) is rolled out using the learned Hamiltonian
  - Each decoded frame is compared to the corresponding GT frame (pixel MSE)
  - KL loss regularises the latent posterior

Validation:
  - Pixel MSE between decoded rollout and ground-truth frames
  - Centroid correlation between decoded frames and true (p, q)
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
from hgn_org import HGN
from diag_common import (
    ActivationMonitor,
    generate_dataset,
    image_centroid,
    log_gradient_stats,
    log_hamiltonian_conservation,
    log_hamiltonian_grad_stats,
    log_gt_pred_video,
    log_histograms,
    log_latent_stats,
    log_marker_video,
    log_weight_norms,
)


# ---- Script-specific logging helpers ----


def log_correlation_plots(writer, pred_frames, q_gt, epoch):
    """Compare x-centroid of decoded frames to ground-truth q (position).

    pred_frames: (N, T, 3, H, W) in [0, 1]
    q_gt:        (N, T)
    """
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


# ---- CLI ----


@click.command()
@shared_options
@click.option(
    "--pos-ch",
    type=int,
    default=16,
    show_default=True,
    help="Position channel depth (paper default: 16).",
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
    help="Weight for Hamiltonian conservation penalty.",
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
):
    assert img_size == 32, "HGN decoder outputs 32x32; set --img-size 32"
    assert seq_len >= n_frames, (
        f"seq_len ({seq_len}) must be >= n_frames ({n_frames})"
    )
    assert seq_len >= train_rollout + 1, (
        f"seq_len ({seq_len}) must be >= train_rollout ({train_rollout}) + 1"
    )

    # ---- Dataset generation ----
    print("Generating datasets...")
    p_train, q_train, frames_train = generate_dataset(
        n_train,
        seq_len,
        dt,
        img_size,
        blob_sigma,
        max_amplitude,
        spring_constant,
        mass,
        margin=4,
    )
    p_val, q_val, frames_val = generate_dataset(
        n_val,
        seq_len,
        dt,
        img_size,
        blob_sigma,
        max_amplitude,
        spring_constant,
        mass,
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
    model = HGN(n_frames=n_frames, pos_ch=pos_ch, img_ch=3, dt=dt).to(device)
    frames_val = frames_val.to(device)
    p_val = p_val.to(device)
    q_val = q_val.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    writer = SummaryWriter(comment="_hgn_org_sho_images")

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

            H_traj = torch.stack([model.H(q, p) for q, p in zip(qs, ps)], dim=1)
            energy_loss = (H_traj - H0.unsqueeze(1)).pow(2).mean()

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
            coord=f"{epoch_coord / n:.4f}",
            energy=f"{epoch_energy / n:.4f}",
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

            # Video 1: GT on top, reconstructed below
            N_VID = min(4, frames_val.shape[0])
            gt_vid = frames_val[:N_VID, : val_rollout_steps + 1].cpu()
            pr_vid = pred_val[:N_VID].cpu()
            log_gt_pred_video(writer, "val/gt_vs_pred", gt_vid, pr_vid, epoch)

            # Video 2: GT frames with X markers for GT centroid and predicted coord
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

            # H(q, p) over time
            log_hamiltonian_conservation(
                writer,
                model.H,
                qs_val,
                ps_val,
                N_VID,
                epoch,
                context_len=n_frames,
            )

            tqdm.write(
                f"  epoch {epoch + 1:3d}  val_mse={val_mse:.4f}"
                f"  corr_q={abs(corr):.3f}  coord_mse={val_coord_mse:.4f}"
            )

    act_monitor.remove()
    writer.close()
    print("\nDone. Run: tensorboard --logdir runs")


if __name__ == "__main__":
    main()
