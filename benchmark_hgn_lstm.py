"""
Benchmark HGN-LSTM on synthetic SHO image sequences.

Identical experimental setup to benchmark_hgn_org.py, but the stacked-frame
CNN encoder is replaced by a recurrent encoder:

    FrameCNN (per frame) → reverse sequence → LSTM → hidden state
    → reshape → mu/log_var → reparameterise → f_ψ → (q0, p0)

The Hamiltonian network, leapfrog integrator, decoder, and all training /
validation logic are unchanged.
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
from checkpoint_common import make_run_dir, save_checkpoint
from hgn_lstm import HGN_LSTM
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
    "--feat-dim",
    type=int,
    default=256,
    show_default=True,
    help="Per-frame CNN embedding size fed to the LSTM.",
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
    feat_dim,
    grad_clip,
    log_every,
    diag_every,
    coord_weight,
    energy_weight,
    max_amplitude,
    spring_constant,
    mass,
):
    # Rename for clarity: context_len is the number of frames shown to the
    # encoder; the model sees frames 0..context_len-1, then integrates backward.
    context_len = n_frames

    assert img_size == 32, "HGN decoder outputs 32x32; set --img-size 32"
    assert seq_len >= context_len, (
        f"seq_len ({seq_len}) must be >= context_len ({context_len})"
    )
    assert seq_len >= train_rollout + 1, (
        f"seq_len ({seq_len}) must be >= train_rollout ({train_rollout}) + 1"
    )
    assert train_rollout < context_len, (
        f"train_rollout ({train_rollout}) must be < context_len ({context_len})"
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
    model = HGN_LSTM(pos_ch=pos_ch, img_ch=3, dt=dt, feat_dim=feat_dim).to(
        device
    )
    frames_val = frames_val.to(device)
    p_val = p_val.to(device)
    q_val = q_val.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    writer = SummaryWriter(comment="_hgn_lstm_sho_images")
    run_dir = make_run_dir("hgn_lstm")
    best_val_mse = float("inf")
    hparams = dict(
        img_size=img_size,
        blob_sigma=blob_sigma,
        seq_len=seq_len,
        context_len=context_len,
        train_rollout=train_rollout,
        dt=dt,
        n_train=n_train,
        n_val=n_val,
        batch_size=batch_size,
        n_epochs=n_epochs,
        lr=lr,
        kl_weight=kl_weight,
        recon_weight=recon_weight,
        free_bits=free_bits,
        grad_clip=grad_clip,
        pos_ch=pos_ch,
        feat_dim=feat_dim,
        coord_weight=coord_weight,
        energy_weight=energy_weight,
        max_amplitude=max_amplitude,
        spring_constant=spring_constant,
        mass=mass,
    )

    hparam_text = (
        "| Hyperparameter | Value |\n"
        "|---|---|\n"
        f"| img_size | {img_size} |\n"
        f"| blob_sigma | {blob_sigma} |\n"
        f"| seq_len | {seq_len} |\n"
        f"| context_len | {context_len} |\n"
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
        f"| feat_dim | {feat_dim} |\n"
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

            q0, p0, kl, mu, log_var = model(frames[:, :context_len])
            kl_loss = kl.clamp(min=free_bits).mean()

            rollout_result = model.rollout(
                q0,
                p0,
                n_steps=rollout_steps,
                dt=-dt,
                return_states=(energy_weight > 0),
            )
            if energy_weight > 0:
                pred_frames_list, pred_coords_list, qs, ps = rollout_result
            else:
                pred_frames_list, pred_coords_list = rollout_result
                qs = ps = []

            pred_frames = torch.stack(pred_frames_list, dim=1)
            pred_coords = torch.stack(pred_coords_list, dim=1)

            # Backward rollout produces [frame T, T-1, ..., T-rollout_steps].
            # Target is the encoder context reversed to match that order.
            target_frames = frames[:, :context_len].flip(dims=[1])[
                :, : rollout_steps + 1
            ]

            with torch.no_grad():
                gt_coords = image_centroid(target_frames)

            recon_loss = F.mse_loss(pred_frames, target_frames)
            coord_loss = F.mse_loss(pred_coords, gt_coords)

            if energy_weight > 0:
                H0 = model.H(q0.detach(), p0.detach()).detach()
                H_traj = torch.stack(
                    [model.H(q, p) for q, p in zip(qs, ps)], dim=1
                )
                energy_loss = (H_traj - H0.unsqueeze(1)).pow(2).mean()
            else:
                energy_loss = torch.zeros(1, device=device)

            loss = (
                recon_weight * recon_loss
                + kl_weight * kl_loss
                + coord_weight * coord_loss
                + energy_weight * energy_loss
            )

            optimizer.zero_grad()
            loss.backward()
            del (
                pred_frames_list,
                pred_coords_list,
                qs,
                ps,
                pred_frames,
                pred_coords,
            )

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
            act_monitor.remove()
            model.eval()
            # Backward rollout covers the encoder context (context_len frames).
            val_rollout_steps = context_len - 1
            N_VID = min(4, frames_val.shape[0])

            # Run validation in mini-batches to avoid OOM on large val sets.
            # Full-set metrics are accumulated incrementally; decoded frames and
            # phase-space trajectories are kept only for the first N_VID samples
            # (used for video / correlation / Hamiltonian logging).
            val_sum_mse = val_sum_coord_mse = val_n = 0
            vis_pred_val = vis_pred_coords_val = None
            qs_val = ps_val = None

            with torch.no_grad():
                for vi in range(0, frames_val.shape[0], batch_size):
                    fv = frames_val[vi : vi + batch_size]
                    q0_i, p0_i, _, _, _ = model(fv[:, :context_len])
                    pred_list_i, pred_coords_list_i, qs_i, ps_i = model.rollout(
                        q0_i,
                        p0_i,
                        n_steps=val_rollout_steps,
                        dt=-dt,
                        return_states=True,
                    )
                    pred_i = torch.stack(pred_list_i, dim=1).clamp(0, 1)
                    pred_coords_i = torch.stack(pred_coords_list_i, dim=1).cpu()
                    # Reversed context is the target: [frame T, T-1, ..., 0]
                    target_i = fv[:, :context_len].flip(dims=[1])
                    gt_coords_i = image_centroid(target_i.cpu())
                    bs = fv.shape[0]
                    val_sum_mse += F.mse_loss(pred_i, target_i).item() * bs
                    val_sum_coord_mse += (
                        F.mse_loss(pred_coords_i, gt_coords_i).item() * bs
                    )
                    val_n += bs
                    if vis_pred_val is None:
                        vis_pred_val = pred_i[:N_VID].cpu()
                        vis_pred_coords_val = pred_coords_i[:N_VID]
                        qs_val = [q[:N_VID] for q in qs_i]
                        ps_val = [p[:N_VID] for p in ps_i]
                    del (
                        pred_list_i,
                        pred_coords_list_i,
                        qs_i,
                        ps_i,
                        pred_i,
                        pred_coords_i,
                    )

            val_mse = val_sum_mse / val_n
            val_coord_mse = val_sum_coord_mse / val_n
            writer.add_scalar("val/pixel_mse", val_mse, epoch)
            writer.add_scalar("val/coord_mse", val_coord_mse, epoch)

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                save_checkpoint(
                    run_dir,
                    epoch,
                    model,
                    hparams,
                    {"val_pixel_mse": val_mse, "val_coord_mse": val_coord_mse},
                )

            # q_gt_val in reversed order to match backward-rollout predictions
            q_gt_val = q_val[:N_VID, :context_len].flip(dims=[1])
            corr = log_correlation_plots(
                writer, vis_pred_val, q_gt_val.cpu(), epoch
            )

            gt_vid = frames_val[:N_VID, :context_len].flip(dims=[1]).cpu()
            log_gt_pred_video(
                writer, "val/gt_vs_pred", gt_vid, vis_pred_val, epoch
            )

            gt_pos = image_centroid(gt_vid)
            T_v = gt_vid.shape[1]
            gt_vid_up = tvf.resize(
                gt_vid.reshape(-1, 3, img_size, img_size), (64, 64)
            ).reshape(N_VID, T_v, 3, 64, 64)
            log_marker_video(
                writer,
                "val/coord_markers",
                gt_vid_up,
                gt_pos,
                vis_pred_coords_val,
                epoch,
                size=4,
            )

            log_hamiltonian_conservation(
                writer,
                model.H,
                qs_val,
                ps_val,
                N_VID,
                epoch,
                context_len=context_len,
            )

            tqdm.write(
                f"  epoch {epoch + 1:3d}  val_mse={val_mse:.4f}"
                f"  corr_q={abs(corr):.3f}  coord_mse={val_coord_mse:.4f}"
            )
            act_monitor = ActivationMonitor(model)

    act_monitor.remove()
    writer.close()
    print("\nDone. Run: tensorboard --logdir runs")


if __name__ == "__main__":
    main()
