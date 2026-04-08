"""
Benchmark RecurrentHGN on synthetic SHO image sequences.

A 1D simple harmonic oscillator is rendered as a Gaussian blob moving
horizontally across a canvas. The model sees only images; ground-truth
(p, q) are withheld from training and used only for validation.

Training signal:
  - The encoder processes the first N_FRAMES frames → (p_0, q_0)
  - (p_0, q_0) is rolled out using the learned Hamiltonian for the rest of
    the sequence, and each rolled-out state is decoded and compared to the
    corresponding ground-truth q value.
  - KL loss regularises the initial latent state.

Validation:
  - Pearson correlation between rolled-out latents and true (p, q)
  - Scatter plots, trajectory comparisons, and videos logged to TensorBoard
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
from hgn import RecurrentHGN
from diag_common import (
    ActivationMonitor,
    generate_dataset,
    image_centroid,
    log_gradient_stats,
    log_gt_pred_video,
    log_hamiltonian_conservation,
    log_hamiltonian_grad_stats,
    log_histograms,
    log_latent_stats,
    log_marker_video,
    log_weight_norms,
    render_frame,
)

# margin used in render_frame — must match the coordinate conversion at inference
_MARGIN = 8


# ---- Script-specific logging helpers ----


def log_correlation_plots(writer, p_enc, q_enc, p_gt, q_gt, epoch):
    """Scatter plots of encoded vs ground-truth (p, q)."""
    p_gt_np = p_gt.reshape(-1).cpu().numpy()
    q_gt_np = q_gt.reshape(-1).cpu().numpy()

    # When latent size > 1, find the best-correlated dimension.
    size = p_enc.shape[-1] if p_enc.dim() == 3 else 1
    if size == 1:
        p_enc_np = p_enc.reshape(-1).cpu().numpy()
        q_enc_np = q_enc.reshape(-1).cpu().numpy()
    else:
        p_enc_dims = p_enc.reshape(-1, size).cpu().numpy()
        q_enc_dims = q_enc.reshape(-1, size).cpu().numpy()
        p_enc_np = max(
            (p_enc_dims[:, i] for i in range(size)),
            key=lambda col: abs(np.corrcoef(p_gt_np, col)[0, 1]),
        )
        q_enc_np = max(
            (q_enc_dims[:, i] for i in range(size)),
            key=lambda col: abs(np.corrcoef(q_gt_np, col)[0, 1]),
        )

    corr_p = float(np.corrcoef(p_gt_np, p_enc_np)[0, 1])
    corr_q = float(np.corrcoef(q_gt_np, q_enc_np)[0, 1])

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, gt, enc, label, corr in zip(
        axes,
        [p_gt_np, q_gt_np],
        [p_enc_np, q_enc_np],
        ["p", "q"],
        [corr_p, corr_q],
    ):
        ax.scatter(gt, enc, alpha=0.15, s=4)
        ax.set_xlabel(f"True {label}")
        ax.set_ylabel(f"Encoded {label}")
        ax.set_ylim(-3.0, 3.0)
        ax.set_title(f"{label} alignment  (epoch {epoch + 1})")
        ax.text(
            0.05, 0.93, f"r = {corr:.3f}", transform=ax.transAxes, fontsize=11
        )

    plt.tight_layout()
    writer.add_figure("val/pq_correlation", fig, epoch)
    writer.add_scalar("val/corr_p", abs(corr_p), epoch)
    writer.add_scalar("val/corr_q", abs(corr_q), epoch)
    plt.close(fig)
    return corr_p, corr_q


def log_sample_trajectory(writer, ps_roll, qs_roll, p_gt, q_gt, epoch, dt):
    """Overlay rolled-out latents vs ground-truth trajectory for one validation sequence."""
    idx = 0
    rollout_len = ps_roll.shape[1]
    t_axis = np.arange(rollout_len) * dt

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, gt, roll, label in zip(
        axes,
        [
            p_gt[idx, :rollout_len].cpu().numpy(),
            q_gt[idx, :rollout_len].cpu().numpy(),
        ],
        [ps_roll[idx, :, 0].cpu().numpy(), qs_roll[idx, :, 0].cpu().numpy()],
        ["p (momentum)", "q (position)"],
    ):
        ax.plot(t_axis, gt, label=f"True {label.split()[0]}")
        ax.plot(
            t_axis, roll, label=f"Rolled-out {label.split()[0]}", linestyle="--"
        )
        ax.set_ylabel(label)
        ax.legend()

    axes[-1].set_xlabel("Time")
    plt.suptitle(f"Rollout trajectory  (epoch {epoch + 1})")
    plt.tight_layout()
    writer.add_figure("val/sample_trajectory", fig, epoch)
    plt.close(fig)


def rollout(model, p0, q0, steps, dt):
    """Roll out `steps` leapfrog steps from (p0, q0). Returns (p, q) each (batch, steps, size)."""
    ps, qs = [p0], [q0]
    p, q = p0, q0
    for _ in range(steps - 1):
        p, q = model.hamiltonian_step(p, q, step_size=dt)
        ps.append(p)
        qs.append(q)
    return torch.stack(ps, dim=1), torch.stack(qs, dim=1)


# ---- CLI ----


@click.command()
@shared_options
@click.option(
    "--p-init-scale",
    type=float,
    default=3.0,
    show_default=True,
    help="Scale factor for p-output weights of f_psi at init.",
)
@click.option(
    "--non-deg-weight",
    type=float,
    default=1e-3,
    show_default=True,
    help="Weight for dH non-degeneracy penalty.",
)
@click.option(
    "--energy-cons-weight",
    type=float,
    default=1e-2,
    show_default=True,
    help="Weight for energy conservation loss: Var_t(H(p_t, q_t)) along rollout.",
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
    grad_clip,
    log_every,
    diag_every,
    max_amplitude,
    spring_constant,
    mass,
    p_init_scale,
    non_deg_weight,
    energy_cons_weight,
):

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
        margin=_MARGIN,
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
        margin=_MARGIN,
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
    model = RecurrentHGN(size=32, p_init_scale=p_init_scale).to(device)
    frames_val = frames_val.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    writer = SummaryWriter(comment="_sho_images_recurrent")

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
        f"| p_init_scale | {p_init_scale} |\n"
        f"| non_deg_weight | {non_deg_weight} |\n"
        f"| energy_cons_weight | {energy_cons_weight} |\n"
        f"| max_amplitude | {max_amplitude} |\n"
        f"| spring_constant | {spring_constant} |\n"
        f"| mass | {mass} |\n"
    )
    writer.add_text("hparams", hparam_text, 0)

    act_monitor = ActivationMonitor(model)

    # Hook on latent_encoder.fc to capture (mu, log_var) before reparameterization.
    _fc_cache: dict = {}

    def _fc_hook(_, __, output):
        _fc_cache["fc_out"] = output.detach()

    _fc_hook_handle = model.latent_encoder.fc.register_forward_hook(_fc_hook)

    # RecurrentHGN.H(p, q) — wrap to the shared (q, p) convention.
    H_fn = lambda q, p: model.H(p, q)  # noqa: E731

    # ---- Training loop ----
    global_step = 0

    for epoch in (epoch_bar := tqdm(range(n_epochs), desc="Epochs")):
        model.train()
        epoch_loss = epoch_kl = epoch_recon = 0.0

        for frames, p_true, q_true in tqdm(
            train_loader, desc=f"  epoch {epoch}", leave=False
        ):
            frames = frames.to(device)
            q_true = q_true.to(device)

            # Encode the first n_frames context frames → (p0, q0) at t=0.
            p0, q0, kl = model(frames[:, :n_frames])
            p0 = p0.requires_grad_(True)
            q0 = q0.requires_grad_(True)
            kl_loss = kl.clamp(min=free_bits).mean()

            ps, qs = rollout(model, p0, q0, steps=train_rollout, dt=dt)

            target_q = q_true[:, :train_rollout]
            pred_q = model.decode(ps, qs)  # (batch, train_rollout, 1)
            recon_loss = F.mse_loss(pred_q.squeeze(-1), target_q)

            H_rollout = model.H(ps, qs)  # (batch, train_rollout)
            energy_cons_loss = H_rollout.var(dim=1).mean()

            p_nd = p0.detach().requires_grad_(True)
            q_nd = q0.detach().requires_grad_(True)
            H_nd = model.H(p_nd, q_nd).sum()
            dH_dp_nd, dH_dq_nd = torch.autograd.grad(
                H_nd, [p_nd, q_nd], create_graph=True
            )
            non_deg_loss = 1.0 / (
                dH_dp_nd.pow(2).mean() + dH_dq_nd.pow(2).mean() + 1e-6
            )

            loss = (
                recon_weight * recon_loss
                + kl_weight * kl_loss
                + energy_cons_weight * energy_cons_loss
                + non_deg_weight * non_deg_loss
            )

            optimizer.zero_grad()
            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            if diag_every > 0 and global_step % diag_every == 0:
                log_gradient_stats(writer, model, global_step)
                act_monitor.log(writer, global_step)
                act_monitor.check_flags(global_step)

                if "fc_out" in _fc_cache:
                    fc_out = _fc_cache["fc_out"]  # (batch, 2*size)
                    sz = model.size
                    mu_z = fc_out[:, :sz]  # mu from variational bottleneck
                    log_var_z = fc_out[
                        :, sz:
                    ]  # log_var from variational bottleneck
                    log_latent_stats(
                        writer, {"z": (mu_z, log_var_z)}, global_step, free_bits
                    )

                log_hamiltonian_grad_stats(
                    writer, H_fn, q0.detach(), p0.detach(), global_step
                )
                log_weight_norms(writer, model, global_step)

            optimizer.step()

            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar(
                "train/recon_loss", recon_loss.item(), global_step
            )
            writer.add_scalar("train/kl_loss", kl_loss.item(), global_step)
            writer.add_scalar(
                "train/energy_cons_loss", energy_cons_loss.item(), global_step
            )
            writer.add_scalar(
                "train/non_deg_loss", non_deg_loss.item(), global_step
            )
            global_step += 1

            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()

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
            val_rollout_len = seq_len

            with torch.no_grad():
                p0_val, q0_val, _ = model(frames_val[:, :n_frames])
                ps_val, qs_val = rollout(
                    model, p0_val, q0_val, steps=val_rollout_len, dt=dt
                )

            p_gt_roll = p_val[:, :val_rollout_len].to(device)
            q_gt_roll = q_val[:, :val_rollout_len].to(device)

            writer.add_scalar("val/mu_p_std", ps_val.std().item(), epoch)
            writer.add_scalar("val/mu_q_std", qs_val.std().item(), epoch)
            corr_p, corr_q = log_correlation_plots(
                writer,
                ps_val,
                qs_val,
                p_gt_roll,
                q_gt_roll,
                epoch,
            )
            log_sample_trajectory(
                writer,
                ps_val,
                qs_val,
                p_val.to(device),
                q_val.to(device),
                epoch,
                dt,
            )

            N_VID = 4

            # Render predicted frames from the decoded scalar position.
            pred_q_dec = (
                model.decode(ps_val[:N_VID], qs_val[:N_VID]).squeeze(-1).cpu()
            )  # (N_VID, val_rollout_len)
            pred_frames = torch.stack(
                [
                    torch.stack(
                        [
                            render_frame(
                                pred_q_dec[n, t].item(),
                                img_size,
                                blob_sigma,
                                max_amplitude,
                                _MARGIN,
                            )
                            for t in range(val_rollout_len)
                        ]
                    )
                    for n in range(N_VID)
                ]
            )  # (N_VID, val_rollout_len, 3, img_size, img_size)

            gt_vid = frames_val[:N_VID, :val_rollout_len].cpu()
            val_pixel_mse = F.mse_loss(pred_frames, gt_vid).item()
            writer.add_scalar("val/pixel_mse", val_pixel_mse, epoch)

            # Side-by-side GT | rendered-prediction video
            log_gt_pred_video(
                writer, "val/gt_vs_pred", gt_vid, pred_frames, epoch
            )

            # GT frames with X markers (GT centroid green, predicted position red)
            frames_val_84 = (
                tvf.resize(
                    frames_val[:N_VID].reshape(-1, 3, img_size, img_size),
                    (84, 84),
                )
                .reshape(N_VID, seq_len, 3, 84, 84)
                .cpu()
            )
            gt_frames_84 = frames_val_84[:, :val_rollout_len]
            pred_x = (
                pred_q_dec.unsqueeze(-1)
                / max_amplitude
                * (img_size / 2 - _MARGIN)
                / img_size
                + 0.5
            )  # (N_VID, val_rollout_len, 1)
            pred_pos = torch.cat([pred_x, torch.full_like(pred_x, 0.5)], dim=-1)
            gt_pos = image_centroid(gt_frames_84)
            log_marker_video(
                writer,
                "val/reconstruction",
                gt_frames_84,
                gt_pos,
                pred_pos,
                epoch,
            )
            log_hamiltonian_conservation(
                writer, H_fn, qs_val, ps_val, N_VID, epoch, context_len=n_frames
            )
            tqdm.write(
                f"  epoch {epoch + 1:3d}  |r_p|={abs(corr_p):.3f}  |r_q|={abs(corr_q):.3f}"
            )

    act_monitor.remove()
    _fc_hook_handle.remove()
    writer.close()
    print("\nDone. Run: tensorboard --logdir runs")


if __name__ == "__main__":
    main()
