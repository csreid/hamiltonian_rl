"""Supervised pendulum encoder/decoder training.

Collects episodes the same way as pendulum_offline.py, then runs two
separate supervised training phases:

Phase 1 — Encoder
    Full episode pixel sequence → FlexLSTMEncoder (BiLSTM) → predicted
    (theta, theta_dot).  Loss: MSE against ground-truth states[:, 0].
    latent_dim=2 so the encoder's mu head directly produces the 2-D
    pendulum phase-space state.

Phase 2 — Decoder
    Ground-truth (theta, theta_dot) at every timestep → FlexDecoder →
    predicted frame.  Loss: MSE against the corresponding ground-truth
    pixel frame.  q_dim=2 so the decoder takes the full phase state.
"""

from __future__ import annotations

import os
import sys

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Allow imports from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from checkpoint_common import make_run_dir, save_checkpoint
from data.pendulum import PendulumDataset, collect_data
from phgn_lstm import FlexDecoder, FlexLSTMEncoder


# ---------------------------------------------------------------------------
# Per-frame state dataset (for decoder training)
# ---------------------------------------------------------------------------


class FrameStateDataset(Dataset):
    """Flat dataset of (state, frame) pairs across all episodes and timesteps.

    states: (N, 2)  float32 — (theta, theta_dot)
    frames: (N, 3, H, W) float32 [0, 1]
    """

    def __init__(self, episodes: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
        all_states, all_frames = [], []
        for frames, _, states in episodes:
            # frames: (T+1, 3, H, W), states: (T+1, 2)
            all_states.append(states)
            all_frames.append(frames)
        self.states = torch.cat(all_states, dim=0)  # (N_total, 2)
        self.frames = torch.cat(all_frames, dim=0)  # (N_total, 3, H, W)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.frames[idx]


# ---------------------------------------------------------------------------
# Training phases
# ---------------------------------------------------------------------------


def _train_encoder_epoch(
    encoder: FlexLSTMEncoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
    device: torch.device,
) -> dict[str, float]:
    """One epoch: episode frame sequence → predict initial (theta, theta_dot)."""
    encoder.train()
    total_loss = 0.0

    for frames, _actions, states in loader:
        frames = frames.to(device)   # (B, T+1, 3, H, W)
        states = states.to(device)   # (B, T+1, 2)

        # Encode the full episode; use mu as the predicted phase state.
        mu, _logvar = encoder(frames)  # (B, 2)

        # Supervise against the initial state of each episode.
        target = states[:, 0]          # (B, 2) — (theta_0, theta_dot_0)
        loss = F.mse_loss(mu, target)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()

    n = len(loader)
    return {"encoder/loss": total_loss / n}


def _train_decoder_epoch(
    decoder: FlexDecoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
    device: torch.device,
) -> dict[str, float]:
    """One epoch: ground-truth (theta, theta_dot) → reconstruct frame."""
    decoder.train()
    total_loss = 0.0

    for states, frames in loader:
        states = states.to(device)  # (B, 2)
        frames = frames.to(device)  # (B, 3, H, W)

        pred = decoder(states)      # (B, 3, H, W)
        loss = F.mse_loss(pred, frames)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()

    n = len(loader)
    return {"decoder/loss": total_loss / n}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def _log_encoder_parity(
    encoder: FlexLSTMEncoder,
    val_trajs: list,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
) -> None:
    """Scatter-plot predicted vs true (theta, theta_dot) for val episodes."""
    encoder.eval()
    pred_all, true_all = [], []

    for frames, _, states in val_trajs:
        ctx = frames.unsqueeze(0).to(device)   # (1, T+1, 3, H, W)
        mu, _ = encoder(ctx)                    # (1, 2)
        pred_all.append(mu.squeeze(0).cpu())
        true_all.append(states[0])              # (theta_0, theta_dot_0)

    pred = torch.stack(pred_all).numpy()
    true = torch.stack(true_all).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    labels = ["theta", "theta_dot"]
    for i, ax in enumerate(axes):
        ax.scatter(true[:, i], pred[:, i], alpha=0.7, s=20)
        lo = min(true[:, i].min(), pred[:, i].min())
        hi = max(true[:, i].max(), pred[:, i].max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
        ax.set_xlabel(f"true {labels[i]}")
        ax.set_ylabel(f"pred {labels[i]}")
        ax.set_title(f"{labels[i]} (epoch {epoch + 1})")
    fig.tight_layout()
    writer.add_figure("encoder/parity", fig, epoch)
    plt.close(fig)


@torch.no_grad()
def _log_decoder_samples(
    decoder: FlexDecoder,
    val_trajs: list,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    fps: int = 10,
) -> None:
    """Log a side-by-side video of ground-truth (left) vs decoded (right) frames."""
    decoder.eval()
    frames, _, states = val_trajs[0]

    gt = frames                                # (T, 3, H, W)
    s = states.to(device)                      # (T, 2)
    pred = decoder(s).cpu()                    # (T, 3, H, W)

    # Concatenate gt (left) and pred (right) along width.
    side_by_side = torch.cat([gt, pred], dim=3).clamp(0, 1)  # (T, 3, H, 2W)

    # add_video expects (N, T, C, H, W).
    video = side_by_side.unsqueeze(0)          # (1, T, 3, H, 2W)
    writer.add_video("decoder/gt_vs_pred", video, epoch, fps=fps)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
# data
@click.option("--n-episodes", type=int, default=200, show_default=True)
@click.option("--img-size", type=int, default=64, show_default=True)
@click.option("--epsilon", type=float, default=0.1, show_default=True)
@click.option("--energy-k", type=float, default=1.0, show_default=True)
@click.option("--max-steps", type=int, default=200, show_default=True)
# model
@click.option("--feat-dim", type=int, default=256, show_default=True,
              help="Per-frame CNN embedding + LSTM hidden size")
@click.option("--pos-ch", type=int, default=8, show_default=True,
              help="Decoder spatial seed channels")
# training
@click.option("--n-enc-epochs", type=int, default=50, show_default=True,
              help="Epochs for encoder phase")
@click.option("--n-dec-epochs", type=int, default=50, show_default=True,
              help="Epochs for decoder phase")
@click.option("--batch-size", type=int, default=16, show_default=True)
@click.option("--lr", type=float, default=1e-4, show_default=True)
@click.option("--grad-clip", type=float, default=1.0, show_default=True)
# logging
@click.option("--log-every", type=int, default=5, show_default=True)
@click.option("--val-every", type=int, default=10, show_default=True)
@click.option("--n-val-episodes", type=int, default=10, show_default=True)
@click.option("--checkpoint-every", type=int, default=10, show_default=True)
def main(**kwargs):
    assert kwargs["img_size"] % 8 == 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    writer = SummaryWriter(comment="_pendulum_supervised")
    run_dir = make_run_dir("pendulum_supervised")

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------
    n_val = kwargs["n_val_episodes"] if kwargs["val_every"] > 0 else 0
    n_total = kwargs["n_episodes"] + n_val
    print(f"\nCollecting {n_total} episodes ({kwargs['n_episodes']} train, {n_val} val)...")
    all_episodes = collect_data(
        n_episodes=n_total,
        img_size=kwargs["img_size"],
        epsilon=kwargs["epsilon"],
        energy_k=kwargs["energy_k"],
        max_steps=kwargs["max_steps"],
    )
    train_eps = all_episodes[: kwargs["n_episodes"]]
    val_trajs = all_episodes[kwargs["n_episodes"] :]

    # Episode-level dataset (for encoder — needs full frame sequence).
    ep_dataset = PendulumDataset(train_eps)
    ep_loader = DataLoader(
        ep_dataset,
        batch_size=kwargs["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    # Frame-level dataset (for decoder — every (state_t, frame_t) pair).
    frame_dataset = FrameStateDataset(train_eps)
    frame_loader = DataLoader(
        frame_dataset,
        batch_size=kwargs["batch_size"] * 8,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    print(f"Episode dataset: {len(ep_dataset)} episodes")
    print(f"Frame dataset:   {len(frame_dataset)} (state, frame) pairs")

    # ------------------------------------------------------------------
    # Models
    #   latent_dim=2  → encoder outputs (theta, theta_dot) directly
    #   q_dim=2       → decoder takes the full 2-D phase state
    # ------------------------------------------------------------------
    LATENT_DIM = 2

    encoder = FlexLSTMEncoder(
        img_ch=3,
        feat_dim=kwargs["feat_dim"],
        latent_dim=LATENT_DIM,
        img_size=kwargs["img_size"],
    ).to(device)

    decoder = FlexDecoder(
        q_dim=LATENT_DIM,           # full (theta, theta_dot) as input
        pos_ch=kwargs["pos_ch"],
        img_ch=3,
        img_size=kwargs["img_size"],
    ).to(device)

    print(f"Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"Decoder params: {sum(p.numel() for p in decoder.parameters()):,}")

    hparams = dict(kwargs)

    # ==================================================================
    # Phase 1: Encoder training
    # ==================================================================
    print("\n--- Phase 1: Encoder training ---")
    enc_optimizer = torch.optim.Adam(encoder.parameters(), lr=kwargs["lr"])
    best_enc_loss = float("inf")

    for epoch in tqdm(range(kwargs["n_enc_epochs"]), desc="Encoder epochs"):
        metrics = _train_encoder_epoch(
            encoder=encoder,
            loader=ep_loader,
            optimizer=enc_optimizer,
            grad_clip=kwargs["grad_clip"],
            device=device,
        )

        if (epoch + 1) % kwargs["log_every"] == 0:
            for k, v in metrics.items():
                writer.add_scalar(k, v, epoch)
            tqdm.write(f"  enc epoch {epoch + 1:3d}  loss={metrics['encoder/loss']:.6f}")

        if (
            val_trajs
            and kwargs["val_every"] > 0
            and (epoch + 1) % kwargs["val_every"] == 0
        ):
            _log_encoder_parity(
                encoder=encoder,
                val_trajs=val_trajs,
                device=device,
                writer=writer,
                epoch=epoch,
            )

        if (
            kwargs["checkpoint_every"] > 0
            and (epoch + 1) % kwargs["checkpoint_every"] == 0
            and metrics["encoder/loss"] < best_enc_loss
        ):
            best_enc_loss = metrics["encoder/loss"]
            save_checkpoint(run_dir, epoch, encoder, hparams, metrics, stem=f"encoder_{epoch}")

    # ==================================================================
    # Phase 2: Decoder training
    # ==================================================================
    print("\n--- Phase 2: Decoder training ---")
    dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=kwargs["lr"])
    best_dec_loss = float("inf")

    # Offset decoder epochs in TensorBoard so they don't overlap with encoder.
    enc_epochs = kwargs["n_enc_epochs"]

    for epoch in tqdm(range(kwargs["n_dec_epochs"]), desc="Decoder epochs"):
        metrics = _train_decoder_epoch(
            decoder=decoder,
            loader=frame_loader,
            optimizer=dec_optimizer,
            grad_clip=kwargs["grad_clip"],
            device=device,
        )

        tb_step = enc_epochs + epoch
        if (epoch + 1) % kwargs["log_every"] == 0:
            for k, v in metrics.items():
                writer.add_scalar(k, v, tb_step)
            tqdm.write(f"  dec epoch {epoch + 1:3d}  loss={metrics['decoder/loss']:.6f}")

        if (
            val_trajs
            and kwargs["val_every"] > 0
            and (epoch + 1) % kwargs["val_every"] == 0
        ):
            _log_decoder_samples(
                decoder=decoder,
                val_trajs=val_trajs,
                device=device,
                writer=writer,
                epoch=tb_step,
            )

        if (
            kwargs["checkpoint_every"] > 0
            and (epoch + 1) % kwargs["checkpoint_every"] == 0
            and metrics["decoder/loss"] < best_dec_loss
        ):
            best_dec_loss = metrics["decoder/loss"]
            save_checkpoint(run_dir, epoch, decoder, hparams, metrics, stem=f"decoder_{epoch}")

    writer.close()
    print("\nDone. Run: tensorboard --logdir runs")
    os._exit(0)


if __name__ == "__main__":
    main()
