"""Offline CartPole world-model training for ControlledDHGN_LSTM.

Collects episodes with a randomised PD controller, then trains the model
as a one-step world model:

    encode context frames → (q0, p0)  via LSTM encoder
    apply one RK4 step    → (q1, p1)  with the recorded action
    decode                → pred_frame
    loss = MSE(pred_frame, next_frame) + kl_weight * KL

No planning, no EM alternation, no prioritised replay.
"""

from __future__ import annotations

import click
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from checkpoint_common import make_run_dir, save_checkpoint
from data.cartpole import CartPoleDataset, collect_data
from phgn_lstm import ControlledDHGN_LSTM


def _train_epoch(
	model: ControlledDHGN_LSTM,
	loader: DataLoader,
	optimizer: torch.optim.Optimizer,
	kl_weight: float,
	free_bits: float,
	grad_clip: float,
	device: torch.device,
) -> dict[str, float]:
	model.train()
	total_recon = total_kl = total_loss = 0.0

	for ctx, action, next_frame, _ in loader:
		ctx = ctx.to(device)  # (B, n_frames, C, H, W)
		action = action.to(device)  # (B, 1)
		next_frame = next_frame.to(device)  # (B, C, H, W)

		q0, p0, kl_raw, _, _ = model(ctx)

		q1, p1 = model.controlled_step(q0, p0, action)
		pred = model.decoder(q1)

		recon = F.mse_loss(pred, next_frame)
		kl = kl_raw.clamp(min=free_bits).mean()
		loss = recon + kl_weight * kl

		optimizer.zero_grad()
		loss.backward()
		if grad_clip > 0:
			torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
		optimizer.step()

		total_recon += recon.item()
		total_kl += kl.item()
		total_loss += loss.item()

	n = len(loader)
	return {
		"train/loss": total_loss / n,
		"train/recon": total_recon / n,
		"train/kl": total_kl / n,
	}


@click.command()
# data
@click.option("--n-episodes", type=int, default=200, show_default=True)
@click.option(
	"--n-frames",
	type=int,
	default=8,
	show_default=True,
	help="Context frames for LSTM encoder",
)
@click.option("--img-size", type=int, default=64, show_default=True)
@click.option("--scripted-fraction", type=float, default=0.5, show_default=True)
# model
@click.option("--pos-ch", type=int, default=8, show_default=True)
@click.option("--feat-dim", type=int, default=256, show_default=True)
@click.option("--dt", type=float, default=0.05, show_default=True)
@click.option("--no-separable", "separable", default=True, flag_value=False)
# training
@click.option("--n-epochs", type=int, default=50, show_default=True)
@click.option("--batch-size", type=int, default=64, show_default=True)
@click.option("--lr", type=float, default=1e-4, show_default=True)
@click.option("--kl-weight", type=float, default=1e-3, show_default=True)
@click.option("--free-bits", type=float, default=0.5, show_default=True)
@click.option("--grad-clip", type=float, default=1.0, show_default=True)
# logging
@click.option("--log-every", type=int, default=5, show_default=True)
@click.option("--checkpoint-every", type=int, default=10, show_default=True)
def main(**kwargs):
	assert kwargs["img_size"] % 8 == 0

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}")

	writer = SummaryWriter(comment="_phgn_lstm_offline")
	run_dir = make_run_dir("cartpole_offline")

	print(f"\nCollecting data from {kwargs['n_episodes']} episodes...")
	transitions = collect_data(
		n_episodes=kwargs["n_episodes"],
		n_frames=kwargs["n_frames"],
		img_size=kwargs["img_size"],
		scripted_fraction=kwargs["scripted_fraction"],
	)
	dataset = CartPoleDataset(transitions)
	loader = DataLoader(
		dataset,
		batch_size=kwargs["batch_size"],
		shuffle=True,
		num_workers=2,
		pin_memory=True,
	)
	print(f"Dataset: {len(dataset):,} transitions")

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

	optimizer = torch.optim.Adam(model.parameters(), lr=kwargs["lr"])
	best_loss = float("inf")

	hparams = {k: v for k, v in kwargs.items()}

	for epoch in tqdm(range(kwargs["n_epochs"]), desc="Epochs"):
		metrics = _train_epoch(
			model=model,
			loader=loader,
			optimizer=optimizer,
			kl_weight=kwargs["kl_weight"],
			free_bits=kwargs["free_bits"],
			grad_clip=kwargs["grad_clip"],
			device=device,
		)

		if (epoch + 1) % kwargs["log_every"] == 0:
			for k, v in metrics.items():
				writer.add_scalar(k, v, epoch)
			tqdm.write(
				f"  epoch {epoch + 1:3d}"
				f"  loss={metrics['train/loss']:.4f}"
				f"  recon={metrics['train/recon']:.4f}"
				f"  kl={metrics['train/kl']:.4f}"
			)

		if (
			kwargs["checkpoint_every"] > 0
			and (epoch + 1) % kwargs["checkpoint_every"] == 0
		):
			if metrics["train/loss"] < best_loss:
				best_loss = metrics["train/loss"]
				save_checkpoint(run_dir, epoch, model, hparams, metrics)

	writer.close()
	print("\nDone. Run: tensorboard --logdir runs")


if __name__ == "__main__":
	main()
