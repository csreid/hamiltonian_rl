"""HGN-LSTM experiment on synthetic SHO image sequences."""

from __future__ import annotations

from pathlib import Path

import click
import torch
import torchvision.transforms.functional as tvf
from torch.utils.data import TensorDataset, DataLoader
from data.base import DataLoaderAdapter
from torch.utils.tensorboard import SummaryWriter

from cli_common import shared_options
from data.sho import generate_dataset
from diag_common import (
	image_centroid,
	log_gt_pred_video,
	log_hamiltonian_conservation,
	log_marker_video,
)
from experiments.base import Experiment, ExperimentConfig
from hgn_lstm import HGN_LSTM
from training.trainer import StandardTrainer, TrainerConfig


def _log_correlation_plots(writer, pred_frames, q_gt, epoch):
	"""Compare x-centroid of decoded frames to ground-truth q (position)."""
	import matplotlib.pyplot as plt
	import numpy as np

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


def _validate_hook(trainer: StandardTrainer, epoch: int) -> None:
	"""Log correlation plots, videos, and Hamiltonian conservation."""
	cfg = trainer.cfg
	vis_pred = trainer._last_vis_pred
	vis_pred_coords = trainer._last_vis_pred_coords
	q_gt_val = trainer._last_q_gt_val
	qs_val = trainer._last_qs_val
	ps_val = trainer._last_ps_val

	corr = _log_correlation_plots(trainer.writer, vis_pred, q_gt_val, epoch)

	N_VID = vis_pred.shape[0]
	context_len = cfg.context_len
	img_size = trainer.val_frames.shape[-1]

	gt_vid = trainer.val_frames[:N_VID, :context_len].flip(dims=[1]).cpu()
	log_gt_pred_video(trainer.writer, "val/gt_vs_pred", gt_vid, vis_pred, epoch)

	gt_pos = image_centroid(gt_vid)
	T_v = gt_vid.shape[1]
	gt_vid_up = tvf.resize(
		gt_vid.reshape(-1, 3, img_size, img_size), (64, 64)
	).reshape(N_VID, T_v, 3, 64, 64)
	log_marker_video(
		trainer.writer,
		"val/coord_markers",
		gt_vid_up,
		gt_pos,
		vis_pred_coords,
		epoch,
		size=4,
	)

	log_hamiltonian_conservation(
		trainer.writer,
		trainer.model.H,
		qs_val,
		ps_val,
		N_VID,
		epoch,
		context_len=context_len,
	)

	from tqdm import tqdm

	val_mse = (
		trainer._last_val_mse if hasattr(trainer, "_last_val_mse") else 0.0
	)
	tqdm.write(f"  corr_q={abs(corr):.3f}")


class HGNLSTMExperiment(Experiment):
	def tb_comment(self) -> str:
		return "_hgn_lstm_sho_images"

	def build_model(self, cfg: ExperimentConfig) -> HGN_LSTM:
		return HGN_LSTM(
			pos_ch=cfg.model_kwargs["pos_ch"],
			img_ch=3,
			dt=cfg.dt,
			feat_dim=cfg.model_kwargs["feat_dim"],
		)

	def build_datasets(self, cfg: ExperimentConfig):
		kw = dict(
			dt=cfg.dt,
			img_size=cfg.img_size,
			blob_sigma=cfg.blob_sigma,
			max_amplitude=cfg.max_amplitude,
			spring_constant=cfg.spring_constant,
			mass=cfg.mass,
			margin=4,
		)
		p_train, q_train, frames_train = generate_dataset(
			cfg.n_train, cfg.seq_len, **kw
		)
		p_val, q_val, frames_val = generate_dataset(
			cfg.n_val, cfg.seq_len, **kw
		)
		print(f"  train frames: {frames_train.shape}")
		print(f"  val   frames: {frames_val.shape}\n")

		loader = DataLoader(
			TensorDataset(frames_train, p_train, q_train),
			batch_size=cfg.batch_size,
			shuffle=True,
		)
		return DataLoaderAdapter(loader), frames_val, q_val

	def build_trainer(
		self,
		cfg,
		trainer_cfg,
		model,
		dataset,
		writer,
		run_dir,
		device,
		hparams,
		val_data,
	):
		frames_val, q_val = val_data
		frames_val = frames_val.to(device)
		q_val = q_val.to(device)
		return StandardTrainer(
			cfg=trainer_cfg,
			model=model,
			train_loader=dataset,
			writer=writer,
			run_dir=run_dir,
			device=device,
			hparams=hparams,
			val_frames=frames_val,
			val_q=q_val,
			rollout_direction="backward",
			validate_hook=_validate_hook,
		)


@click.command()
@shared_options
@click.option("--pos-ch", type=int, default=16, show_default=True)
@click.option("--feat-dim", type=int, default=256, show_default=True)
@click.option("--coord-weight", type=float, default=0.0, show_default=True)
@click.option("--energy-weight", type=float, default=0.0, show_default=True)
def main(**kwargs):
	context_len = kwargs["n_frames"]
	assert kwargs["img_size"] == 32, (
		"HGN decoder outputs 32x32; set --img-size 32"
	)
	assert kwargs["seq_len"] >= context_len
	assert kwargs["train_rollout"] < context_len

	cfg = ExperimentConfig.from_kwargs(
		kwargs,
		model_kwargs={
			"pos_ch": kwargs["pos_ch"],
			"feat_dim": kwargs["feat_dim"],
		},
	)
	HGNLSTMExperiment().run(cfg)


if __name__ == "__main__":
	main()
