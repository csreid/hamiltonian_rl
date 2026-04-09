"""HGN (original CNN encoder) experiment on synthetic SHO image sequences."""

from __future__ import annotations

import click
import torch
from torch.utils.data import TensorDataset, DataLoader
from data.base import DataLoaderAdapter

from cli_common import shared_options
from data.sho import generate_dataset
from experiments.base import Experiment, ExperimentConfig
from experiments.sho_hgn_lstm import _validate_hook
from hgn_org import HGN
from training.trainer import StandardTrainer, TrainerConfig


class HGNOrgExperiment(Experiment):
	def tb_comment(self) -> str:
		return "_hgn_org_sho_images"

	def build_model(self, cfg: ExperimentConfig) -> HGN:
		return HGN(
			n_frames=cfg.n_frames,
			pos_ch=cfg.model_kwargs["pos_ch"],
			img_ch=3,
			dt=cfg.dt,
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
			rollout_direction="forward",
			validate_hook=_validate_hook,
		)


@click.command()
@shared_options
@click.option("--pos-ch", type=int, default=16, show_default=True)
@click.option("--coord-weight", type=float, default=0.0, show_default=True)
@click.option("--energy-weight", type=float, default=0.0, show_default=True)
def main(**kwargs):
	assert kwargs["img_size"] == 32, (
		"HGN decoder outputs 32x32; set --img-size 32"
	)

	cfg = ExperimentConfig.from_kwargs(
		kwargs,
		model_kwargs={"pos_ch": kwargs["pos_ch"]},
	)
	HGNOrgExperiment().run(cfg)


if __name__ == "__main__":
	main()
