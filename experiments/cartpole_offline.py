"""PHGN-LSTM offline EM experiment on CartPole with PID data collection."""

from __future__ import annotations

import random

import click
import numpy as np
from tqdm import tqdm

from data.cartpole import collect_pid_episode
from experiments.base import Experiment
from mppi import MPPI
from phgn_lstm import ControlledDHGN_LSTM
from prioritized_replay_buffer import PrioritizedEpisodeReplayBuffer
from training.em_trainer import OfflineEMConfig, OfflineEMTrainer


class CartPoleOfflineExperiment(Experiment):
	def tb_comment(self) -> str:
		return "_phgn_lstm_offline"

	def build_model(self, cfg: OfflineEMConfig) -> ControlledDHGN_LSTM:
		return ControlledDHGN_LSTM(
			pos_ch=cfg.pos_ch,
			img_ch=3,
			dt=cfg.dt,
			feat_dim=cfg.feat_dim,
			img_size=cfg.img_size,
			control_dim=1,
			obs_state_dim=cfg.obs_state_dim,
			separable=cfg.separable,
		)

	def run(self, cfg: OfflineEMConfig) -> None:
		import torch

		device, writer, run_dir = self._setup()

		model = self.build_model(cfg).to(device)
		print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

		# ── Replay buffers ────────────────────────────────────────────────────
		buffer = PrioritizedEpisodeReplayBuffer(
			capacity=int(
				cfg.n_transitions
				/ max(1, cfg.max_episode_steps)
				* (1 - cfg.test_fraction)
				+ 1
			),
			min_seq_len=1,
			alpha=cfg.per_alpha,
			beta=cfg.per_beta,
			beta_annealing=cfg.per_beta_annealing,
		)
		test_capacity = max(
			1,
			int(
				cfg.n_transitions
				/ max(1, cfg.max_episode_steps)
				* cfg.test_fraction
				+ 1
			),
		)
		test_buffer = PrioritizedEpisodeReplayBuffer(
			capacity=test_capacity,
			min_seq_len=1,
			alpha=0.0,
			beta=1.0,
			beta_annealing=0.0,
		)

		# ── PID data collection ───────────────────────────────────────────────
		print(
			f"\nCollecting ~{cfg.n_transitions:,} transitions with randomised PID..."
		)
		total_collected = 0
		n_episodes = 0
		ep_lens: list[int] = []

		with tqdm(
			total=cfg.n_transitions, desc="Collecting", unit="steps"
		) as pbar:
			while total_collected < cfg.n_transitions:
				ep = collect_pid_episode(
					kp_theta=random.uniform(*cfg.pid_kp_theta_range),
					kd_theta=random.uniform(*cfg.pid_kd_theta_range),
					kp_x=random.uniform(*cfg.pid_kp_x_range),
					kd_x=random.uniform(*cfg.pid_kd_x_range),
					epsilon=cfg.pid_epsilon,
					img_size=cfg.img_size,
					seq_len=cfg.seq_len,
					max_steps=cfg.max_episode_steps,
				)
				if random.random() < cfg.test_fraction:
					test_buffer.push(ep)
				else:
					buffer.push(ep)
				ep_len = len(ep)
				total_collected += ep_len
				n_episodes += 1
				ep_lens.append(ep_len)
				pbar.update(ep_len)

		print(
			f"Collection done: {n_episodes} episodes, {total_collected:,} steps, "
			f"mean len={np.mean(ep_lens):.1f}, "
			f"train={len(buffer)} eps, test={len(test_buffer)} eps"
		)
		writer.add_scalar("collect/total_transitions", total_collected, 0)
		writer.add_scalar("collect/n_episodes", n_episodes, 0)
		writer.add_scalar("collect/mean_ep_len", float(np.mean(ep_lens)), 0)

		# ── MPPI planner (evaluation only) ────────────────────────────────────
		planner = MPPI(
			model=model,
			horizon=cfg.mppi_horizon,
			n_samples=cfg.mppi_samples,
			temperature=cfg.mppi_temperature,
			noise_sigma=cfg.mppi_sigma,
			control_dim=1,
			control_min=-1.0,
			control_max=1.0,
			device=device,
		)

		model_hparams = {
			"img_size": cfg.img_size,
			"pos_ch": cfg.pos_ch,
			"feat_dim": cfg.feat_dim,
			"dt": cfg.dt,
			"seq_len": cfg.seq_len,
			"separable": cfg.separable,
			"obs_state_dim": cfg.obs_state_dim,
		}

		trainer = OfflineEMTrainer(
			cfg=cfg,
			model=model,
			writer=writer,
			run_dir=run_dir,
			device=device,
			buffer=buffer,
			test_buffer=test_buffer,
			planner=planner,
			cost_fn=None,
			model_hparams=model_hparams,
		)
		trainer.fit()
		print("\nDone. Run: tensorboard --logdir runs")


@click.command()
# model
@click.option("--img-size", type=int, default=64, show_default=True)
@click.option("--pos-ch", type=int, default=8, show_default=True)
@click.option("--feat-dim", type=int, default=256, show_default=True)
@click.option("--seq-len", type=int, default=8, show_default=True)
@click.option("--dt", type=float, default=0.05, show_default=True)
@click.option("--no-separable", "separable", default=True, flag_value=False)
@click.option("--obs-state-dim", type=int, default=4, show_default=True)
# data collection
@click.option("--n-transitions", type=int, default=100_000, show_default=True)
@click.option("--max-episode-steps", type=int, default=500, show_default=True)
@click.option(
	"--pid-kp-theta-range",
	nargs=2,
	type=float,
	default=(5.0, 25.0),
	show_default=True,
)
@click.option(
	"--pid-kd-theta-range",
	nargs=2,
	type=float,
	default=(0.5, 5.0),
	show_default=True,
)
@click.option(
	"--pid-kp-x-range",
	nargs=2,
	type=float,
	default=(-2.0, 2.0),
	show_default=True,
)
@click.option(
	"--pid-kd-x-range",
	nargs=2,
	type=float,
	default=(-1.0, 1.0),
	show_default=True,
)
@click.option("--pid-epsilon", type=float, default=0.05, show_default=True)
@click.option("--test-fraction", type=float, default=0.2, show_default=True)
# training
@click.option("--n-pretrain-steps", type=int, default=500, show_default=True)
@click.option("--n-iterations", type=int, default=500, show_default=True)
@click.option("--em-e-steps", type=int, default=25, show_default=True)
@click.option("--em-m-steps", type=int, default=25, show_default=True)
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option("--lr", type=float, default=1e-4, show_default=True)
@click.option("--lr-dynamics", type=float, default=1e-4, show_default=True)
@click.option("--kl-weight", type=float, default=1e-3, show_default=True)
@click.option("--recon-weight", type=float, default=1.0, show_default=True)
@click.option("--state-weight", type=float, default=0.5, show_default=True)
@click.option("--fwd-weight", type=float, default=0.5, show_default=True)
@click.option("--anchor-context", type=int, default=3, show_default=True)
@click.option("--free-bits", type=float, default=0.5, show_default=True)
@click.option("--grad-clip", type=float, default=1.0, show_default=True)
# PER
@click.option("--per-alpha", type=float, default=0.6, show_default=True)
@click.option("--per-beta", type=float, default=0.4, show_default=True)
@click.option(
	"--per-beta-annealing", type=float, default=0.0, show_default=True
)
# MPPI (eval only)
@click.option("--mppi-horizon", type=int, default=20, show_default=True)
@click.option("--mppi-samples", type=int, default=256, show_default=True)
@click.option("--mppi-temperature", type=float, default=0.05, show_default=True)
@click.option("--mppi-sigma", type=float, default=0.5, show_default=True)
# eval / logging
@click.option("--eval-every", type=int, default=10, show_default=True)
@click.option("--n-eval-episodes", type=int, default=10, show_default=True)
@click.option("--log-every", type=int, default=1, show_default=True)
@click.option("--checkpoint-every", type=int, default=50, show_default=True)
def main(**kwargs):
	assert kwargs["seq_len"] >= 2
	assert kwargs["img_size"] % 8 == 0
	CartPoleOfflineExperiment().run(OfflineEMConfig.from_kwargs(kwargs))


if __name__ == "__main__":
	main()
