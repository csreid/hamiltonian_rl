"""
CartPole pixel benchmark: ControlledDHGN_LSTM + MPPI vs PPO.

Same pipeline as benchmark_cartpole.py but with the LSTM-based encoder:

  - Default resolution is 128×128; NatureCNN works natively (no custom CNN).
  - The LSTM encoder processes n_frames sequentially → current latent state.
  - One controlled_step forward → predicted next frame.

Everything else (data collection, MPPI, PPO baseline) is identical.
"""

from __future__ import annotations

import click
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from benchmark_cartpole import (
	CartPoleDataset,
	CartPolePixelEnv,
	SampleEfficiencyCallback,
	collect_data,
	collect_val_trajectories,
	evaluate_mppi,
	make_cartpole_cost,
)
from diag_common import log_gt_pred_video
from mppi import MPPI
from phgn_lstm import ControlledDHGN_LSTM


# ── World model training ──────────────────────────────────────────────────────


def train_world_model(
	model: ControlledDHGN_LSTM,
	dataset: CartPoleDataset,
	writer: SummaryWriter,
	n_epochs: int = 30,
	batch_size: int = 32,
	lr: float = 1e-4,
	kl_weight: float = 1e-3,
	recon_weight: float = 1.0,
	state_weight: float = 0.1,
	free_bits: float = 0.5,
	grad_clip: float = 1.0,
	device: torch.device = torch.device("cpu"),
) -> None:
	loader = DataLoader(
		dataset, batch_size=batch_size, shuffle=True, drop_last=True
	)
	optimiser = torch.optim.Adam(model.parameters(), lr=lr)
	global_step = 0

	for epoch in (bar := tqdm(range(n_epochs), desc="World model")):
		ep_recon = ep_kl = ep_state = ep_total = 0.0

		for ctx, acts, nxt, states in loader:
			ctx = ctx.to(device)  # (B, T, 3, H, W)
			acts = acts.to(device)  # (B, 1)
			nxt = nxt.to(device)  # (B, 3, H, W)
			states = states.to(device)  # (B, 4)

			q0, p0, kl, mu, log_var = model(ctx)
			kl_loss = kl.clamp(min=free_bits).mean()

			q1, p1 = model.controlled_step(q0, p0, acts)

			pred_frame = model.decoder(q1)
			recon_loss = F.mse_loss(pred_frame, nxt)

			if model.state_decoder is not None:
				pred_state = model.decode_state(q1, p1)
				state_loss = F.mse_loss(pred_state, states)
			else:
				state_loss = torch.tensor(0.0, device=device)

			loss = (
				recon_weight * recon_loss
				+ kl_weight * kl_loss
				+ state_weight * state_loss
			)

			optimiser.zero_grad()
			loss.backward()
			if grad_clip > 0:
				torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
			optimiser.step()

			writer.add_scalar("wm/recon_loss", recon_loss.item(), global_step)
			writer.add_scalar("wm/kl_loss", kl_loss.item(), global_step)
			writer.add_scalar("wm/state_loss", state_loss.item(), global_step)
			writer.add_scalar("wm/total_loss", loss.item(), global_step)
			global_step += 1

			ep_recon += recon_loss.item()
			ep_kl += kl_loss.item()
			ep_state += state_loss.item()
			ep_total += loss.item()

		n = len(loader)
		bar.set_postfix(
			loss=f"{ep_total / n:.4f}",
			recon=f"{ep_recon / n:.4f}",
			state=f"{ep_state / n:.4f}",
		)


# ── Rollout visualisation ─────────────────────────────────────────────────────


def log_rollout_reconstructions(
	model: ControlledDHGN_LSTM,
	trajectories,
	writer: SummaryWriter,
	epoch: int,
	n_frames: int,
	device: torch.device,
	n_trajs: int = 4,
	rollout_steps: int = 20,
) -> None:
	model.eval()
	n_trajs = min(n_trajs, len(trajectories))
	trajs = sorted(trajectories, key=lambda t: len(t[1]), reverse=True)[
		:n_trajs
	]
	max_steps = min(rollout_steps, min(len(a) for _, a in trajs))

	gt_vids, pred_vids = [], []

	with torch.no_grad():
		for frames, actions in trajs:
			ctx = frames[:n_frames].unsqueeze(0).to(device)  # (1, T, 3, H, W)
			q, p = model.encode_mean(ctx)

			gt_seq, pred_seq = [], []
			for t in range(max_steps):
				u = actions[t].reshape(1, 1).to(device)
				q, p = model.controlled_step(q, p, u)
				pred_seq.append(model.decoder(q).clamp(0, 1).squeeze(0).cpu())
				gt_seq.append(frames[n_frames + t])

			gt_vids.append(torch.stack(gt_seq))
			pred_vids.append(torch.stack(pred_seq))

	log_gt_pred_video(
		writer,
		"val/gt_vs_pred_rollout",
		torch.stack(gt_vids),
		torch.stack(pred_vids),
		epoch,
	)


# ── PPO baseline (no custom CNN — NatureCNN works at ≥84px) ──────────────────


class PPOVideoCallback(BaseCallback):
	"""Render a few deterministic policy episodes and log them as a TB video.

	Uses a separate raw gym env (not VecFrameStack) for rendering, and
	queries self.model (the PPO agent) via the VecFrameStack eval_env for
	actions so we see what the policy has actually learned.
	"""

	def __init__(
		self,
		img_size: int,
		n_frames: int,
		writer: SummaryWriter,
		log_freq: int = 20_000,
		n_episodes: int = 3,
		fps: int = 15,
	):
		super().__init__(verbose=0)
		self.img_size = img_size
		self.n_frames = n_frames
		self.writer = writer
		self.log_freq = log_freq
		self.n_episodes = n_episodes
		self.fps = fps

	def _on_step(self) -> bool:
		if self.n_calls % self.log_freq != 0:
			return True

		# One VecFrameStack env for policy queries, one raw env for rendering.
		env_fn = lambda: CartPolePixelEnv(img_size=self.img_size)
		vec_env = VecFrameStack(DummyVecEnv([env_fn]), n_stack=self.n_frames)
		render_env = gym.make("CartPole-v1", render_mode="rgb_array")

		episodes = []
		for _ in range(self.n_episodes):
			frames = []
			obs = vec_env.reset()
			render_env.reset()
			done = False
			while not done:
				frame = render_env.render()
				frames.append(
					torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
				)
				action, _ = self.model.predict(obs, deterministic=True)
				obs, _, dones, _ = vec_env.step(action)
				# Mirror the same action in the render env.
				_, _, term, trunc, _ = render_env.step(int(action[0]))
				done = dones[0] or term or trunc

			frames_t = torch.stack(frames)  # (T, 3, H_orig, W_orig)
			frames_t = F.interpolate(
				frames_t,
				size=(self.img_size, self.img_size),
				mode="bilinear",
				align_corners=False,
			)
			episodes.append(frames_t)

		vec_env.close()
		render_env.close()

		max_t = max(f.shape[0] for f in episodes)
		padded = [
			torch.cat([f, f[-1:].expand(max_t - f.shape[0], -1, -1, -1)], dim=0)
			for f in episodes
		]
		video = torch.stack(padded)  # (N, T, 3, H, W)
		self.writer.add_video(
			"ppo/episode",
			(video.clamp(0, 1) * 255).to(torch.uint8),
			self.num_timesteps,
			fps=self.fps,
		)
		return True


def train_and_eval_ppo(
	writer,
	img_size: int,
	n_train_steps: int,
	n_eval_episodes: int,
	n_frames: int = 4,
	ppo_eval_freq: int = 5_000,
	device: str = "auto",
) -> float:
	env_fn = lambda: CartPolePixelEnv(img_size=img_size)
	vec_env = VecFrameStack(DummyVecEnv([env_fn]), n_stack=n_frames)
	eval_env = VecFrameStack(DummyVecEnv([env_fn]), n_stack=n_frames)

	ppo = PPO(
		"CnnPolicy",
		vec_env,
		n_steps=512,
		batch_size=64,
		n_epochs=4,
		learning_rate=3e-4,
		ent_coef=0.01,
		verbose=0,
		device=device,
	)

	callback = CallbackList(
		[
			SampleEfficiencyCallback(
				eval_env=eval_env,
				writer=writer,
				eval_freq=ppo_eval_freq,
				n_eval_episodes=n_eval_episodes,
			),
			PPOVideoCallback(
				img_size=img_size,
				n_frames=n_frames,
				writer=writer,
				log_freq=ppo_eval_freq,
			),
		]
	)

	print(
		f"\nTraining PPO for {n_train_steps:,} env steps "
		f"(eval every {ppo_eval_freq:,} steps)..."
	)
	ppo.learn(total_timesteps=n_train_steps, callback=callback)

	rewards = []
	for _ in tqdm(range(n_eval_episodes), desc="PPO final eval"):
		obs = eval_env.reset()
		total_r = 0.0
		while True:
			action, _ = ppo.predict(obs, deterministic=True)
			obs, r, done, _ = eval_env.step(action)
			total_r += float(r[0])
			if done[0]:
				break
		rewards.append(total_r)

	eval_env.close()
	mean_r = float(np.mean(rewards))
	writer.add_scalar("comparison/mean_reward", mean_r, n_train_steps)
	return mean_r


# ── CLI ──────────────────────────────────────────────────────────────────────


@click.command()
@click.option(
	"--img-size",
	type=int,
	default=128,
	show_default=True,
	help="Frame resolution (H=W). Must be 4·2^k, e.g. 32, 64, 128.",
)
@click.option(
	"--n-frames",
	type=int,
	default=4,
	show_default=True,
	help="Context frames fed to the LSTM encoder.",
)
@click.option(
	"--pos-ch",
	type=int,
	default=16,
	show_default=True,
	help="Position channels in the latent state.",
)
@click.option(
	"--feat-dim",
	type=int,
	default=256,
	show_default=True,
	help="Per-frame CNN embedding size.",
)
@click.option("--dt", type=float, default=0.05, show_default=True)
# ── data collection ──
@click.option("--n-collect", type=int, default=200, show_default=True)
@click.option("--scripted-fraction", type=float, default=0.5, show_default=True)
# ── world model ──
@click.option("--wm-epochs", type=int, default=30, show_default=True)
@click.option("--wm-batch", type=int, default=32, show_default=True)
@click.option("--wm-lr", type=float, default=1e-4, show_default=True)
@click.option("--kl-weight", type=float, default=1e-3, show_default=True)
@click.option("--recon-weight", type=float, default=1.0, show_default=True)
@click.option("--state-weight", type=float, default=0.5, show_default=True)
@click.option("--free-bits", type=float, default=0.5, show_default=True)
@click.option("--grad-clip", type=float, default=1.0, show_default=True)
# ── MPPI ──
@click.option("--mppi-horizon", type=int, default=20, show_default=True)
@click.option("--mppi-samples", type=int, default=256, show_default=True)
@click.option("--mppi-temperature", type=float, default=0.05, show_default=True)
@click.option("--mppi-sigma", type=float, default=0.5, show_default=True)
@click.option("--n-eval-mppi", type=int, default=20, show_default=True)
# ── PPO ──
@click.option("--n-ppo-steps", type=int, default=200_000, show_default=True)
@click.option("--n-eval-ppo", type=int, default=20, show_default=True)
@click.option("--ppo-eval-freq", type=int, default=5_000, show_default=True)
@click.option("--skip-ppo", is_flag=True, default=False)
def main(
	img_size,
	n_frames,
	pos_ch,
	feat_dim,
	dt,
	n_collect,
	scripted_fraction,
	wm_epochs,
	wm_batch,
	wm_lr,
	kl_weight,
	recon_weight,
	state_weight,
	free_bits,
	grad_clip,
	mppi_horizon,
	mppi_samples,
	mppi_temperature,
	mppi_sigma,
	n_eval_mppi,
	n_ppo_steps,
	n_eval_ppo,
	ppo_eval_freq,
	skip_ppo,
):
	import math

	assert img_size >= 8 and img_size == 4 * (
		2 ** int(math.log2(img_size // 4))
	), f"img_size must be 4·2^k (e.g. 32, 64, 128), got {img_size}"

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}\n")

	writer = SummaryWriter(comment="_phgn_lstm_cartpole")

	hparam_text = (
		"| Hyperparameter | Value |\n|---|---|\n"
		f"| img_size | {img_size} |\n"
		f"| n_frames | {n_frames} |\n"
		f"| pos_ch | {pos_ch} |\n"
		f"| feat_dim | {feat_dim} |\n"
		f"| dt | {dt} |\n"
		f"| n_collect | {n_collect} |\n"
		f"| scripted_fraction | {scripted_fraction} |\n"
		f"| wm_epochs | {wm_epochs} |\n"
		f"| wm_lr | {wm_lr} |\n"
		f"| kl_weight | {kl_weight} |\n"
		f"| recon_weight | {recon_weight} |\n"
		f"| state_weight | {state_weight} |\n"
		f"| mppi_horizon | {mppi_horizon} |\n"
		f"| mppi_samples | {mppi_samples} |\n"
		f"| mppi_temperature | {mppi_temperature} |\n"
		f"| mppi_sigma | {mppi_sigma} |\n"
		f"| n_ppo_steps | {n_ppo_steps} |\n"
		f"| ppo_eval_freq | {ppo_eval_freq} |\n"
	)
	writer.add_text("hparams", hparam_text, 0)

	# ── Phase 1: Data collection ──────────────────────────────────────────
	print("=== Phase 1: Data collection ===")
	transitions = collect_data(n_collect, n_frames, img_size, scripted_fraction)
	dataset = CartPoleDataset(transitions)

	print("\nCollecting validation trajectories...")
	val_trajectories = collect_val_trajectories(
		n_episodes=8, n_frames=n_frames, img_size=img_size
	)

	# ── Phase 2: World model training ────────────────────────────────────
	print("\n=== Phase 2: World model training ===")
	model = ControlledDHGN_LSTM(
		pos_ch=pos_ch,
		img_ch=3,
		dt=dt,
		feat_dim=feat_dim,
		img_size=img_size,
		control_dim=1,
		obs_state_dim=4,
	).to(device)
	print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
	print(f"State dim D = {model.state_dim}  (img_size={img_size})\n")

	train_world_model(
		model,
		dataset,
		writer,
		n_epochs=wm_epochs,
		batch_size=wm_batch,
		lr=wm_lr,
		kl_weight=kl_weight,
		recon_weight=recon_weight,
		state_weight=state_weight,
		free_bits=free_bits,
		grad_clip=grad_clip,
		device=device,
	)

	# ── Phase 2b: Rollout visualisation ──────────────────────────────────
	print("\n=== Phase 2b: Logging rollout reconstructions ===")
	log_rollout_reconstructions(
		model,
		val_trajectories,
		writer,
		epoch=wm_epochs - 1,
		n_frames=n_frames,
		device=device,
	)

	# ── Phase 3: MPPI evaluation ──────────────────────────────────────────
	print("\n=== Phase 3: MPPI evaluation ===")
	cost_fn = make_cartpole_cost(model, device)
	planner = MPPI(
		model=model,
		horizon=mppi_horizon,
		n_samples=mppi_samples,
		temperature=mppi_temperature,
		noise_sigma=mppi_sigma,
		control_dim=1,
		control_min=-1.0,
		control_max=1.0,
		device=device,
	)
	mppi_mean = evaluate_mppi(
		model,
		planner,
		cost_fn,
		writer,
		n_eval_mppi,
		n_frames,
		img_size,
		device,
		env_step_budget=len(transitions),
	)

	# ── Phase 4: PPO baseline ─────────────────────────────────────────────
	if not skip_ppo:
		print("\n=== Phase 4: PPO baseline ===")
		ppo_mean = train_and_eval_ppo(
			writer,
			img_size=img_size,
			n_train_steps=n_ppo_steps,
			n_eval_episodes=n_eval_ppo,
			n_frames=n_frames,
			ppo_eval_freq=ppo_eval_freq,
			device="cuda" if torch.cuda.is_available() else "cpu",
		)

		summary = (
			f"| Method | Mean reward |\n|---|---|\n"
			f"| PHGN-LSTM + MPPI | {mppi_mean:.1f} |\n"
			f"| PPO (SB3) | {ppo_mean:.1f} |\n"
		)
		writer.add_text("eval/comparison", summary)
		print(f"\nPHGN-LSTM + MPPI  →  {mppi_mean:.1f}")
		print(f"PPO (SB3)         →  {ppo_mean:.1f}")

	writer.close()
	print("\nDone. Run: tensorboard --logdir runs")


if __name__ == "__main__":
	main()
