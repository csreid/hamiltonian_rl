"""Online PHGN-LSTM benchmark: MPPI data collection + replay buffer training.

Alternates between:
  1. Collecting CartPole episodes by running MPPI under the current world model
  2. Training the world model on contiguous subsequences sampled from the buffer

The replay buffer stores full episodes.  Training samples windows of
``seq_len`` frames (+ ``seq_len-1`` actions) and applies three losses:

  recon_loss  — forward rollout from the encoded state vs. subsequent frames
  state_loss  — decode_state at each forward step vs. true CartPole state
  kl_loss     — VAE regularisation

Training signal: LSTM encodes ``anchor_context`` frames to get (q_0, p_0),
then integrates the port-Hamiltonian forward for seq_len-anchor_context steps,
supervising decoded frames and states against ground truth at each step.

Warmup: the first ``n_warmup`` episodes are collected with random actions so
the buffer is populated before MPPI starts planning.  Once the buffer holds
at least ``min_buffer`` episodes, MPPI takes over collection.
"""

from __future__ import annotations

import random

import click
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from benchmark_cartpole import (
	FrameBuffer,
	preprocess_frame,
)
from checkpoint_common import make_run_dir, save_checkpoint
from mppi import MPPI
from phgn_lstm import ControlledDHGN_LSTM
from prioritized_replay_buffer import PrioritizedEpisodeReplayBuffer
from replay_buffer import Episode

_CART_X_LIMIT = 2.4
_POLE_THETA_LIMIT = 0.2094  # 12 degrees


# ── Cost function ─────────────────────────────────────────────────────────────


def make_cartpole_cost(model: ControlledDHGN_LSTM, device: torch.device):
	def cost_fn(qs, ps):
		K = qs[0].shape[0]
		costs = torch.zeros(K, device=device)
		for q, p in zip(qs[1:], ps[1:]):
			state = model.decode_state(q, p)  # (K, 4)
			x, theta = state[:, 0], state[:, 2]
			costs += theta.pow(2) + 0.1 * x.pow(2)
			failed = (x.abs() > _CART_X_LIMIT) | (
				theta.abs() > _POLE_THETA_LIMIT
			)
			costs += 100.0 * failed.float()
		return costs

	return cost_fn


# ── Episode collection ────────────────────────────────────────────────────────


def collect_episode(
	model: ControlledDHGN_LSTM,
	planner: MPPI | None,
	seq_len: int,
	img_size: int,
	max_steps: int,
	device: torch.device,
	cost_fn=None,
) -> Episode:
	"""Run one CartPole episode and return it as an Episode.

	If ``planner`` is None (or the model is in warm-up), actions are sampled
	uniformly from {-1.0, +1.0}.  Otherwise MPPI plans the action.
	"""
	env = gym.make("CartPole-v1", render_mode="rgb_array")
	buf = FrameBuffer(seq_len, img_size)

	_, _ = env.reset()
	first_frame = env.render()
	buf.reset(first_frame)

	frames = [preprocess_frame(first_frame, img_size)]  # (3, H, W) each
	actions: list[float] = []
	states: list[np.ndarray] = []

	if planner is not None:
		planner.reset()

	model.eval()
	with torch.no_grad():
		for _ in range(max_steps):
			true_state = np.array(env.unwrapped.state, dtype=np.float32)
			states.append(true_state)

			if planner is not None and cost_fn is not None:
				ctx = buf.get().unsqueeze(0).to(device)  # (1, T, 3, H, W)
				q0, p0 = model.encode_mean(ctx)
				action_t = planner.plan(q0, p0, cost_fn)
				action_float = float(action_t.item())
			else:
				action_float = float(random.choice([-1.0, 1.0]))

			gym_action = 1 if action_float > 0 else 0
			_, _, terminated, truncated, _ = env.step(gym_action)

			frame = env.render()
			buf.push(frame)
			frames.append(preprocess_frame(frame, img_size))
			actions.append(action_float)

			if terminated or truncated:
				break

	env.close()

	# One true_state per frame except the last (no state after terminal).
	# Pad with the last state so lengths match frames.
	states.append(states[-1])

	return Episode(
		frames=torch.stack(frames),  # (T+1, 3, H, W)
		actions=torch.tensor(actions, dtype=torch.float32).unsqueeze(
			-1
		),  # (T, 1)
		states=torch.from_numpy(np.stack(states)),  # (T+1, 4)
	)


# ── Training step ─────────────────────────────────────────────────────────────


def train_step(
	model: ControlledDHGN_LSTM,
	buffer: PrioritizedEpisodeReplayBuffer,
	optimizer: torch.optim.Optimizer,
	seq_len: int,
	dt: float,
	batch_size: int,
	kl_weight: float,
	recon_weight: float,
	state_weight: float,
	anchor_context: int,
	free_bits: float,
	grad_clip: float,
	device: torch.device,
) -> tuple[dict[str, float], list[int], np.ndarray]:
	"""One gradient step with PER IS-weighted loss.

	Encodes anchor_context frames → (q_0, p_0), then rolls the Hamiltonian
	forward for seq_len-anchor_context steps, supervising decoded frames and
	states against the ground-truth sequence.

	Returns:
	    losses:           dict of scalar loss components
	    sampled_indices:  episode buffer indices for priority update
	    per_sample_loss:  (B,) numpy array of per-sample total losses
	"""
	frames, actions, states, sampled_indices, is_weights, lengths = buffer.sample_sequences(
		batch_size, seq_len=seq_len
	)
	frames = frames.to(device)       # (B, seq_len+1, 3, H, W)
	actions = actions.to(device)     # (B, seq_len, 1)
	is_weights = is_weights.to(device)  # (B,)
	lengths = lengths.to(device)     # (B,) actual episode lengths, <= seq_len
	if states is not None:
		states = states.to(device)   # (B, seq_len+1, 4)

	B = frames.shape[0]
	model.train()

	do_state = (
		state_weight > 0
		and states is not None
		and model.state_decoder is not None
	)

	# ── Encode anchor_context frames → (q_0, p_0) ─────────────────────────
	# q_0, p_0 represent the phase-space state after seeing frames 0..anchor_context-1.
	q, p, kl, *_ = model(frames[:, :anchor_context])
	kl_per = kl.clamp(min=free_bits)  # (B,)

	# ── Forward rollout: q_0 → q_{seq_len-anchor_context} ─────────────────
	# pred_frames_list[0] decodes q_0, predicting frames[:, anchor_context-1].
	# Each subsequent step applies action anchor_context-1+i and predicts
	# frames[:, anchor_context+i].
	n_fwd = seq_len - anchor_context
	pred_frames_list = [model.decoder(q)]
	state_preds      = [model.decode_state(q, p)] if do_state else []
	for i in range(n_fwd):
		q, p = model.controlled_step(q, p, actions[:, anchor_context - 1 + i], dt=dt)
		pred_frames_list.append(model.decoder(q))
		if do_state:
			state_preds.append(model.decode_state(q, p))

	pred_frames   = torch.stack(pred_frames_list, dim=1)        # (B, n_fwd+1, 3, H, W)
	target_frames = frames[:, anchor_context - 1 : seq_len]     # (B, n_fwd+1, 3, H, W)

	# Mask padded steps: frame anchor_context-1+t is valid iff its index < lengths[b].
	t_fwd      = torch.arange(n_fwd + 1, device=device)
	valid_mask = (anchor_context - 1 + t_fwd < lengths.unsqueeze(1)).float()  # (B, n_fwd+1)
	denom      = valid_mask.sum(dim=1).clamp(min=1)

	recon_per = (
		F.mse_loss(pred_frames, target_frames, reduction="none").mean(dim=[2, 3, 4])
		* valid_mask
	).sum(dim=1) / denom  # (B,)

	if do_state:
		pred_states   = torch.stack(state_preds, dim=1)             # (B, n_fwd+1, 4)
		target_states = states[:, anchor_context - 1 : seq_len]     # (B, n_fwd+1, 4)
		state_per = (
			F.mse_loss(pred_states, target_states, reduction="none").mean(dim=2)
			* valid_mask
		).sum(dim=1) / denom  # (B,)
	else:
		state_per = torch.zeros(B, device=device)

	# ── IS-weighted total loss ─────────────────────────────────────────────
	per_sample = recon_weight * recon_per + kl_weight * kl_per + state_weight * state_per

	loss = (is_weights * per_sample).mean()

	optimizer.zero_grad()
	loss.backward()
	if grad_clip > 0:
		torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
	optimizer.step()

	return (
		{
			"loss": loss.item(),
			"recon": recon_per.mean().item(),
			"kl": kl_per.mean().item(),
			"state": state_per.mean().item(),
		},
		sampled_indices,
		per_sample.detach().cpu().numpy(),
	)


# ── Evaluation ────────────────────────────────────────────────────────────────


def evaluate(
	model: ControlledDHGN_LSTM,
	planner: MPPI,
	cost_fn,
	seq_len: int,
	img_size: int,
	n_episodes: int,
	max_steps: int,
	device: torch.device,
) -> tuple[float, list[int], torch.Tensor]:
	"""Returns (mean_reward, all_rewards, video_tensor) where video_tensor is (1, T, 3, H, W) uint8."""
	rewards: list[int] = []
	sample_frames = None
	for i in range(n_episodes):
		ep = collect_episode(
			model, planner, seq_len, img_size, max_steps, device, cost_fn
		)
		rewards.append(len(ep))  # CartPole reward = number of steps survived
		if i == 0:
			sample_frames = ep.frames  # (T, 3, H, W) float in [0, 1]
	video = (sample_frames.unsqueeze(0).clamp(0, 1) * 255).to(
		torch.uint8
	)  # (1, T, 3, H, W)
	return float(np.mean(rewards)), rewards, video


# ── Forward validation ────────────────────────────────────────────────────────


_STATE_LABELS = ["cart_pos (x)", "cart_vel (ẋ)", "pole_angle (θ)", "pole_vel (θ̇)"]


def log_forward_validation(
	model: ControlledDHGN_LSTM,
	buffer: PrioritizedEpisodeReplayBuffer,
	writer: SummaryWriter,
	seq_len: int,
	anchor_context: int,
	dt: float,
	device: torch.device,
	step: int,
) -> None:
	"""Mirror the training forward rollout on an eval sample and log
	ground-truth vs. predicted frames and CartPole state trajectories.

	Encodes anchor_context frames → (q_0, p_0), then rolls forward using
	all available actions.  A vertical line marks where context ends and
	true forward prediction begins.

	Tries to sample 2*seq_len transitions for a longer forward horizon.
	Falls back to any episode of length >= anchor_context if needed.
	"""
	long_seq = 2 * seq_len

	if buffer.can_sample(long_seq):
		frames, actions, states, *_ = buffer.sample_sequences(1, seq_len=long_seq)
	elif buffer.can_sample(anchor_context):
		eligible = [ep for ep in buffer._episodes if len(ep) >= anchor_context]
		ep = random.choice(eligible)
		end = min(long_seq, len(ep))
		frames  = ep.frames[: end + 1].unsqueeze(0)
		actions = ep.actions[:end].unsqueeze(0)
		states  = ep.states[: end + 1].unsqueeze(0) if ep.states is not None else None
	else:
		return

	frames = frames.to(device)
	actions = actions.to(device)
	if states is not None:
		states = states.to(device)

	total_frames = frames.shape[1]

	model.eval()
	with torch.no_grad():
		# Encode anchor_context frames → (q_0, p_0)
		q, p, *_ = model(frames[:, :anchor_context])

		# Roll forward from q_0 using all available actions
		pred_frames_list: list[torch.Tensor] = [model.decoder(q)]
		pred_states_list: list[torch.Tensor] = []
		if model.state_decoder is not None:
			pred_states_list.append(model.decode_state(q, p))
		for t in range(anchor_context - 1, total_frames - 1):
			q, p = model.controlled_step(q, p, actions[:, t], dt=dt)
			pred_frames_list.append(model.decoder(q))
			if model.state_decoder is not None:
				pred_states_list.append(model.decode_state(q, p))

	n = len(pred_frames_list)
	pred_frames_t = torch.stack(pred_frames_list, dim=1)   # (1, n, 3, H, W)
	gt_frames     = frames[:, anchor_context - 1 : anchor_context - 1 + n]

	writer.add_video(
		"eval/fwd_gt_rollout",   (gt_frames.clamp(0, 1) * 255).to(torch.uint8),   step, fps=15
	)
	writer.add_video(
		"eval/fwd_pred_rollout", (pred_frames_t.clamp(0, 1) * 255).to(torch.uint8), step, fps=15
	)

	if states is not None and pred_states_list:
		gt_s = states[0, anchor_context - 1 : anchor_context - 1 + n].cpu().numpy()
		pr_s = torch.cat(pred_states_list, dim=0).cpu().numpy()

		t_range = np.arange(n)
		fig, axes = plt.subplots(2, 2, figsize=(10, 7))
		for i, (ax, label) in enumerate(zip(axes.flat, _STATE_LABELS)):
			ax.plot(t_range, gt_s[:, i], label="ground truth", color="steelblue")
			ax.plot(t_range, pr_s[:, i], label="predicted", color="darkorange", linestyle="--")
			ax.axvline(
				seq_len - anchor_context, color="gray", linestyle=":", alpha=0.7,
				label=f"training horizon end (t={seq_len - anchor_context})"
			)
			ax.set_title(label)
			ax.set_xlabel("step from context end")
			ax.legend(fontsize=8)
		fig.suptitle(f"Forward rollout: GT vs predicted state (iter {step})")
		fig.tight_layout()
		writer.add_figure("eval/fwd_state_rollout", fig, step)
		plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────


@click.command()
# model
@click.option("--img-size", type=int, default=64, show_default=True)
@click.option("--pos-ch", type=int, default=8, show_default=True)
@click.option("--feat-dim", type=int, default=256, show_default=True)
@click.option(
	"--seq-len",
	type=int,
	default=16,
	show_default=True,
	help="Total frames sampled per training window (context + forward rollout).",
)
@click.option("--dt", type=float, default=0.05, show_default=True)
@click.option(
	"--no-separable",
	"separable",
	default=True,
	flag_value=False,
	help="Use a non-separable Hamiltonian H(q,p) instead of T(q,p)+V(q).",
)
@click.option(
	"--obs-state-dim",
	type=int,
	default=4,
	show_default=True,
	help="CartPole state dimension for the state-decoder head.",
)
# online loop
@click.option(
	"--n-iterations",
	type=int,
	default=300,
	show_default=True,
	help="Number of collect→train iterations.",
)
@click.option(
	"--collect-per-iter",
	type=int,
	default=5,
	show_default=True,
	help="Episodes to collect per iteration.",
)
@click.option(
	"--train-steps-per-iter",
	type=int,
	default=50,
	show_default=True,
	help="Gradient steps per iteration.",
)
@click.option(
	"--n-warmup",
	type=int,
	default=20,
	show_default=True,
	help="Random-action episodes before MPPI takes over.",
)
@click.option(
	"--min-buffer",
	type=int,
	default=20,
	show_default=True,
	help="Minimum buffer episodes before training begins.",
)
@click.option("--buffer-capacity", type=int, default=2000, show_default=True)
@click.option("--max-episode-steps", type=int, default=500, show_default=True)
# training
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option("--lr", type=float, default=1e-4, show_default=True)
@click.option("--kl-weight", type=float, default=1e-3, show_default=True)
@click.option("--recon-weight", type=float, default=1.0, show_default=True)
@click.option(
	"--state-weight",
	type=float,
	default=0.5,
	show_default=True,
	help="Weight for state supervision along the forward rollout.",
)
@click.option(
	"--anchor-context",
	type=int,
	default=3,
	show_default=True,
	help=(
		"Number of context frames fed to the LSTM encoder. "
		"The hidden state after these frames is decoded to (q_0, p_0) and "
		"integrated forward for seq_len-anchor_context steps."
	),
)
@click.option("--free-bits", type=float, default=0.5, show_default=True)
@click.option("--grad-clip", type=float, default=1.0, show_default=True)
# PER
@click.option(
	"--per-alpha",
	type=float,
	default=0.6,
	show_default=True,
	help="PER priority exponent α. 0 = uniform sampling, 1 = full prioritization.",
)
@click.option(
	"--per-beta",
	type=float,
	default=0.4,
	show_default=True,
	help="PER IS-weight exponent β (initial value). Annealed toward 1 during training.",
)
@click.option(
	"--per-beta-annealing",
	type=float,
	default=0.0,
	show_default=True,
	help="Amount added to β after each training step. Set to (1-β0)/total_steps to reach 1 at end.",
)
# MPPI
@click.option("--mppi-horizon", type=int, default=20, show_default=True)
@click.option("--mppi-samples", type=int, default=256, show_default=True)
@click.option("--mppi-temperature", type=float, default=0.05, show_default=True)
@click.option("--mppi-sigma", type=float, default=0.5, show_default=True)
# eval / logging
@click.option(
	"--eval-every",
	type=int,
	default=10,
	show_default=True,
	help="Evaluate MPPI every N iterations.",
)
@click.option("--n-eval-episodes", type=int, default=10, show_default=True)
@click.option(
	"--log-every",
	type=int,
	default=1,
	show_default=True,
	help="Log training scalars every N iterations.",
)
@click.option(
	"--checkpoint-every",
	type=int,
	default=50,
	show_default=True,
	help="Save a periodic checkpoint every N iterations (0 = disabled).",
)
def main(
	img_size,
	pos_ch,
	feat_dim,
	seq_len,
	dt,
	separable,
	obs_state_dim,
	n_iterations,
	collect_per_iter,
	train_steps_per_iter,
	n_warmup,
	min_buffer,
	buffer_capacity,
	max_episode_steps,
	batch_size,
	lr,
	kl_weight,
	recon_weight,
	state_weight,
	anchor_context,
	free_bits,
	grad_clip,
	per_alpha,
	per_beta,
	per_beta_annealing,
	mppi_horizon,
	mppi_samples,
	mppi_temperature,
	mppi_sigma,
	eval_every,
	n_eval_episodes,
	log_every,
	checkpoint_every,
):
	assert img_size % 8 == 0, "--img-size must be a multiple of 8"
	assert anchor_context >= 1, "--anchor-context must be >= 1"
	assert anchor_context < seq_len, "--anchor-context must be < --seq-len"

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}")
	print(f"Context frames: {anchor_context}  →  forward rollout: {seq_len - anchor_context} steps")

	writer = SummaryWriter(comment="_phgn_lstm_online")
	run_dir = make_run_dir("phgn_lstm_online")

	# Snapshot of all model constructor args — used by the checkpoint visualiser.
	_model_hparams = {
		"img_size": img_size,
		"pos_ch": pos_ch,
		"feat_dim": feat_dim,
		"dt": dt,
		"seq_len": seq_len,
		"separable": separable,
		"obs_state_dim": obs_state_dim,
	}

	# ── Model ─────────────────────────────────────────────────────────────
	model = ControlledDHGN_LSTM(
		pos_ch=pos_ch,
		img_ch=3,
		dt=dt,
		feat_dim=feat_dim,
		img_size=img_size,
		control_dim=1,
		obs_state_dim=obs_state_dim,
		separable=separable,
	).to(device)
	print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
	print(f"State dim D = {model.state_dim}")

	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	# ── Replay buffer ──────────────────────────────────────────────────────
	buffer = PrioritizedEpisodeReplayBuffer(
		capacity=buffer_capacity,
		min_seq_len=1,
		alpha=per_alpha,
		beta=per_beta,
		beta_annealing=per_beta_annealing,
	)

	# ── MPPI planner ───────────────────────────────────────────────────────
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

	# ── Warmup: fill buffer with random-policy episodes ────────────────────
	print(f"\nWarmup: collecting {n_warmup} random episodes...")
	for i in tqdm(range(n_warmup), desc="Warmup"):
		ep = collect_episode(
			model,
			planner=None,
			seq_len=seq_len,
			img_size=img_size,
			max_steps=max_episode_steps,
			device=device,
		)
		buffer.push(ep)
		writer.add_scalar("collect/episode_len", len(ep), i)

	print(f"Buffer: {len(buffer)} episodes, {buffer.num_steps()} steps\n")

	# ── Main online loop ───────────────────────────────────────────────────
	use_mppi = len(buffer) >= min_buffer
	global_train_step = 0
	best_mean_reward = 0.0
	total_env_steps = buffer.num_steps()

	for iteration in (pbar := tqdm(range(n_iterations), desc="Iterations")):
		cost_fn = make_cartpole_cost(model, device)

		# ── Collect ────────────────────────────────────────────────────────
		ep_lens = []
		for _ in range(collect_per_iter):
			ep = collect_episode(
				model,
				planner=planner if use_mppi else None,
				seq_len=seq_len,
				img_size=img_size,
				max_steps=max_episode_steps,
				device=device,
				cost_fn=cost_fn if use_mppi else None,
			)
			buffer.push(ep)
			ep_lens.append(len(ep))
			total_env_steps += len(ep)

		if not use_mppi and len(buffer) >= min_buffer:
			use_mppi = True
			print(
				f"\n[iter {iteration}] Buffer ready — switching to MPPI collection"
			)

		writer.add_scalar(
			"collect/mean_episode_len", float(np.mean(ep_lens)), iteration
		)
		writer.add_scalar("collect/buffer_episodes", len(buffer), iteration)
		writer.add_scalar("collect/total_env_steps", total_env_steps, iteration)

		# ── Train ──────────────────────────────────────────────────────────
		if len(buffer) == 0:
			continue

		losses = {k: 0.0 for k in ("loss", "recon", "kl", "state")}
		for _ in range(train_steps_per_iter):
			step_losses, ep_indices, per_sample_losses = train_step(
				model,
				buffer,
				optimizer,
				seq_len=seq_len,
				dt=dt,
				batch_size=batch_size,
				kl_weight=kl_weight,
				recon_weight=recon_weight,
				state_weight=state_weight,
				anchor_context=anchor_context,
				free_bits=free_bits,
				grad_clip=grad_clip,
				device=device,
			)
			buffer.update_priorities(ep_indices, per_sample_losses)
			for k, v in step_losses.items():
				losses[k] += v
			global_train_step += 1

		if log_every > 0 and iteration % log_every == 0:
			n = train_steps_per_iter
			for k, v in losses.items():
				writer.add_scalar(f"train/{k}", v / n, iteration)
			pri_stats = buffer.priority_stats()
			writer.add_scalar("per/priority_mean", pri_stats["mean"], iteration)
			writer.add_scalar("per/priority_max", pri_stats["max"], iteration)
			writer.add_scalar("per/priority_min", pri_stats["min"], iteration)
			writer.add_scalar("per/beta", buffer.beta, iteration)

		# ── Periodic checkpoint ────────────────────────────────────────────
		if checkpoint_every > 0 and (iteration + 1) % checkpoint_every == 0:
			save_checkpoint(
				run_dir,
				iteration,
				model,
				{
					"seq_len": seq_len,
					"img_size": img_size,
					"pos_ch": pos_ch,
					"dt": dt,
				},
				{"iteration": iteration},
				stem=f"iter_{iteration + 1}",
			)

		# ── Evaluate ───────────────────────────────────────────────────────
		if (iteration + 1) % eval_every == 0:
			mean_r, all_rewards, video = evaluate(
				model,
				planner,
				cost_fn,
				seq_len=seq_len,
				img_size=img_size,
				n_episodes=n_eval_episodes,
				max_steps=max_episode_steps,
				device=device,
			)
			writer.add_scalar("eval/mean_reward", mean_r, iteration)
			writer.add_scalar(
				"eval/mean_reward_vs_env_steps", mean_r, total_env_steps
			)
			writer.add_histogram(
				"eval/reward_dist",
				np.array(all_rewards, dtype=np.float32),
				iteration,
			)
			writer.add_video("eval/rollout", video, iteration, fps=30)
			log_forward_validation(
				model,
				buffer,
				writer,
				seq_len=seq_len,
				anchor_context=anchor_context,
				dt=dt,
				device=device,
				step=iteration,
			)

			if mean_r > best_mean_reward:
				best_mean_reward = mean_r
				save_checkpoint(
					run_dir,
					iteration,
					model,
					_model_hparams,
					{"mean_reward": mean_r},
					stem="best",
				)

			pbar.set_postfix(
				mean_r=f"{mean_r:.1f}",
				best=f"{best_mean_reward:.1f}",
				buf=len(buffer),
				mppi=use_mppi,
			)
			tqdm.write(
				f"  [iter {iteration + 1:4d}]  mean_reward={mean_r:.1f}"
				f"  best={best_mean_reward:.1f}"
				f"  env_steps={total_env_steps:,}"
			)

	# ── Save final checkpoint ──────────────────────────────────────────────
	save_checkpoint(
		run_dir,
		n_iterations - 1,
		model,
		{
			"seq_len": seq_len,
			"img_size": img_size,
			"pos_ch": pos_ch,
			"dt": dt,
		},
		{"best_mean_reward": best_mean_reward},
		stem="last",
	)

	writer.close()
	print(f"\nDone. Best mean reward: {best_mean_reward:.1f}")
	print("Run: tensorboard --logdir runs")


if __name__ == "__main__":
	main()
