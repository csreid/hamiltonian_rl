"""Online PHGN-LSTM benchmark: MPPI data collection + replay buffer training.

Alternates between:
  1. Collecting CartPole episodes by running MPPI under the current world model
  2. Training the world model on contiguous subsequences sampled from the buffer

The replay buffer stores full episodes.  Training samples windows of
``seq_len`` frames (+ ``seq_len-1`` actions) and applies three losses:

  recon_loss  — backward rollout from the encoded state vs. reversed frames
  state_loss  — decode_state at each backward step vs. true CartPole state
  kl_loss     — VAE regularisation

The backward rollout covers the full sequence: LSTM encodes seq_len frames to
get (q, p) at time T, then integrates the port-Hamiltonian backwards seq_len-1
steps, supervising decoded frames and states at each step.

When ``fwd_weight > 0``, an additional forward-rollout loss is computed: the
LSTM hidden state after ``anchor_context`` frames is decoded to (q_k, p_k) and
integrated forward to seq_len-1, with recon and state losses scaled by
``fwd_weight``.

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
	fwd_weight: float,
	anchor_context: int,
	free_bits: float,
	grad_clip: float,
	device: torch.device,
) -> tuple[dict[str, float], list[int], np.ndarray]:
	"""One gradient step with PER IS-weighted loss.

	Encodes seq_len frames → (q_T, p_T), then rolls backward seq_len-1
	steps, supervising decoded frames and states against the reversed sequence.

	When fwd_weight > 0 and anchor_context < seq_len, also decodes the LSTM
	hidden state after anchor_context frames to (q_k, p_k) and rolls the
	Hamiltonian forward to seq_len-1, supervising against ground truth.

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
	rollout_steps = seq_len - 1
	model.train()

	do_state = (
		state_weight > 0
		and states is not None
		and model.state_decoder is not None
	)

	# ── Encode context (seq_len frames) → (q_T, p_T) ──────────────────────
	use_anchor = fwd_weight > 0 and anchor_context < seq_len - 1
	if use_anchor:
		# Capture all intermediate hidden states in a single encoder pass.
		all_mu, all_logvar = model.encoder.forward_all(frames[:, :seq_len])
		all_logvar = all_logvar.clamp(-10, 10)
		mu      = all_mu[:, -1]
		log_var = all_logvar[:, -1]
		z       = mu + torch.randn_like(mu) * (0.5 * log_var).exp()
		s0      = model.f_psi(z)
		q_T, p_T = model._split(s0)
		kl_per  = (-0.5 * (1 + log_var - mu.pow(2) - log_var.exp())).sum(dim=[1, 2, 3]).clamp(min=free_bits)
	else:
		q_T, p_T, kl, *_ = model(frames[:, :seq_len])
		kl_per = kl.clamp(min=free_bits)

	# ── Backward reconstruction rollout ───────────────────────────────────
	q, p = q_T, p_T
	pred_frames_list = [model.decoder(q)]
	state_preds = [model.decode_state(q, p)] if do_state else []
	for i in range(rollout_steps):
		u_rev = actions[:, seq_len - 2 - i]
		q, p = model.controlled_step(q, p, u_rev, dt=-dt)
		pred_frames_list.append(model.decoder(q))
		if do_state:
			state_preds.append(model.decode_state(q, p))

	pred_frames   = torch.stack(pred_frames_list, dim=1)   # (B, seq_len, 3, H, W)
	target_frames = frames[:, :seq_len].flip(dims=[1])     # (B, seq_len, 3, H, W)

	# Mask out padded time steps.
	t_bwd      = torch.arange(seq_len, device=device)
	valid_mask = (t_bwd >= (seq_len - lengths.unsqueeze(1))).float()  # (B, seq_len)

	recon_elem = F.mse_loss(pred_frames, target_frames, reduction="none").mean(dim=[2, 3, 4])
	recon_per  = (recon_elem * valid_mask).sum(dim=1) / lengths.float()

	if do_state:
		pred_states   = torch.stack(state_preds, dim=1)        # (B, seq_len, 4)
		target_states = states[:, :seq_len].flip(dims=[1])     # (B, seq_len, 4)
		state_elem = F.mse_loss(pred_states, target_states, reduction="none").mean(dim=2)
		state_per  = (state_elem * valid_mask).sum(dim=1) / lengths.float()
	else:
		state_per = torch.zeros(B, device=device)

	# ── Anchor forward rollout from h_{anchor_context} ────────────────────
	if use_anchor:
		k    = anchor_context
		mu_k = all_mu[:, k]          # (B, latent_ch, 4, 4) — posterior mean
		s_k  = model.f_psi(mu_k)
		q_k, p_k = model._split(s_k)

		n_fwd = seq_len - 1 - k
		fwd_pred_list  = [model.decoder(q_k)]
		fwd_state_list = [model.decode_state(q_k, p_k)] if do_state else []
		q, p = q_k, p_k
		for i in range(n_fwd):
			q, p = model.controlled_step(q, p, actions[:, k + i], dt=dt)
			fwd_pred_list.append(model.decoder(q))
			if do_state:
				fwd_state_list.append(model.decode_state(q, p))

		fwd_pred   = torch.stack(fwd_pred_list, dim=1)  # (B, n_fwd+1, 3, H, W)
		fwd_target = frames[:, k:seq_len]               # (B, n_fwd+1, 3, H, W)

		t_fwd     = torch.arange(n_fwd + 1, device=device)
		fwd_valid = (k + t_fwd < lengths.unsqueeze(1)).float()
		denom_fwd = fwd_valid.sum(dim=1).clamp(min=1)

		anchor_recon_per = (
			F.mse_loss(fwd_pred, fwd_target, reduction="none").mean(dim=[2, 3, 4])
			* fwd_valid
		).sum(dim=1) / denom_fwd

		if do_state:
			fwd_st_pred = torch.stack(fwd_state_list, dim=1)
			fwd_st_tgt  = states[:, k:seq_len]
			anchor_state_per = (
				F.mse_loss(fwd_st_pred, fwd_st_tgt, reduction="none").mean(dim=2)
				* fwd_valid
			).sum(dim=1) / denom_fwd
		else:
			anchor_state_per = torch.zeros(B, device=device)
	else:
		anchor_recon_per = torch.zeros(B, device=device)
		anchor_state_per = torch.zeros(B, device=device)

	# ── IS-weighted total loss ─────────────────────────────────────────────
	per_sample = (
		recon_weight * recon_per
		+ kl_weight * kl_per
		+ state_weight * state_per
		+ fwd_weight * (recon_weight * anchor_recon_per + state_weight * anchor_state_per)
	)

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
			"anchor_recon": anchor_recon_per.mean().item(),
			"anchor_state": anchor_state_per.mean().item(),
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


# ── Backward rollout visualisation ────────────────────────────────────────────


_STATE_LABELS = ["cart_pos (x)", "cart_vel (ẋ)", "pole_angle (θ)", "pole_vel (θ̇)"]


def log_backward_rollout(
	model: ControlledDHGN_LSTM,
	buffer: PrioritizedEpisodeReplayBuffer,
	writer: SummaryWriter,
	seq_len: int,
	dt: float,
	device: torch.device,
	step: int,
) -> None:
	"""Replicate the training backward rollout on an eval sample and log
	ground-truth vs. reconstructed frames and CartPole state trajectories.
	"""
	if not buffer.can_sample(seq_len):
		return

	model.eval()
	with torch.no_grad():
		frames, actions, states, *_ = buffer.sample_sequences(1, seq_len=seq_len)
		frames  = frames.to(device)
		actions = actions.to(device)
		if states is not None:
			states = states.to(device)

		q_T, p_T, *_ = model(frames[:, :seq_len])

		q, p = q_T, p_T
		pred_frames_list: list[torch.Tensor] = [model.decoder(q)]
		pred_states_list: list[torch.Tensor] = []
		if model.state_decoder is not None:
			pred_states_list.append(model.decode_state(q, p))

		for i in range(seq_len - 1):
			u_rev = actions[:, seq_len - 2 - i]
			q, p  = model.controlled_step(q, p, u_rev, dt=-dt)
			pred_frames_list.append(model.decoder(q))
			if model.state_decoder is not None:
				pred_states_list.append(model.decode_state(q, p))

	pred_frames_t = torch.stack(pred_frames_list, dim=1)   # (1, seq_len, 3, H, W)
	gt_frames     = frames[:, :seq_len].flip(dims=[1])

	writer.add_video("eval/bwd_gt_rollout",   (gt_frames.clamp(0, 1) * 255).to(torch.uint8),   step, fps=15)
	writer.add_video("eval/bwd_pred_rollout", (pred_frames_t.clamp(0, 1) * 255).to(torch.uint8), step, fps=15)

	if states is not None and pred_states_list:
		gt_s = states[0, :seq_len].flip(dims=[0]).cpu().numpy()
		pr_s = torch.cat(pred_states_list, dim=0).cpu().numpy()

		t_range = np.arange(seq_len)
		fig, axes = plt.subplots(2, 2, figsize=(10, 7))
		for i, (ax, label) in enumerate(zip(axes.flat, _STATE_LABELS)):
			ax.plot(t_range, gt_s[:, i], label="ground truth", color="steelblue")
			ax.plot(t_range, pr_s[:, i], label="predicted",    color="darkorange", linestyle="--")
			ax.set_title(label)
			ax.set_xlabel("backward rollout step")
			ax.legend(fontsize=8)
		fig.suptitle(f"Backward rollout: GT vs predicted state (iter {step})")
		fig.tight_layout()
		writer.add_figure("eval/bwd_state_rollout", fig, step)
		plt.close(fig)


# ── Forward validation ────────────────────────────────────────────────────────


def log_forward_validation(
	model: ControlledDHGN_LSTM,
	buffer: PrioritizedEpisodeReplayBuffer,
	writer: SummaryWriter,
	seq_len: int,
	dt: float,
	device: torch.device,
	step: int,
) -> None:
	"""Validate that the learned dynamics extrapolate forward correctly.

	Encodes the first seq_len frames to get (q_T, p_T), rolls backward
	seq_len-1 steps to recover (q_0, p_0), then rolls forward using the
	recorded actions for the full available horizon and compares decoded
	states to ground truth.  A vertical line marks where the encoding
	context ends and true forward prediction begins.

	Tries to sample 2*seq_len transitions for a meaningful forward horizon.
	Falls back to any episode of length >= seq_len if none are long enough.
	"""
	long_seq = 2 * seq_len

	if buffer.can_sample(long_seq):
		frames, actions, states, *_ = buffer.sample_sequences(1, seq_len=long_seq)
	elif buffer.can_sample(seq_len):
		eligible = [ep for ep in buffer._episodes if len(ep) >= seq_len]
		ep  = random.choice(eligible)
		end = min(long_seq, len(ep))
		frames  = ep.frames[: end + 1].unsqueeze(0)
		actions = ep.actions[:end].unsqueeze(0)
		states  = ep.states[: end + 1].unsqueeze(0) if ep.states is not None else None
	else:
		return

	frames  = frames.to(device)
	actions = actions.to(device)
	if states is not None:
		states = states.to(device)

	total_frames = frames.shape[1]

	model.eval()
	with torch.no_grad():
		# 1. Encode first seq_len frames → (q_T, p_T)
		q_T, p_T, *_ = model(frames[:, :seq_len])

		# 2. Roll backward seq_len-1 steps → (q_0, p_0)
		q, p = q_T, p_T
		for i in range(seq_len - 1):
			u_rev = actions[:, seq_len - 2 - i]
			q, p  = model.controlled_step(q, p, u_rev, dt=-dt)

		# 3. Roll forward from (q_0, p_0) using all available actions
		pred_states_list: list[torch.Tensor] = []
		if model.state_decoder is not None:
			pred_states_list.append(model.decode_state(q, p))
		for t in range(total_frames - 1):
			q, p = model.controlled_step(q, p, actions[:, t], dt=dt)
			if model.state_decoder is not None:
				pred_states_list.append(model.decode_state(q, p))

	if states is not None and pred_states_list:
		n    = len(pred_states_list)
		gt_s = states[0, :n].cpu().numpy()
		pr_s = torch.cat(pred_states_list, dim=0).cpu().numpy()

		t_range = np.arange(n)
		fig, axes = plt.subplots(2, 2, figsize=(10, 7))
		for i, (ax, label) in enumerate(zip(axes.flat, _STATE_LABELS)):
			ax.plot(t_range, gt_s[:, i], label="ground truth", color="steelblue")
			ax.plot(t_range, pr_s[:, i], label="predicted", color="darkorange", linestyle="--")
			ax.axvline(
				seq_len - 1, color="gray", linestyle=":", alpha=0.7,
				label=f"context end (t={seq_len - 1})"
			)
			ax.set_title(label)
			ax.set_xlabel("step")
			ax.legend(fontsize=8)
		fig.suptitle(f"Forward validation: GT vs predicted state (iter {step})")
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
	default=8,
	show_default=True,
	help="Frames fed to the LSTM encoder; backward rollout covers seq_len-1 steps.",
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
	help="Weight for state supervision along the backward rollout trajectory.",
)
@click.option(
	"--fwd-weight",
	type=float,
	default=0.5,
	show_default=True,
	help="Weight for anchor forward-rollout supervision loss.",
)
@click.option(
	"--anchor-context",
	type=int,
	default=3,
	show_default=True,
	help=(
		"Number of context frames for the forward-rollout anchor. "
		"The LSTM hidden state after this many frames is decoded to phase space "
		"and integrated forward to seq_len-1."
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
	fwd_weight,
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
	assert seq_len >= 2, "--seq-len must be >= 2"
	assert img_size % 8 == 0, "--img-size must be a multiple of 8"

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}")
	print(f"Sequence length: {seq_len}  (backward rollout: {seq_len - 1} steps)")

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

		losses = {k: 0.0 for k in ("loss", "recon", "kl", "state", "anchor_recon", "anchor_state")}
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
				fwd_weight=fwd_weight,
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
			log_backward_rollout(
				model,
				buffer,
				writer,
				seq_len=seq_len,
				dt=dt,
				device=device,
				step=iteration,
			)
			log_forward_validation(
				model,
				buffer,
				writer,
				seq_len=seq_len,
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
