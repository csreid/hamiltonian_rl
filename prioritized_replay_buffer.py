"""Prioritized Experience Replay (PER) buffer for episode-based sequence sampling.

Uses episode-level priorities: each stored episode has a single scalar priority
that is updated after training using the mean per-sample loss from subsequences
drawn from that episode.  Sampling probability is proportional to priority^alpha.
Importance-sampling (IS) weights are returned to correct for the introduced bias.

Typical usage
-------------
buffer = PrioritizedEpisodeReplayBuffer(
    capacity=2000, min_seq_len=context_len, alpha=0.6, beta=0.4
)

# After each episode:
buffer.push(episode)

# In the training loop:
frames, actions, states, indices, is_weights = buffer.sample_sequences(
    batch_size=32, seq_len=context_len,
)
# ... compute per_sample_losses (B,) ...
buffer.update_priorities(indices, per_sample_losses)

References
----------
Schaul et al., "Prioritized Experience Replay", ICLR 2016.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch

from data.base import SampledBatch
from replay_buffer import Episode


class PrioritizedEpisodeReplayBuffer:
	"""Fixed-capacity replay buffer with prioritized episode sampling.

	Each episode is assigned one priority scalar.  New episodes receive the
	maximum priority seen so far, ensuring they are sampled at least once
	before their priority is estimated from actual training losses.

	Sampling selects episodes with probability proportional to p_i^alpha,
	then draws a uniformly random contiguous window within the chosen episode.
	IS weights w_i = (N * p_i)^{-beta} / max_j(w_j) are returned so the
	caller can correct the gradient bias.  beta is optionally annealed toward
	1 over training.

	Args:
	    capacity:        maximum number of episodes (FIFO eviction)
	    min_seq_len:     episodes shorter than this are silently rejected on push
	    alpha:           priority exponent in [0, 1].  0 = uniform, 1 = full PER
	    beta:            IS weight exponent in [0, 1].  0 = no correction,
	                     1 = full unbiased correction
	    beta_annealing:  amount added to beta after each call to sample_sequences;
	                     set to (1 - beta_0) / total_train_steps to reach 1 at end
	    epsilon:         small constant added to every priority to prevent zeros
	"""

	def __init__(
		self,
		capacity: int,
		min_seq_len: int = 1,
		alpha: float = 0.6,
		beta: float = 0.4,
		beta_annealing: float = 0.0,
		epsilon: float = 1e-6,
	) -> None:
		self.capacity = capacity
		self.min_seq_len = min_seq_len
		self.alpha = alpha
		self.beta = beta
		self.beta_annealing = beta_annealing
		self.epsilon = epsilon

		self._episodes: list[Episode] = []
		self._priorities: list[float] = []
		self._cursor: int = 0
		self._max_priority: float = 1.0

	# ── Insertion ─────────────────────────────────────────────────────────────

	def push(self, episode: Episode) -> bool:
		"""Add an episode.  Returns False (and discards) if too short."""
		if len(episode) < self.min_seq_len:
			return False
		if len(self._episodes) < self.capacity:
			self._episodes.append(episode)
			self._priorities.append(self._max_priority)
		else:
			self._episodes[self._cursor] = episode
			self._priorities[self._cursor] = self._max_priority
		self._cursor = (self._cursor + 1) % self.capacity
		return True

	# ── Queries ───────────────────────────────────────────────────────────────

	def __len__(self) -> int:
		return len(self._episodes)

	def num_steps(self) -> int:
		"""Total transition steps stored across all episodes."""
		return sum(len(ep) for ep in self._episodes)

	def can_sample(self, seq_len: int) -> bool:
		"""True if at least one episode has >= seq_len steps."""
		return any(len(ep) >= seq_len for ep in self._episodes)

	# ── Sampling ──────────────────────────────────────────────────────────────

	def sample_sequences(
		self,
		batch_size: int,
		seq_len: int,
	) -> tuple[
		torch.Tensor,
		torch.Tensor,
		Optional[torch.Tensor],
		list[int],
		torch.Tensor,
		torch.Tensor,
	]:
		"""Sample a prioritized batch of contiguous subsequences.

		Episodes shorter than seq_len are included: their full length is used
		and the returned tensors are right-padded to seq_len (last frame
		repeated, actions zero-padded).  The returned ``lengths`` tensor
		records the actual (unpadded) length of each sample so callers can
		mask losses accordingly.

		Algorithm:
		  1. Sample from ALL episodes proportional to priority^alpha.
		  2. For each chosen episode, pick a random start offset; effective
		     length is min(seq_len, len(ep)).
		  3. Right-pad short sequences to seq_len.
		  4. Compute IS weights w_i = (N * P(i))^{-beta} / max_j(w_j).
		  5. Anneal beta.

		Args:
		    batch_size: number of sequences in the batch
		    seq_len:    desired number of *steps* per sequence

		Returns:
		    frames:     (B, seq_len+1, C, H, W) float32
		    actions:    (B, seq_len, control_dim) float32
		    states:     (B, seq_len+1, S) float32, or None if not stored
		    indices:    list[int] of length B — buffer indices for update_priorities()
		    is_weights: (B,) float32 IS correction weights in (0, 1]
		    lengths:    (B,) int64 actual (unpadded) sequence lengths

		Raises:
		    ValueError if the buffer is empty.
		"""
		if not self._episodes:
			raise ValueError("Buffer is empty")

		# Priority-proportional sampling over ALL episodes
		n = len(self._episodes)
		raw_pri = np.array(self._priorities[:n], dtype=np.float64)
		probs = raw_pri**self.alpha
		probs /= probs.sum()

		# IS weights: w_i = (N * P(i))^{-beta}, normalised by max weight
		min_prob = float(probs.min())
		max_weight = (n * min_prob) ** (-self.beta)

		chosen = np.random.choice(n, size=batch_size, p=probs)
		is_weights = np.array(
			[(n * probs[k]) ** (-self.beta) / max_weight for k in chosen],
			dtype=np.float32,
		)

		indices = list(chosen)

		has_states = self._episodes[indices[0]].states is not None

		frames_list: list[torch.Tensor] = []
		actions_list: list[torch.Tensor] = []
		states_list: list[torch.Tensor] = []
		lengths_list: list[int] = []

		for idx in indices:
			ep = self._episodes[idx]
			eff_len = min(seq_len, len(ep))
			start = random.randint(0, len(ep) - eff_len)

			f = ep.frames[start : start + eff_len + 1]  # (eff_len+1, C, H, W)
			a = ep.actions[start : start + eff_len]  # (eff_len, control_dim)

			if eff_len < seq_len:
				pad = seq_len - eff_len
				f = torch.cat([f, f[-1:].expand(pad, *f.shape[1:])], dim=0)
				a = torch.cat([a, torch.zeros(pad, a.shape[-1])], dim=0)

			frames_list.append(f)
			actions_list.append(a)
			lengths_list.append(eff_len)

			if has_states and ep.states is not None:
				s = ep.states[start : start + eff_len + 1]
				if eff_len < seq_len:
					s = torch.cat([s, s[-1:].expand(pad, *s.shape[1:])], dim=0)
				states_list.append(s)

		frames = torch.stack(frames_list)  # (B, seq_len+1, C, H, W)
		actions = torch.stack(actions_list)  # (B, seq_len, control_dim)
		states = torch.stack(states_list) if states_list else None
		lengths = torch.tensor(lengths_list, dtype=torch.long)

		# Anneal beta toward 1
		self.beta = min(1.0, self.beta + self.beta_annealing)

		return (
			frames,
			actions,
			states,
			indices,
			torch.from_numpy(is_weights),
			lengths,
		)

	def sample(self, batch_size: int, seq_len: int) -> SampledBatch:
		"""Sample a prioritized batch and return a SampledBatch with commit() support.

		Equivalent to sample_sequences() but returns a SampledBatch whose
		commit(per_sample_losses) method calls update_priorities() automatically,
		removing the need for callers to track buffer indices.
		"""
		frames, actions, states, indices, is_weights, lengths = (
			self.sample_sequences(batch_size, seq_len)
		)
		return SampledBatch(
			frames=frames,
			actions=actions,
			states=states,
			is_weights=is_weights,
			lengths=lengths,
			_commit_fn=lambda losses: self.update_priorities(indices, losses),
		)

	def sample_full_episodes(self, batch_size: int) -> SampledBatch:
		"""Sample full episodes, padded to the longest episode in the batch.

		Unlike sample_sequences(), no random start offset is applied — the
		entire episode is returned.  This lets a bidirectional encoder see the
		complete trajectory before committing to a latent state.

		Returns:
		    SampledBatch where:
		      frames:  (B, max_len+1, C, H, W)
		      actions: (B, max_len, control_dim)
		      states:  (B, max_len+1, S) or None
		      lengths: (B,) actual episode lengths (unpadded step count)
		"""
		if not self._episodes:
			raise ValueError("Buffer is empty")

		n = len(self._episodes)
		raw_pri = np.array(self._priorities[:n], dtype=np.float64)
		probs = raw_pri ** self.alpha
		probs /= probs.sum()

		min_prob = float(probs.min())
		max_weight = (n * min_prob) ** (-self.beta)

		chosen = np.random.choice(n, size=batch_size, p=probs)
		is_weights = np.array(
			[(n * probs[k]) ** (-self.beta) / max_weight for k in chosen],
			dtype=np.float32,
		)
		indices = list(chosen)

		max_len = max(len(self._episodes[i]) for i in indices)
		has_states = self._episodes[indices[0]].states is not None

		frames_list: list[torch.Tensor] = []
		actions_list: list[torch.Tensor] = []
		states_list: list[torch.Tensor] = []
		lengths_list: list[int] = []

		for idx in indices:
			ep = self._episodes[idx]
			eff_len = len(ep)
			lengths_list.append(eff_len)

			f = ep.frames   # (eff_len+1, C, H, W)
			a = ep.actions  # (eff_len, control_dim)

			if eff_len < max_len:
				pad = max_len - eff_len
				f = torch.cat([f, f[-1:].expand(pad, *f.shape[1:])], dim=0)
				a = torch.cat([a, torch.zeros(pad, a.shape[-1])], dim=0)

			frames_list.append(f)
			actions_list.append(a)

			if has_states and ep.states is not None:
				s = ep.states
				if eff_len < max_len:
					s = torch.cat([s, s[-1:].expand(pad, *s.shape[1:])], dim=0)
				states_list.append(s)

		frames = torch.stack(frames_list)   # (B, max_len+1, C, H, W)
		actions = torch.stack(actions_list)  # (B, max_len, control_dim)
		states = torch.stack(states_list) if states_list else None
		lengths = torch.tensor(lengths_list, dtype=torch.long)

		self.beta = min(1.0, self.beta + self.beta_annealing)

		return SampledBatch(
			frames=frames,
			actions=actions,
			states=states,
			is_weights=torch.from_numpy(is_weights),
			lengths=lengths,
			_commit_fn=lambda losses: self.update_priorities(indices, losses),
		)

	# ── Priority update ───────────────────────────────────────────────────────

	def update_priorities(
		self,
		indices: list[int],
		losses: list[float] | np.ndarray,
	) -> None:
		"""Update episode priorities from per-sample training losses.

		When multiple samples in a batch come from the same episode, their
		losses are averaged before updating that episode's priority.

		Args:
		    indices: episode buffer indices returned by sample_sequences
		    losses:  per-sample scalar losses, same length as indices
		"""
		idx_to_losses: dict[int, list[float]] = {}
		for idx, loss in zip(indices, losses):
			idx_to_losses.setdefault(idx, []).append(float(loss))

		for idx, ep_losses in idx_to_losses.items():
			if idx < len(self._priorities):
				new_priority = float(np.mean(ep_losses)) + self.epsilon
				self._priorities[idx] = new_priority
				self._max_priority = max(self._max_priority, new_priority)

	# ── Diagnostics ───────────────────────────────────────────────────────────

	def priority_stats(self) -> dict[str, float]:
		"""Return summary statistics of current priorities."""
		if not self._priorities:
			return {"mean": 0.0, "max": 0.0, "min": 0.0}
		p = np.array(self._priorities, dtype=np.float64)
		return {
			"mean": float(p.mean()),
			"max": float(p.max()),
			"min": float(p.min()),
		}
