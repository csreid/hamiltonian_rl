"""Episode replay buffer for per-episode sequence sampling.

Stores complete episodes and supports sampling contiguous subsequences
for multi-step reverse-time rollout training.

Typical usage
-------------
buffer = EpisodeReplayBuffer(capacity=1000, min_seq_len=context_len)

# After each MPPI episode:
buffer.push(Episode(frames=..., actions=..., states=...))

# In the training loop:
frames, actions, states = buffer.sample_sequences(
    batch_size=32,
    seq_len=context_len + rollout_steps,
)
# frames:  (B, seq_len + 1, C, H, W)
# actions: (B, seq_len, control_dim)
# states:  (B, seq_len + 1, S) or None
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Episode:
	"""A single collected episode.

	Attributes:
	    frames:  (T+1, C, H, W) float32 in [0, 1] — one frame per
	             *state*, so T+1 frames for T steps.
	    actions: (T, control_dim) float32 — action applied at each step.
	    states:  (T+1, obs_state_dim) float32 or None — ground-truth
	             observed state if available (e.g. CartPole 4-vector).
	"""

	frames: torch.Tensor
	actions: torch.Tensor
	states: Optional[torch.Tensor] = None

	def __len__(self) -> int:
		"""Number of transition steps (= T)."""
		return self.actions.shape[0]

	def __post_init__(self) -> None:
		assert self.frames.shape[0] == self.actions.shape[0] + 1, (
			f"Expected frames.shape[0] == actions.shape[0] + 1, "
			f"got {self.frames.shape[0]} and {self.actions.shape[0]}"
		)
		if self.states is not None:
			assert self.states.shape[0] == self.frames.shape[0], (
				f"states.shape[0] must equal frames.shape[0], "
				f"got {self.states.shape[0]} and {self.frames.shape[0]}"
			)


class EpisodeReplayBuffer:
	"""Fixed-capacity replay buffer storing complete episodes.

	New episodes evict the oldest when the buffer is full (FIFO ring).
	Sampling draws contiguous subsequences from randomly chosen episodes,
	which is the primitive needed for multi-step reverse-time rollouts:

	    frames[start : start + seq_len + 1]   # seq_len+1 frames
	    actions[start : start + seq_len]       # seq_len actions

	Args:
	    capacity:    maximum number of episodes (FIFO eviction)
	    min_seq_len: episodes shorter than this are silently rejected
	                 on push.  Set to context_len + rollout_steps so
	                 every stored episode is always sampleable.
	"""

	def __init__(self, capacity: int, min_seq_len: int = 1) -> None:
		self.capacity = capacity
		self.min_seq_len = min_seq_len
		self._episodes: list[Episode] = []
		self._cursor: int = 0

	# ── Insertion ────────────────────────────────────────────────────────

	def push(self, episode: Episode) -> bool:
		"""Add an episode.  Returns False (and discards) if too short."""
		if len(episode) < self.min_seq_len:
			return False
		if len(self._episodes) < self.capacity:
			self._episodes.append(episode)
		else:
			self._episodes[self._cursor] = episode
		self._cursor = (self._cursor + 1) % self.capacity
		return True

	# ── Queries ──────────────────────────────────────────────────────────

	def __len__(self) -> int:
		return len(self._episodes)

	def num_steps(self) -> int:
		"""Total transition steps stored across all episodes."""
		return sum(len(ep) for ep in self._episodes)

	def can_sample(self, seq_len: int) -> bool:
		"""True if at least one episode has >= seq_len steps."""
		return any(len(ep) >= seq_len for ep in self._episodes)

	# ── Sampling ─────────────────────────────────────────────────────────

	def sample_sequences(
		self,
		batch_size: int,
		seq_len: int,
	) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
		"""Sample a batch of contiguous subsequences from random episodes.

		Algorithm:
		  1. Filter to episodes with at least ``seq_len`` steps.
		  2. For each sample, pick a random eligible episode and a random
		     start index, then slice:
		       frames[start : start + seq_len + 1]
		       actions[start : start + seq_len]
		       states[start : start + seq_len + 1]  (if available)

		Args:
		    batch_size: number of sequences in the batch
		    seq_len:    number of *steps* per sequence.
		                Caller should set this to context_len + rollout_steps
		                (or whatever window the training loop needs).

		Returns:
		    frames:  (B, seq_len + 1, C, H, W) float32
		    actions: (B, seq_len, control_dim) float32
		    states:  (B, seq_len + 1, S) float32, or None if not stored

		Raises:
		    ValueError if no episode in the buffer is long enough.
		"""
		eligible = [ep for ep in self._episodes if len(ep) >= seq_len]
		if not eligible:
			raise ValueError(
				f"No episodes with >= {seq_len} steps in buffer "
				f"({len(self._episodes)} episodes, "
				f"longest has {max((len(e) for e in self._episodes), default=0)} steps)"
			)

		has_states = eligible[0].states is not None

		frames_list: list[torch.Tensor] = []
		actions_list: list[torch.Tensor] = []
		states_list: list[torch.Tensor] = []

		for _ in range(batch_size):
			ep = random.choice(eligible)
			start = random.randint(0, len(ep) - seq_len)

			frames_list.append(ep.frames[start : start + seq_len + 1])
			actions_list.append(ep.actions[start : start + seq_len])
			if has_states and ep.states is not None:
				states_list.append(ep.states[start : start + seq_len + 1])

		frames = torch.stack(frames_list)  # (B, seq_len+1, C, H, W)
		actions = torch.stack(actions_list)  # (B, seq_len, control_dim)
		states = torch.stack(states_list) if states_list else None

		return frames, actions, states
