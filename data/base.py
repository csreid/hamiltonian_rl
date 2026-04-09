"""Abstract base types for training datasets.

Defines SampledBatch (a batch with a back-channel for priority updates) and
the TrainingDataset Protocol shared by replay buffers and static data loaders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Protocol

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader


def _noop_commit(losses: np.ndarray) -> None:
	pass


@dataclass
class SampledBatch:
	"""A batch drawn from any TrainingDataset.

	The _commit_fn field closes over the buffer indices (when relevant) and
	calls update_priorities when commit() is invoked.  For static datasets
	the function is a no-op so callers never need to branch on dataset type.
	"""

	frames: Tensor
	actions: Tensor | None
	states: Tensor | None
	is_weights: Tensor
	lengths: Tensor | None
	_commit_fn: Callable[[np.ndarray], None] = field(
		default=_noop_commit, repr=False, compare=False
	)

	def commit(self, per_sample_losses: np.ndarray) -> None:
		"""Push per-sample losses back to the buffer for priority update."""
		self._commit_fn(per_sample_losses)


class TrainingDataset(Protocol):
	"""Minimal interface shared by replay buffers and static data loaders.

	Replay buffers (PrioritizedEpisodeReplayBuffer) implement all methods.
	Static datasets (DataLoaderAdapter) make push / can_sample no-ops and
	set is_weights=ones in every returned SampledBatch.
	"""

	def sample(self, batch_size: int, **kwargs) -> SampledBatch:
		"""Draw a batch.  Caller should call batch.commit(losses) after the
		training step if priority feedback is relevant."""
		...

	def push(self, item: Any) -> None:
		"""Add one item (episode/sample).  No-op for static datasets."""
		...

	def can_sample(self, n: int) -> bool:
		"""True if at least one item of effective length >= n is available."""
		...

	def __len__(self) -> int: ...


class DataLoaderAdapter:
	"""Wraps a PyTorch DataLoader to satisfy the TrainingDataset Protocol.

	Maintains a cycling internal iterator so consecutive sample() calls cross
	epoch boundaries seamlessly.  __iter__ (used by BaseTrainer.fit()) yields
	SampledBatch objects so StandardTrainer.train_step receives a uniform type.

	Batches from the underlying DataLoader are expected to have frames as their
	first element; remaining elements are discarded (they are not used by
	StandardTrainer but are present in SHO TensorDatasets).
	"""

	def __init__(self, loader: DataLoader) -> None:
		self._loader = loader
		self._iter: Iterator | None = None

	# ── TrainingDataset Protocol ──────────────────────────────────────────────

	def sample(self, batch_size: int, **kwargs) -> SampledBatch:
		if self._iter is None:
			self._iter = iter(self._loader)
		try:
			raw = next(self._iter)
		except StopIteration:
			self._iter = iter(self._loader)
			raw = next(self._iter)
		return self._wrap(raw)

	def push(self, item: Any) -> None:
		pass

	def can_sample(self, n: int) -> bool:
		return len(self._loader.dataset) > 0

	def __len__(self) -> int:
		"""Number of batches per epoch (mirrors DataLoader.__len__)."""
		return len(self._loader)

	# ── Iterable for BaseTrainer.fit() epoch loop ─────────────────────────────

	def __iter__(self) -> Iterator[SampledBatch]:
		for raw in self._loader:
			yield self._wrap(raw)

	# ── Internal ──────────────────────────────────────────────────────────────

	@staticmethod
	def _wrap(raw) -> SampledBatch:
		frames = raw[0]
		B = frames.shape[0]
		return SampledBatch(
			frames=frames,
			actions=None,
			states=None,
			is_weights=torch.ones(B),
			lengths=None,
		)
