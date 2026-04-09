"""Logging mixin and decorator for HGN trainers."""

from __future__ import annotations

import functools

from diag_common import (
	ActivationMonitor,
	log_gradient_stats,
	log_hamiltonian_grad_stats,
	log_histograms,
	log_latent_stats,
	log_weight_norms,
)
from training.losses import LossResult


def increment_step(method):
	"""Decorator: increments self.global_step after each train_step call."""

	@functools.wraps(method)
	def wrapper(self, batch):
		result = method(self, batch)
		self.global_step += 1
		return result

	return wrapper


class LoggingMixin:
	"""Mixin that provides standardised TensorBoard logging for trainers.

	The host class must have:
	    self.writer:       SummaryWriter
	    self.global_step:  int
	    self.model:        nn.Module
	    self.cfg:          object with .diag_every, .log_every, .loss.free_bits
	    self._act_monitor: ActivationMonitor  (managed by BaseTrainer)
	"""

	def log_scalar(
		self, tag: str, value: float, step: int | None = None
	) -> None:
		step = step if step is not None else self.global_step
		self.writer.add_scalar(tag, value, step)

	def log_scalars(self, d: dict[str, float], step: int | None = None) -> None:
		for tag, val in d.items():
			self.log_scalar(tag, val, step)

	def log_loss_result(self, result: LossResult) -> None:
		self.log_scalars(result.as_dict())

	def maybe_log_diagnostics(
		self,
		mu=None,
		log_var=None,
		q0=None,
		p0=None,
	) -> None:
		"""Conditionally run expensive diagnostics if diag_every fires."""
		if self.cfg.diag_every <= 0:
			return
		if self.global_step % self.cfg.diag_every != 0:
			return

		log_gradient_stats(self.writer, self.model, self.global_step)
		self._act_monitor.log(self.writer, self.global_step)
		self._act_monitor.check_flags(self.global_step)

		if mu is not None and log_var is not None:
			log_latent_stats(
				self.writer,
				{"z": (mu.detach(), log_var.detach())},
				self.global_step,
				self.cfg.loss.free_bits,
			)

		if q0 is not None and p0 is not None and hasattr(self.model, "H"):
			log_hamiltonian_grad_stats(
				self.writer,
				self.model.H,
				q0.detach(),
				p0.detach(),
				self.global_step,
			)

		log_weight_norms(self.writer, self.model, self.global_step)

	def maybe_log_histograms(self, epoch: int) -> None:
		"""Log weight/gradient histograms every log_every epochs."""
		if (epoch + 1) % self.cfg.log_every == 0:
			log_histograms(self.writer, self.model, epoch)
