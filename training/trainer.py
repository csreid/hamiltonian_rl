"""Base and standard trainers for HGN models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from checkpoint_common import make_run_dir, save_checkpoint
from diag_common import ActivationMonitor, image_centroid
from training.logging_mixin import LoggingMixin, increment_step
from training.losses import LossConfig, LossResult, compute_standard_loss


@dataclass
class TrainerConfig:
	"""All hyperparameters consumed by BaseTrainer and StandardTrainer.

	Populated from Click kwargs via TrainerConfig.from_kwargs().
	"""

	n_epochs: int = 5
	lr: float = 1.5e-4
	grad_clip: float = 1.0
	log_every: int = 10
	diag_every: int = 1
	batch_size: int = 16
	dt: float = 0.125
	train_rollout: int = 30
	context_len: int = 31
	loss: LossConfig = field(default_factory=LossConfig)

	@classmethod
	def from_kwargs(cls, kwargs: dict) -> "TrainerConfig":
		loss = LossConfig(
			recon_weight=kwargs.get("recon_weight", 1.0),
			kl_weight=kwargs.get("kl_weight", 1e-3),
			free_bits=kwargs.get("free_bits", 1.0),
			coord_weight=kwargs.get("coord_weight", 0.0),
			energy_weight=kwargs.get("energy_weight", 0.0),
			state_weight=kwargs.get("state_weight", 0.0),
		)
		return cls(
			n_epochs=kwargs["n_epochs"],
			lr=kwargs["lr"],
			grad_clip=kwargs.get("grad_clip", 1.0),
			log_every=kwargs.get("log_every", 10),
			diag_every=kwargs.get("diag_every", 1),
			batch_size=kwargs["batch_size"],
			dt=kwargs["dt"],
			train_rollout=kwargs.get("train_rollout", 30),
			context_len=kwargs.get("n_frames", 31),
			loss=loss,
		)


class BaseTrainer(LoggingMixin):
	"""Abstract base trainer. Subclasses implement train_step and validate.

	Provides:
	- Standard fit() loop (epoch → batch, periodic validation + checkpoint)
	- _backward_and_step() gradient update helper
	- Automatic ActivationMonitor lifecycle
	- hparam markdown table logged to TensorBoard
	"""

	def __init__(
		self,
		cfg: TrainerConfig,
		model: nn.Module,
		train_loader,
		writer: SummaryWriter,
		run_dir: Path,
		device: torch.device,
		hparams: dict,
	):
		self.cfg = cfg
		self.model = model
		self.train_loader = train_loader
		self.writer = writer
		self.run_dir = run_dir
		self.device = device
		self.hparams = hparams
		self.global_step = 0
		self.best_val_metric = float("inf")
		self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
		self._act_monitor = ActivationMonitor(model)
		self._write_hparam_table(hparams)

	# ── Abstract interface ────────────────────────────────────────────────────

	def train_step(self, batch) -> LossResult:
		raise NotImplementedError

	def validate(self, epoch: int) -> dict[str, float]:
		raise NotImplementedError

	# ── Main loop ─────────────────────────────────────────────────────────────

	def fit(self) -> None:
		for epoch in (
			epoch_bar := tqdm(range(self.cfg.n_epochs), desc="Epochs")
		):
			self.model.train()
			self._act_monitor = ActivationMonitor(self.model)
			epoch_totals: dict[str, float] = {}

			for batch in tqdm(
				self.train_loader, desc=f"  epoch {epoch}", leave=False
			):
				result = self.train_step(batch)
				for k, v in result.as_dict().items():
					epoch_totals[k] = epoch_totals.get(k, 0.0) + v

			n = len(self.train_loader)
			epoch_bar.set_postfix(
				loss=f"{epoch_totals.get('train/loss', 0.0) / n:.4f}"
			)

			self.maybe_log_histograms(epoch)

			if (epoch + 1) % self.cfg.log_every == 0:
				self._act_monitor.remove()
				val_metrics = self.validate(epoch)
				self.log_scalars(val_metrics, step=epoch)
				self._handle_checkpoint(epoch, val_metrics)
				self._act_monitor = ActivationMonitor(self.model)

		self._act_monitor.remove()
		self.writer.close()

	# ── Shared helpers ────────────────────────────────────────────────────────

	def _backward_and_step(self, loss: torch.Tensor) -> None:
		self.optimizer.zero_grad()
		loss.backward()
		if self.cfg.grad_clip > 0:
			nn.utils.clip_grad_norm_(
				self.model.parameters(), self.cfg.grad_clip
			)
		self.optimizer.step()

	def _handle_checkpoint(
		self, epoch: int, val_metrics: dict[str, float]
	) -> None:
		primary = val_metrics.get("val/pixel_mse", float("inf"))
		if primary < self.best_val_metric:
			self.best_val_metric = primary
			save_checkpoint(
				self.run_dir, epoch, self.model, self.hparams, val_metrics
			)

	def _write_hparam_table(self, hparams: dict) -> None:
		rows = "| Hyperparameter | Value |\n|---|---|\n"
		rows += "\n".join(f"| {k} | {v} |" for k, v in hparams.items())
		self.writer.add_text("hparams", rows, 0)


class StandardTrainer(BaseTrainer):
	"""Trainer for SHO-image HGN models with backward or forward rollout.

	Handles:
	- Backward-rollout training (default, HGN_LSTM family): dt is negated
	- Forward-rollout training (HGN_org family): dt used as-is
	- Optional coordinate head loss
	- Optional Hamiltonian energy conservation loss

	After each call to validate(), the following attributes are set and
	available to a _validate_hook callback:
	    self._last_vis_pred        (N_VID, T, C, H, W) predicted frames on CPU
	    self._last_vis_pred_coords (N_VID, T, 2) predicted coords on CPU
	    self._last_q_gt_val        (N_VID, T) ground-truth q on CPU
	    self._last_qs_val, self._last_ps_val  lists of (N_VID, ...) tensors
	"""

	def __init__(
		self,
		cfg: TrainerConfig,
		model: nn.Module,
		train_loader,
		writer: SummaryWriter,
		run_dir: Path,
		device: torch.device,
		hparams: dict,
		val_frames: torch.Tensor,
		val_q: torch.Tensor,
		rollout_direction: str = "backward",
		validate_hook: Callable | None = None,
	):
		super().__init__(
			cfg, model, train_loader, writer, run_dir, device, hparams
		)
		self.val_frames = val_frames
		self.val_q = val_q
		self.rollout_direction = rollout_direction
		self._validate_hook = validate_hook
		self._rollout_dt = (
			-cfg.dt if rollout_direction == "backward" else cfg.dt
		)

		# Set by validate() for use in _validate_hook
		self._last_vis_pred = None
		self._last_vis_pred_coords = None
		self._last_q_gt_val = None
		self._last_qs_val = None
		self._last_ps_val = None

	@increment_step
	def train_step(self, batch) -> LossResult:
		frames = batch.frames.to(self.device)

		q0, p0, kl, mu, log_var = self.model(frames[:, : self.cfg.context_len])

		need_states = self.cfg.loss.energy_weight > 0
		rollout_out = self.model.rollout(
			q0,
			p0,
			n_steps=self.cfg.train_rollout,
			dt=self._rollout_dt,
			return_states=need_states,
		)

		if need_states:
			pred_frames_list, pred_coords_list, qs, ps = rollout_out
		else:
			pred_frames_list, pred_coords_list = rollout_out
			qs = ps = None

		pred_frames = torch.stack(pred_frames_list, dim=1)
		pred_coords = torch.stack(pred_coords_list, dim=1)

		if self.rollout_direction == "backward":
			target_frames = frames[:, : self.cfg.context_len].flip(dims=[1])[
				:, : self.cfg.train_rollout + 1
			]
		else:
			target_frames = frames[:, : self.cfg.train_rollout + 1]

		with torch.no_grad():
			gt_coords = image_centroid(target_frames)

		result = compute_standard_loss(
			pred_frames,
			target_frames,
			kl,
			cfg=self.cfg.loss,
			pred_coords=pred_coords,
			gt_coords=gt_coords,
			qs=qs,
			ps=ps,
			H_fn=self.model.H if need_states else None,
			q0=q0,
			p0=p0,
		)

		self._backward_and_step(result.total)
		self.log_loss_result(result)
		self.maybe_log_diagnostics(mu=mu, log_var=log_var, q0=q0, p0=p0)
		return result

	def validate(self, epoch: int) -> dict[str, float]:
		self.model.eval()
		context_len = self.cfg.context_len
		val_rollout_steps = context_len - 1
		batch_size = self.cfg.batch_size
		N_VID = min(4, self.val_frames.shape[0])

		val_sum_mse = val_sum_coord_mse = val_n = 0
		vis_pred = vis_pred_coords = None
		qs_vid = ps_vid = None

		with torch.no_grad():
			for vi in range(0, self.val_frames.shape[0], batch_size):
				fv = self.val_frames[vi : vi + batch_size]
				q0_i, p0_i, _, _, _ = self.model(fv[:, :context_len])
				pred_list_i, pred_coords_list_i, qs_i, ps_i = (
					self.model.rollout(
						q0_i,
						p0_i,
						n_steps=val_rollout_steps,
						dt=self._rollout_dt,
						return_states=True,
					)
				)
				pred_i = torch.stack(pred_list_i, dim=1).clamp(0, 1)
				pred_coords_i = torch.stack(pred_coords_list_i, dim=1).cpu()

				if self.rollout_direction == "backward":
					target_i = fv[:, :context_len].flip(dims=[1])
				else:
					target_i = fv[:, :context_len]

				gt_coords_i = image_centroid(target_i.cpu())
				bs = fv.shape[0]
				val_sum_mse += F.mse_loss(pred_i, target_i).item() * bs
				val_sum_coord_mse += (
					F.mse_loss(pred_coords_i, gt_coords_i).item() * bs
				)
				val_n += bs

				if vis_pred is None:
					vis_pred = pred_i[:N_VID].cpu()
					vis_pred_coords = pred_coords_i[:N_VID]
					qs_vid = [q[:N_VID] for q in qs_i]
					ps_vid = [p[:N_VID] for p in ps_i]

		val_mse = val_sum_mse / val_n
		val_coord_mse = val_sum_coord_mse / val_n

		if self.rollout_direction == "backward":
			q_gt_val = self.val_q[:N_VID, :context_len].flip(dims=[1]).cpu()
		else:
			q_gt_val = self.val_q[:N_VID, :context_len].cpu()

		self._last_vis_pred = vis_pred
		self._last_vis_pred_coords = vis_pred_coords
		self._last_q_gt_val = q_gt_val
		self._last_qs_val = qs_vid
		self._last_ps_val = ps_vid

		if self._validate_hook is not None:
			self._validate_hook(self, epoch)

		tqdm.write(
			f"  epoch {epoch + 1:3d}  val_mse={val_mse:.4f}"
			f"  coord_mse={val_coord_mse:.4f}"
		)
		return {"val/pixel_mse": val_mse, "val/coord_mse": val_coord_mse}
