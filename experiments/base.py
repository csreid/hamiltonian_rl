"""Base experiment class and config for HGN benchmark experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from checkpoint_common import make_run_dir
from training.trainer import TrainerConfig, BaseTrainer


@dataclass
class ExperimentConfig:
    """All settings for one experiment run.

    Combines physics/data params (not in TrainerConfig) with all trainer params.
    Populated entirely from Click kwargs — no logic here.
    """

    # Physics / dataset
    img_size: int = 32
    blob_sigma: float = 2.0
    seq_len: int = 31
    n_frames: int = 31
    train_rollout: int = 30
    dt: float = 0.125
    n_train: int = 50000
    n_val: int = 10000
    max_amplitude: float = 1.0
    spring_constant: float = 2.0
    mass: float = 0.5

    # Training (forwarded to TrainerConfig)
    batch_size: int = 16
    n_epochs: int = 5
    lr: float = 1.5e-4
    kl_weight: float = 1e-3
    recon_weight: float = 1.0
    free_bits: float = 1.0
    coord_weight: float = 0.0
    energy_weight: float = 0.0
    state_weight: float = 0.0
    grad_clip: float = 1.0
    log_every: int = 10
    diag_every: int = 1

    # Model-specific extras
    model_kwargs: dict = field(default_factory=dict)

    @classmethod
    def from_kwargs(
        cls, kwargs: dict, model_kwargs: dict | None = None
    ) -> "ExperimentConfig":
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        cfg_kwargs = {k: v for k, v in kwargs.items() if k in fields}
        cfg_kwargs["model_kwargs"] = model_kwargs or {}
        return cls(**cfg_kwargs)


class Experiment:
    """Wires together data generation, model construction, and trainer.

    Subclasses override:
        build_model(cfg)    -> nn.Module
        build_datasets(cfg) -> (train_loader, val_frames, val_q, ...)
        build_trainer(...)  -> BaseTrainer
        tb_comment()        -> str  (appended to TensorBoard run name)

    The run() entry point is provided and should not be overridden.
    """

    def build_model(self, cfg: ExperimentConfig) -> nn.Module:
        raise NotImplementedError

    def build_datasets(self, cfg: ExperimentConfig):
        """Returns (dataset, *val_data).

        dataset must satisfy the TrainingDataset Protocol (DataLoaderAdapter for
        static SHO datasets, PrioritizedEpisodeReplayBuffer for CartPole).
        val_data is experiment-specific and forwarded to build_trainer().
        """
        raise NotImplementedError

    def build_trainer(
        self,
        cfg: ExperimentConfig,
        trainer_cfg: TrainerConfig,
        model: nn.Module,
        dataset,
        writer: SummaryWriter,
        run_dir: Path,
        device: torch.device,
        hparams: dict,
        val_data: tuple,
    ) -> BaseTrainer:
        raise NotImplementedError

    def tb_comment(self) -> str:
        return ""

    def _setup(self) -> tuple:
        """Shared setup: device, TensorBoard writer, run directory."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        writer = SummaryWriter(comment=self.tb_comment())
        run_dir = make_run_dir(self.__class__.__name__.lower())
        return device, writer, run_dir

    def run(self, cfg: ExperimentConfig) -> None:
        device, writer, run_dir = self._setup()

        print("Generating datasets...")
        dataset, *val_data = self.build_datasets(cfg)

        model = self.build_model(cfg).to(device)
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        hparams = {k: v for k, v in vars(cfg).items() if k != "model_kwargs"}
        hparams.update(cfg.model_kwargs)

        trainer_cfg = TrainerConfig.from_kwargs(vars(cfg))
        trainer = self.build_trainer(
            cfg,
            trainer_cfg,
            model,
            dataset,
            writer,
            run_dir,
            device,
            hparams,
            tuple(val_data),
        )
        trainer.fit()
        print("\nDone. Run: tensorboard --logdir runs")
