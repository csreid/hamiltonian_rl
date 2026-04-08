"""Shared checkpointing utilities."""

from datetime import datetime
from pathlib import Path

import torch
import yaml


def make_run_dir(identifier: str) -> Path:
    """Create and return models/<identifier>/<timestamp>/."""
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path("models") / identifier / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_checkpoint(
    run_dir: Path,
    epoch: int,
    model: torch.nn.Module,
    hparams: dict,
    metrics: dict,
    stem: str | None = None,
) -> None:
    """Save model weights and a YAML sidecar with hparams + metrics.

    Args:
        stem: filename stem (no extension). Defaults to ``checkpoint_{epoch}``.
    """
    if stem is None:
        stem = f"checkpoint_{epoch}"
    torch.save(model.state_dict(), run_dir / f"{stem}.pt")
    with open(run_dir / f"{stem}.yaml", "w") as f:
        yaml.dump(
            {"hparams": hparams, "metrics": metrics},
            f,
            default_flow_style=False,
            sort_keys=False,
        )
