"""Checkpointing: run directories and the unified world-model format.

A world-model checkpoint is a single ``.pt`` holding both halves of the model
plus everything needed to rebuild them — no YAML sidecar or out-of-band
hyperparameters required to load:

    {
      "format_version": 1,
      "config": {
        "autoencoder": {...LSTMAutoencoder ctor args...},
        "dynamics":    {...HamiltonianFlowModel ctor args...} | None,
        "data":        {...how the training episodes were collected...},
      },
      "autoencoder": <state_dict>,
      "dynamics":    <state_dict> | None,
      "hparams": {...}, "metrics": {...}, "epoch": int,
    }

Phase 1 writes ``dynamics: None``; Phase 2 fills it in, so its checkpoint is
the one file the dashboard needs.  A YAML sidecar (hparams + metrics only) is
still written for human eyeballing.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch
import yaml

FORMAT_VERSION = 1


def make_run_dir(identifier: str) -> Path:
    """Create and return models/<identifier>/<timestamp>/."""
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path("models") / identifier / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_yaml_sidecar(run_dir: Path, stem: str, hparams: dict, metrics: dict) -> None:
    with open(run_dir / f"{stem}.yaml", "w") as f:
        yaml.dump(
            {"hparams": hparams, "metrics": metrics},
            f,
            default_flow_style=False,
            sort_keys=False,
        )


def save_checkpoint(
    run_dir: Path,
    epoch: int,
    model: torch.nn.Module,
    hparams: dict,
    metrics: dict,
    stem: str | None = None,
) -> None:
    """Save bare model weights and a YAML sidecar with hparams + metrics.

    Legacy format for models outside the world-model pipeline (e.g. the
    state-based experiment).  The world-model pipeline uses save_world_model.

    Args:
        stem: filename stem (no extension). Defaults to ``checkpoint_{epoch}``.
    """
    if stem is None:
        stem = f"checkpoint_{epoch}"
    torch.save(model.state_dict(), run_dir / f"{stem}.pt")
    _write_yaml_sidecar(run_dir, stem, hparams, metrics)


def save_world_model(
    run_dir: Path,
    stem: str,
    model,
    hparams: dict,
    metrics: dict,
    epoch: int,
) -> None:
    """Save a WorldModel (autoencoder + optional dynamics) as one .pt file."""
    payload = {
        "format_version": FORMAT_VERSION,
        "config": {
            "autoencoder": model.autoencoder.config,
            "dynamics": model.dynamics.config if model.dynamics is not None else None,
            "data": model.data_config,
        },
        "autoencoder": model.autoencoder.state_dict(),
        "dynamics": model.dynamics.state_dict() if model.dynamics is not None else None,
        "hparams": hparams,
        "metrics": metrics,
        "epoch": epoch,
    }
    torch.save(payload, Path(run_dir) / f"{stem}.pt")
    _write_yaml_sidecar(Path(run_dir), stem, hparams, metrics)


def load_world_model(path, device: torch.device | None = None):
    """Load a unified checkpoint into a WorldModel (dynamics may be None)."""
    from hamilton_rl.models import HamiltonianFlowModel, LSTMAutoencoder, WorldModel

    path = Path(path)
    ckpt = torch.load(path, map_location=device or "cpu", weights_only=True)
    if not isinstance(ckpt, dict) or "format_version" not in ckpt:
        raise ValueError(
            f"{path} is not a unified world-model checkpoint. "
            "Re-train with the current pipeline (old-format run dirs are obsolete)."
        )

    config = ckpt["config"]
    autoencoder = LSTMAutoencoder(**config["autoencoder"])
    autoencoder.load_state_dict(ckpt["autoencoder"])

    dynamics = None
    if ckpt["dynamics"] is not None:
        dynamics = HamiltonianFlowModel(**config["dynamics"])
        dynamics.load_state_dict(ckpt["dynamics"])

    model = WorldModel(autoencoder, dynamics, data_config=config.get("data") or {})
    if device is not None:
        model = model.to(device)
    model.eval()
    return model
