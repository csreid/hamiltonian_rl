"""Checkpointing: run directories and the unified single-file checkpoint formats.

Two self-describing formats, both a single ``.pt`` holding model weights plus
everything needed to rebuild them — no YAML sidecar or out-of-band
hyperparameters required to load. ``kind`` distinguishes them so a loader
given the wrong file fails with a clear error instead of a shape mismatch.

World model (pixel pipeline — ``save_world_model`` / ``load_world_model``):

    {
      "format_version": 1,
      "kind": "world_model",
      "config": {
        "autoencoder": {...TemporalAutoencoder ctor args...},
        "dynamics":    {...HamiltonianFlowModel ctor args...} | None,
        "data":        {...how the training episodes were collected...},
      },
      "autoencoder": <state_dict>,
      "dynamics":    <state_dict> | None,
      "hparams": {...}, "metrics": {...}, "epoch": int,
    }

Phase 1 writes ``dynamics: None``; Phase 2 fills it in, so its checkpoint is
the one file the dashboard needs.

State model (ground-truth phase-space pipeline — ``save_state_model`` /
``load_state_model``):

    {
      "format_version": 1,
      "kind": "state_model",
      "config": {
        "model": {...StatePHGN ctor args...},
        "data":  {...how the training episodes were collected...},
      },
      "model": <state_dict>,
      "hparams": {...}, "metrics": {...}, "epoch": int,
    }

Both also get a YAML sidecar (hparams + metrics only) for human eyeballing.
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


def _load_checked(path, expected_kind: str, device: torch.device | None) -> dict:
    path = Path(path)
    ckpt = torch.load(path, map_location=device or "cpu", weights_only=True)
    if not isinstance(ckpt, dict) or "format_version" not in ckpt:
        raise ValueError(
            f"{path} is not a unified checkpoint. "
            "Re-train with the current pipeline (old-format run dirs are obsolete)."
        )
    kind = ckpt.get("kind", "world_model")  # pre-"kind" checkpoints are all world models
    if kind != expected_kind:
        raise ValueError(f"{path} is a {kind!r} checkpoint, not {expected_kind!r}.")
    return ckpt


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
        "kind": "world_model",
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
    from hamilton_rl.models import HamiltonianFlowModel, TemporalAutoencoder, WorldModel

    ckpt = _load_checked(path, "world_model", device)
    config = ckpt["config"]
    autoencoder = TemporalAutoencoder(**config["autoencoder"])
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


def save_state_model(
    run_dir: Path,
    stem: str,
    model,
    hparams: dict,
    metrics: dict,
    epoch: int,
    data_config: dict | None = None,
) -> None:
    """Save a StatePHGN (ground-truth phase-space dynamics) as one .pt file."""
    payload = {
        "format_version": FORMAT_VERSION,
        "kind": "state_model",
        "config": {
            "model": model.config,
            "data": data_config or {},
        },
        "model": model.state_dict(),
        "hparams": hparams,
        "metrics": metrics,
        "epoch": epoch,
    }
    torch.save(payload, Path(run_dir) / f"{stem}.pt")
    _write_yaml_sidecar(Path(run_dir), stem, hparams, metrics)


def load_state_model(path, device: torch.device | None = None):
    """Load a unified checkpoint into a StatePHGN."""
    from hamilton_rl.models import StatePHGN

    ckpt = _load_checked(path, "state_model", device)
    model = StatePHGN(**ckpt["config"]["model"])
    model.load_state_dict(ckpt["model"])
    if device is not None:
        model = model.to(device)
    model.eval()
    model.data_config = ckpt["config"].get("data") or {}
    return model


def save_projected_model(
    run_dir: Path,
    stem: str,
    model,
    projection: torch.Tensor,
    hparams: dict,
    metrics: dict,
    epoch: int,
    data_config: dict | None = None,
) -> None:
    """Save a HamiltonianFlowModel trained on a noisy random-linear-projection
    proxy for a pixel encoder's latent (see
    ``experiments/pendulum_projected_offline.py``). ``projection`` is the
    fixed (D, 2) matrix mapping ground-truth (θ, θ̇) into the D-dim latent —
    saved alongside the model so eval can reproduce the exact same latent
    space."""
    payload = {
        "format_version": FORMAT_VERSION,
        "kind": "projected_model",
        "config": {
            "model": model.config,
            "data": data_config or {},
        },
        "model": model.state_dict(),
        "projection": projection,
        "hparams": hparams,
        "metrics": metrics,
        "epoch": epoch,
    }
    torch.save(payload, Path(run_dir) / f"{stem}.pt")
    _write_yaml_sidecar(Path(run_dir), stem, hparams, metrics)


def load_projected_model(path, device: torch.device | None = None):
    """Load a unified checkpoint into a (HamiltonianFlowModel, projection) pair."""
    from hamilton_rl.models import HamiltonianFlowModel

    ckpt = _load_checked(path, "projected_model", device)
    model = HamiltonianFlowModel(**ckpt["config"]["model"])
    model.load_state_dict(ckpt["model"])
    if device is not None:
        model = model.to(device)
    model.eval()
    model.data_config = ckpt["config"].get("data") or {}
    return model, ckpt["projection"]
