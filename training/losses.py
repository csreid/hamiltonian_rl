"""Composable loss functions for HGN training."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class LossConfig:
    """Weights for each loss term. Zero weight skips computation.

    Mirrors the CLI flags already present in shared_options.
    """

    recon_weight: float = 1.0
    kl_weight: float = 1e-3
    free_bits: float = 1.0
    coord_weight: float = 0.0
    energy_weight: float = 0.0
    state_weight: float = 0.0


@dataclass
class LossResult:
    """All computed loss terms and the total scalar loss."""

    total: torch.Tensor
    recon: float = 0.0
    kl: float = 0.0
    coord: float = 0.0
    energy: float = 0.0
    state: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "train/loss": self.total.item(),
            "train/recon_loss": self.recon,
            "train/kl_loss": self.kl,
            "train/coord_loss": self.coord,
            "train/energy_loss": self.energy,
            "train/state_loss": self.state,
        }


def compute_standard_loss(
    pred_frames: torch.Tensor,
    target_frames: torch.Tensor,
    kl: torch.Tensor,
    cfg: LossConfig,
    pred_coords: torch.Tensor | None = None,
    gt_coords: torch.Tensor | None = None,
    qs: list | None = None,
    ps: list | None = None,
    H_fn=None,
    q0: torch.Tensor | None = None,
    p0: torch.Tensor | None = None,
) -> LossResult:
    """Compute the composite reconstruction + KL + optional auxiliary loss.

    Args:
        pred_frames:  (B, T, C, H, W) predicted frames
        target_frames: (B, T, C, H, W) ground-truth frames
        kl:           (B,) per-sample KL divergence
        cfg:          weight configuration
        pred_coords:  (B, T, 2) predicted centroid coordinates (optional)
        gt_coords:    (B, T, 2) ground-truth centroid coordinates (optional)
        qs, ps:       list of (B, ...) latent tensors per rollout step (optional)
        H_fn:         callable H_fn(q, p) -> (B,) for energy conservation loss
        q0, p0:       initial latent state for energy baseline (optional)
    """
    recon_loss = F.mse_loss(pred_frames, target_frames)
    kl_loss = kl.clamp(min=cfg.free_bits).mean()
    total = cfg.recon_weight * recon_loss + cfg.kl_weight * kl_loss

    coord_val = energy_val = state_val = 0.0

    if (
        cfg.coord_weight > 0
        and pred_coords is not None
        and gt_coords is not None
    ):
        coord_loss = F.mse_loss(pred_coords, gt_coords)
        total = total + cfg.coord_weight * coord_loss
        coord_val = coord_loss.item()

    if (
        cfg.energy_weight > 0
        and qs is not None
        and ps is not None
        and H_fn is not None
        and q0 is not None
        and p0 is not None
    ):
        H0 = H_fn(q0.detach(), p0.detach()).detach()
        H_traj = torch.stack([H_fn(q, p) for q, p in zip(qs, ps)], dim=1)
        energy_loss = (H_traj - H0.unsqueeze(1)).pow(2).mean()
        total = total + cfg.energy_weight * energy_loss
        energy_val = energy_loss.item()

    return LossResult(
        total=total,
        recon=recon_loss.item(),
        kl=kl_loss.item(),
        coord=coord_val,
        energy=energy_val,
        state=state_val,
    )
