"""Generic MPPI (Model Predictive Path Integral) planning core.

Domain-agnostic: knows nothing about pendulums, dynamics models, or cost
functions. A caller supplies ``rollout_cost_fn``, a closure that rolls
candidate action sequences forward through whatever dynamics/cost it wants
and returns one scalar cost per candidate.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MPPIConfig:
    horizon: int
    n_samples: int
    n_iterations: int = 1
    noise_std: float = 1.0
    temperature: float = 1.0
    action_low: float = -2.0
    action_high: float = 2.0


def mppi_plan(
    init_mean: torch.Tensor,
    rollout_cost_fn,
    cfg: MPPIConfig,
) -> torch.Tensor:
    """Optimize an action-sequence mean via MPPI.

    Args:
        init_mean: (H, action_dim) current best guess (typically the shifted
            mean from the previous planning step, for warm-starting).
        rollout_cost_fn: callable, (K, H, action_dim) candidate actions ->
            (K,) total cost per candidate.
        cfg: MPPIConfig.

    Returns:
        (H, action_dim) updated mean action sequence.
    """
    mean = init_mean.clone()
    for _ in range(cfg.n_iterations):
        eps = torch.randn(
            cfg.n_samples, *mean.shape, device=mean.device, dtype=mean.dtype
        ) * cfg.noise_std
        candidates = (mean.unsqueeze(0) + eps).clamp(cfg.action_low, cfg.action_high)
        costs = rollout_cost_fn(candidates)  # (K,)
        # Subtract the min before the softmax: shift-invariant, just keeps
        # exp() away from overflow when costs are large (e.g. pixel-space
        # sums vs. angle-space sums live on very different scales).
        weights = torch.softmax(-(costs - costs.min()) / cfg.temperature, dim=0)
        mean = torch.einsum("k,kha->ha", weights, candidates)
    return mean


def shift_mean(mean: torch.Tensor) -> torch.Tensor:
    """Warm-start the next planning step: drop the executed action, pad the tail."""
    return torch.cat([mean[1:], mean[-1:].clone()], dim=0)
