"""Model protocols for structural type-checking.

These Protocols formalise the interface shared by all HGN model classes
without requiring any changes to existing model files.  Python's structural
subtyping (typing.Protocol) means any class with the right methods satisfies
the protocol — no explicit inheritance needed.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class HGNModel(Protocol):
    """Protocol satisfied structurally by HGN, HGN_LSTM, DHGN, DHGN_LSTM."""

    def forward(
        self, imgs: torch.Tensor
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Encode context frames.

        Returns:
            (q0, p0, kl, mu, log_var)
        """
        ...

    def rollout(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        n_steps: int,
        dt: float | None = None,
        return_states: bool = False,
    ) -> tuple | list:
        """Roll out dynamics for n_steps from initial (q, p)."""
        ...

    def H(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Evaluate the Hamiltonian H(q, p) -> (B,)."""
        ...


@runtime_checkable
class ControlledHGNModel(HGNModel, Protocol):
    """Extended protocol for models with a control input port (PHGN family)."""

    def controlled_step(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        u: torch.Tensor,
        dt: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one controlled dynamics step."""
        ...

    def decode_state(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Decode latent (q, p) → observed state vector (B, obs_state_dim)."""
        ...

    def encode_mean(
        self, imgs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Deterministic encoding (no reparameterisation) for planning."""
        ...
