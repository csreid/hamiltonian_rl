"""Dissipative HGN with LSTM encoder (DHGN-LSTM).

Combines the recurrent encoder from HGN-LSTM with the port-Hamiltonian
dissipative dynamics from DissipativeHGN:

    FrameCNN (per frame) → LSTM (chronological) → hidden state at frame T
    → reshape → mu/log_var → reparameterise → f_ψ → (qT, pT)

    Rollout: dz/dt = (J − R) ∇H,  integrated with RK4 at ±dt

Because the LSTM hidden state represents the *current* phase-space position
(frame T), training rollouts are run backwards in time (dt < 0) to reconstruct
the history seen by the encoder.

Structure matrices:
    J      = A − Aᵀ            skew-symmetric  (conservative)
    L      = tril(L_param),  diag → softplus
    R      = L Lᵀ              positive semi-definite  (dissipative)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from hgn_lstm import HGN_LSTM


class DHGN_LSTM(HGN_LSTM):
    """Port-Hamiltonian HGN with LSTM encoder and learned dissipation.

    Inherits the LSTM encoder, f_ψ state transform, Hamiltonian network,
    decoder and coord head from HGN_LSTM.  Replaces the leapfrog integrator
    with an RK4 step driven by the port-Hamiltonian ODE:

        dz/dt = (J − R) ∇_z H(z)

    where J is skew-symmetric (conservative) and R is positive semi-definite
    (dissipative).  Both are learned from data.

    Args:
        pos_ch:   channel depth for q (and p)
        img_ch:   image channels
        dt:       default integration step size
        feat_dim: per-frame CNN embedding size fed to the LSTM
    """

    def __init__(
        self,
        pos_ch: int = 16,
        img_ch: int = 3,
        dt: float = 0.125,
        feat_dim: int = 256,
        separable: bool = True,
    ):
        super().__init__(
            pos_ch=pos_ch,
            img_ch=img_ch,
            dt=dt,
            feat_dim=feat_dim,
            separable=separable,
        )

        D = self.latent_ch * 4 * 4  # 2 * pos_ch * 16
        self.state_dim = D

        # ── Structure matrices ──────────────────────────────────────────────
        # A      → J = A − Aᵀ  (skew-symmetric by construction)
        # L_param → L = tril with softplus diagonal → R = L Lᵀ (PSD)
        self.A = nn.Parameter(torch.zeros(D, D))
        self.L_param = nn.Parameter(torch.zeros(D, D))

        nn.init.normal_(self.A, std=1e-2)
        nn.init.normal_(self.L_param, std=1e-2)

    # ── Structure matrix helpers ────────────────────────────────────────────

    def get_J(self) -> torch.Tensor:
        """Return the skew-symmetric J = A − Aᵀ, shape (D, D)."""
        return self.A - self.A.T

    def get_L(self) -> torch.Tensor:
        """Return the lower-triangular factor L with positive diagonal."""
        L_lower = self.L_param.tril(-1)
        diag_pos = F.softplus(self.L_param.diagonal())
        return L_lower + torch.diag(diag_pos)

    def get_R(self) -> torch.Tensor:
        """Return R = L Lᵀ, positive semi-definite, shape (D, D)."""
        L = self.get_L()
        return L @ L.T

    def get_J_minus_R(self) -> torch.Tensor:
        """Return M = J − R, the combined structure matrix."""
        return self.get_J() - self.get_R()

    # ── Dynamics ────────────────────────────────────────────────────────────

    def _dynamics(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        M: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute dz/dt = (J − R) ∇H at (q, p).

        Args:
            q: (B, pos_ch, 4, 4)
            p: (B, pos_ch, 4, 4)
            M: (D, D)  — J − R, pre-computed for the step

        Returns:
            dq: (B, pos_ch, 4, 4)
            dp: (B, pos_ch, 4, 4)
        """
        B = q.shape[0]
        q_ = q.detach().requires_grad_(True)
        p_ = p.detach().requires_grad_(True)

        q_n = q_
        p_n = p_

        H_val = self.H(q_n, p_n).sum()
        dH_dq, dH_dp = torch.autograd.grad(
            H_val, [q_, p_], create_graph=self.training
        )

        half = self.state_dim // 2
        grad_H = torch.cat(
            [dH_dq.reshape(B, half), dH_dp.reshape(B, half)], dim=1
        )  # (B, D)

        dz = torch.einsum("ij,bj->bi", M, grad_H)  # (B, D)

        dq = dz[:, :half].reshape_as(q)
        dp = dz[:, half:].reshape_as(p)
        return dq, dp

    @torch.enable_grad()
    def dissipative_step(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        dt: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One RK4 step of the port-Hamiltonian ODE dz/dt = (J − R) ∇H.

        Passing dt < 0 integrates backward in time, which is used during
        training to reconstruct the history from the current state.

        Args:
            q, p: phase-space state, each (B, pos_ch, 4, 4)
            dt:   step size (negative for backward integration)

        Returns:
            q_next, p_next: (B, pos_ch, 4, 4)
        """
        if dt is None:
            dt = self.dt

        M = self.get_J_minus_R()  # computed once, shared by all RK4 sub-steps

        dq1, dp1 = self._dynamics(q, p, M)
        dq2, dp2 = self._dynamics(q + 0.5 * dt * dq1, p + 0.5 * dt * dp1, M)
        dq3, dp3 = self._dynamics(q + 0.5 * dt * dq2, p + 0.5 * dt * dp2, M)
        dq4, dp4 = self._dynamics(q + dt * dq3, p + dt * dp3, M)

        q_next = q + (dt / 6.0) * (dq1 + 2 * dq2 + 2 * dq3 + dq4)
        p_next = p + (dt / 6.0) * (dp1 + 2 * dp2 + 2 * dp3 + dp4)
        return q_next, p_next

    # ── Rollout override ────────────────────────────────────────────────────

    def rollout(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        n_steps: int,
        dt: float | None = None,
        return_states: bool = False,
    ):
        """Decode initial state then roll out using dissipative RK4 dynamics.

        Pass dt < 0 to integrate backward in time.
        """
        frames = [self.decoder(q)]
        coords = [self.coord_head(q.detach())]
        qs = [q]
        ps = [p]
        for _ in range(n_steps):
            q, p = self.dissipative_step(q, p, dt=dt)
            frames.append(self.decoder(q))
            coords.append(self.coord_head(q.detach()))
            qs.append(q)
            ps.append(p)
        if return_states:
            return frames, coords, qs, ps
        return frames, coords
