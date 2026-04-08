"""Dissipative Hamiltonian Generative Network (DHGN).

Extends HGN (Toth et al., ICLR 2020) to port-Hamiltonian systems with
dissipation.  The leapfrog integrator is replaced by an RK4 integration of
the port-Hamiltonian ODE:

    dz/dt = (J − R) ∇_z H(z)

where z = [q; p] is the flattened phase-space state and:

    J = A − Aᵀ        skew-symmetric  (conservative, symplectic structure)
    R = L Lᵀ          positive semi-definite  (dissipative)

A and L are learned parameter matrices of shape (D, D), D = 2·pos_ch·4·4.
L is lower-triangular with a softplus-constrained diagonal so that R ⪰ 0
is guaranteed regardless of the values of L_param.

For a damped harmonic oscillator the ground-truth structure is:
    J = [[0, I], [−I, 0]]
    R = [[0, 0], [0, γ·I]]
so the model has sufficient capacity to recover the correct physics.

Gradient convention: ∂H/∂q and ∂H/∂p are computed via autograd with the
same detach-then-create_graph pattern used in the parent HGN leapfrog.
This lets gradients flow through J, R and the Hamiltonian network during
training, while avoiding double-backward through the state trajectory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from hgn_org import HGN


class DissipativeHGN(HGN):
    """Port-Hamiltonian Generative Network with learned dissipation.

    Inherits the encoder, f_ψ state transform, Hamiltonian network and
    decoder from HGN.  Adds two (D × D) parameter matrices A and L_param
    that define the structure J and dissipation R:

        J      = A − Aᵀ                    (skew-symmetric)
        L      = tril(L_param),  diag → softplus
        R      = L Lᵀ                       (positive semi-definite)

    The forward pass and rollout are inherited; only the integrator is
    replaced.

    Args:
        n_frames: context frames for the encoder  (same as HGN)
        pos_ch:   channel depth for q (and p)     (same as HGN)
        img_ch:   image channels                   (same as HGN)
        dt:       default integration step size    (same as HGN)
    """

    def __init__(
        self,
        n_frames: int = 4,
        pos_ch: int = 16,
        img_ch: int = 3,
        dt: float = 0.125,
    ):
        super().__init__(n_frames=n_frames, pos_ch=pos_ch, img_ch=img_ch, dt=dt)

        D = self.latent_ch * 4 * 4  # 2 * pos_ch * 16
        self.state_dim = D

        # ── Structure matrices ──────────────────────────────────────────────
        # A  → J = A − Aᵀ  (skew-symmetric by construction)
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
        """Return the lower-triangular factor L with positive diagonal.

        L = strictly-lower-tri(L_param) + diag(softplus(diag(L_param)))

        Separating the strictly-lower part from the diagonal is cleaner
        than masking in-place, and is fully differentiable.
        """
        L_lower = self.L_param.tril(-1)  # off-diagonal lower tri
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

        Detaches q and p before differentiating H (same pattern as the
        parent leapfrog) so that gradients flow through the Hamiltonian
        network and M, but not back through the state trajectory.

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

        H_val = self.H(q_, p_).sum()
        dH_dq, dH_dp = torch.autograd.grad(
            H_val, [q_, p_], create_graph=self.training
        )

        # Flatten ∇H = [∂H/∂q; ∂H/∂p]  →  (B, D)
        half = self.state_dim // 2
        grad_H = torch.cat(
            [dH_dq.reshape(B, half), dH_dp.reshape(B, half)], dim=1
        )

        # dz/dt = M ∇H,  einsum handles batch broadcast
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

        M = J − R is computed once per call; the four RK4 sub-evaluations
        all share the same M so structure-matrix gradients are summed
        across the sub-steps.

        Args:
            q, p: phase-space state, each (B, pos_ch, 4, 4)
            dt:   step size override; if None uses self.dt

        Returns:
            q_next, p_next: (B, pos_ch, 4, 4)
        """
        if dt is None:
            dt = self.dt

        M = self.get_J_minus_R()  # (D, D) — computed once, shared by all ki

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
        """Decode initial state, then roll out using dissipative dynamics.

        Identical interface to HGN.rollout; uses dissipative_step in place
        of leapfrog_step.
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
