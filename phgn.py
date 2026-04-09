"""Port-Hamiltonian Generative Network with control input (PHGN).

Extends DissipativeHGN to include a control input port B:

    dz/dt = (J − R) ∇H(z) + B u

where:
    B ∈ R^{D × m}   learned input matrix (m = control_dim)
    u ∈ R^m          control signal (zero-order hold over one dt step)

This corresponds to the standard port-Hamiltonian formulation with an
explicit input port.  For a damped spring actuated at the cart:
    B ≈ [0; 0; ...; 1; ...; 0]   (only the position channel is actuated)
"""

import torch
import torch.nn as nn

from dhgn import DissipativeHGN


class ControlledDissipativeHGN(DissipativeHGN):
	"""Port-Hamiltonian network with learned control input port B.

	Inherits encoder, f_ψ, Hamiltonian network, decoder, J, R from
	DissipativeHGN.  Adds:

	* B  (D × control_dim) — input matrix, state-independent

	The controlled ODE is integrated with RK4 (zero-order hold on u).

	Args:
	    control_dim: dimension of the control input u
	"""

	def __init__(
		self,
		n_frames: int = 4,
		pos_ch: int = 4,
		img_ch: int = 3,
		dt: float = 0.05,
		control_dim: int = 1,
	):
		super().__init__(n_frames=n_frames, pos_ch=pos_ch, img_ch=img_ch, dt=dt)
		self.control_dim = control_dim

		# B: input matrix.  Small init so control starts negligible.
		self.B = nn.Parameter(torch.zeros(self.state_dim, control_dim))
		nn.init.normal_(self.B, std=1e-2)

	# ── Controlled dynamics ──────────────────────────────────────────────

	def _controlled_dynamics(
		self,
		q: torch.Tensor,
		p: torch.Tensor,
		u: torch.Tensor,
		M: torch.Tensor,
	) -> tuple[torch.Tensor, torch.Tensor]:
		"""Compute dz/dt = (J − R) ∇H + B u.

		Args:
		    q, p: (B, pos_ch, 4, 4)
		    u:    (B, control_dim)
		    M:    (D, D) — J − R, pre-computed for the step

		Returns:
		    dq, dp: (B, pos_ch, 4, 4)
		"""
		dq, dp = self._dynamics(q, p, M)

		# Control contribution: B u split into q and p parts.
		Bu = u @ self.B.T  # (B_batch, D)
		half = self.state_dim // 2
		dq = dq + Bu[:, :half].reshape_as(q)
		dp = dp + Bu[:, half:].reshape_as(p)
		return dq, dp

	@torch.enable_grad()
	def controlled_step(
		self,
		q: torch.Tensor,
		p: torch.Tensor,
		u: torch.Tensor,
		dt: float | None = None,
	) -> tuple[torch.Tensor, torch.Tensor]:
		"""One RK4 step of dz/dt = (J − R) ∇H + B u.

		The control u is held constant over the step (zero-order hold).

		Args:
		    q, p: phase-space state, each (B, pos_ch, 4, 4)
		    u:    control input (B, control_dim)
		    dt:   step size override; if None uses self.dt

		Returns:
		    q_next, p_next: (B, pos_ch, 4, 4)
		"""
		if dt is None:
			dt = self.dt

		M = self.get_J_minus_R()  # computed once; shared across ki

		dq1, dp1 = self._controlled_dynamics(q, p, u, M)
		dq2, dp2 = self._controlled_dynamics(
			q + 0.5 * dt * dq1, p + 0.5 * dt * dp1, u, M
		)
		dq3, dp3 = self._controlled_dynamics(
			q + 0.5 * dt * dq2, p + 0.5 * dt * dp2, u, M
		)
		dq4, dp4 = self._controlled_dynamics(q + dt * dq3, p + dt * dp3, u, M)

		q_next = q + (dt / 6.0) * (dq1 + 2 * dq2 + 2 * dq3 + dq4)
		p_next = p + (dt / 6.0) * (dp1 + 2 * dp2 + 2 * dp3 + dp4)
		return q_next, p_next

	def rollout_controlled(
		self,
		q: torch.Tensor,
		p: torch.Tensor,
		us: torch.Tensor,
		dt: float | None = None,
		return_states: bool = False,
	):
		"""Roll out from (q, p) applying control sequence us.

		Args:
		    q, p: initial state (B, pos_ch, 4, 4)
		    us:   (B, H, control_dim) — control at each of H steps
		    dt:   step size override

		Returns:
		    frames:        list of H+1 decoded images
		    qs, ps:        (if return_states) lists of H+1 state tensors
		"""
		frames = [self.decoder(q)]
		qs, ps = [q], [p]
		for t in range(us.shape[1]):
			u_t = us[:, t]
			q, p = self.controlled_step(q, p, u_t, dt=dt)
			frames.append(self.decoder(q))
			qs.append(q)
			ps.append(p)
		if return_states:
			return frames, qs, ps
		return frames

	# ── Deterministic encoding for eval ─────────────────────────────────

	def encode_mean(
		self, imgs: torch.Tensor
	) -> tuple[torch.Tensor, torch.Tensor]:
		"""Encode using the posterior mean — no reparameterisation noise.

		Args:
		    imgs: (B, T, C, H, W)

		Returns:
		    q0, p0: (B, pos_ch, 4, 4)
		"""
		B, T, C, H, W = imgs.shape
		stacked = imgs.reshape(B, T * C, H, W)
		mu, _ = self.encoder(stacked)
		s0 = self.f_psi(mu)
		return self._split(s0)
