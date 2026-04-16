"""Model Predictive Path Integral (MPPI) controller.

Williams, G. et al. (2017) "Information Theoretic MPC for Model-Based
Reinforcement Learning."  ICRA 2017.

The controller maintains a running warm-start control sequence
U ∈ R^{H × m} and refines it each call to `plan` by importance-weighted
averaging over K sample trajectories drawn from a Gaussian perturbation
distribution centred on U:

    V_k = clip(U + ε_k,  u_min, u_max),   ε_k ~ N(0, σ²I)
    w_k  = exp(−(J_k − min J) / λ)        (shifted for numerical stability)
    U   ← U + Σ_k (w_k / Σ w) · ε_k

The cost function is decoupled from this class: any callable
``cost_fn(qs, ps) → Tensor (K,)`` is accepted, where qs and ps are
lists of (K, pos_ch, 4, 4) tensors — one per horizon step including
the initial state.

Usage:
    planner = MPPI(model, horizon=20, n_samples=512, ...)

    # Inside the control loop:
    action = planner.plan(q0, p0, cost_fn)
    planner.reset()   # call at episode start

The action is a float tensor of shape (control_dim,).  Map to a discrete
gym action externally if needed (e.g. sign → {0, 1} for CartPole).
"""

import torch


class MPPI:
	"""Model Predictive Path Integral controller.

	Args:
	    model:        ControlledDissipativeHGN (or compatible) instance.
	                  Must expose `.controlled_step(q, p, u)`.
	    horizon:      planning horizon H
	    n_samples:    number of sample trajectories K
	    temperature:  MPPI temperature λ (lower → greedier exploitation)
	    noise_sigma:  std-dev of per-step perturbation noise
	    control_dim:  dimension of the control vector
	    control_min:  lower bound on control (scalar or tensor)
	    control_max:  upper bound on control (scalar or tensor)
	    device:       torch device for all tensors
	"""

	def __init__(
		self,
		model,
		horizon: int = 20,
		n_samples: int = 256,
		temperature: float = 0.05,
		noise_sigma: float = 0.5,
		control_dim: int = 1,
		control_min: float = -1.0,
		control_max: float = 1.0,
		device: str | torch.device = "cpu",
	):
		self.model = model
		self.horizon = horizon
		self.n_samples = n_samples
		self.temperature = temperature
		self.noise_sigma = noise_sigma
		self.control_dim = control_dim
		self.control_min = control_min
		self.control_max = control_max
		self.device = torch.device(device)

		# Warm-start control sequence; refined in-place each call.
		self.U = torch.zeros(horizon, control_dim, device=self.device)

	def reset(self) -> None:
		"""Reset warm-start to zero (call at episode start)."""
		self.U.zero_()

	@torch.no_grad()
	def plan(
		self,
		q0: torch.Tensor,
		p0: torch.Tensor,
		cost_fn,
	) -> torch.Tensor:
		"""Refine U and return the first action to apply.

		Args:
		    q0:      (1, pos_ch, 4, 4) initial latent position
		    p0:      (1, pos_ch, 4, 4) initial latent momentum
		    cost_fn: callable(qs, ps) → (K,) cost tensor.
		             qs, ps are lists of (K, ...) tensors, one per step
		             (H+1 entries including the initial state).

		Returns:
		    action: (control_dim,) tensor — action to apply this step.
		"""
		K = self.n_samples

		# Broadcast initial state to all K samples.
		q = q0.expand(K, *q0.shape[1:]).clone()
		p = p0.expand(K, *p0.shape[1:]).clone()

		# Sample perturbations ε_k ~ N(0, σ²I).
		eps = (
			torch.randn(K, self.horizon, self.control_dim, device=self.device)
			* self.noise_sigma
		)

		# Perturbed and clipped control sequences: (K, H, control_dim).
		V = (self.U.unsqueeze(0) + eps).clamp(
			self.control_min, self.control_max
		)

		# Roll out all K trajectories under the world model.
		qs, ps = [q], [p]
		for t in range(self.horizon):
			q, p = self.model.controlled_step(q, p, V[:, t])
			qs.append(q)
			ps.append(p)

		# Evaluate costs: (K,)
		costs = cost_fn(qs, ps)

		# Importance weights (numerically stable via beta shift).
		beta = costs.min()
		w = torch.exp(-(costs - beta) / (self.temperature + 1e-8))
		w = w / (w.sum() + 1e-10)

		# Weighted update of the running control sequence.
		delta = (w.view(K, 1, 1) * eps).sum(0)  # (H, control_dim)
		self.U = (self.U + delta).clamp(self.control_min, self.control_max)

		# Extract the first action, then shift the window.
		action = self.U[0].clone()
		self.U = torch.roll(self.U, -1, dims=0)
		self.U[-1].zero_()

		return action
