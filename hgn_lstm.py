import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Sub-modules (unchanged from hgn_org.py)
# ---------------------------------------------------------------------------


class StateTransform(nn.Module):
	"""f_ψ: 3-layer CNN that maps sampled z to the initial state s0."""

	def __init__(self, latent_ch: int = 32):
		super().__init__()
		self.net = nn.Sequential(
			nn.Conv2d(latent_ch, 64, 3, padding=1),
			nn.ReLU(),
			nn.Conv2d(64, 64, 3, padding=1),
			nn.ReLU(),
			nn.Conv2d(64, latent_ch, 3, padding=1),
		)

	def forward(self, z: torch.Tensor) -> torch.Tensor:
		return self.net(z)  # (B, latent_ch, 4, 4)


class KineticNet(nn.Module):
	"""T_γ(q, p): kinetic energy — 6-layer conv net over the full state [q; p]."""

	def __init__(self, latent_ch: int = 32):
		super().__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(latent_ch, 64, 3, padding=1),
			nn.Softplus(),
			nn.Conv2d(64, 64, 3, padding=1),
			nn.Softplus(),
			nn.Conv2d(64, 64, 3, padding=1),
			nn.Softplus(),
			nn.Conv2d(64, 64, 3, padding=1),
			nn.Softplus(),
			nn.Conv2d(64, 64, 3, padding=1),
			nn.Softplus(),
			nn.Conv2d(64, 64, 3, padding=1),
			nn.Softplus(),
		)
		self.to_scalar = nn.Conv2d(64, 1, 1)

	def forward(self, s: torch.Tensor) -> torch.Tensor:
		return self.to_scalar(self.conv(s)).sum(dim=[1, 2, 3])


class PotentialNet(nn.Module):
	"""V_γ(q): potential energy — 6-layer conv net over position q only."""

	def __init__(self, pos_ch: int = 16):
		super().__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(pos_ch, 64, 3, padding=1),
			nn.Softplus(),
			nn.Conv2d(64, 64, 3, padding=1),
			nn.Softplus(),
			nn.Conv2d(64, 64, 3, padding=1),
			nn.Softplus(),
			nn.Conv2d(64, 64, 3, padding=1),
			nn.Softplus(),
			nn.Conv2d(64, 64, 3, padding=1),
			nn.Softplus(),
			nn.Conv2d(64, 64, 3, padding=1),
			nn.Softplus(),
		)
		self.to_scalar = nn.Conv2d(64, 1, 1)

	def forward(self, q: torch.Tensor) -> torch.Tensor:
		return self.to_scalar(self.conv(q)).sum(dim=[1, 2, 3])


class HamiltonianNet(nn.Module):
	"""Convolutional Hamiltonian H_γ(q, p) = T_γ(q, p) + V_γ(q) → scalar."""

	def __init__(self, latent_ch: int = 32):
		super().__init__()
		pos_ch = latent_ch // 2
		self.kinetic = KineticNet(latent_ch=latent_ch)
		self.potential = PotentialNet(pos_ch=pos_ch)

	def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
		s = torch.cat([q, p], dim=1)
		return self.kinetic(s) + self.potential(q)


class FullHamiltonianNet(nn.Module):
	"""Convolutional Hamiltonian H_γ(q, p) — non-separable, takes full state [q; p]."""

	def __init__(self, latent_ch: int = 32):
		super().__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(latent_ch, 64, 3, padding=1),
			nn.Softplus(),
			nn.Conv2d(64, 64, 3, padding=1),
			nn.Softplus(),
			nn.Conv2d(64, 64, 3, padding=1),
			nn.Softplus(),
			nn.Conv2d(64, 64, 3, padding=1),
			nn.Softplus(),
			nn.Conv2d(64, 64, 3, padding=1),
			nn.Softplus(),
			nn.Conv2d(64, 64, 3, padding=1),
			nn.Softplus(),
		)
		self.to_scalar = nn.Conv2d(64, 1, 1)

	def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
		s = torch.cat([q, p], dim=1)
		return self.to_scalar(self.conv(s)).sum(dim=[1, 2, 3])


class CoordHead(nn.Module):
	"""Maps position q_t (B, pos_ch, 4, 4) -> pixel coords (B, 2) in [0, 1]."""

	def __init__(self, pos_ch: int = 16):
		super().__init__()
		self.net = nn.Sequential(
			nn.Flatten(),
			nn.Linear(pos_ch * 4 * 4, 64),
			nn.ReLU(),
			nn.Linear(64, 2),
			nn.Sigmoid(),
		)

	def forward(self, q: torch.Tensor) -> torch.Tensor:
		return self.net(q)


class DecoderBlock(nn.Module):
	"""One progressive upsampling block."""

	def __init__(self, in_ch: int):
		super().__init__()
		self.conv1 = nn.Conv2d(in_ch, 64, 3, padding=1)
		self.conv2 = nn.Conv2d(64, 64, 3, padding=1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = F.interpolate(x, scale_factor=2, mode="nearest")
		x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
		x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
		return torch.sigmoid(x)


class Decoder(nn.Module):
	"""Progressive decoder: q_t → reconstructed image."""

	def __init__(self, pos_ch: int = 16, img_ch: int = 3):
		super().__init__()
		self.block1 = DecoderBlock(pos_ch)  # 4  → 8
		self.block2 = DecoderBlock(64)  # 8  → 16
		self.block3 = DecoderBlock(64)  # 16 → 32
		self.out_conv = nn.Conv2d(64, img_ch, 1)

	def forward(self, q: torch.Tensor) -> torch.Tensor:
		x = self.block1(q)
		x = self.block2(x)
		x = self.block3(x)
		return self.out_conv(x)


# ---------------------------------------------------------------------------
# LSTM encoder
# ---------------------------------------------------------------------------


class FrameCNN(nn.Module):
	"""Per-frame CNN: (B, 3, 32, 32) → (B, feat_dim).

	Mirrors the first few layers of the HGN encoder but processes one frame
	at a time, producing a flat feature vector that feeds the LSTM.
	"""

	def __init__(self, img_ch: int = 3, feat_dim: int = 256):
		super().__init__()
		self.net = nn.Sequential(
			# 32×32 → 16×16
			nn.Conv2d(img_ch, 32, 3, stride=2, padding=1),
			nn.ReLU(),
			# 16×16 → 8×8
			nn.Conv2d(32, 64, 3, stride=2, padding=1),
			nn.ReLU(),
			# 8×8 → 4×4
			nn.Conv2d(64, 64, 3, stride=2, padding=1),
			nn.ReLU(),
			nn.Conv2d(64, 64, 3, padding=1),
			nn.ReLU(),
			nn.Flatten(),  # 64 * 4 * 4 = 1024
			nn.Linear(1024, feat_dim),
			nn.ReLU(),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Args:
		    x: (B, 3, 32, 32)
		Returns:
		    features: (B, feat_dim)
		"""
		return self.net(x)


class LSTMEncoder(nn.Module):
	"""Recurrent encoder: image sequence → (mu, log_var) in latent space.

	Processing order:
	  1. Each frame is embedded independently by FrameCNN.
	  2. An LSTM processes the sequence in chronological order; we take the
	     hidden state after the last step (which corresponds to frame T).
	  3. The hidden state is reshaped to (B, latent_ch, 4, 4) and fed to
	     mu/log_var conv heads — matching the shape expected by f_psi.

	The hidden state therefore represents the *current* phase-space position
	(frame T). During training the Hamiltonian is integrated backwards in time
	(-dt) to reconstruct the history seen by the encoder.

	Args:
	    img_ch:    image channels (3 for RGB)
	    feat_dim:  per-frame CNN output size
	    latent_ch: total latent channels (2 * pos_ch); hidden_size = latent_ch * 4 * 4
	"""

	def __init__(
		self,
		img_ch: int = 3,
		feat_dim: int = 256,
		latent_ch: int = 32,
	):
		super().__init__()
		self.latent_ch = latent_ch
		self.hidden_size = latent_ch * 4 * 4

		self.frame_cnn = FrameCNN(img_ch=img_ch, feat_dim=feat_dim)
		self.lstm = nn.LSTM(
			input_size=feat_dim,
			hidden_size=self.hidden_size,
			num_layers=1,
			batch_first=True,
		)
		self.mu_head = nn.Conv2d(latent_ch, latent_ch, 1)
		self.logvar_head = nn.Conv2d(latent_ch, latent_ch, 1)

	def forward(self, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		"""Args:
		    imgs: (B, T, C, H, W)
		Returns:
		    mu, log_var: each (B, latent_ch, 4, 4)
		"""
		B, T, C, H, W = imgs.shape

		# Embed every frame independently.
		frames_flat = imgs.reshape(B * T, C, H, W)
		feats = self.frame_cnn(frames_flat)  # (B*T, feat_dim)
		feats = feats.reshape(B, T, -1)  # (B, T, feat_dim)

		# Run LSTM in chronological order; the final hidden state therefore
		# encodes the *current* state (frame T) rather than frame 0.
		_, (h_n, _) = self.lstm(feats)  # h_n: (1, B, hidden_size)
		h = h_n.squeeze(0)  # (B, hidden_size)

		# Reshape to spatial tensor expected by f_psi / Hamiltonian.
		h_spatial = h.reshape(B, self.latent_ch, 4, 4)

		mu = self.mu_head(h_spatial)
		log_var = self.logvar_head(h_spatial)
		return mu, log_var


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class HGN_LSTM(nn.Module):
	"""HGN with an LSTM-based recurrent encoder.

	Identical to HGN (Toth et al., ICLR 2020) except the stacked-frame CNN
	encoder is replaced by:

	    FrameCNN (per frame) → reverse sequence → LSTM → hidden state
	    → reshape → mu/log_var heads → reparameterise → f_ψ → (q0, p0)

	The Hamiltonian network, leapfrog integrator, and decoder are unchanged.

	Args:
	    pos_ch:    channel depth for q (and for p); paper default 16
	    img_ch:    image channels, 3 for RGB
	    dt:        leapfrog step size
	    feat_dim:  per-frame CNN embedding size fed to the LSTM
	"""

	def __init__(
		self,
		pos_ch: int = 16,
		img_ch: int = 3,
		dt: float = 0.125,
		feat_dim: int = 256,
		separable: bool = True,
	):
		super().__init__()
		self.pos_ch = pos_ch
		self.latent_ch = 2 * pos_ch
		self.dt = dt

		self.encoder = LSTMEncoder(
			img_ch=img_ch,
			feat_dim=feat_dim,
			latent_ch=self.latent_ch,
		)
		self.f_psi = StateTransform(latent_ch=self.latent_ch)
		if separable:
			self.hamiltonian = HamiltonianNet(latent_ch=self.latent_ch)
		else:
			self.hamiltonian = FullHamiltonianNet(latent_ch=self.latent_ch)
		self.decoder = Decoder(pos_ch=pos_ch, img_ch=img_ch)
		self.coord_head = CoordHead(pos_ch=pos_ch)

	# ── phase-space helpers ──────────────────────────────────────────────────

	def _split(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		return s[:, : self.pos_ch], s[:, self.pos_ch :]

	def _join(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
		return torch.cat([q, p], dim=1)

	def H(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
		return self.hamiltonian(q, p)

	# ── leapfrog integrator (identical to hgn_org.py) ───────────────────────

	@torch.enable_grad()
	def leapfrog_step(
		self,
		q: torch.Tensor,
		p: torch.Tensor,
		dt: float | None = None,
	) -> tuple[torch.Tensor, torch.Tensor]:
		if dt is None:
			dt = self.dt

		q_ = q.detach().requires_grad_(True)
		p_ = p.detach().requires_grad_(True)
		dH_dq = torch.autograd.grad(
			self.H(q_, p_).sum(), q_, create_graph=self.training
		)[0]
		p_half = p - 0.5 * dt * dH_dq

		q_ = q.detach().requires_grad_(True)
		p_half_ = p_half.detach().requires_grad_(True)
		dH_dp = torch.autograd.grad(
			self.H(q_, p_half_).sum(), p_half_, create_graph=self.training
		)[0]
		q_next = q + dt * dH_dp

		q_next_ = q_next.detach().requires_grad_(True)
		p_half_ = p_half.detach().requires_grad_(True)
		dH_dq2 = torch.autograd.grad(
			self.H(q_next_, p_half_).sum(), q_next_, create_graph=self.training
		)[0]
		p_next = p_half - 0.5 * dt * dH_dq2

		return q_next, p_next

	# ── forward (inference) pass ─────────────────────────────────────────────

	def forward(
		self, imgs: torch.Tensor
	) -> tuple[
		torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
	]:
		"""Infer the initial phase-space state from an image sequence.

		Args:
		    imgs: (B, T, C, H, W)

		Returns:
		    q0:      (B, pos_ch, 4, 4)
		    p0:      (B, pos_ch, 4, 4)
		    kl:      (B,)
		    mu:      (B, 2*pos_ch, 4, 4)
		    log_var: (B, 2*pos_ch, 4, 4)
		"""
		mu, log_var = self.encoder(imgs)
		log_var = log_var.clamp(-10, 10)

		z = mu + torch.randn_like(mu) * (0.5 * log_var).exp()
		s0 = self.f_psi(z)
		q0, p0 = self._split(s0)

		kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
		kl = kl.sum(dim=[1, 2, 3])

		return q0, p0, kl, mu, log_var

	# ── rollout (identical to hgn_org.py) ───────────────────────────────────

	def rollout(
		self,
		q: torch.Tensor,
		p: torch.Tensor,
		n_steps: int,
		dt: float | None = None,
		return_states: bool = False,
	):
		frames = [self.decoder(q)]
		coords = [self.coord_head(q.detach())]
		qs = [q]
		ps = [p]
		for _ in range(n_steps):
			q, p = self.leapfrog_step(q, p, dt=dt)
			frames.append(self.decoder(q))
			coords.append(self.coord_head(q.detach()))
			qs.append(q)
			ps.append(p)
		if return_states:
			return frames, coords, qs, ps
		return frames, coords
