"""Controlled port-Hamiltonian GN with LSTM encoder (PHGN-LSTM).

Extends DHGN_LSTM with a control input port:

    dz/dt = (J − R) ∇H(z) + B u

Adds:
    B  (D × control_dim)    — learned input matrix
    state_decoder (optional) — latent z → observed state MLP

Also replaces the 32×32-hardcoded encoder/decoder with size-flexible
versions, so the model can work at whatever resolution the environment
renders at (no forced downscale to 32px²).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dhgn_lstm import DHGN_LSTM


# ---------------------------------------------------------------------------
# Size-flexible encoder
# ---------------------------------------------------------------------------


class FlexFrameCNN(nn.Module):
	"""Per-frame CNN: (B, C, H, W) → (B, feat_dim).

	Works for any H, W that are multiples of 8.  The flatten size is
	computed from a dry-run so no hardcoded 1024 assumption.
	"""

	def __init__(
		self, img_ch: int = 3, feat_dim: int = 256, img_size: int = 64
	):
		super().__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(img_ch, 32, 3, stride=2, padding=1),  # H/2
			nn.ReLU(),
			nn.Conv2d(32, 64, 3, stride=2, padding=1),  # H/4
			nn.ReLU(),
			nn.Conv2d(64, 64, 3, stride=2, padding=1),  # H/8
			nn.ReLU(),
			nn.Conv2d(64, 64, 3, padding=1),
			nn.ReLU(),
			nn.Flatten(),
		)
		with torch.no_grad():
			flat = self.conv(torch.zeros(1, img_ch, img_size, img_size)).shape[
				1
			]
		self.fc = nn.Sequential(nn.Linear(flat, feat_dim), nn.ReLU())

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.fc(self.conv(x))


class FlexLSTMEncoder(nn.Module):
	"""LSTM encoder that accepts any image size.

	Same chronological-order processing as LSTMEncoder in hgn_lstm.py;
	only FrameCNN is replaced with the size-flexible variant.
	"""

	def __init__(
		self,
		img_ch: int = 3,
		feat_dim: int = 256,
		latent_ch: int = 32,
		img_size: int = 64,
	):
		super().__init__()
		self.latent_ch = latent_ch
		self.hidden_size = latent_ch * 4 * 4

		self.frame_cnn = FlexFrameCNN(
			img_ch=img_ch, feat_dim=feat_dim, img_size=img_size
		)
		self.lstm = nn.LSTM(
			input_size=feat_dim,
			hidden_size=self.hidden_size,
			num_layers=1,
			batch_first=True,
		)
		self.mu_head = nn.Conv2d(latent_ch, latent_ch, 1)
		self.logvar_head = nn.Conv2d(latent_ch, latent_ch, 1)

	def forward(self, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		B, T, C, H, W = imgs.shape
		feats = self.frame_cnn(imgs.reshape(B * T, C, H, W)).reshape(B, T, -1)
		_, (h_n, _) = self.lstm(feats)
		h_spatial = h_n.squeeze(0).reshape(B, self.latent_ch, 4, 4)
		return self.mu_head(h_spatial), self.logvar_head(h_spatial)

	def forward_all(self, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		"""Like forward() but returns mu/logvar for every LSTM timestep.

		Args:
		    imgs: (B, T, C, H, W)

		Returns:
		    all_mu:     (B, T, latent_ch, 4, 4)
		    all_logvar: (B, T, latent_ch, 4, 4)

		outputs[:, t] is the hidden state after processing t+1 frames, so
		index -1 matches what forward() returns.
		"""
		B, T, C, H, W = imgs.shape
		feats = self.frame_cnn(imgs.reshape(B * T, C, H, W)).reshape(B, T, -1)
		outputs, _ = self.lstm(feats)  # (B, T, hidden_size)
		h_all = outputs.reshape(B * T, self.latent_ch, 4, 4)
		all_mu     = self.mu_head(h_all).reshape(B, T, self.latent_ch, 4, 4)
		all_logvar = self.logvar_head(h_all).reshape(B, T, self.latent_ch, 4, 4)
		return all_mu, all_logvar


# ---------------------------------------------------------------------------
# Size-flexible decoder
# ---------------------------------------------------------------------------


class _DecoderBlock(nn.Module):
	def __init__(self, in_ch: int):
		super().__init__()
		self.conv1 = nn.Conv2d(in_ch, 64, 3, padding=1)
		self.conv2 = nn.Conv2d(64, 64, 3, padding=1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = F.interpolate(x, scale_factor=2, mode="nearest")
		x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
		x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
		return x


class FlexDecoder(nn.Module):
	"""Decoder from (B, pos_ch, 4, 4) → (B, img_ch, img_size, img_size).

	img_size must equal 4 * 2^k for some k ≥ 1 (e.g. 8, 16, 32, 64, 128).
	"""

	def __init__(self, pos_ch: int = 16, img_ch: int = 3, img_size: int = 64):
		super().__init__()
		n_blocks = int(math.log2(img_size // 4))
		assert 4 * (2**n_blocks) == img_size, (
			f"img_size must be 4·2^k, got {img_size}"
		)
		blocks = [_DecoderBlock(pos_ch)]
		for _ in range(n_blocks - 1):
			blocks.append(_DecoderBlock(64))
		self.blocks = nn.ModuleList(blocks)
		self.out_conv = nn.Conv2d(64, img_ch, 1)

	def forward(self, q: torch.Tensor) -> torch.Tensor:
		x = self.blocks[0](q)
		for block in self.blocks[1:]:
			x = block(x)
		return torch.sigmoid(self.out_conv(x))


# ---------------------------------------------------------------------------
# ControlledDHGN_LSTM
# ---------------------------------------------------------------------------


class ControlledDHGN_LSTM(DHGN_LSTM):
	"""Port-Hamiltonian HGN with LSTM encoder, dissipation, and control.

	Inherits J, R structure matrices and RK4 dissipative integrator from
	DHGN_LSTM.  Adds:

	    B  (D × control_dim)        control input matrix
	    state_decoder (optional)    latent z → obs_state_dim MLP

	The controlled ODE integrated with RK4 (zero-order hold on u):

	    dz/dt = (J − R) ∇H(z) + B u

	The encoder and decoder are replaced with size-flexible versions so
	the model is not limited to 32×32 inputs/outputs.

	Args:
	    pos_ch:        latent position channel depth
	    img_ch:        image channels (3 for RGB)
	    dt:            default integration step size
	    feat_dim:      per-frame CNN embedding size
	    img_size:      spatial resolution of input/output frames
	    control_dim:   dimension of control input u
	    obs_state_dim: if > 0, adds an MLP head for observed-state prediction
	"""

	def __init__(
		self,
		pos_ch: int = 16,
		img_ch: int = 3,
		dt: float = 0.05,
		feat_dim: int = 256,
		img_size: int = 64,
		control_dim: int = 1,
		obs_state_dim: int = 0,
		separable: bool = True,
	):
		super().__init__(pos_ch=pos_ch, img_ch=img_ch, dt=dt, feat_dim=feat_dim, separable=separable)

		# Replace 32×32-hardcoded encoder/decoder with flexible versions.
		self.encoder = FlexLSTMEncoder(
			img_ch=img_ch,
			feat_dim=feat_dim,
			latent_ch=self.latent_ch,
			img_size=img_size,
		)
		self.decoder = FlexDecoder(
			pos_ch=pos_ch, img_ch=img_ch, img_size=img_size
		)

		self.control_dim = control_dim
		self.B = nn.Parameter(torch.zeros(self.state_dim // 2, control_dim))
		nn.init.normal_(self.B, std=1e-2)

		self.obs_state_dim = obs_state_dim
		if obs_state_dim > 0:
			self.state_decoder = nn.Sequential(
				nn.Linear(self.state_dim, 256),
				nn.SiLU(),
				nn.Linear(256, 128),
				nn.SiLU(),
				nn.Linear(128, obs_state_dim),
			)
		else:
			self.state_decoder = None

	# ── Controlled dynamics ──────────────────────────────────────────────────

	def _controlled_dynamics(
		self,
		q: torch.Tensor,
		p: torch.Tensor,
		u: torch.Tensor,
		M: torch.Tensor,
	) -> tuple[torch.Tensor, torch.Tensor]:
		"""dz/dt = (J − R) ∇H + B u."""
		dq, dp = self._dynamics(q, p, M)
		Bu = u @ self.B.T  # (B_batch, D)
		dp = dp + Bu.reshape_as(p)
		return dq, dp

	@torch.enable_grad()
	def controlled_step(
		self,
		q: torch.Tensor,
		p: torch.Tensor,
		u: torch.Tensor,
		dt: float | None = None,
	) -> tuple[torch.Tensor, torch.Tensor]:
		"""One RK4 step of dz/dt = (J − R) ∇H + B u (zero-order hold on u)."""
		if dt is None:
			dt = self.dt
		M = self.get_J_minus_R()
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
		"""Roll out applying control sequence us (B, H, control_dim)."""
		frames = [self.decoder(q)]
		qs, ps = [q], [p]
		for t in range(us.shape[1]):
			q, p = self.controlled_step(q, p, us[:, t], dt=dt)
			frames.append(self.decoder(q))
			qs.append(q)
			ps.append(p)
		if return_states:
			return frames, qs, ps
		return frames

	# ── State decoder ────────────────────────────────────────────────────────

	def decode_state(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
		if self.state_decoder is None:
			raise RuntimeError(
				"No state_decoder; construct with obs_state_dim > 0"
			)
		B = q.shape[0]
		z = torch.cat([q.reshape(B, -1), p.reshape(B, -1)], dim=1)
		return self.state_decoder(z)

	# ── Deterministic encoding for planning / eval ───────────────────────────

	def encode_mean(
		self, imgs: torch.Tensor
	) -> tuple[torch.Tensor, torch.Tensor]:
		"""Encode using posterior mean — no reparameterisation noise.

		Args:
		    imgs: (B, T, C, H, W)

		Returns:
		    q, p: (B, pos_ch, 4, 4)
		"""
		mu, _ = self.encoder(imgs)
		s0 = self.f_psi(mu)
		return self._split(s0)
