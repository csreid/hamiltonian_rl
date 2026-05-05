"""Controlled port-Hamiltonian GN with LSTM encoder (PHGN-LSTM).

Extends DHGN_LSTM with a control input port:

    dz/dt = (J − R) ∇H(z) + B u

Phase space is a flat latent_dim-dimensional vector (q and p each latent_dim//2).
All Hamiltonian machinery uses MLPs rather than convolutions — the 4×4 spatial
scaffolding was arbitrary and caused full receptive field collapse after 2 conv layers.

Adds:
    B  (q_dim × control_dim)    — learned input matrix
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from dhgn_lstm import DHGN_LSTM


# ---------------------------------------------------------------------------
# Per-frame CNN
# ---------------------------------------------------------------------------


class FlexFrameCNN(nn.Module):
    """Per-frame CNN: (B, C, H, W) → (B, feat_dim).

    Works for any H, W that are multiples of 8.  The flatten size is
    computed from a dry-run so no hardcoded assumption.
    """

    def __init__(
        self, img_ch: int = 3, feat_dim: int = 256, img_size: int = 64
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(img_ch, 32, 3, stride=2, padding=1),  # H/2
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # H/4
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # H/8
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, img_ch, img_size, img_size)).shape[1]
        self.fc = nn.Sequential(nn.Linear(flat, feat_dim), nn.LeakyReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


# ---------------------------------------------------------------------------
# Encoder: image sequence → flat latent vector
# ---------------------------------------------------------------------------


class FlexLSTMEncoder(nn.Module):
    """Bidirectional LSTM encoder: image sequence → (mu, logvar) flat vectors.

    (B, T, C, H, W) → (B, latent_dim), (B, latent_dim)

    The LSTM hidden size matches feat_dim per direction (total 2*feat_dim),
    then linear heads project to latent_dim.
    """

    def __init__(
        self,
        img_ch: int = 3,
        feat_dim: int = 256,
        latent_dim: int = 32,
        img_size: int = 64,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.frame_cnn = FlexFrameCNN(img_ch=img_ch, feat_dim=feat_dim, img_size=img_size)
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=feat_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.mu_head = nn.Linear(2 * feat_dim, latent_dim)
        self.logvar_head = nn.Linear(2 * feat_dim, latent_dim)
        # Directional heads: each takes only one half of the BiLSTM output
        self.mu_fwd_head = nn.Linear(feat_dim, latent_dim)
        self.logvar_fwd_head = nn.Linear(feat_dim, latent_dim)
        self.mu_bwd_head = nn.Linear(feat_dim, latent_dim)
        self.logvar_bwd_head = nn.Linear(feat_dim, latent_dim)

    def _embed_frames(self, imgs: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = imgs.shape
        return self.frame_cnn(imgs.reshape(B * T, C, H, W)).reshape(B, T, -1)

    def forward(
        self, imgs: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a sequence of frames to (mu, logvar).

        Args:
            imgs:    (B, T, C, H, W)
            lengths: (B,) actual sequence lengths; if None the full T is used.

        Returns:
            mu, logvar: each (B, latent_dim)
        """
        B = imgs.shape[0]
        feats = self._embed_frames(imgs)  # (B, T, feat_dim)

        if lengths is not None:
            packed = pack_padded_sequence(
                feats, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(feats)

        # h_n: (2, B, feat_dim) — forward and backward final hidden states
        h = h_n.permute(1, 0, 2).reshape(B, -1)  # (B, 2*feat_dim)
        return self.mu_head(h), self.logvar_head(h)

    def forward_all(
        self, imgs: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode each timestep → per-step (mu, logvar).

        Runs the full LSTM and projects every output hidden state, giving
        one latent distribution per frame rather than one per sequence.

        Args:
            imgs:    (B, T, C, H, W)
            lengths: (B,) actual sequence lengths; if None, full T is used.

        Returns:
            mu_all, logvar_all: each (B, T, latent_dim)
        """
        B, T = imgs.shape[:2]
        feats = self._embed_frames(imgs)  # (B, T, feat_dim)

        if lengths is not None:
            packed = pack_padded_sequence(
                feats, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.lstm(packed)
            out, _ = pad_packed_sequence(out, batch_first=True, total_length=T)
        else:
            out, _ = self.lstm(feats)

        # out: (B, T, 2*feat_dim) — all-timestep hidden states
        return self.mu_head(out), self.logvar_head(out)

    def forward_all_split(
        self, imgs: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, ...]:
        """Encode each timestep with full, forward-only, and backward-only hidden states.

        Returns mu/logvar for three prediction targets:
          - full:    both directions → predicts current frame
          - fwd:     forward half only → predicts next frame (caller uses [:, :-1])
          - bwd:     backward half only → predicts previous frame (caller uses [:, 1:])

        Returns:
            mu_full, logvar_full, mu_fwd, logvar_fwd, mu_bwd, logvar_bwd:
            each (B, T, latent_dim)
        """
        B, T = imgs.shape[:2]
        feats = self._embed_frames(imgs)

        if lengths is not None:
            packed = pack_padded_sequence(
                feats, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.lstm(packed)
            out, _ = pad_packed_sequence(out, batch_first=True, total_length=T)
        else:
            out, _ = self.lstm(feats)

        fwd = out[:, :, :self.feat_dim]
        bwd = out[:, :, self.feat_dim:]

        return (
            self.mu_head(out),
            self.logvar_head(out),
            self.mu_fwd_head(fwd),
            self.logvar_fwd_head(fwd),
            self.mu_bwd_head(bwd),
            self.logvar_bwd_head(bwd),
        )


# ---------------------------------------------------------------------------
# MLP modules replacing the conv-based HGN components
# ---------------------------------------------------------------------------


class MLPStateTransform(nn.Module):
    """f_ψ: 3-layer MLP mapping sampled z → initial phase-space state s0.

    (B, latent_dim) → (B, latent_dim)
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class AffineCouplingLayer(nn.Module):
    """Affine coupling layer for a RealNVP-style normalizing flow.

    Splits input in half; one half conditions scale/translate for the other.
    Alternating which half passes through gives a universal bijection.
    Zero-initialised output layers so the flow starts as identity.
    """

    def __init__(self, dim: int, mask_first: bool):
        super().__init__()
        d1 = dim // 2
        d2 = dim - d1
        self.d1 = d1
        self.mask_first = mask_first
        d_cond = d1 if mask_first else d2
        d_out = d2 if mask_first else d1

        self.scale_net = nn.Sequential(
            nn.Linear(d_cond, 128), nn.ReLU(),
            nn.Linear(128, d_out), nn.Tanh(),  # bounded → exp never blows up
        )
        self.translate_net = nn.Sequential(
            nn.Linear(d_cond, 128), nn.ReLU(),
            nn.Linear(128, d_out),
        )
        nn.init.zeros_(self.scale_net[-2].weight)
        nn.init.zeros_(self.scale_net[-2].bias)
        nn.init.zeros_(self.translate_net[-1].weight)
        nn.init.zeros_(self.translate_net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., :self.d1], x[..., self.d1:]
        if self.mask_first:
            s, t = self.scale_net(x1), self.translate_net(x1)
            return torch.cat([x1, x2 * s.exp() + t], dim=-1)
        else:
            s, t = self.scale_net(x2), self.translate_net(x2)
            return torch.cat([x1 * s.exp() + t, x2], dim=-1)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        y1, y2 = y[..., :self.d1], y[..., self.d1:]
        if self.mask_first:
            s, t = self.scale_net(y1), self.translate_net(y1)
            return torch.cat([y1, (y2 - t) * (-s).exp()], dim=-1)
        else:
            s, t = self.scale_net(y2), self.translate_net(y2)
            return torch.cat([(y1 - t) * (-s).exp(), y2], dim=-1)


class NormalizingFlow(nn.Module):
    """Stack of affine coupling layers: LSTM latent z ↔ Hamiltonian phase space (q, p).

    Bijective differentiable map so the two spaces carry identical information.
    forward() maps z → (q, p); inverse() maps (q, p) → z.
    """

    def __init__(self, dim: int, n_layers: int = 6):
        super().__init__()
        self.layers = nn.ModuleList([
            AffineCouplingLayer(dim, mask_first=(i % 2 == 0))
            for i in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        for layer in reversed(self.layers):
            y = layer.inverse(y)
        return y


class MLPHamiltonianNet(nn.Module):
    """H(q, p) implemented as an MLP.

    Separable mode: H = T(q, p) + V(q), matching the physical structure
    where kinetic energy depends on both q and p but potential only on q.

    Args:
        latent_dim: total phase-space dimension (q_dim = p_dim = latent_dim // 2)
        separable:  if True, use T + V decomposition
    """

    def __init__(self, latent_dim: int, separable: bool = True):
        super().__init__()
        self.separable = separable
        q_dim = latent_dim // 2

        if separable:
            self.kinetic = nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.Softplus(),
                nn.Linear(256, 256),
                nn.Softplus(),
                nn.Linear(256, 1),
            )
            self.potential = nn.Sequential(
                nn.Linear(q_dim, 256),
                nn.Softplus(),
                nn.Linear(256, 256),
                nn.Softplus(),
                nn.Linear(256, 1),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.Softplus(),
                nn.Linear(256, 256),
                nn.Softplus(),
                nn.Linear(256, 1),
            )

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        if self.separable:
            T = self.kinetic(torch.cat([q, p], dim=-1)).squeeze(-1)
            V = self.potential(q).squeeze(-1)
            return T + V
        return self.net(torch.cat([q, p], dim=-1)).squeeze(-1)


class MLPStateDecoder(nn.Module):
    """Maps (q, p) → observed state vector (B, obs_state_dim).

    Used for supervised auxiliary loss when ground-truth state labels are
    available (e.g. CartPole 4-vector).

    Args:
        latent_dim:    total phase-space dimension (q_dim + p_dim)
        obs_state_dim: dimensionality of the target state vector
    """

    def __init__(self, latent_dim: int, obs_state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, obs_state_dim),
        )

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([q, p], dim=-1))


class MLPCoordHead(nn.Module):
    """Maps position q (B, q_dim) → pixel coords (B, 2) in [0, 1]."""

    def __init__(self, q_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(q_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid(),
        )

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.net(q)


# ---------------------------------------------------------------------------
# Decoder: flat latent vector → image
# ---------------------------------------------------------------------------


def _leaky_hard_sigmoid(x: torch.Tensor, outer_slope: float = 0.01) -> torch.Tensor:
    """Hard sigmoid in [-3, 3] (slope=1/6) with a leaky tail outside.

    Exactly matches nn.Hardsigmoid in the inner region:
        f(x) = x/6 + 0.5   for x ∈ [-3, 3]

    Outside, lines connect continuously at (-3, 0) and (3, 1) with slope
    `outer_slope`, so gradients never vanish completely.
    """
    inner = x / 6.0 + 0.5
    lo = outer_slope * (x + 3.0)           # passes through (-3, 0)
    hi = outer_slope * (x - 3.0) + 1.0    # passes through (3, 1)
    return torch.where(x < -3.0, lo, torch.where(x > 3.0, hi, inner))


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
    """Decoder: (B, q_dim) → (B, img_ch, img_size, img_size).

    A linear layer expands the flat q vector to pos_ch * 4 * 4, which is
    reshaped to (B, pos_ch, 4, 4) as the spatial seed for progressive upsampling.

    img_size must equal 4 * 2^k for some k ≥ 1 (e.g. 8, 16, 32, 64, 128).
    """

    def __init__(
        self,
        q_dim: int = 16,
        pos_ch: int = 16,
        img_ch: int = 3,
        img_size: int = 64,
    ):
        super().__init__()
        self.pos_ch = pos_ch
        n_blocks = int(math.log2(img_size // 4))
        assert 4 * (2**n_blocks) == img_size, f"img_size must be 4·2^k, got {img_size}"

        self.expand = nn.Linear(q_dim, pos_ch * 4 * 4)
        blocks = [_DecoderBlock(pos_ch)]
        for _ in range(n_blocks - 1):
            blocks.append(_DecoderBlock(64))
        self.blocks = nn.ModuleList(blocks)
        self.out_conv = nn.Conv2d(64, img_ch, 1)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        B = q.shape[0]
        x = self.expand(q).reshape(B, self.pos_ch, 4, 4)
        for block in self.blocks:
            x = block(x)
        return _leaky_hard_sigmoid(self.out_conv(x))


# ---------------------------------------------------------------------------
# ControlledDHGN_LSTM
# ---------------------------------------------------------------------------


class ControlledDHGN_LSTM(DHGN_LSTM):
    """Port-Hamiltonian world model with LSTM encoder, dissipation, and control.

    Inherits J, R structure matrices and RK4 dissipative integrator from
    DHGN_LSTM.  Replaces all conv-based components with MLPs operating on a
    flat latent_dim-dimensional phase space:

        q, p ∈ ℝ^(latent_dim/2)

    The controlled ODE integrated with RK4 (zero-order hold on u):

        dz/dt = (J − R) ∇H(z) + B u

    Args:
        pos_ch:      spatial channel depth for the decoder's 4×4 seed
        img_ch:      image channels (3 for RGB)
        dt:          default integration step size
        feat_dim:    per-frame CNN embedding size and LSTM hidden size
        img_size:    spatial resolution of input/output frames
        latent_dim:  flat phase-space dimension (q and p each latent_dim//2)
        control_dim: dimension of control input u
        separable:   if True, use T + V Hamiltonian decomposition
    """

    def __init__(
        self,
        pos_ch: int = 16,
        img_ch: int = 3,
        dt: float = 0.05,
        feat_dim: int = 256,
        img_size: int = 64,
        latent_dim: int = 32,
        control_dim: int = 1,
        separable: bool = True,
        obs_state_dim: int | None = None,
        learn_structure: bool = True,
        damping: float = 0.0,
    ):
        super().__init__(
            pos_ch=pos_ch,
            img_ch=img_ch,
            dt=dt,
            feat_dim=feat_dim,
            separable=separable,
        )

        self.latent_dim = latent_dim
        self.learn_structure = learn_structure
        q_dim = latent_dim // 2

        # Override state_dim and structure matrices with flat latent_dim.
        # (Parent created these sized to latent_ch * 4 * 4; reassignment
        # via nn.Module.__setattr__ cleanly replaces them in _parameters.)
        self.state_dim = latent_dim
        if learn_structure:
            self.A = nn.Parameter(torch.zeros(latent_dim, latent_dim))
            self.L_param = nn.Parameter(torch.zeros(latent_dim, latent_dim))
            nn.init.normal_(self.A, std=1e-2)
            nn.init.normal_(self.L_param, std=1e-2)
        else:
            # Canonical symplectic J = [[0, I], [-I, 0]], R = 0, fixed.
            J_fixed = torch.zeros(latent_dim, latent_dim)
            J_fixed[:q_dim, q_dim:] = torch.eye(q_dim)
            J_fixed[q_dim:, :q_dim] = -torch.eye(q_dim)
            self.register_buffer("J_fixed", J_fixed)
            R_fixed = torch.zeros(latent_dim, latent_dim)
            R_fixed[q_dim:, q_dim:] = damping * torch.eye(q_dim)
            self.register_buffer("R_fixed", R_fixed)

        # Replace all conv-based modules with flat MLP equivalents.
        self.encoder = FlexLSTMEncoder(
            img_ch=img_ch,
            feat_dim=feat_dim,
            latent_dim=latent_dim,
            img_size=img_size,
        )
        self.f_psi = NormalizingFlow(latent_dim)
        self.hamiltonian = MLPHamiltonianNet(latent_dim, separable=separable)
        self.decoder = FlexDecoder(
            q_dim=q_dim, pos_ch=pos_ch, img_ch=img_ch, img_size=img_size
        )
        self.coord_head = MLPCoordHead(q_dim)

        self.control_dim = control_dim
        if learn_structure:
            self.B = nn.Parameter(torch.zeros(q_dim, control_dim))
            nn.init.normal_(self.B, std=1e-2)
        else:
            self.register_buffer("B_fixed", torch.ones(q_dim, control_dim))

        self.state_decoder = (
            MLPStateDecoder(latent_dim, obs_state_dim)
            if obs_state_dim is not None
            else None
        )

    # ── Structure matrix accessors (override parent to support fixed mode) ──

    def get_J(self) -> torch.Tensor:
        if not self.learn_structure:
            return self.J_fixed
        return self.A - self.A.T

    def get_R(self) -> torch.Tensor:
        if not self.learn_structure:
            return self.R_fixed
        L = self.get_L()
        return L @ L.T

    def get_B(self) -> torch.Tensor:
        if not self.learn_structure:
            return self.B_fixed
        return self.B

    # ── Dynamics (override parent to preserve gradient graph across rollout) ─

    @torch.enable_grad()
    def _dynamics(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        M: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """dz/dt = M ∇H(z). Does NOT detach q/p so gradients flow across steps."""
        half = self.latent_dim // 2
        z_ = torch.cat([q, p], dim=-1).requires_grad_(True)
        H_val = self.hamiltonian(z_[:, :half], z_[:, half:]).sum()
        grad_H = torch.autograd.grad(H_val, z_, create_graph=self.training)[0]
        dz = torch.einsum("ij,bj->bi", M, grad_H)
        return dz[:, :half], dz[:, half:]

    # ── Phase-space helpers ─────────────────────────────────────────────────

    def _split(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        half = self.latent_dim // 2
        return s[:, :half], s[:, half:]

    # ── Forward (encode → initial state) ───────────────────────────────────

    def forward(
        self, imgs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Infer initial phase-space state from an image sequence.

        Args:
            imgs: (B, T, C, H, W)

        Returns:
            q0, p0: (B, latent_dim//2) each
            kl:     (B,)
            mu:     (B, latent_dim)
            log_var:(B, latent_dim)
        """
        mu, log_var = self.encoder(imgs)
        log_var = log_var.clamp(-10, 10)
        z = mu + torch.randn_like(mu) * (0.5 * log_var).exp()
        s0 = self.f_psi(z)
        q0, p0 = self._split(s0)
        kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
        kl = kl.sum(dim=1)  # (B,) — sum over latent_dim
        return q0, p0, kl, mu, log_var

    # ── Controlled dynamics ─────────────────────────────────────────────────

    def _controlled_dynamics(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        u: torch.Tensor,
        M: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """dz/dt = (J − R) ∇H + B u."""
        dq, dp = self._dynamics(q, p, M)
        Bu = u @ self.get_B().T  # (B, q_dim)
        dp = dp + Bu
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

    # ── Deterministic encoding for eval ────────────────────────────────────

    def decode_state(
        self, q: torch.Tensor, p: torch.Tensor
    ) -> torch.Tensor | None:
        """Decode (q, p) to observed state via state_decoder, or None."""
        if self.state_decoder is None:
            return None
        return self.state_decoder(q, p)

    def encode_mean(
        self, imgs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode using posterior mean — no reparameterisation noise.

        Args:
            imgs: (B, T, C, H, W)

        Returns:
            q, p: (B, latent_dim//2) each
        """
        mu, _ = self.encoder(imgs)
        s0 = self.f_psi(mu)
        return self._split(s0)


# ---------------------------------------------------------------------------
# HamiltonianFlowModel — Phase 2: dynamics-only model
# ---------------------------------------------------------------------------


class HamiltonianFlowModel(nn.Module):
    """Phase 2 model: learns Φ mapping precomputed h_t → (q, p) for Hamiltonian dynamics.

    Completely separate from Phase 1 (ControlledDHGN_LSTM). Takes precomputed
    LSTM encoder outputs h_t as input — no encoder or decoder.

    The controlled ODE integrated with RK4: dz/dt = (J − R) ∇H(z) + B u

    Args:
        latent_dim:      dimension of h_t (= ControlledDHGN_LSTM.latent_dim)
        control_dim:     dimension of control input u
        separable:       if True, use T + V Hamiltonian decomposition
        learn_structure: if True, learn J/R/B; if False, use canonical J, R=0
        dt:              integration step size
        damping:         diagonal dissipation for fixed R (only when not learn_structure)
    """

    def __init__(
        self,
        latent_dim: int = 32,
        control_dim: int = 1,
        separable: bool = True,
        learn_structure: bool = True,
        dt: float = 0.05,
        damping: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.dt = dt
        self.learn_structure = learn_structure
        q_dim = latent_dim // 2

        self.phi = NormalizingFlow(latent_dim)
        self.hamiltonian = MLPHamiltonianNet(latent_dim, separable=separable)

        if learn_structure:
            self.A = nn.Parameter(torch.zeros(latent_dim, latent_dim))
            self.L_param = nn.Parameter(torch.zeros(latent_dim, latent_dim))
            nn.init.normal_(self.A, std=1e-2)
            nn.init.normal_(self.L_param, std=1e-2)
            self.B = nn.Parameter(torch.zeros(q_dim, control_dim))
            nn.init.normal_(self.B, std=1e-2)
        else:
            J_fixed = torch.zeros(latent_dim, latent_dim)
            J_fixed[:q_dim, q_dim:] = torch.eye(q_dim)
            J_fixed[q_dim:, :q_dim] = -torch.eye(q_dim)
            self.register_buffer("J_fixed", J_fixed)
            R_fixed = torch.zeros(latent_dim, latent_dim)
            R_fixed[q_dim:, q_dim:] = damping * torch.eye(q_dim)
            self.register_buffer("R_fixed", R_fixed)
            self.register_buffer("B_fixed", torch.ones(q_dim, control_dim))

    # ── Structure matrix accessors ──────────────────────────────────────────

    def get_J(self) -> torch.Tensor:
        if not self.learn_structure:
            return self.J_fixed
        return self.A - self.A.T

    def get_L(self) -> torch.Tensor:
        L_lower = self.L_param.tril(-1)
        diag_pos = F.softplus(self.L_param.diagonal())
        return L_lower + torch.diag(diag_pos)

    def get_R(self) -> torch.Tensor:
        if not self.learn_structure:
            return self.R_fixed
        L = self.get_L()
        return L @ L.T

    def get_B(self) -> torch.Tensor:
        if not self.learn_structure:
            return self.B_fixed
        return self.B

    def get_J_minus_R(self) -> torch.Tensor:
        return self.get_J() - self.get_R()

    # ── Dynamics ────────────────────────────────────────────────────────────

    @torch.enable_grad()
    def _dynamics(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        M: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        half = self.latent_dim // 2
        z_ = torch.cat([q, p], dim=-1).requires_grad_(True)
        H_val = self.hamiltonian(z_[:, :half], z_[:, half:]).sum()
        grad_H = torch.autograd.grad(H_val, z_, create_graph=self.training)[0]
        dz = torch.einsum("ij,bj->bi", M, grad_H)
        return dz[:, :half], dz[:, half:]

    def _controlled_dynamics(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        u: torch.Tensor,
        M: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dq, dp = self._dynamics(q, p, M)
        Bu = u @ self.get_B().T
        dp = dp + Bu
        return dq, dp

    @torch.enable_grad()
    def controlled_step(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        u: torch.Tensor,
        dt: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One RK4 step of dz/dt = (J − R) ∇H + B u."""
        if dt is None:
            dt = self.dt
        M = self.get_J_minus_R()
        dq1, dp1 = self._controlled_dynamics(q, p, u, M)
        dq2, dp2 = self._controlled_dynamics(q + 0.5 * dt * dq1, p + 0.5 * dt * dp1, u, M)
        dq3, dp3 = self._controlled_dynamics(q + 0.5 * dt * dq2, p + 0.5 * dt * dp2, u, M)
        dq4, dp4 = self._controlled_dynamics(q + dt * dq3, p + dt * dp3, u, M)
        q_next = q + (dt / 6.0) * (dq1 + 2 * dq2 + 2 * dq3 + dq4)
        p_next = p + (dt / 6.0) * (dp1 + 2 * dp2 + 2 * dp3 + dp4)
        return q_next, p_next

    # ── Phase-space helpers ─────────────────────────────────────────────────

    def encode(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """h_t → (q, p) via Φ."""
        s = self.phi(h)
        q_dim = self.latent_dim // 2
        return s[:, :q_dim], s[:, q_dim:]

    def decode(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """(q, p) → h_t via Φ⁻¹."""
        return self.phi.inverse(torch.cat([q, p], dim=-1))
