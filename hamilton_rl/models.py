"""Pendulum world-model components.

Two independently trained halves plus a wrapper that stitches them:

    TemporalAutoencoder       Phase 1 — pixels ↔ latent h_t
        encoder             causal LSTM over per-frame CNN features → (mu, logvar),
                            or a memoryless two-frame stack (encoder_type="framestack")
        f_psi               normalizing flow h → s (decoder input is s[:q_dim])
        decoder             q → frame
        next_frame_decoder  (h_t, a_t) → frame_{t+1} (auxiliary predictive head)

    HamiltonianFlowModel  Phase 2 — port-Hamiltonian dynamics on precomputed h_t
        phi                 normalizing flow h ↔ (q, p)
        hamiltonian         H(q, p), optionally separable T(p) + V(q)
        J/R/B               canonical symplectic J; learned or fixed dissipation
                            (optionally state-dependent, R = R(z)) and control
                            matrices.  dz/dt = (J − R(z)) ∇H(z) + B u

    WorldModel            autoencoder + dynamics; owns the dreaming stitch and
                          single-file checkpoint save/load.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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


class HardConcreteGate(nn.Module):
    """Per-dimension L0 gate (Louizos et al. 2017, https://arxiv.org/abs/1712.01312).

    Learns one logit per latent dim controlling whether that dim is used at
    all, separately from its value — unlike L1, kept dims pay no magnitude
    penalty and dropped dims are exactly zero, not just discouraged.
    """

    def __init__(
        self,
        dim: int,
        temperature: float = 2.0 / 3.0,
        stretch: tuple[float, float] = (-0.1, 1.1),
        init_open_prob: float = 0.5,
        init_std: float = 0.01,
    ):
        super().__init__()
        self.temperature = temperature
        self.gamma, self.zeta = stretch
        # init_open_prob=0.5 (log_alpha≈0) keeps the initial stochastic gate
        # samples centered in the unclipped region of the stretch interval —
        # starting near-saturated-open (e.g. 0.9) makes samples land above 1
        # and get hard-clamped almost every step, zeroing the reconstruction
        # gradient into log_alpha and leaving only the smooth, dim-symmetric
        # L0 penalty gradient, which drags every dim down in lockstep instead
        # of letting useful dims diverge from useless ones. Small per-dim
        # jitter (std=0.01, per Louizos et al.) breaks the tie so dims don't
        # start bit-identical either.
        init_logit = math.log(init_open_prob / (1 - init_open_prob))
        self.log_alpha = nn.Parameter(torch.full((dim,), init_logit) + init_std * torch.randn(dim))

    def _stretch(self, s: torch.Tensor) -> torch.Tensor:
        return (s * (self.zeta - self.gamma) + self.gamma).clamp(0.0, 1.0)

    def forward(self) -> torch.Tensor:
        """Returns the (dim,) gate mask — stochastic if training, hard if eval."""
        if self.training:
            u = torch.rand_like(self.log_alpha).clamp(1e-6, 1 - 1e-6)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.log_alpha) / self.temperature)
        else:
            s = torch.sigmoid(self.log_alpha)
        return self._stretch(s)

    def l0_penalty(self) -> torch.Tensor:
        """Expected number of active dims (differentiable) — the L0 regularizer."""
        return torch.sigmoid(self.log_alpha - self.temperature * math.log(-self.gamma / self.zeta)).sum()

    @torch.no_grad()
    def effective_dim(self) -> float:
        return float(self.l0_penalty().item())


class FlexLSTMEncoder(nn.Module):
    """Causal LSTM encoder: image sequence → (mu, logvar) flat vectors.

    (B, T, C, H, W) → (B, latent_dim), (B, latent_dim)

    Forward-only so that h_t at training matches h_t at inference. Actions are
    excluded — momentum is recoverable from frame differences, and actions enter
    the dynamics model separately via the control port.
    """

    def __init__(
        self,
        img_ch: int = 3,
        feat_dim: int = 256,
        latent_dim: int = 32,
        img_size: int = 64,
        num_layers: int = 1,
        use_gate: bool = False,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_layers = num_layers
        self.frame_cnn = FlexFrameCNN(img_ch=img_ch, feat_dim=feat_dim, img_size=img_size)
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=feat_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.mu_head = nn.Linear(feat_dim, latent_dim)
        self.logvar_head = nn.Linear(feat_dim, latent_dim)
        self.gate = HardConcreteGate(latent_dim) if use_gate else None

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
        feats = self._embed_frames(imgs)  # (B, T, feat_dim)

        if lengths is not None:
            packed = pack_padded_sequence(
                feats, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(feats)

        h = h_n[-1]  # (B, feat_dim) — final layer's hidden state at the last step
        mu = self.mu_head(h)
        if self.gate is not None:
            mu = mu * self.gate()
        return mu, self.logvar_head(h)

    def forward_all(
        self, imgs: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode each timestep → per-step (mu, logvar).

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

        # out: (B, T, feat_dim) — per-timestep forward hidden states
        mu_all = self.mu_head(out)
        if self.gate is not None:
            mu_all = mu_all * self.gate()
        return mu_all, self.logvar_head(out)


class FrameStackEncoder(nn.Module):
    """Two-frame stacked encoder: (frame_{t-1}, frame_t) → (mu, logvar) at t.

    Drop-in replacement for FlexLSTMEncoder (same forward_all interface and
    frame_cnn attribute) that replaces the recurrent hidden state with a
    memoryless function of the current and previous frame embeddings —
    momentum is identifiable from two consecutive frames, so in principle no
    longer history is needed. At t=0 the frame is paired with itself, which
    carries no motion evidence, matching the LSTM's blindness to velocity at
    the first step.
    """

    def __init__(
        self,
        img_ch: int = 3,
        feat_dim: int = 256,
        latent_dim: int = 32,
        img_size: int = 64,
        use_gate: bool = False,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.frame_cnn = FlexFrameCNN(img_ch=img_ch, feat_dim=feat_dim, img_size=img_size)
        self.fuse = nn.Sequential(
            nn.Linear(2 * feat_dim, feat_dim),
            nn.LeakyReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.LeakyReLU(),
        )
        self.mu_head = nn.Linear(feat_dim, latent_dim)
        self.logvar_head = nn.Linear(feat_dim, latent_dim)
        self.gate = HardConcreteGate(latent_dim) if use_gate else None

    def _embed_frames(self, imgs: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = imgs.shape
        return self.frame_cnn(imgs.reshape(B * T, C, H, W)).reshape(B, T, -1)

    def forward_all(
        self, imgs: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode each timestep → per-step (mu, logvar).

        Args:
            imgs:    (B, T, C, H, W)
            lengths: accepted for interface parity with FlexLSTMEncoder and
                     ignored — each output depends only on frames t-1 and t,
                     so padded positions never contaminate valid ones.

        Returns:
            mu_all, logvar_all: each (B, T, latent_dim)
        """
        feats = self._embed_frames(imgs)                       # (B, T, feat_dim)
        prev = torch.cat([feats[:, :1], feats[:, :-1]], dim=1)  # frame_{-1} := frame_0
        out = self.fuse(torch.cat([prev, feats], dim=-1))      # (B, T, feat_dim)
        mu_all = self.mu_head(out)
        if self.gate is not None:
            mu_all = mu_all * self.gate()
        return mu_all, self.logvar_head(out)

    def forward(
        self, imgs: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a sequence to the final step's (mu, logvar), each (B, latent_dim)."""
        if lengths is None:
            mu_all, logvar_all = self.forward_all(imgs[:, -2:])
            return mu_all[:, -1], logvar_all[:, -1]
        mu_all, logvar_all = self.forward_all(imgs)
        idx = (lengths.to(imgs.device) - 1).view(-1, 1, 1).expand(-1, 1, mu_all.shape[-1])
        return mu_all.gather(1, idx).squeeze(1), logvar_all.gather(1, idx).squeeze(1)


# ---------------------------------------------------------------------------
# Normalizing flow (RealNVP-style affine coupling)
# ---------------------------------------------------------------------------


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

    def forward_with_logdet(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = x[..., :self.d1], x[..., self.d1:]
        if self.mask_first:
            s, t = self.scale_net(x1), self.translate_net(x1)
            out = torch.cat([x1, x2 * s.exp() + t], dim=-1)
        else:
            s, t = self.scale_net(x2), self.translate_net(x2)
            out = torch.cat([x1 * s.exp() + t, x2], dim=-1)
        return out, s.sum(dim=-1)  # log|det J| contribution for this layer

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

    def forward_with_logdet(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for layer in self.layers:
            x, ld = layer.forward_with_logdet(x)
            log_det = log_det + ld
        return x, log_det

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        for layer in reversed(self.layers):
            y = layer.inverse(y)
        return y


# ---------------------------------------------------------------------------
# Hamiltonian nets
# ---------------------------------------------------------------------------


class QuadraticKinetic(nn.Module):
    """Physical kinetic energy T(p) = ½ pᵀ M⁻¹ p with a learned constant mass.

    M⁻¹ = L Lᵀ where L is lower-triangular with softplus-positive diagonal, so
    the inverse mass matrix is symmetric positive definite by construction.
    Compared to a free MLP this enforces: convexity of T in p, T(0) = 0 and
    ∇T(0) = 0 (a state at rest stays at rest), and T(p) = T(−p) (time-reversal
    symmetry of the undamped flow).  Initialised to M⁻¹ = I, i.e. T = ½‖p‖².

    Output is (B, 1) to match the MLP kinetic head interface.
    """

    def __init__(self, q_dim: int):
        super().__init__()
        self.mass_chol = nn.Parameter(torch.zeros(q_dim, q_dim))
        with torch.no_grad():
            # softplus(x) = 1 ⇔ x = log(e − 1): start at M⁻¹ = I
            self.mass_chol.diagonal().fill_(math.log(math.e - 1))

    def _L(self) -> torch.Tensor:
        return self.mass_chol.tril(-1) + torch.diag(F.softplus(self.mass_chol.diagonal()))

    def M_inv(self) -> torch.Tensor:
        """Inverse mass matrix M⁻¹ = L Lᵀ (symmetric positive definite)."""
        L = self._L()
        return L @ L.T

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        Lp = p @ self._L()  # rows are (Lᵀp)ᵀ, so ‖Lᵀp‖² = pᵀ L Lᵀ p
        return 0.5 * Lp.pow(2).sum(dim=-1, keepdim=True)


class MLPHamiltonianNet(nn.Module):
    """H(q, p) implemented as an MLP.

    Separable mode: H = T(p) + V(q), matching the physical structure where
    kinetic energy depends only on momentum and potential only on position.
    This true separability is what makes an explicit symplectic (leapfrog)
    step possible: ∂H/∂p depends only on p and ∂H/∂q only on q.

    Args:
        latent_dim:  total phase-space dimension (q_dim = p_dim = latent_dim // 2)
        separable:   if True, use T + V decomposition
        quadratic_t: if True (requires separable), T is a PSD quadratic form
                     (QuadraticKinetic) instead of a free MLP
    """

    def __init__(self, latent_dim: int, separable: bool = True, quadratic_t: bool = False):
        super().__init__()
        if quadratic_t and not separable:
            raise ValueError(
                "quadratic_t requires a separable Hamiltonian H = T(p) + V(q); "
                "pass separable=True or quadratic_t=False."
            )
        self.separable = separable
        q_dim = latent_dim // 2

        if separable:
            self.kinetic = QuadraticKinetic(q_dim) if quadratic_t else nn.Sequential(
                nn.Linear(q_dim, 256),
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
            T = self.kinetic(p).squeeze(-1)
            V = self.potential(q).squeeze(-1)
            return T + V
        return self.net(torch.cat([q, p], dim=-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Decoders: flat latent vector → image
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


class NextFrameDecoder(nn.Module):
    """Predicts frame_{t+1} from (h_t, a_t).

    Projects the concatenated latent + action to a spatial seed, then
    progressively upsamples to the full image resolution — same architecture
    as FlexDecoder but conditioned on the action taken.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        control_dim: int = 1,
        pos_ch: int = 16,
        img_ch: int = 3,
        img_size: int = 64,
    ):
        super().__init__()
        self.pos_ch = pos_ch
        n_blocks = int(math.log2(img_size // 4))
        assert 4 * (2**n_blocks) == img_size, f"img_size must be 4·2^k, got {img_size}"

        in_dim = latent_dim + control_dim
        self.expand = nn.Linear(in_dim, pos_ch * 4 * 4)
        blocks = [_DecoderBlock(pos_ch)]
        for _ in range(n_blocks - 1):
            blocks.append(_DecoderBlock(64))
        self.blocks = nn.ModuleList(blocks)
        self.out_conv = nn.Conv2d(64, img_ch, 1)

    def forward(self, h: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, latent_dim)
            a: (B, control_dim) or (B,) — action at time t
        """
        if a.dim() == 1:
            a = a.unsqueeze(-1)
        x = self.expand(torch.cat([h, a], dim=-1)).reshape(h.shape[0], self.pos_ch, 4, 4)
        for block in self.blocks:
            x = block(x)
        return _leaky_hard_sigmoid(self.out_conv(x))


# ---------------------------------------------------------------------------
# TemporalAutoencoder — Phase 1: reconstruction-only model
# ---------------------------------------------------------------------------


class TemporalAutoencoder(nn.Module):
    """Phase 1 model: temporal encoder + normalizing flow + CNN decoders. No dynamics.

    The latent h_t produced by the encoder is what the Phase 2 dynamics model
    (HamiltonianFlowModel) consumes.  Decoding goes h → f_psi(h)[:q_dim] → frame.

    Args:
        latent_dim:   flat latent dimension of h_t (decoder sees latent_dim // 2)
        feat_dim:     per-frame CNN embedding size and encoder hidden size
        pos_ch:       spatial channel depth for the decoders' 4×4 seed
        img_size:     spatial resolution of input/output frames
        img_ch:       image channels (3 for RGB)
        control_dim:  dimension of the action fed to next_frame_decoder
        num_layers:   number of stacked LSTM layers (lstm encoder only)
        encoder_type: "lstm" (causal LSTM over frame embeddings) or "framestack"
                      (memoryless two-consecutive-frame encoder)
        use_gate:     learn a per-dim L0 hard-concrete gate on the latent mean
                      (see HardConcreteGate), in place of/alongside L1 sparsity
    """

    def __init__(
        self,
        latent_dim: int = 32,
        feat_dim: int = 256,
        pos_ch: int = 8,
        img_size: int = 64,
        img_ch: int = 3,
        control_dim: int = 1,
        num_layers: int = 1,
        encoder_type: str = "lstm",
        use_gate: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.config = {
            "latent_dim": latent_dim,
            "feat_dim": feat_dim,
            "pos_ch": pos_ch,
            "img_size": img_size,
            "img_ch": img_ch,
            "control_dim": control_dim,
            "num_layers": num_layers,
            "encoder_type": encoder_type,
            "use_gate": use_gate,
        }
        q_dim = latent_dim // 2

        if encoder_type == "lstm":
            self.encoder = FlexLSTMEncoder(
                img_ch=img_ch,
                feat_dim=feat_dim,
                latent_dim=latent_dim,
                img_size=img_size,
                num_layers=num_layers,
                use_gate=use_gate,
            )
        elif encoder_type == "framestack":
            self.encoder = FrameStackEncoder(
                img_ch=img_ch,
                feat_dim=feat_dim,
                latent_dim=latent_dim,
                img_size=img_size,
                use_gate=use_gate,
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type!r}")
        self.f_psi = NormalizingFlow(latent_dim)
        self.decoder = FlexDecoder(
            q_dim=q_dim, pos_ch=pos_ch, img_ch=img_ch, img_size=img_size
        )
        self.next_frame_decoder = NextFrameDecoder(
            latent_dim=latent_dim, control_dim=control_dim,
            pos_ch=pos_ch, img_ch=img_ch, img_size=img_size,
        )

    @property
    def q_dim(self) -> int:
        return self.latent_dim // 2

    def decode_latent(self, h: torch.Tensor) -> torch.Tensor:
        """h (B, latent_dim) → frame (B, C, H, W) via f_psi + decoder."""
        s = self.f_psi(h)
        return self.decoder(s[:, :self.q_dim])


# ---------------------------------------------------------------------------
# HamiltonianFlowModel — Phase 2: dynamics-only model
# ---------------------------------------------------------------------------


class HamiltonianFlowModel(nn.Module):
    """Phase 2 model: learns Φ mapping precomputed h_t → (q, p) for Hamiltonian dynamics.

    Completely separate from Phase 1 (TemporalAutoencoder). Takes precomputed
    LSTM encoder outputs h_t as input — no encoder or decoder.

    The controlled ODE integrated with RK4: dz/dt = (J − R(z)) ∇H(z) + B u

    Args:
        latent_dim:      dimension of h_t (= TemporalAutoencoder.latent_dim)
        control_dim:     dimension of control input u
        separable:       if True, use T + V Hamiltonian decomposition
        learn_structure: if True, learn R/B; if False, R is fixed from damping
        dt:              integration step size
        damping:         diagonal dissipation for fixed R (only when not learn_structure)
        quadratic_t:     if True (requires separable), kinetic energy is a PSD
                         quadratic form T(p) = ½ pᵀM⁻¹p with learned constant mass
        state_dep_r:     if True (requires learn_structure), the dissipation
                         R_pp = L(z) L(z)ᵀ is a function of the phase-space
                         point z = (q, p) via a small MLP, instead of a
                         constant matrix.  A constant R can only express
                         linear (viscous) damping ṗ ∝ −p; state dependence is
                         needed for e.g. the env's quadratic drag ṗ ∝ −p|p|,
                         whose effective damping coefficient grows with speed.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        control_dim: int = 1,
        separable: bool = True,
        learn_structure: bool = True,
        dt: float = 0.05,
        damping: float = 0.0,
        integrator: str = "rk4",
        quadratic_t: bool = False,
        state_dep_r: bool = False,
    ):
        super().__init__()
        if integrator not in ("rk4", "leapfrog"):
            raise ValueError(f"integrator must be 'rk4' or 'leapfrog', got {integrator!r}")
        if integrator == "leapfrog" and not separable:
            raise ValueError(
                "leapfrog requires a separable Hamiltonian H = T(p) + V(q); "
                "pass separable=True or use integrator='rk4'."
            )
        if state_dep_r and not learn_structure:
            raise ValueError("state_dep_r requires learn_structure=True")
        self.latent_dim = latent_dim
        self.dt = dt
        self.learn_structure = learn_structure
        self.separable = separable
        self.integrator = integrator
        self.state_dep_r = state_dep_r
        self.config = {
            "latent_dim": latent_dim,
            "control_dim": control_dim,
            "separable": separable,
            "learn_structure": learn_structure,
            "dt": dt,
            "damping": damping,
            "integrator": integrator,
            "quadratic_t": quadratic_t,
            "state_dep_r": state_dep_r,
        }
        q_dim = latent_dim // 2

        self.phi = NormalizingFlow(latent_dim)
        self.hamiltonian = MLPHamiltonianNet(
            latent_dim, separable=separable, quadratic_t=quadratic_t
        )

        # J is ALWAYS the canonical symplectic structure [[0, I], [-I, 0]].
        # A learned constant J buys nothing over canonical here: the change of
        # variables p' = C^-T p turns any block coupling C into canonical while
        # keeping H separable, so C is absorbed by the kinetic net's first layer
        # (and by phi).  Fixing J keeps leapfrog trivially valid and symplectic.
        J_fixed = torch.zeros(latent_dim, latent_dim)
        J_fixed[:q_dim, q_dim:] = torch.eye(q_dim)
        J_fixed[q_dim:, :q_dim] = -torch.eye(q_dim)
        self.register_buffer("J_fixed", J_fixed)

        if learn_structure:
            # Only dissipation R and control B are learned now.  R is restricted
            # to the momentum block, R = [[0, 0], [0, L Lᵀ]]: a nonzero qq-block
            # would break the kinematic identity q̇ = ∂H/∂p — the very thing that
            # makes q "position" — and physical (Rayleigh) damping acts only on
            # momenta, mirroring the fixed-structure branch below.
            if state_dep_r:
                # L(z): small MLP z = (q, p) → lower-triangular entries, so
                # R_pp(z) = L(z) L(z)ᵀ is PSD at every state.  The output layer
                # is zero-initialised: the bias alone then plays the role of
                # the constant L_param (softplus(0)·I diagonal), so training
                # starts from exactly the constant-R model and the weights
                # learn only the state-dependent deviation.
                n_tril = q_dim * (q_dim + 1) // 2
                self.r_net = nn.Sequential(
                    nn.Linear(latent_dim, 64),
                    nn.Tanh(),
                    nn.Linear(64, n_tril),
                )
                nn.init.zeros_(self.r_net[-1].weight)
                nn.init.zeros_(self.r_net[-1].bias)
                self.register_buffer(
                    "_tril_idx", torch.tril_indices(q_dim, q_dim), persistent=False
                )
            else:
                self.L_param = nn.Parameter(torch.zeros(q_dim, q_dim))
                nn.init.normal_(self.L_param, std=1e-2)
            self.B = nn.Parameter(torch.zeros(q_dim, control_dim))
            nn.init.normal_(self.B, std=1e-2)
            self._has_dissipation = True
        else:
            R_fixed = torch.zeros(latent_dim, latent_dim)
            R_fixed[q_dim:, q_dim:] = damping * torch.eye(q_dim)
            self.register_buffer("R_fixed", R_fixed)
            self.register_buffer("B_fixed", torch.ones(q_dim, control_dim))
            self._has_dissipation = damping > 0

    # ── Structure matrix accessors ──────────────────────────────────────────

    def get_J(self) -> torch.Tensor:
        return self.J_fixed

    def get_L(self, z: torch.Tensor | None = None) -> torch.Tensor:
        """Cholesky factor of the momentum-block dissipation.

        Constant R: (q_dim, q_dim), z is ignored.  State-dependent R: pass the
        phase-space points z = cat(q, p) of shape (B, latent_dim) to get
        (B, q_dim, q_dim); z=None evaluates at z = 0 (the constant baseline
        carried by the r_net output bias) and returns (q_dim, q_dim).
        """
        if not self.state_dep_r:
            L_lower = self.L_param.tril(-1)
            diag_pos = F.softplus(self.L_param.diagonal())
            return L_lower + torch.diag(diag_pos)
        squeeze = z is None
        if z is None:
            z = self.J_fixed.new_zeros(1, self.latent_dim)
        q_dim = self.latent_dim // 2
        entries = self.r_net(z)  # (B, q_dim*(q_dim+1)//2)
        L = entries.new_zeros(z.shape[0], q_dim, q_dim)
        L[:, self._tril_idx[0], self._tril_idx[1]] = entries
        diag_pos = F.softplus(L.diagonal(dim1=-2, dim2=-1))
        L = L.tril(-1) + torch.diag_embed(diag_pos)
        return L.squeeze(0) if squeeze else L

    def get_R_pp(self, z: torch.Tensor | None = None) -> torch.Tensor:
        """Momentum-block of R — the only part that can be nonzero.

        (q_dim, q_dim), or (B, q_dim, q_dim) for state-dependent R with z given
        (see get_L for the z convention).
        """
        q_dim = self.latent_dim // 2
        if not self.learn_structure:
            return self.R_fixed[q_dim:, q_dim:]
        L = self.get_L(z)
        return L @ L.transpose(-2, -1)

    def get_R(self, z: torch.Tensor | None = None) -> torch.Tensor:
        if not self.learn_structure:
            return self.R_fixed
        q_dim = self.latent_dim // 2
        R_pp = self.get_R_pp(z)
        R = R_pp.new_zeros(*R_pp.shape[:-2], self.latent_dim, self.latent_dim)
        R[..., q_dim:, q_dim:] = R_pp
        return R

    def get_B(self) -> torch.Tensor:
        if not self.learn_structure:
            return self.B_fixed
        return self.B

    def structural_parameters(self) -> list[nn.Parameter]:
        """The learned structure parameters (R and B) — for the structural_lr group."""
        r_params = (
            list(self.r_net.parameters()) if self.state_dep_r else [self.L_param]
        )
        return r_params + [self.B]

    def _apply_R_pp(self, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """R_pp(z) @ v per sample; z = cat(q, p) is only consulted when R is state-dependent."""
        R_pp = self.get_R_pp(z if self.state_dep_r else None)
        if R_pp.dim() == 2:
            return v @ R_pp  # R_pp symmetric
        return (R_pp @ v.unsqueeze(-1)).squeeze(-1)

    # ── Dynamics ────────────────────────────────────────────────────────────

    @torch.enable_grad()
    def _dynamics(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(J − R(z)) ∇H at z = (q, p), written out blockwise for canonical J:

            q̇ = ∂H/∂p,   ṗ = −∂H/∂q − R_pp(z) ∂H/∂p

        R is evaluated at the *current* state, so each RK4 stage sees the
        dissipation appropriate to its own stage point.
        """
        half = self.latent_dim // 2
        z_ = torch.cat([q, p], dim=-1).requires_grad_(True)
        H_val = self.hamiltonian(z_[:, :half], z_[:, half:]).sum()
        grad_H = torch.autograd.grad(H_val, z_, create_graph=self.training)[0]
        g_q, g_p = grad_H[:, :half], grad_H[:, half:]
        dq = g_p
        dp = -g_q
        if self._has_dissipation:
            dp = dp - self._apply_R_pp(z_, g_p)
        return dq, dp

    def _controlled_dynamics(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        u: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dq, dp = self._dynamics(q, p)
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
        """One integration step of dz/dt = (J − R) ∇H + B u.

        Dispatches on self.integrator: 'rk4' (classic 4-stage, works for any
        structure) or 'leapfrog' (Strang split — symplectic Störmer-Verlet on
        the conservative + control part, with the dissipative R substep folded
        symmetrically around it; requires canonical J and separable H).
        """
        if dt is None:
            dt = self.dt
        if self.integrator == "leapfrog":
            return self._leapfrog_step(q, p, u, dt)
        return self._rk4_step(q, p, u, dt)

    def _rk4_step(
        self, q: torch.Tensor, p: torch.Tensor, u: torch.Tensor, dt: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Classic 4-stage explicit RK4 on the full field (J − R(z))∇H + Bu."""
        dq1, dp1 = self._controlled_dynamics(q, p, u)
        dq2, dp2 = self._controlled_dynamics(q + 0.5 * dt * dq1, p + 0.5 * dt * dp1, u)
        dq3, dp3 = self._controlled_dynamics(q + 0.5 * dt * dq2, p + 0.5 * dt * dp2, u)
        dq4, dp4 = self._controlled_dynamics(q + dt * dq3, p + dt * dp3, u)
        q_next = q + (dt / 6.0) * (dq1 + 2 * dq2 + 2 * dq3 + dq4)
        p_next = p + (dt / 6.0) * (dp1 + 2 * dp2 + 2 * dp3 + dp4)
        return q_next, p_next

    def _grad_H(
        self, q: torch.Tensor, p: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(∂H/∂q, ∂H/∂p) at (q, p). Keeps the graph for backprop through steps.

        With separable H = T(p) + V(q), ∂H/∂q = ∇V(q) and ∂H/∂p = ∇T(p), so the
        returned halves are exactly the per-coordinate forces leapfrog needs.
        """
        half = self.latent_dim // 2
        z_ = torch.cat([q, p], dim=-1).requires_grad_(True)
        H_val = self.hamiltonian(z_[:, :half], z_[:, half:]).sum()
        g = torch.autograd.grad(H_val, z_, create_graph=self.training)[0]
        return g[:, :half], g[:, half:]

    @torch.enable_grad()
    def _grad_V(self, q: torch.Tensor) -> torch.Tensor:
        """∇V(q) alone — skips the kinetic net entirely (separable H only)."""
        q_ = q.clone().requires_grad_(True)
        V_val = self.hamiltonian.potential(q_).sum()
        return torch.autograd.grad(V_val, q_, create_graph=self.training)[0]

    @torch.enable_grad()
    def _grad_T(self, p: torch.Tensor) -> torch.Tensor:
        """∇T(p) alone — skips the potential net entirely (separable H only)."""
        p_ = p.clone().requires_grad_(True)
        T_val = self.hamiltonian.kinetic(p_).sum()
        return torch.autograd.grad(T_val, p_, create_graph=self.training)[0]

    def _dissipation_substep(
        self, q: torch.Tensor, p: torch.Tensor, tau: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Substep of the dissipative flow ż = −R ∇H over (possibly negative) time tau.

        R acts only on the momentum block, so q is untouched and ṗ = −R_pp ∂H/∂p.
        Only called from the leapfrog path (separable H), so ∂H/∂p = ∇T(p).

        With a constant R and a quadratic kinetic energy the flow is linear,
        ṗ = −R_pp M⁻¹ p, and is integrated EXACTLY via a matrix exponential —
        tau < 0 then gives the exact anti-damping inverse, keeping the damped
        step exactly reversible.  (This also matches the env's exponential
        damping θ̇ *= exp(−c·dt).)

        With a state-dependent R (still quadratic kinetic) R_pp(z) is frozen at
        the substep's entry state and the same matrix-exp flow is applied.
        This is only O(tau²)-accurate in R's state variation, but — unlike an
        explicit Euler step — remains strictly dissipative for tau > 0 no
        matter how large R_pp(z) gets, since exp(−tau A) never overshoots
        through zero.  The tau < 0 substep freezes R at *its* entry state
        (the damped endpoint), so the inverse is exact only for constant R and
        O(tau²) otherwise — same order as the MLP-kinetic fallback below.

        With an MLP kinetic the flow is nonlinear; falls back to explicit
        Euler, whose tau < 0 step inverts the forward one only to O(tau²).
        """
        kin = self.hamiltonian.kinetic
        z = torch.cat([q, p], dim=-1) if self.state_dep_r else None
        R_pp = self.get_R_pp(z)
        if isinstance(kin, QuadraticKinetic):
            A = R_pp @ kin.M_inv()  # (d, d) or (B, d, d)
            if A.dim() == 2:
                return q, p @ torch.matrix_exp(-tau * A).T
            return q, (torch.matrix_exp(-tau * A) @ p.unsqueeze(-1)).squeeze(-1)
        g_T = self._grad_T(p)
        if R_pp.dim() == 2:
            return q, p - tau * g_T @ R_pp  # R_pp symmetric
        return q, p - tau * (R_pp @ g_T.unsqueeze(-1)).squeeze(-1)

    def _leapfrog_step(
        self, q: torch.Tensor, p: torch.Tensor, u: torch.Tensor, dt: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Strang split: D(dt/2) ∘ Leapfrog(dt) ∘ D(dt/2).

        The symplectic core is canonical kick-drift-kick on ż = J∇H, with the
        constant control force Bu folded into the two half-kicks (exact ZOH).
        The dissipative flow ż = −R∇H is Strang-split symmetrically around it so
        the composite stays 2nd-order and reduces to pure symplectic leapfrog
        when R = 0.

        Each kick/drift needs only one half of ∇H, and separable H (required
        for leapfrog) makes the halves independent, so _grad_V/_grad_T run just
        the sub-network they need instead of the full Hamiltonian.
        """
        if self._has_dissipation:
            q, p = self._dissipation_substep(q, p, dt / 2)

        Bu = u @ self.get_B().T  # constant force on p (zero-order hold)
        p = p - (dt / 2) * self._grad_V(q) + (dt / 2) * Bu
        q = q + dt * self._grad_T(p)
        p = p - (dt / 2) * self._grad_V(q) + (dt / 2) * Bu

        if self._has_dissipation:
            q, p = self._dissipation_substep(q, p, dt / 2)
        return q, p

    @torch.enable_grad()
    def reverse_step(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        u: torch.Tensor,
        dt: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inverse of controlled_step: recover (q_t, p_t) from (q_{t+1}, p_{t+1}) and u_t.

        Leapfrog: exact inverse.  The symplectic core is ρ-reversible
        (ρ: p ↦ −p, same u since forces depend only on q), so it inverts by
        running kick-drift-kick with flipped signs; the dissipation substeps
        invert with negative tau — exactly when T is quadratic (matrix-exp
        flow), to O(dt²) with an MLP kinetic.  RK4: approximate inverse via a
        −dt step (no exact inverse exists for an explicit RK step).
        """
        if dt is None:
            dt = self.dt
        if self.integrator == "leapfrog":
            return self._leapfrog_step_inverse(q, p, u, dt)
        return self._rk4_step(q, p, u, -dt)

    def _leapfrog_step_inverse(
        self, q: torch.Tensor, p: torch.Tensor, u: torch.Tensor, dt: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact inverse of _leapfrog_step: D(−dt/2) ∘ core⁻¹ ∘ D(−dt/2)."""
        if self._has_dissipation:
            q, p = self._dissipation_substep(q, p, -dt / 2)

        Bu = u @ self.get_B().T
        p = p + (dt / 2) * self._grad_V(q) - (dt / 2) * Bu
        q = q - dt * self._grad_T(p)
        p = p + (dt / 2) * self._grad_V(q) - (dt / 2) * Bu

        if self._has_dissipation:
            q, p = self._dissipation_substep(q, p, -dt / 2)
        return q, p

    # ── Phase-space helpers ─────────────────────────────────────────────────

    def encode(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """h_t → (q, p) via Φ."""
        s = self.phi(h)
        q_dim = self.latent_dim // 2
        return s[:, :q_dim], s[:, q_dim:]

    def encode_with_logdet(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """h_t → (q, p, log_det) via Φ. log_det is per-sample log|det J_Φ|."""
        s, log_det = self.phi.forward_with_logdet(h)
        q_dim = self.latent_dim // 2
        return s[:, :q_dim], s[:, q_dim:], log_det

    def decode(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """(q, p) → h_t via Φ⁻¹."""
        return self.phi.inverse(torch.cat([q, p], dim=-1))


# ---------------------------------------------------------------------------
# WorldModel — autoencoder + dynamics, one checkpoint
# ---------------------------------------------------------------------------


class WorldModel(nn.Module):
    """LSTM autoencoder + port-Hamiltonian dynamics, saved/loaded as one file.

    ``dynamics`` is None after Phase 1 only; a complete world model has both.
    ``data_config`` records how the training data was collected (img_size,
    max_steps, epsilon, energy_k, damping, ...) so downstream consumers can
    reproduce matching episodes without out-of-band hyperparameters.
    """

    def __init__(
        self,
        autoencoder: TemporalAutoencoder,
        dynamics: HamiltonianFlowModel | None = None,
        data_config: dict | None = None,
    ):
        super().__init__()
        self.autoencoder = autoencoder
        self.dynamics = dynamics
        self.data_config = dict(data_config or {})

    @property
    def latent_dim(self) -> int:
        return self.autoencoder.latent_dim

    @torch.no_grad()
    def dream(
        self,
        frames: torch.Tensor,
        actions: torch.Tensor,
        n_context: int,
        n_steps: int | None = None,
    ) -> torch.Tensor:
        """Encode n_context frames, roll out dynamics, decode back to pixels.

        Pipeline per dreamed step k:
            u = actions[n_context - 1 + k]
            (q, p) = dynamics.controlled_step(q, p, u)
            h_pred = dynamics.decode(q, p)              [phi^{-1}]
            frame  = autoencoder.decode_latent(h_pred)  [f_psi → decoder]

        Args:
            frames:    (T+1, C, H, W) ground-truth frames (any device)
            actions:   (T,) actions
            n_context: frames fed to the LSTM encoder before dreaming
            n_steps:   rollout length; clipped to the available actions
                       (None = as far as the actions allow)

        Returns:
            (n, C, H, W) dreamed frames on CPU; n may be 0 if no actions remain.
        """
        if self.dynamics is None:
            raise RuntimeError("WorldModel has no dynamics — Phase 2 has not been trained.")
        device = next(self.autoencoder.parameters()).device

        ctx = frames[:n_context].unsqueeze(0).to(device)     # (1, n_context, C, H, W)
        mu_ctx, _ = self.autoencoder.encoder.forward_all(ctx)
        h = mu_ctx[:, -1]                                     # (1, latent_dim)
        q, p = self.dynamics.encode(h)

        max_steps = len(actions) - (n_context - 1)
        n = max_steps if n_steps is None else min(n_steps, max_steps)

        dreamed = []
        for k in range(max(n, 0)):
            u = actions[n_context - 1 + k].view(1, 1).to(device=device, dtype=torch.float32)
            q, p = self.dynamics.controlled_step(q, p, u)
            h_pred = self.dynamics.decode(q, p)
            dreamed.append(self.autoencoder.decode_latent(h_pred).squeeze(0).cpu())

        if not dreamed:
            C, H, W = frames.shape[1:]
            return torch.empty(0, C, H, W)
        return torch.stack(dreamed)

    def save(
        self,
        run_dir,
        stem: str,
        hparams: dict,
        metrics: dict,
        epoch: int,
    ) -> None:
        from hamilton_rl.checkpoint import save_world_model

        save_world_model(run_dir, stem, self, hparams, metrics, epoch)

    @classmethod
    def load(cls, path, device: torch.device | None = None) -> "WorldModel":
        from hamilton_rl.checkpoint import load_world_model

        return load_world_model(path, device)
