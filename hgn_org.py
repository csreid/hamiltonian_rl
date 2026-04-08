import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------


class Encoder(nn.Module):
    """8-layer CNN encoder (Appendix A.1).

    Layer counts and filter widths follow the paper exactly:
        layer 1       — 32 filters
        layers 2-7    — 64 filters each
        layer 8       — 48 filters

    Three stride-2 convolutions downsample 32×32 → 16×16 → 8×8 → 4×4.
    Two 1×1 conv heads produce mu and log_var, each (B, latent_ch, 4, 4).
    """

    def __init__(self, n_frames: int = 4, latent_ch: int = 32):
        super().__init__()
        in_ch = n_frames * 3
        self.body = nn.Sequential(
            # layer 1 — 32×32 → 16×16
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            # layer 2 — 16×16 → 8×8
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            # layer 3
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            # layer 4 — 8×8 → 4×4
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            # layer 5
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            # layer 6
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            # layer 7
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            # layer 8 — 48 filters as stated in the paper
            nn.Conv2d(64, 48, 3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.mu_head = nn.Conv2d(48, latent_ch, 1)
        self.logvar_head = nn.Conv2d(48, latent_ch, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, n_frames*3, 32, 32)
        Returns:
            mu, log_var: each (B, latent_ch, 4, 4)
        """
        h = self.body(x)
        return self.mu_head(h), self.logvar_head(h)


class StateTransform(nn.Module):
    """f_ψ: 3-layer CNN that maps sampled z to the initial state s0.

    Appendix A.1: "The final encoder transformer network is a convolutional
    neural network with 3 layers and 64 filters on each layer."

        s0 = f_ψ(z),   z, s0 ∈ R^{B × latent_ch × 4 × 4}
    """

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
        """Args:
            s: (B, latent_ch, 4, 4)  — concatenated [q, p]
        Returns:
            T: (B,)
        """
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
        """Args:
            q: (B, pos_ch, 4, 4)
        Returns:
            V: (B,)
        """
        return self.to_scalar(self.conv(q)).sum(dim=[1, 2, 3])


class HamiltonianNet(nn.Module):
    """Convolutional Hamiltonian H_γ(q, p) = T_γ(q, p) + V_γ(q) → scalar.

    Formulated as kinetic + potential energy with independent weights.
    KineticNet takes the full phase-space state [q; p]; PotentialNet takes
    only the position q.  Softplus activations throughout (smooth, positive,
    suitable for second-order autodiff through the leapfrog integrator).

    Maps (q, p) ∈ R^{B × pos_ch × 4 × 4} × R^{B × pos_ch × 4 × 4} → R^B.
    """

    def __init__(self, latent_ch: int = 32):
        super().__init__()
        pos_ch = latent_ch // 2
        self.kinetic = KineticNet(latent_ch=latent_ch)
        self.potential = PotentialNet(pos_ch=pos_ch)

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Args:
            q: (B, pos_ch, 4, 4)
            p: (B, pos_ch, 4, 4)
        Returns:
            H = T(q, p) + V(q): (B,)
        """
        s = torch.cat([q, p], dim=1)  # (B, latent_ch, 4, 4)
        return self.kinetic(s) + self.potential(q)


class CoordHead(nn.Module):
    """Maps position q_t (B, pos_ch, 4, 4) -> pixel coords (B, 2) in [0, 1].

    A small MLP that flattens the spatial position latent and predicts
    normalised (x, y) pixel coordinates via sigmoid.
    """

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
        """Args:
            q: (B, pos_ch, 4, 4)
        Returns:
            coords: (B, 2) in [0, 1], as (x, y)
        """
        return self.net(q)


class DecoderBlock(nn.Module):
    """One progressive upsampling block (Appendix A.1).

    Nearest-neighbour ×2 upsample, then two conv layers (64 filters,
    LeakyReLU), closing with sigmoid — as described in the paper.
    """

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
    """Progressive decoder: q_t → reconstructed image (Appendix A.1).

    Three DecoderBlocks progressively upsample 4×4 → 8×8 → 16×16 → 32×32.
    A final 1×1 conv maps the 64-channel output to img_ch channels.

    Only the position component q_t is consumed; momentum p_t is not
    observable (Section 3.2: p_θ(x_t) = d_θ(q_t)).

    Input:  q_t  (B, pos_ch, 4, 4)
    Output: x̂_t  (B, img_ch, 32, 32)
    """

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
        return self.out_conv(x)  # (B, img_ch, 32, 32)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class HGN(nn.Module):
    """Hamiltonian Generative Network (Toth et al., ICLR 2020).

    Three components (Section 3.2 + Appendix A.1):

    1. Inference network
       The encoder takes the full image sequence concatenated along the
       channel dimension and outputs a diagonal-Gaussian posterior
       q_φ(z | x_0 … x_T).  Samples are passed through an additional
       convolutional network f_ψ to produce the initial phase-space state
       s_0 = f_ψ(z).

    2. Hamiltonian network
       A convolutional network H_γ maps abstract state s_t to a scalar
       energy.  The leapfrog (Störmer-Verlet) integrator rolls out future
       states using the Hamiltonian equations:
           dq/dt =  ∂H/∂p
           dp/dt = −∂H/∂q

    3. Decoder
       A progressive upsampling CNN maps the position component q_t back
       to pixel space: p_θ(x_t) = d_θ(q_t).  Momentum is never decoded.

    Latent state shape: s_t ∈ R^{B × 2·pos_ch × 4 × 4}
    The channel axis is split into position q (first pos_ch channels) and
    momentum p (last pos_ch channels).

    Objective (Equation 4):
        L = (1/(T+1)) Σ_t E[log p_θ(x_t | q_t)] − KL(q_φ(z) || p(z))

    Args:
        n_frames:  number of context frames stacked for inference
        pos_ch:    channel depth for q (and for p) — 16 in the paper
        img_ch:    image channels, 3 for RGB
        dt:        leapfrog step size — 0.125 in the paper; pass a negative
                   value to get a backward rollout (Figure 7 of the paper)
    """

    def __init__(
        self,
        n_frames: int = 4,
        pos_ch: int = 16,
        img_ch: int = 3,
        dt: float = 0.125,
    ):
        super().__init__()
        self.pos_ch = pos_ch
        self.latent_ch = 2 * pos_ch
        self.dt = dt

        self.encoder = Encoder(n_frames=n_frames, latent_ch=self.latent_ch)
        self.f_psi = StateTransform(latent_ch=self.latent_ch)
        self.hamiltonian = HamiltonianNet(latent_ch=self.latent_ch)
        self.decoder = Decoder(pos_ch=pos_ch, img_ch=img_ch)
        self.coord_head = CoordHead(pos_ch=pos_ch)

    # ── phase-space helpers ──────────────────────────────────────────────────

    def _split(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split (B, 2*pos_ch, H, W) → (q, p), each (B, pos_ch, H, W)."""
        return s[:, : self.pos_ch], s[:, self.pos_ch :]

    def _join(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return torch.cat([q, p], dim=1)

    def H(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Scalar Hamiltonian H_γ(q, p) = T(q,p) + V(q). Returns (B,)."""
        return self.hamiltonian(q, p)

    # ── leapfrog integrator ──────────────────────────────────────────────────

    @torch.enable_grad()
    def leapfrog_step(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        dt: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One Störmer-Verlet leapfrog step (Appendix A.6).

            p_{t+dt/2} = p_t       − (dt/2) · ∂H/∂q  at (q_t,       p_t      )
            q_{t+dt}   = q_t       +  dt    · ∂H/∂p  at (q_t,       p_{t+dt/2})
            p_{t+dt}   = p_{t+dt/2}− (dt/2) · ∂H/∂q  at (q_{t+dt},  p_{t+dt/2})

        Passing dt < 0 gives time-reversed dynamics (Figure 7).
        Passing |dt| > self.dt speeds up or slows down the rollout.
        """
        if dt is None:
            dt = self.dt

        # — half-step in p ——————————————————————————————————————————————————
        q_ = q.detach().requires_grad_(True)
        p_ = p.detach().requires_grad_(True)
        dH_dq = torch.autograd.grad(
            self.H(q_, p_).sum(), q_, create_graph=self.training
        )[0]
        p_half = p - 0.5 * dt * dH_dq

        # — full step in q ——————————————————————————————————————————————————
        q_ = q.detach().requires_grad_(True)
        p_half_ = p_half.detach().requires_grad_(True)
        dH_dp = torch.autograd.grad(
            self.H(q_, p_half_).sum(), p_half_, create_graph=self.training
        )[0]
        q_next = q + dt * dH_dp

        # — second half-step in p ———————————————————————————————————————————
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
        """Infer the initial phase-space state from the full image sequence.

        Section 3.2 + Figure 2: all T frames are channel-concatenated and
        passed through the encoder in one shot.

        Args:
            imgs: (B, T, C, H, W)

        Returns:
            q0:      (B, pos_ch, 4, 4)      — initial abstract position
            p0:      (B, pos_ch, 4, 4)      — initial abstract momentum
            kl:      (B,)                   — KL( q_φ || N(0,I) )
            mu:      (B, 2*pos_ch, 4, 4)
            log_var: (B, 2*pos_ch, 4, 4)
        """
        B, T, C, H, W = imgs.shape

        # Channel-concatenate all frames (Section 3.2).
        stacked = imgs.reshape(B, T * C, H, W)

        # Encoder → posterior parameters.
        mu, log_var = self.encoder(stacked)  # each (B, 2*pos_ch, 4, 4)

        # Clamp log_var to prevent exp() overflow / underflow during reparameterization.
        log_var = log_var.clamp(-10, 10)

        # Reparameterization trick.
        z = mu + torch.randn_like(mu) * (0.5 * log_var).exp()

        # f_ψ: sample → initial state (Section 3.2: "s0 = f_ψ(z)").
        s0 = self.f_psi(z)
        q0, p0 = self._split(s0)

        # KL( N(mu, sigma²) || N(0,I) ), summed over all latent dims.
        kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
        kl = kl.sum(dim=[1, 2, 3])  # (B,)

        return q0, p0, kl, mu, log_var

    def decode_coords(self, q: torch.Tensor) -> torch.Tensor:
        """Decode position latent to pixel coordinates.

        Args:
            q: (B, pos_ch, 4, 4)
        Returns:
            coords: (B, 2) normalised (x, y) in [0, 1]
        """
        return self.coord_head(q.detach())

    # ── rollout ──────────────────────────────────────────────────────────────

    def rollout(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        n_steps: int,
        dt: float | None = None,
        return_states: bool = False,
    ) -> (
        tuple[list[torch.Tensor], list[torch.Tensor]]
        | tuple[
            list[torch.Tensor],
            list[torch.Tensor],
            list[torch.Tensor],
            list[torch.Tensor],
        ]
    ):
        """Decode the initial state, then roll out for n_steps leapfrog steps.

        Args:
            q, p:          initial phase-space state, each (B, pos_ch, 4, 4)
            n_steps:       number of leapfrog integration steps
            dt:            step size override; negative → backward rollout;
                           ±2*self.dt → double speed (Figure 7 of the paper)
            return_states: if True, also return the q and p trajectories

        Returns:
            frames: list of (n_steps + 1) decoded images, each (B, img_ch, 32, 32).
            coords: list of (n_steps + 1) pixel coord tensors, each (B, 2) in [0, 1].
            qs:     (only if return_states=True) list of (n_steps + 1) q tensors.
            ps:     (only if return_states=True) list of (n_steps + 1) p tensors.
        """
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
