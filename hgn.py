import torch
import torch.nn as nn
import torchvision.transforms.functional as tvf
from cnn import ImpalaCNN


class StateTransformer(nn.Module):
    def __init__(self, phase_size=32):
        super().__init__()

        self.phase_size = phase_size

        self.fc = nn.Sequential(
            nn.Linear(phase_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, phase_size * 2),
        )

    def forward(self, z):
        out = self.fc(z)
        return out[..., : self.phase_size], out[
            ..., self.phase_size :
        ]  # returns p, q


class LatentDistribution(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size

        self.fc = nn.Linear(in_size, out_size * 2)

    def forward(self, X):
        # Reparameterization trick
        out = self.fc(X)
        mu_z = out[..., : self.out_size]
        logvar_z = out[..., self.out_size :]

        eps_z = torch.randn_like(mu_z)
        z = mu_z + eps_z * torch.exp(0.5 * logvar_z)

        # KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl = -0.5 * (1 + logvar_z - mu_z.pow(2) - logvar_z.exp()).sum(-1)

        return z, kl


class RecurrentHGN(nn.Module):
    def __init__(self, size=32, p_init_scale=3.0):
        super().__init__()
        self.size = size

        self.img_encoder = ImpalaCNN(features_dim=32)

        self.recurrent = nn.LSTM(32, 64)
        self.latent_encoder = LatentDistribution(64, size)

        self.f_psi = StateTransformer(size)

        self.hamiltonian = nn.Sequential(
            nn.Linear(2 * size, 256),
            nn.Softplus(),
            nn.Linear(256, 256),
            nn.Softplus(),
            nn.Linear(256, 1),
        )
        # Linear map from q only — no hidden layer, no sigmoid.
        # Deep MLPs can collapse to constant output (killing gradient to H).
        self.pos_decoder = nn.Linear(size, 1)

        # Large p at init: scale up the p-output half of f_psi so the leapfrog
        # immediately produces varied q across rollout steps, preventing the
        # zero-gradient fixed point from the very first iteration.
        with torch.no_grad():
            last = self.f_psi.fc[-1]
            last.weight.data[:size] *= p_init_scale
            last.bias.data[:size] = p_init_scale

    def decode(self, p, q):
        """Decode latent q to scalar physical position."""
        return self.pos_decoder(q)  # (..., 1)

    def H(self, p, q):
        """Compute scalar Hamiltonian from phase-space coordinates."""
        x = torch.cat([p, q], dim=-1)
        return self.hamiltonian(x).squeeze(-1)

    @torch.enable_grad()
    def hamiltonian_step(self, p, q, step_size=0.1):
        """
        Leapfrog (Störmer-Verlet) integration step.
        Preserves the symplectic structure of Hamiltonian dynamics.

        dq/dt =  ∂H/∂p
        dp/dt = -∂H/∂q
        """
        # Half-step for p using gradient w.r.t. q at (p, q)
        q_ = q.detach().requires_grad_(True)
        p_ = p.detach().requires_grad_(True)
        H_val = self.H(p_, q_).sum()
        dH_dq = torch.autograd.grad(H_val, q_, create_graph=self.training)[0]
        p_half = p - 0.5 * step_size * dH_dq

        # Full step for q using gradient w.r.t. p at (p_half, q)
        q_ = q.detach().requires_grad_(True)
        p_half_ = p_half.detach().requires_grad_(True)
        H_val = self.H(p_half_, q_).sum()
        dH_dp = torch.autograd.grad(H_val, p_half_, create_graph=self.training)[
            0
        ]
        q_next = q + step_size * dH_dp

        # Half-step for p using gradient w.r.t. q at (p_half, q_next)
        q_next_ = q_next.detach().requires_grad_(True)
        p_half_ = p_half.detach().requires_grad_(True)
        H_val = self.H(p_half_, q_next_).sum()
        dH_dq = torch.autograd.grad(H_val, q_next_, create_graph=self.training)[
            0
        ]
        p_next = p_half - 0.5 * step_size * dH_dq

        return p_next, q_next

    def forward(self, imgs):
        """Encode a context window of frames into a single initial phase-space state.

        All `seqlen` frames are processed by the LSTM; the final hidden state
        summarises the full context, mirroring the frame-stacking encoder in
        hgn_org.py.  The returned (p0, q0) represent the initial state at t=0
        from which the leapfrog rollout should start.

        Args:
            imgs: (B, seqlen, C, H, W)
        Returns:
            p0:  (B, size)
            q0:  (B, size)
            kl:  (B,)
        """
        batchsize, seqlen, C, H, W = imgs.shape

        # Visual encoding — flatten batch and time, encode, restore.
        out = imgs.view(seqlen * batchsize, C, H, W)
        out = tvf.resize(out, (84, 84))
        out = self.img_encoder(out)

        embed_dim = out.shape[-1]
        out = out.view(seqlen, batchsize, embed_dim)

        # LSTM over the full context; take the final hidden state as the summary.
        out, _ = self.recurrent(out)  # (seqlen, batchsize, hidden)
        out = out[-1]  # (batchsize, hidden)

        # Variational bottleneck → initial phase-space state.
        z, kl = self.latent_encoder(out)  # (batchsize, size), (batchsize,)
        p0, q0 = self.f_psi(z)  # each (batchsize, size)

        return p0, q0, kl
