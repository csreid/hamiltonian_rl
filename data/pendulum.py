"""Pendulum-v1 data collection with pixel observations."""

from __future__ import annotations

import random

import gymnasium as gym
import numpy as np
import torch
import torchvision.transforms.functional as TF
from gymnasium import spaces
from torch.utils.data import Dataset
from tqdm import tqdm

from hamilton_rl.mppi import MPPIConfig, mppi_plan, shift_mean

# Pendulum-v1 physical constants (gymnasium defaults)
_G = 10.0
_H_STAR = 20.0  # 2 * m * g * l  with m=l=1, g=10
_DT = 0.05  # integration timestep (matches Pendulum-v1's default)
_MAX_SPEED = 8.0  # nominal velocity scale used for normalization/plot ranges
                   # (Pendulum-v1's old hard clip value; no longer physically
                   # enforced — see _DRAG_COEFF, which bounds speed smoothly
                   # instead)
_DRAG_COEFF = 0.05  # quadratic (Rayleigh) drag coefficient, ṗ_drag = -c·p·|p|
                     # replaces the old hard |theta_dot| clip. Calibrated so
                     # typical energy-pump episodes are barely affected
                     # (mean peak ~7.9 rad/s vs ~8.6 undamped) while sustained
                     # max-torque spin-up saturates around ~10-12 rad/s
                     # instead of growing unboundedly.


# ── Analytic (batched) dynamics ──────────────────────────────────────────────
#
# Canonical Hamiltonian coordinates: q = theta, p = theta_dot, with
# T(p) = p²/2 (unit "mass") and V(q) = 1.5·g·cos(theta), so that the
# unforced/undamped flow q̇ = ∂T/∂p = p, ṗ = -∂V/∂q = 1.5·g·sin(theta)
# matches Pendulum-v1's own equation of motion (m=l=1). Control enters as
# ṗ += 3·u (matching gym's B=3), dissipation (linear ``damping`` and/or
# quadratic ``drag``) acts on p alone, exactly as in
# ``HamiltonianFlowModel``'s R (see hamilton_rl/models.py).
#
# Integrated with the same Strang-split leapfrog (kick-drift-kick, with the
# dissipative flow split symmetrically around it) as
# ``HamiltonianFlowModel._leapfrog_step`` — this is the single source of
# truth for pendulum physics: the real env (``PendulumPixelEnv.step``), the
# state-only collectors, and the planner's MPPI ground-truth imagination all
# call this same function, so there is no way for them to drift apart.


def angle_normalize(theta: torch.Tensor) -> torch.Tensor:
    return ((theta + torch.pi) % (2 * torch.pi)) - torch.pi


def _grad_V(theta: torch.Tensor, g: float = _G) -> torch.Tensor:
    return -1.5 * g * torch.sin(theta)


def _drag_substep(p: torch.Tensor, tau: float, c: float) -> torch.Tensor:
    """Exact flow of dp/dt = -c·p·|p| over time ``tau`` (any sign).

    Closed form: p(t) = sign(p0) / (1/|p0| + c·t). Evaluating the same
    formula at -t recovers p0 exactly (up to float precision) — the same
    functional form inverts itself, so this is exactly reversible.
    """
    if c == 0.0:
        return p
    sign = torch.sign(p)
    inv_abs_p = 1.0 / p.abs().clamp_min(1e-12) + c * tau
    return sign / inv_abs_p.clamp_min(1e-12)


def _dissipation_substep(
    q: torch.Tensor, p: torch.Tensor, tau: float, damping: float, drag: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward dissipative flow over time ``tau``: exponential damping, then drag."""
    if damping != 0.0:
        p = p * float(np.exp(-damping * tau))
    if drag != 0.0:
        p = _drag_substep(p, tau, drag)
    return q, p


def _dissipation_substep_inverse(
    q: torch.Tensor, p: torch.Tensor, tau: float, damping: float, drag: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact inverse of ``_dissipation_substep`` (reverses composition order)."""
    if drag != 0.0:
        p = _drag_substep(p, -tau, drag)
    if damping != 0.0:
        p = p * float(np.exp(damping * tau))
    return q, p


def analytic_pendulum_step(
    theta: torch.Tensor,
    theta_dot: torch.Tensor,
    u: torch.Tensor,
    dt: float = _DT,
    damping: float = 0.0,
    drag: float = _DRAG_COEFF,
    g: float = _G,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One leapfrog (Strang-split) step of the pendulum's port-Hamiltonian dynamics.

    Structurally identical to ``HamiltonianFlowModel._leapfrog_step``: a
    dissipative half-step, canonical kick-drift-kick (with the constant
    control force folded into the two half-kicks as an exact zero-order
    hold), then the other dissipative half-step. Plain tensor ops over
    arbitrarily-shaped ``theta``/``theta_dot``/``u`` — lets a planner
    simulate many candidate rollouts in parallel without stepping copies of
    the real env.
    """
    u = u.clamp(-2.0, 2.0)
    q, p = theta, theta_dot
    has_dissipation = damping != 0.0 or drag != 0.0

    if has_dissipation:
        q, p = _dissipation_substep(q, p, dt / 2, damping, drag)

    Bu = 3.0 * u
    p = p - (dt / 2) * _grad_V(q, g) + (dt / 2) * Bu
    q = q + dt * p
    p = p - (dt / 2) * _grad_V(q, g) + (dt / 2) * Bu

    if has_dissipation:
        q, p = _dissipation_substep(q, p, dt / 2, damping, drag)

    return q, p


def analytic_pendulum_step_inverse(
    theta: torch.Tensor,
    theta_dot: torch.Tensor,
    u: torch.Tensor,
    dt: float = _DT,
    damping: float = 0.0,
    drag: float = _DRAG_COEFF,
    g: float = _G,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact inverse of ``analytic_pendulum_step``: recover (theta_t, theta_dot_t)
    from (theta_{t+1}, theta_dot_{t+1}) and u_t.

    Mirrors ``HamiltonianFlowModel._leapfrog_step_inverse``: D(-dt/2) ∘
    core⁻¹ ∘ D(-dt/2), exact up to float precision (the symplectic core is
    ρ-reversible; the dissipation substep inverts exactly via
    ``_dissipation_substep_inverse``).
    """
    u = u.clamp(-2.0, 2.0)
    q, p = theta, theta_dot
    has_dissipation = damping != 0.0 or drag != 0.0

    if has_dissipation:
        q, p = _dissipation_substep_inverse(q, p, dt / 2, damping, drag)

    Bu = 3.0 * u
    p = p + (dt / 2) * _grad_V(q, g) - (dt / 2) * Bu
    q = q - dt * p
    p = p + (dt / 2) * _grad_V(q, g) - (dt / 2) * Bu

    if has_dissipation:
        q, p = _dissipation_substep_inverse(q, p, dt / 2, damping, drag)

    return q, p


# ── Image preprocessing ──────────────────────────────────────────────────────


def preprocess_frame(frame: np.ndarray, img_size: int = 64) -> torch.Tensor:
    """Convert a raw Pendulum RGB frame to a normalised (3, H, W) tensor."""
    t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    return TF.resize(t, [img_size, img_size], antialias=True)


# ── Pixel wrapper ─────────────────────────────────────────────────────────────


class PendulumPixelEnv(gym.Wrapper):
    """Pendulum-v1 with (3, img_size, img_size) uint8 pixel observations.

    Physics is driven entirely by ``analytic_pendulum_step`` (see module
    docstring above ``analytic_pendulum_step``) — gym's own built-in step is
    bypassed so the real env can never drift from the "ground truth" dynamics
    used elsewhere (e.g. the planner's MPPI imagination).

    Args:
        img_size:   Side length of the square pixel observation.
        damping:    Linear viscous damping coefficient. ``theta_dot *=
                    exp(-damping * dt)`` each step. 0.0 (default) disables it.
        drag:       Quadratic (Rayleigh) drag coefficient — see
                    ``_DRAG_COEFF``. Smoothly bounds angular velocity in
                    place of gym's old hard clip; 0.0 disables it.
    """

    def __init__(self, img_size: int = 64, damping: float = 0.0, drag: float = _DRAG_COEFF):
        env = gym.make("Pendulum-v1", render_mode="rgb_array")
        super().__init__(env)
        self.img_size = img_size
        self.damping = damping
        self.drag = drag
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(3, img_size, img_size),
            dtype=np.uint8,
        )

    def _obs(self) -> np.ndarray:
        # Suppress gym's torque-arrow overlay: its direction/magnitude encodes
        # the applied action, so a faithful autoencoder would bake the action
        # into the latent h.  We want h to be pure physical state, with the
        # action entering the Phase-2 dynamics only through the B matrix.
        self.env.unwrapped.last_u = None  # type: ignore[union-attr]
        frame = self.env.render()  # (H, W, 3) uint8
        t = preprocess_frame(frame, self.img_size)
        return (t * 255).byte().numpy()

    def render_with_action(self, u: float) -> np.ndarray:
        """Render the *current* state with gym's torque-arrow overlay for display.

        Display-only counterpart to `_obs()`: every observation that could
        reach an encoder must stay arrow-free (see `_obs()`'s docstring), but
        a human-facing view benefits from seeing the applied action. Restores
        `last_u = None` afterward so `_obs()`'s no-arrow invariant still holds
        for any observation taken after this call.
        """
        self.env.unwrapped.last_u = u  # type: ignore[union-attr]
        frame = self.env.render()  # (H, W, 3) uint8
        t = preprocess_frame(frame, self.img_size)
        self.env.unwrapped.last_u = None  # type: ignore[union-attr]
        return (t * 255).byte().numpy()

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        return self._obs(), {}

    def set_state(self, theta: float, theta_dot: float) -> np.ndarray:
        """Force the environment into (theta, theta_dot) and return the resulting observation."""
        self.env.unwrapped.state = np.array([theta, theta_dot], dtype=np.float64)  # type: ignore[union-attr]
        return self._obs()

    def step(self, action):
        theta, theta_dot = self.env.unwrapped.state  # type: ignore[union-attr]
        u = torch.as_tensor([float(action[0])], dtype=torch.float64)
        new_theta, new_theta_dot = analytic_pendulum_step(
            torch.as_tensor([theta], dtype=torch.float64),
            torch.as_tensor([theta_dot], dtype=torch.float64),
            u,
            damping=self.damping,
            drag=self.drag,
        )
        self.env.unwrapped.state = np.array(  # type: ignore[union-attr]
            [new_theta.item(), new_theta_dot.item()], dtype=np.float64
        )
        return self._obs(), 0.0, False, False, {}


# ── Energy-based controller ───────────────────────────────────────────────────


def _energy(theta: float, theta_dot: float) -> float:
    # H = 0 at the bottom (theta=pi), H = 2g at the upright (theta=0)
    # using gymnasium's convention where theta=0 is upright
    return 0.5 * theta_dot**2 + _G * (1.0 + np.cos(theta))


def _energy_pumping_action(
    theta: float, theta_dot: float, k: float = 1.0
) -> float:
    """u = k * (H - H*) * theta_dot * cos(theta), clamped to [-2, 2]."""
    H = _energy(theta, theta_dot)
    u = k * (H - _H_STAR) * theta_dot * np.cos(theta)
    return float(np.clip(u, -2.0, 2.0))


def _pd_action(
    theta: float,
    theta_dot: float,
    kp: float,
    kd: float,
) -> float:
    """Simple PD controller around upright (theta=0), clamped to [-2, 2]."""
    u = -(kp * theta + kd * theta_dot)
    return float(np.clip(u, -2.0, 2.0))


_UPRIGHT_THRESHOLD = 0.3  # radians — switch to PD when |theta| < this


def _grid_seeds(
    n_points: int,
    min_theta: float = -np.pi,
    max_theta: float = np.pi,
    min_theta_dot: float = -10.0,
    max_theta_dot: float = 10.0,
) -> np.ndarray:
    """As-square-as-possible covering grid of (theta, theta_dot) initial states.

    Picks grid dimensions so cells are as close to square as possible given
    the aspect ratio of the two ranges (theta_dot spans 20, theta spans
    ~6.28, so the grid is wider than it is tall), then tiles/truncates to
    exactly n_points and shuffles so consecutive episodes (which some
    downstream code splits into interleaved train/val halves) aren't grouped
    into a single region of the grid.

    Returns (n_points, 2) array of (theta, theta_dot) pairs.
    """
    theta_range = max_theta - min_theta
    theta_dot_range = max_theta_dot - min_theta_dot
    aspect = theta_dot_range / theta_range
    n_theta = max(1, round(np.sqrt(n_points / aspect)))
    n_theta_dot = max(1, round(n_points / n_theta))

    thetas = np.linspace(min_theta, max_theta, n_theta)
    theta_dots = np.linspace(min_theta_dot, max_theta_dot, n_theta_dot)
    grid_t, grid_td = np.meshgrid(thetas, theta_dots, indexing="xy")
    grid = np.stack([grid_t.ravel(), grid_td.ravel()], axis=-1)  # (n_theta * n_theta_dot, 2)

    reps = int(np.ceil(n_points / len(grid)))
    seeds = np.tile(grid, (reps, 1))[:n_points]
    np.random.shuffle(seeds)
    return seeds


# ── Data collection ──────────────────────────────────────────────────────────

# Zero-action steps at the start of each grid-seeded episode. The covering
# grid is treated as *targets* reached this many steps in (backward-solved
# via ``_seeds_reaching_targets``, exactly as in
# ``collect_zero_trajectories_targeted``) rather than as raw initial states:
# a raw high-energy seed near theta=0 swings away within a few steps, and a
# causal encoder has no motion context at t=0 anyway, so direct seeding
# leaves that corner of the grid unrepresented at any encodable timestep.
# Matches the energy-landscape drawer's default context (context_frames=5,
# i.e. 4 steps of motion).
_GRID_WARMUP_STEPS = 4


def _targeted_grid_seeds(
    n_episodes: int,
    warmup_steps: int,
    damping: float,
    drag: float,
    max_vel: float = _MAX_SPEED,
) -> np.ndarray:
    """Backward-solved seeds whose zero-action state at step ``warmup_steps``
    covers the standard grid (see ``collect_zero_trajectories_targeted``)."""
    targets = _grid_seeds(n_episodes, min_theta_dot=-max_vel, max_theta_dot=max_vel)
    return _seeds_reaching_targets(
        targets, max_steps=warmup_steps, damping=damping, drag=drag
    )


def _collect_episodes(
    n_episodes: int,
    img_size: int,
    epsilon: float,
    max_steps: int,
    energy_k: float,
    desc: str,
    damping: float = 0.0,
    warmup_steps: int = _GRID_WARMUP_STEPS,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Shared episode collection loop.

    The first ``warmup_steps`` actions of every episode are zero, and the
    seed is backward-solved so the episode passes exactly through its grid
    point at the end of that warmup (see ``_targeted_grid_seeds``); the
    policy takes over afterwards.

    Returns a list of (frames, actions, states) tuples:
        frames  : (T+1, 3, img_size, img_size) float32 [0,1]
        actions : (T,)  float32
        states  : (T+1, 3) float32 — (cos(theta), sin(theta), theta_dot) at each frame, post-damping
    """
    env = PendulumPixelEnv(img_size=img_size, damping=damping)
    episodes = []
    warmup = min(warmup_steps, max_steps)
    seeds = _targeted_grid_seeds(n_episodes, warmup, damping=damping, drag=env.drag)

    try:
        for i in tqdm(range(n_episodes), desc=desc):
            kp = random.uniform(2.0, 15.0)
            kd = random.uniform(0.5, 5.0)

            env.reset()
            theta0, theta_dot0 = seeds[i]
            obs = env.set_state(theta0, theta_dot0)
            frames = [torch.from_numpy(obs).float() / 255.0]
            actions = []
            states = [np.array([np.cos(theta0), np.sin(theta0), theta_dot0], dtype=np.float32)]

            for t in range(max_steps):
                theta, theta_dot = env.unwrapped.state  # type: ignore[union-attr]

                if t < warmup:
                    action = 0.0
                elif random.random() < epsilon:
                    action = float(np.random.uniform(-2.0, 2.0))
                elif abs(theta) < _UPRIGHT_THRESHOLD:
                    action = _pd_action(theta, theta_dot, kp, kd)
                else:
                    action = _energy_pumping_action(theta, theta_dot, energy_k)

                obs, _, _, _, _ = env.step(np.array([action], dtype=np.float32))
                theta_next, theta_dot_next = env.unwrapped.state  # type: ignore[union-attr]  # post-damping
                frames.append(torch.from_numpy(obs).float() / 255.0)
                actions.append(action)
                states.append(
                    np.array([np.cos(theta_next), np.sin(theta_next), theta_dot_next], dtype=np.float32)
                )

            episodes.append(
                (
                    torch.stack(frames),  # (T+1, 3, H, W)
                    torch.tensor(actions, dtype=torch.float32),  # (T,)
                    torch.from_numpy(np.stack(states)),  # (T+1, 3)
                )
            )
    finally:
        env.close()

    return episodes


def collect_val_trajectories(
    n_episodes: int,
    img_size: int,
    max_steps: int = 200,
    energy_k: float = 1.0,
    damping: float = 0.0,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Collect validation episodes (no random actions)."""
    return _collect_episodes(
        n_episodes=n_episodes,
        img_size=img_size,
        epsilon=0.0,
        max_steps=max_steps,
        energy_k=energy_k,
        desc="Val trajectories (energy-pump)",
        damping=damping,
    )


def collect_random_trajectories(
    n_episodes: int,
    img_size: int,
    max_steps: int = 200,
    damping: float = 0.0,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Collect episodes driven by purely random actions (epsilon=1)."""
    return _collect_episodes(
        n_episodes=n_episodes,
        img_size=img_size,
        epsilon=1.0,
        max_steps=max_steps,
        energy_k=1.0,  # unused when epsilon=1
        desc="Val trajectories (random)",
        damping=damping,
    )


def _spin_action(theta_dot: float, k: float = 1.0) -> float:
    """Always torque in the direction of current angular velocity to maximise spin."""
    u = k * 2.0 * np.sign(theta_dot) if theta_dot != 0.0 else float(np.random.choice([-2.0, 2.0]))
    return float(np.clip(u, -2.0, 2.0))


def _collect_spin_episodes(
    n_episodes: int,
    img_size: int,
    max_steps: int,
    damping: float = 0.0,
    warmup_steps: int = _GRID_WARMUP_STEPS,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Collect episodes with a spin-maximising controller (maximises |theta_dot|).

    Grid-targeted seeding with a zero-action warmup, same as
    ``_collect_episodes``.
    """
    env = PendulumPixelEnv(img_size=img_size, damping=damping)
    episodes = []
    warmup = min(warmup_steps, max_steps)
    seeds = _targeted_grid_seeds(n_episodes, warmup, damping=damping, drag=env.drag)

    try:
        for i in tqdm(range(n_episodes), desc="Val trajectories (spin)"):
            env.reset()
            theta0, theta_dot0 = seeds[i]
            obs = env.set_state(theta0, theta_dot0)
            frames = [torch.from_numpy(obs).float() / 255.0]
            actions = []
            states = [np.array([np.cos(theta0), np.sin(theta0), theta_dot0], dtype=np.float32)]

            for t in range(max_steps):
                theta, theta_dot = env.unwrapped.state  # type: ignore[union-attr]
                action = 0.0 if t < warmup else _spin_action(theta_dot)
                obs, _, _, _, _ = env.step(np.array([action], dtype=np.float32))
                theta_next, theta_dot_next = env.unwrapped.state  # type: ignore[union-attr]
                frames.append(torch.from_numpy(obs).float() / 255.0)
                actions.append(action)
                states.append(np.array([np.cos(theta_next), np.sin(theta_next), theta_dot_next], dtype=np.float32))

            episodes.append(
                (
                    torch.stack(frames),
                    torch.tensor(actions, dtype=torch.float32),
                    torch.from_numpy(np.stack(states)),  # (T+1, 3)
                )
            )
    finally:
        env.close()

    return episodes


def collect_spin_trajectories(
    n_episodes: int,
    img_size: int,
    max_steps: int = 200,
    damping: float = 0.0,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Collect episodes driven by a spin-maximising controller."""
    return _collect_spin_episodes(
        n_episodes=n_episodes,
        img_size=img_size,
        max_steps=max_steps,
        damping=damping,
    )


# ── Single switching-policy long rollout ──────────────────────────────────────
#
# Ground-truth MPPI cost, shared with pendulum_planner.py's
# ``run_ground_truth_mppi`` (which imports ``_mppi_gt_cost_fn`` from here
# rather than defining its own) so there is one definition of "what does it
# mean to balance the pendulum" for both data collection and the planner demo.


def _mppi_gt_cost_fn(
    theta0: torch.Tensor,
    theta_dot0: torch.Tensor,
    damping: float,
    action_weight: float,
    drag: float = _DRAG_COEFF,
):
    def cost_fn(candidates: torch.Tensor) -> torch.Tensor:
        K, H, _ = candidates.shape
        theta = theta0.expand(K).clone()
        theta_dot = theta_dot0.expand(K).clone()
        total = torch.zeros(K, device=candidates.device, dtype=candidates.dtype)
        for t in range(H):
            u = candidates[:, t, 0]
            theta, theta_dot = analytic_pendulum_step(theta, theta_dot, u, damping=damping, drag=drag)
            total = total + angle_normalize(theta) ** 2 + 0.1 * theta_dot**2 + action_weight * u**2
        return total

    return cost_fn


_BALANCE_MPPI_CFG = MPPIConfig(
    horizon=20, n_samples=256, n_iterations=1, noise_std=1.0, temperature=1.0,
    action_low=-2.0, action_high=2.0,
)
_BALANCE_ACTION_WEIGHT = 0.001


class _MPPIBalancePolicy:
    """Closed-loop receding-horizon MPPI controller, warm-started across steps.

    Plans against the same ground-truth analytic dynamics the real env
    steps with (see module docstring above ``analytic_pendulum_step``), so
    it needs no learned model to drive the "try to balance" phase of
    ``_collect_switching_rollout``.
    """

    def __init__(self, damping: float, drag: float):
        self.damping = damping
        self.drag = drag
        self.mean = torch.zeros(_BALANCE_MPPI_CFG.horizon, 1)

    def act(self, theta: float, theta_dot: float) -> float:
        cost_fn = _mppi_gt_cost_fn(
            torch.tensor(float(theta)), torch.tensor(float(theta_dot)),
            self.damping, _BALANCE_ACTION_WEIGHT, drag=self.drag,
        )
        self.mean = mppi_plan(self.mean, cost_fn, _BALANCE_MPPI_CFG)
        u0 = float(self.mean[0, 0].clamp(_BALANCE_MPPI_CFG.action_low, _BALANCE_MPPI_CFG.action_high))
        self.mean = shift_mean(self.mean)
        return u0


# Order matters only in that it determines which quarter of the rollout each
# behavior occupies; equal-length blocks, MPPI-balance placed after both spin
# directions so it always gets to recover from whichever spin state it's left in.
_SWITCHING_PHASES = ("spin_cw", "spin_ccw", "balance", "random")


def _switching_phase_action(
    phase: str, theta: float, theta_dot: float, balance_policy: _MPPIBalancePolicy
) -> float:
    if phase == "spin_cw":
        return 2.0
    if phase == "spin_ccw":
        return -2.0
    if phase == "balance":
        return balance_policy.act(theta, theta_dot)
    return float(np.random.uniform(-2.0, 2.0))  # random


def collect_switching_rollout(
    n_steps: int,
    img_size: int,
    damping: float = 0.0,
    drag: float = _DRAG_COEFF,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One continuous rollout of ``n_steps`` that cycles through four behaviors
    in equal-length blocks: constant max-torque CW spin, constant max-torque
    CCW spin, an MPPI controller balancing at the upright, and uniform-random
    actions (see ``_SWITCHING_PHASES``).

    Unlike the per-episode collectors above, there is exactly one reset —
    every step after the first carries real motion history, so any window
    later sliced out of the middle of this rollout (see
    ``PendulumRolloutDataset``) has genuine causal context for the encoder
    without needing the zero-action warmup those collectors rely on.

    Returns the same (frames, actions, states) shapes as ``_collect_episodes``,
    but for the full ``n_steps``-long trajectory rather than one episode.
    """
    env = PendulumPixelEnv(img_size=img_size, damping=damping, drag=drag)
    balance_policy = _MPPIBalancePolicy(damping=damping, drag=drag)
    n_phases = len(_SWITCHING_PHASES)
    block = max(1, n_steps // n_phases)

    try:
        env.reset()
        theta0 = float(np.random.uniform(-np.pi, np.pi))
        theta_dot0 = float(np.random.uniform(-_MAX_SPEED, _MAX_SPEED))
        obs = env.set_state(theta0, theta_dot0)

        frames = [torch.from_numpy(obs).float() / 255.0]
        actions = []
        states = [np.array([np.cos(theta0), np.sin(theta0), theta_dot0], dtype=np.float32)]

        for t in tqdm(range(n_steps), desc="Collecting switching rollout"):
            phase = _SWITCHING_PHASES[min(t // block, n_phases - 1)]
            theta, theta_dot = env.unwrapped.state  # type: ignore[union-attr]
            action = _switching_phase_action(phase, theta, theta_dot, balance_policy)

            obs, _, _, _, _ = env.step(np.array([action], dtype=np.float32))
            theta_next, theta_dot_next = env.unwrapped.state  # type: ignore[union-attr]
            frames.append(torch.from_numpy(obs).float() / 255.0)
            actions.append(action)
            states.append(
                np.array([np.cos(theta_next), np.sin(theta_next), theta_dot_next], dtype=np.float32)
            )
    finally:
        env.close()

    return (
        torch.stack(frames),  # (n_steps+1, 3, H, W)
        torch.tensor(actions, dtype=torch.float32),  # (n_steps,)
        torch.from_numpy(np.stack(states)),  # (n_steps+1, 3)
    )


class PendulumRolloutDataset(Dataset):
    """Random windows of one long switching rollout — no fixed episodes.

    Holds a single (frames, actions, states) trajectory (see
    ``collect_switching_rollout``) and draws a *fresh* random offset on every
    ``__getitem__`` — the index is ignored, so each epoch trains on a new set
    of windows rather than a frozen slicing. ``n_windows`` only sets the
    nominal epoch size for the DataLoader.

    Each item matches one old "episode" shape-for-shape:
        frames  : (window_len+1, 3, H, W) float32 [0,1]
        actions : (window_len,)  float32
        states  : (window_len+1, 3) float32
    """

    def __init__(
        self,
        rollout: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        window_len: int,
        n_windows: int,
    ):
        frames, actions, states = rollout
        rollout_steps = actions.shape[0]
        if rollout_steps < window_len:
            raise ValueError(
                f"rollout has {rollout_steps} steps but window_len is {window_len}"
            )
        self.frames = frames
        self.actions = actions
        self.states = states
        self.window_len = window_len
        self.n_windows = n_windows
        self.max_offset = rollout_steps - window_len

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        start = random.randint(0, self.max_offset)
        end = start + self.window_len
        return (
            self.frames[start : end + 1],
            self.actions[start:end],
            self.states[start : end + 1],
        )


def _seeds_reaching_targets(
    targets: np.ndarray,
    max_steps: int,
    damping: float = 0.0,
    drag: float = _DRAG_COEFF,
) -> np.ndarray:
    """Find zero-action seeds that land at ``targets`` after ``max_steps`` steps.

    Walks backward from each target one step at a time via
    ``analytic_pendulum_step_inverse`` — exact up to float precision (see
    that function's docstring), unlike the old semi-implicit-Euler map,
    which had no exact multi-step inverse.

    Returns an (N, 2) array of (theta0, theta_dot0) seeds, same shape as
    ``targets``.
    """
    theta = torch.as_tensor(targets[:, 0].copy(), dtype=torch.float64)
    theta_dot = torch.as_tensor(targets[:, 1].copy(), dtype=torch.float64)
    zero_u = torch.zeros_like(theta)
    for _ in range(max_steps):
        theta, theta_dot = analytic_pendulum_step_inverse(
            theta, theta_dot, zero_u, damping=damping, drag=drag
        )
    return np.stack([theta.numpy(), theta_dot.numpy()], axis=-1)


def _collect_zero_episodes(
    n_episodes: int,
    img_size: int,
    max_steps: int,
    damping: float = 0.0,
    seeds: np.ndarray | None = None,
    drag: float = _DRAG_COEFF,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Collect episodes with the action fixed at zero (uncontrolled dynamics).

    ``seeds``: explicit (N, 2) array of (theta0, theta_dot0) to start from;
    defaults to the standard covering grid (``_grid_seeds``).
    """
    env = PendulumPixelEnv(img_size=img_size, damping=damping, drag=drag)
    episodes = []
    if seeds is None:
        seeds = _grid_seeds(n_episodes)

    try:
        for i in tqdm(range(n_episodes), desc="Val trajectories (zero action)"):
            env.reset()
            theta0, theta_dot0 = seeds[i]
            obs = env.set_state(theta0, theta_dot0)
            frames = [torch.from_numpy(obs).float() / 255.0]
            actions = []
            states = [np.array([np.cos(theta0), np.sin(theta0), theta_dot0], dtype=np.float32)]

            for _ in range(max_steps):
                action = 0.0
                obs, _, _, _, _ = env.step(np.array([action], dtype=np.float32))
                theta_next, theta_dot_next = env.unwrapped.state  # type: ignore[union-attr]
                frames.append(torch.from_numpy(obs).float() / 255.0)
                actions.append(action)
                states.append(np.array([np.cos(theta_next), np.sin(theta_next), theta_dot_next], dtype=np.float32))

            episodes.append(
                (
                    torch.stack(frames),
                    torch.tensor(actions, dtype=torch.float32),
                    torch.from_numpy(np.stack(states)),  # (T+1, 3)
                )
            )
    finally:
        env.close()

    return episodes


def collect_zero_trajectories(
    n_episodes: int,
    img_size: int,
    max_steps: int = 200,
    damping: float = 0.0,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Collect episodes with action always zero (uncontrolled dynamics)."""
    return _collect_zero_episodes(
        n_episodes=n_episodes,
        img_size=img_size,
        max_steps=max_steps,
        damping=damping,
    )


def collect_zero_trajectory_from(
    theta0: float,
    theta_dot0: float,
    img_size: int,
    max_steps: int,
    damping: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single zero-action rollout from an explicit seed (theta0, theta_dot0).

    Unlike ``collect_zero_trajectories`` (which seeds from the standard
    covering grid), this starts from a caller-chosen state — for building a
    real zero-action motion context to feed the causal LSTM encoder ahead of
    e.g. an MPPI planning rollout, matching the reasoning in
    ``_collect_energy_grid_episodes``: the encoder needs real motion, not a
    single still frame, to infer velocity.
    """
    seeds = np.array([[theta0, theta_dot0]])
    return _collect_zero_episodes(
        n_episodes=1, img_size=img_size, max_steps=max_steps,
        damping=damping, seeds=seeds,
    )[0]


def collect_zero_trajectories_targeted(
    n_episodes: int,
    img_size: int,
    max_steps: int,
    damping: float = 0.0,
    drag: float = _DRAG_COEFF,
    max_vel: float = _MAX_SPEED,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Zero-action episodes whose *final* state, not seed, covers the grid.

    Plain ``collect_zero_trajectories`` seeds the covering grid and lets
    zero-action dynamics carry each point wherever gravity takes it over
    ``max_steps`` — near the unstable equilibrium (theta=0) at high angular
    velocity, that's a large excursion in just a few steps, so the highest-
    energy corner of the grid ends up unrepresented in the final states.
    This instead treats the grid as the *targets* and backward-solves
    (``_seeds_reaching_targets``) for the seed that reaches each one, so the
    full grid — including the high-energy region near theta=0 — is actually
    covered by the states each episode ends on.

    Because ``analytic_pendulum_step``/``..._inverse`` are exact inverses of
    each other regardless of speed (no hard clip to saturate against), the
    backward-solved seed lands exactly on every target — including the
    high-energy corner near theta=0 — with no special-casing needed.
    ``max_vel`` just sets how wide a range of target velocities to cover.
    """
    seeds = _targeted_grid_seeds(
        n_episodes, max_steps, damping=damping, drag=drag, max_vel=max_vel
    )
    return _collect_zero_episodes(
        n_episodes=n_episodes,
        img_size=img_size,
        max_steps=max_steps,
        damping=damping,
        drag=drag,
        seeds=seeds,
    )


# ── State-only data collection (no pixel rendering) ──────────────────────────


def _state_step(theta: float, theta_dot: float, action: float, damping: float, drag: float) -> tuple[float, float]:
    """Scalar convenience wrapper around ``analytic_pendulum_step``."""
    new_theta, new_theta_dot = analytic_pendulum_step(
        torch.tensor([theta], dtype=torch.float64),
        torch.tensor([theta_dot], dtype=torch.float64),
        torch.tensor([action], dtype=torch.float64),
        damping=damping,
        drag=drag,
    )
    return new_theta.item(), new_theta_dot.item()


def _collect_state_only_episodes(
    n_episodes: int,
    epsilon: float,
    max_steps: int,
    energy_k: float,
    desc: str,
    damping: float = 0.0,
    drag: float = _DRAG_COEFF,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Collect (states, actions) without rendering frames.

    Returns a list of (states, actions) tuples:
        states  : (T+1, 2) float32 — (theta, theta_dot)
        actions : (T,)  float32
    """
    episodes = []
    seeds = _grid_seeds(n_episodes)

    for i in tqdm(range(n_episodes), desc=desc):
        kp = random.uniform(2.0, 15.0)
        kd = random.uniform(0.5, 5.0)

        theta, theta_dot = (float(x) for x in seeds[i])
        states = [np.array([theta, theta_dot], dtype=np.float32)]
        actions = []

        for _ in range(max_steps):
            if random.random() < epsilon:
                action = float(np.random.uniform(-2.0, 2.0))
            elif abs(theta) < _UPRIGHT_THRESHOLD:
                action = _pd_action(theta, theta_dot, kp, kd)
            else:
                action = _energy_pumping_action(theta, theta_dot, energy_k)

            theta, theta_dot = _state_step(theta, theta_dot, action, damping, drag)
            actions.append(action)
            states.append(np.array([theta, theta_dot], dtype=np.float32))

        episodes.append((
            torch.from_numpy(np.stack(states)),          # (T+1, 3)
            torch.tensor(actions, dtype=torch.float32),  # (T,)
        ))

    return episodes


def collect_state_data(
    n_episodes: int,
    epsilon: float = 0.1,
    max_steps: int = 200,
    energy_k: float = 1.0,
    damping: float = 0.0,
    drag: float = _DRAG_COEFF,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Collect state-only Pendulum episodes (no pixel rendering)."""
    episodes = _collect_state_only_episodes(
        n_episodes=n_episodes,
        epsilon=epsilon,
        max_steps=max_steps,
        energy_k=energy_k,
        desc="Collecting state data",
        damping=damping,
        drag=drag,
    )
    print(f"  Collected {n_episodes} episodes ({max_steps} steps each).")
    return episodes


def collect_state_val_trajectories(
    n_episodes: int,
    max_steps: int = 200,
    energy_k: float = 1.0,
    damping: float = 0.0,
    drag: float = _DRAG_COEFF,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return _collect_state_only_episodes(
        n_episodes=n_episodes,
        epsilon=0.0,
        max_steps=max_steps,
        energy_k=energy_k,
        desc="Val trajectories (energy-pump)",
        damping=damping,
        drag=drag,
    )


def collect_state_random_trajectories(
    n_episodes: int,
    max_steps: int = 200,
    damping: float = 0.0,
    drag: float = _DRAG_COEFF,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return _collect_state_only_episodes(
        n_episodes=n_episodes,
        epsilon=1.0,
        max_steps=max_steps,
        energy_k=1.0,
        desc="Val trajectories (random)",
        damping=damping,
        drag=drag,
    )


def _collect_state_spin_episodes(
    n_episodes: int,
    max_steps: int,
    damping: float = 0.0,
    drag: float = _DRAG_COEFF,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    episodes = []
    seeds = _grid_seeds(n_episodes)

    for i in tqdm(range(n_episodes), desc="Val trajectories (spin)"):
        theta, theta_dot = (float(x) for x in seeds[i])
        states = [np.array([theta, theta_dot], dtype=np.float32)]
        actions = []

        for _ in range(max_steps):
            action = _spin_action(theta_dot)
            theta, theta_dot = _state_step(theta, theta_dot, action, damping, drag)
            actions.append(action)
            states.append(np.array([theta, theta_dot], dtype=np.float32))

        episodes.append((
            torch.from_numpy(np.stack(states)),
            torch.tensor(actions, dtype=torch.float32),
        ))

    return episodes


def collect_state_spin_trajectories(
    n_episodes: int,
    max_steps: int = 200,
    damping: float = 0.0,
    drag: float = _DRAG_COEFF,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return _collect_state_spin_episodes(
        n_episodes=n_episodes,
        max_steps=max_steps,
        damping=damping,
        drag=drag,
    )


# ── Dataset ──────────────────────────────────────────────────────────────────


class PendulumStateDataset(Dataset):
    """State-only episode dataset (no frames).

    states  : (T+1, 2) float32 — (theta, theta_dot)
    actions : (T,)  float32
    """

    def __init__(self, episodes: list[tuple[torch.Tensor, torch.Tensor]]):
        self.states = torch.stack([e[0] for e in episodes])   # (N, T+1, 2)
        self.actions = torch.stack([e[1] for e in episodes])  # (N, T)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


