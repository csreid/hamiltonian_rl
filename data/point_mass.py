"""2D point-mass pixel-observation environment: a Pendulum without the angle.

Same port-Hamiltonian design as ``data/pendulum.py`` — analytic batched
dynamics as the single source of truth, a pixel wrapper around it, and
matching data-collection helpers — but with no periodic coordinate anywhere
in the state. State is q = (x, y) position, p = (vx, vy) momentum (unit
mass, so p = v); there is no angle to wrap, so none of the pendulum
pipeline's angle-discontinuity handling (``angle_normalize``, sin/cos
encoding) has an analog here.

Physics:
    H(q, p) = T(p) + V(q) = 0.5·|p|² + V_wall(q)     (no gravity)

``V_wall`` is a smooth confining potential (see ``_grad_V``) that replaces
the pendulum's hard nothing-at-all (an unbounded plane can't be rendered to
a fixed-size frame) with a physically-justified spring-like wall: exactly
zero in the interior, engaging with a C¹-smooth quartic restoring force only
near the edges. Likewise, velocity is bounded by the same quadratic
(Rayleigh) drag as the pendulum's, not a hard clip. Reuses
``data.pendulum``'s dissipative-substep machinery unchanged — those
functions are plain elementwise tensor ops on ``p`` with no dependence on
its last dimension being size 1, so they generalize to the 2-vector
momentum here with zero modification.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pygame
import torch
from gymnasium import spaces
from tqdm import tqdm

from data.pendulum import PendulumMultiRolloutDataset as MultiRolloutDataset  # noqa: F401
from data.pendulum import PendulumStateDataset as StateDataset  # noqa: F401
from data.pendulum import (
    _dissipation_substep,
    _dissipation_substep_inverse,
    preprocess_frame,
)

# MultiRolloutDataset/StateDataset above are dimension-generic slicing
# datasets (they don't care about the width of actions/states, only their
# leading time dimension) — reused directly rather than duplicated.

# Physical constants (free choices for this toy system, not inherited from
# anywhere — unit mass, so B = 1 is simply Newton's second law dv/dt = F/m).
_DT = 0.05  # matches data.pendulum's integration timestep
_DRAG_COEFF = 0.05  # quadratic drag, same magnitude as the pendulum's default
_U_MAX = 2.0  # actuator saturation (matches the pendulum's torque clamp)
B_TRUE = 1.0  # true control gain: dp/dt += B·u (unit mass -> B = 1)
_L_WALL = 2.0  # half-width of the force-free interior region, per axis
_K_WALL = 50.0  # wall stiffness


# ── Analytic (batched) dynamics ──────────────────────────────────────────────
#
# Canonical Hamiltonian coordinates: q = (x, y), p = (vx, vy), with
# T(p) = 0.5|p|² (unit mass) and V(q) = Vw(x) + Vw(y), a soft-wall potential
# that is exactly zero inside [-L, L] on each axis and a quartic spring
# outside it (see ``_grad_V``). Control enters as ṗ += B·u; dissipation
# (quadratic drag, optionally linear damping) acts on p alone, exactly as in
# ``data.pendulum``. Integrated with the same Strang-split leapfrog
# (kick-drift-kick, dissipative flow split symmetrically around it) — this is
# the single source of truth for point-mass physics: the real env
# (``PointMassPixelEnv.step``) and the state-only collectors below both call
# this same function.


def _grad_V(q: torch.Tensor, k: float = _K_WALL, L: float = _L_WALL) -> torch.Tensor:
    """dV/dq for the per-axis soft-wall potential Vw(s) = 0.25·k·(|s|-L)⁴ for |s|>L, else 0.

    dVw/ds = k·sign(s)·(|s|-L)³ for |s| > L, else 0 — continuous and C¹ at
    the boundary (both Vw and dVw/ds vanish at |s| = L), so the restoring
    force engages smoothly with no discontinuity. Applies independently and
    elementwise to every component of q, so this works unmodified for the
    2-vector (x, y) case.
    """
    beyond = (q.abs() - L).clamp_min(0.0)
    return k * torch.sign(q) * beyond**3


def analytic_point_mass_step(
    q: torch.Tensor,
    p: torch.Tensor,
    u: torch.Tensor,
    dt: float = _DT,
    damping: float = 0.0,
    drag: float = _DRAG_COEFF,
    k: float = _K_WALL,
    L: float = _L_WALL,
    u_max: float = _U_MAX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One leapfrog (Strang-split) step of the point mass's port-Hamiltonian dynamics.

    Structurally identical to ``data.pendulum.analytic_pendulum_step``: a
    dissipative half-step, canonical kick-drift-kick (with the constant
    control force folded into the two half-kicks as an exact zero-order
    hold), then the other dissipative half-step. Plain tensor ops over
    arbitrarily-shaped ``(..., 2)`` q/p/u.
    """
    u = u.clamp(-u_max, u_max)
    has_dissipation = damping != 0.0 or drag != 0.0

    if has_dissipation:
        q, p = _dissipation_substep(q, p, dt / 2, damping, drag)

    Bu = B_TRUE * u
    p = p - (dt / 2) * _grad_V(q, k, L) + (dt / 2) * Bu
    q = q + dt * p
    p = p - (dt / 2) * _grad_V(q, k, L) + (dt / 2) * Bu

    if has_dissipation:
        q, p = _dissipation_substep(q, p, dt / 2, damping, drag)

    return q, p


def analytic_point_mass_step_inverse(
    q: torch.Tensor,
    p: torch.Tensor,
    u: torch.Tensor,
    dt: float = _DT,
    damping: float = 0.0,
    drag: float = _DRAG_COEFF,
    k: float = _K_WALL,
    L: float = _L_WALL,
    u_max: float = _U_MAX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact inverse of ``analytic_point_mass_step``, mirroring
    ``data.pendulum.analytic_pendulum_step_inverse``."""
    u = u.clamp(-u_max, u_max)
    has_dissipation = damping != 0.0 or drag != 0.0

    if has_dissipation:
        q, p = _dissipation_substep_inverse(q, p, dt / 2, damping, drag)

    Bu = B_TRUE * u
    p = p + (dt / 2) * _grad_V(q, k, L) - (dt / 2) * Bu
    q = q - dt * p
    p = p + (dt / 2) * _grad_V(q, k, L) - (dt / 2) * Bu

    if has_dissipation:
        q, p = _dissipation_substep_inverse(q, p, dt / 2, damping, drag)

    return q, p


def H_true(q: torch.Tensor, p: torch.Tensor, k: float = _K_WALL, L: float = _L_WALL) -> torch.Tensor:
    """Canonical Hamiltonian H(q, p) = T(p) + V(q) matching ``analytic_point_mass_step``."""
    beyond = (q.abs() - L).clamp_min(0.0)
    V = (0.25 * k * beyond**4).sum(dim=-1)
    T = 0.5 * (p**2).sum(dim=-1)
    return T + V


def grad_H_true(
    q: torch.Tensor, p: torch.Tensor, k: float = _K_WALL, L: float = _L_WALL
) -> tuple[torch.Tensor, torch.Tensor]:
    """(∂H/∂q, ∂H/∂p) = (grad_V(q), p), matching ``_grad_V``."""
    return _grad_V(q, k, L), p


def dissipation_rate_true(p: torch.Tensor, damping: float, drag: float) -> torch.Tensor:
    """Per-axis energy dissipation rate damping·p² + drag·|p|³, summed over axes."""
    return (damping * p**2 + drag * p.abs() ** 3).sum(dim=-1)


def R_pp_true(p: torch.Tensor, damping: float, drag: float) -> torch.Tensor:
    """Effective R_pp(z) such that ṗ_diss = -R_pp·∂H/∂p = -R_pp·p reproduces the
    linear damping + quadratic drag dissipative flow, decoupled per axis.

    p: (..., 2) -> R_pp: (..., 2, 2), diagonal (each axis drags independently
    on its own component's magnitude, matching ``_drag_substep``'s elementwise
    application — there is no coupling between x and y drag).
    """
    diag = damping + drag * p.abs()  # (..., 2)
    return torch.diag_embed(diag)  # (..., 2, 2)


# ── Pixel env ─────────────────────────────────────────────────────────────────


def _world_to_px(coord: float, half_extent: float, px: int) -> int:
    return int((coord + half_extent) / (2 * half_extent) * px)


class PointMassPixelEnv(gym.Env):
    """A 2D point mass with (3, img_size, img_size) uint8 pixel observations.

    Unlike ``PendulumPixelEnv``, there is no existing Gymnasium env to wrap —
    this is a standalone ``gym.Env`` whose physics and rendering are both
    defined here. Rendering uses pygame drawing primitives directly onto an
    off-screen ``Surface`` (no ``pygame.display`` call, so this works
    headlessly with no video driver, same as Gymnasium's own classic-control
    ``rgb_array`` rendering).

    State is the raw ``(x, y, vx, vy)`` tuple; ``step`` calls
    ``analytic_point_mass_step`` directly and always returns reward 0.0 —
    this is a physics/world-model-learning env, not a task with a reward,
    matching ``PendulumPixelEnv``'s convention.

    Args:
        img_size:  Side length of the square pixel observation.
        damping:   Linear viscous damping coefficient (0.0 disables it).
        drag:      Quadratic (Rayleigh) drag coefficient — see ``_DRAG_COEFF``.
        k:         Wall stiffness (see ``_grad_V``).
        L:         Half-width of the force-free interior region per axis.
        u_max:     Actuator saturation on each action component.
        margin:    Extra world-space margin around ``[-L, L]²`` shown in the
                   rendered frame, so the wall's engagement zone is visible.
        mass_radius: Rendered radius of the point mass, in world units.
        render_px: Resolution the scene is drawn at before ``preprocess_frame``
                   resizes to ``img_size`` (kept higher than img_size so
                   downsizing anti-aliases the disc rather than aliasing it).
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        img_size: int = 64,
        damping: float = 0.0,
        drag: float = _DRAG_COEFF,
        k: float = _K_WALL,
        L: float = _L_WALL,
        u_max: float = _U_MAX,
        margin: float = 1.0,
        mass_radius: float = 0.15,
        render_px: int = 200,
    ):
        super().__init__()
        self.img_size = img_size
        self.damping = damping
        self.drag = drag
        self.k = k
        self.L = L
        self.u_max = u_max
        self.half_extent = L + margin
        self.mass_radius = mass_radius
        self.render_px = render_px
        self.action_space = spaces.Box(low=-u_max, high=u_max, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(3, img_size, img_size), dtype=np.uint8
        )
        self.state = np.zeros(4, dtype=np.float64)  # x, y, vx, vy

    def _render_frame(self) -> np.ndarray:
        surf = pygame.Surface((self.render_px, self.render_px))
        surf.fill((255, 255, 255))
        x, y = self.state[0], self.state[1]
        px = _world_to_px(x, self.half_extent, self.render_px)
        py = _world_to_px(-y, self.half_extent, self.render_px)  # flip y: world-up = screen-up
        radius_px = max(1, int(self.mass_radius / (2 * self.half_extent) * self.render_px))
        pygame.draw.circle(surf, (30, 30, 180), (px, py), radius_px)
        arr = pygame.surfarray.array3d(surf)  # (W, H, 3), x-major
        return np.transpose(arr, (1, 0, 2))  # (H, W, 3)

    def _obs(self) -> np.ndarray:
        frame = self._render_frame()
        t = preprocess_frame(frame, self.img_size)
        return (t * 255).byte().numpy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(4, dtype=np.float64)
        return self._obs(), {}

    def set_state(self, x: float, y: float, vx: float, vy: float) -> np.ndarray:
        """Force the environment into (x, y, vx, vy) and return the resulting observation."""
        self.state = np.array([x, y, vx, vy], dtype=np.float64)
        return self._obs()

    def step(self, action):
        x, y, vx, vy = self.state
        q = torch.as_tensor([[x, y]], dtype=torch.float64)
        p = torch.as_tensor([[vx, vy]], dtype=torch.float64)
        u = torch.as_tensor([[float(action[0]), float(action[1])]], dtype=torch.float64)
        q_next, p_next = analytic_point_mass_step(
            q, p, u, damping=self.damping, drag=self.drag, k=self.k, L=self.L, u_max=self.u_max
        )
        self.state = np.array(
            [q_next[0, 0].item(), q_next[0, 1].item(), p_next[0, 0].item(), p_next[0, 1].item()],
            dtype=np.float64,
        )
        return self._obs(), 0.0, False, False, {}


# ── Seeding ───────────────────────────────────────────────────────────────────


def _random_seeds(n_points: int, L: float = _L_WALL, max_speed: float = 4.0) -> np.ndarray:
    """Uniform random (x, y, vx, vy) seeds inside the wall-free interior.

    Unlike the pendulum's ``_grid_seeds`` (a covering grid over its 2D phase
    space), a covering grid over this 4D phase space would need many more
    points for the same per-axis resolution — uniform random sampling is used
    instead. Positions are drawn from the interior ``[-L, L]²`` (away from the
    wall, where the physics is trivial T(p)-only) rather than the full
    rendered frame.
    """
    xy = np.random.uniform(-L, L, size=(n_points, 2))
    v = np.random.uniform(-max_speed, max_speed, size=(n_points, 2))
    return np.concatenate([xy, v], axis=-1)


# ── Data collection ──────────────────────────────────────────────────────────


def collect_seeded_random_rollouts(
    n_samples: int,
    rollout_len: int,
    img_size: int,
    damping: float = 0.0,
    drag: float = _DRAG_COEFF,
    k: float = _K_WALL,
    L: float = _L_WALL,
    u_max: float = _U_MAX,
    max_speed: float = 4.0,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Many short, purely-random-action rollouts seeded across phase space.

    Mirrors ``data.pendulum.collect_seeded_random_rollouts`` — the one Phase 1
    of the offline pipeline actually consumes. Returns a list of ``n_seeds``
    (frames, actions, states) tuples:
        frames  : (rollout_len+1, 3, img_size, img_size) float32 [0,1]
        actions : (rollout_len, 2) float32
        states  : (rollout_len+1, 4) float32 — (x, y, vx, vy)
    """
    n_seeds = max(1, n_samples // rollout_len)
    seeds = _random_seeds(n_seeds, L=L, max_speed=max_speed)

    env = PointMassPixelEnv(img_size=img_size, damping=damping, drag=drag, k=k, L=L, u_max=u_max)
    rollouts = []
    for x0, y0, vx0, vy0 in tqdm(seeds, desc="Collecting seeded random rollouts", dynamic_ncols=True):
        env.reset()
        obs = env.set_state(float(x0), float(y0), float(vx0), float(vy0))
        frames = [torch.from_numpy(obs).float() / 255.0]
        actions = []
        states = [np.array([x0, y0, vx0, vy0], dtype=np.float32)]

        for _ in range(rollout_len):
            action = np.random.uniform(-u_max, u_max, size=2).astype(np.float32)
            obs, _, _, _, _ = env.step(action)
            frames.append(torch.from_numpy(obs).float() / 255.0)
            actions.append(action)
            states.append(env.state.astype(np.float32).copy())

        rollouts.append(
            (
                torch.stack(frames),  # (rollout_len+1, 3, H, W)
                torch.from_numpy(np.stack(actions)),  # (rollout_len, 2)
                torch.from_numpy(np.stack(states)),  # (rollout_len+1, 4)
            )
        )

    return rollouts


# ── State-only data collection (no pixel rendering) ──────────────────────────


def _state_step(
    x: float, y: float, vx: float, vy: float, ux: float, uy: float, damping: float, drag: float
) -> tuple[float, float, float, float]:
    """Scalar convenience wrapper around ``analytic_point_mass_step``."""
    q_next, p_next = analytic_point_mass_step(
        torch.tensor([[x, y]], dtype=torch.float64),
        torch.tensor([[vx, vy]], dtype=torch.float64),
        torch.tensor([[ux, uy]], dtype=torch.float64),
        damping=damping,
        drag=drag,
    )
    return q_next[0, 0].item(), q_next[0, 1].item(), p_next[0, 0].item(), p_next[0, 1].item()


def collect_state_data(
    n_episodes: int,
    max_steps: int = 200,
    damping: float = 0.0,
    drag: float = _DRAG_COEFF,
    u_max: float = _U_MAX,
    max_speed: float = 4.0,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Collect state-only point-mass episodes (no pixel rendering), random actions throughout.

    Returns a list of (states, actions) tuples:
        states  : (T+1, 4) float32 — (x, y, vx, vy)
        actions : (T, 2)  float32
    """
    episodes = []
    seeds = _random_seeds(n_episodes, max_speed=max_speed)

    for i in tqdm(range(n_episodes), desc="Collecting state data", dynamic_ncols=True):
        x, y, vx, vy = (float(v) for v in seeds[i])
        states = [np.array([x, y, vx, vy], dtype=np.float32)]
        actions = []

        for _ in range(max_steps):
            ux, uy = np.random.uniform(-u_max, u_max, size=2)
            x, y, vx, vy = _state_step(x, y, vx, vy, ux, uy, damping, drag)
            actions.append(np.array([ux, uy], dtype=np.float32))
            states.append(np.array([x, y, vx, vy], dtype=np.float32))

        episodes.append(
            (
                torch.from_numpy(np.stack(states)),  # (T+1, 4)
                torch.from_numpy(np.stack(actions)),  # (T, 2)
            )
        )

    print(f"  Collected {n_episodes} episodes ({max_steps} steps each).")
    return episodes
