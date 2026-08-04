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

# Pendulum-v1 physical constants (gymnasium defaults)
_G = 10.0
_H_STAR = 20.0  # 2 * m * g * l  with m=l=1, g=10
_DT = 0.05  # Pendulum-v1 integration timestep
_MAX_SPEED = 8.0  # Pendulum-v1's hard clip on |theta_dot| (gymnasium default)


# ── Analytic (batched) dynamics ──────────────────────────────────────────────


def angle_normalize(theta: torch.Tensor) -> torch.Tensor:
    return ((theta + torch.pi) % (2 * torch.pi)) - torch.pi


def analytic_pendulum_step(
    theta: torch.Tensor,
    theta_dot: torch.Tensor,
    u: torch.Tensor,
    dt: float = _DT,
    damping: float = 0.0,
    g: float = _G,
    max_speed: float = _MAX_SPEED,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched replica of Pendulum-v1's own semi-implicit Euler step (m=l=1).

    Matches ``gymnasium.envs.classic_control.pendulum.PendulumEnv.step`` (and
    ``PendulumPixelEnv``'s post-step damping) exactly, but as plain tensor ops
    over arbitrarily-shaped ``theta``/``theta_dot``/``u`` — lets a planner
    simulate many candidate rollouts in parallel without stepping copies of
    the real env.
    """
    u = u.clamp(-2.0, 2.0)
    new_theta_dot = theta_dot + (1.5 * g * torch.sin(theta) + 3.0 * u) * dt
    new_theta_dot = new_theta_dot.clamp(-max_speed, max_speed)
    new_theta = theta + new_theta_dot * dt
    if damping != 0.0:
        new_theta_dot = new_theta_dot * np.exp(-damping * dt)
    return new_theta, new_theta_dot


# ── Image preprocessing ──────────────────────────────────────────────────────


def preprocess_frame(frame: np.ndarray, img_size: int = 64) -> torch.Tensor:
    """Convert a raw Pendulum RGB frame to a normalised (3, H, W) tensor."""
    t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    return TF.resize(t, [img_size, img_size], antialias=True)


# ── Pixel wrapper ─────────────────────────────────────────────────────────────


class PendulumPixelEnv(gym.Wrapper):
    """Pendulum-v1 with (3, img_size, img_size) uint8 pixel observations.

    Args:
        img_size: Side length of the square pixel observation.
        damping:  Linear viscous damping coefficient applied post-step.
                  ``theta_dot *= exp(-damping * dt)`` each step.
                  0.0 (default) reproduces the standard frictionless pendulum.
    """

    def __init__(self, img_size: int = 64, damping: float = 0.0):
        env = gym.make("Pendulum-v1", render_mode="rgb_array")
        super().__init__(env)
        self.img_size = img_size
        self.damping = damping
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

    def _apply_damping(self) -> None:
        if self.damping != 0.0:
            theta, theta_dot = self.env.unwrapped.state  # type: ignore[union-attr]
            theta_dot *= np.exp(-self.damping * _DT)
            self.env.unwrapped.state = np.array([theta, theta_dot])  # type: ignore[union-attr]

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        return self._obs(), {}

    def set_state(self, theta: float, theta_dot: float) -> np.ndarray:
        """Force the environment into (theta, theta_dot) and return the resulting observation."""
        self.env.unwrapped.state = np.array([theta, theta_dot], dtype=np.float64)  # type: ignore[union-attr]
        return self._obs()

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        self._apply_damping()
        return self._obs(), reward, terminated, truncated, info


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


def _collect_episodes(
    n_episodes: int,
    img_size: int,
    epsilon: float,
    max_steps: int,
    energy_k: float,
    desc: str,
    damping: float = 0.0,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Shared episode collection loop.

    Returns a list of (frames, actions, states) tuples:
        frames  : (T+1, 3, img_size, img_size) float32 [0,1]
        actions : (T,)  float32
        states  : (T+1, 3) float32 — (cos(theta), sin(theta), theta_dot) at each frame, post-damping
    """
    env = PendulumPixelEnv(img_size=img_size, damping=damping)
    episodes = []
    seeds = _grid_seeds(n_episodes)

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

            for _ in range(max_steps):
                theta, theta_dot = env.unwrapped.state  # type: ignore[union-attr]

                if random.random() < epsilon:
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


def collect_data(
    n_episodes: int,
    img_size: int,
    epsilon: float = 0.1,
    max_steps: int = 200,
    energy_k: float = 1.0,
    damping: float = 0.0,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Collect Pendulum episodes for world-model training.

    Policy (per step):
      - With probability ``epsilon``: random action from U(-2, 2).
      - Otherwise, if |theta| < upright_threshold: PD controller with
        random gains (kp, kd drawn fresh each episode for coverage).
      - Otherwise: energy-pumping law.
    """
    episodes = _collect_episodes(
        n_episodes=n_episodes,
        img_size=img_size,
        epsilon=epsilon,
        max_steps=max_steps,
        energy_k=energy_k,
        desc="Collecting data",
        damping=damping,
    )
    print(f"  Collected {n_episodes} episodes ({max_steps} steps each).")
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
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Collect episodes with a spin-maximising controller (maximises |theta_dot|)."""
    env = PendulumPixelEnv(img_size=img_size, damping=damping)
    episodes = []
    seeds = _grid_seeds(n_episodes)

    try:
        for i in tqdm(range(n_episodes), desc="Val trajectories (spin)"):
            env.reset()
            theta0, theta_dot0 = seeds[i]
            obs = env.set_state(theta0, theta_dot0)
            frames = [torch.from_numpy(obs).float() / 255.0]
            actions = []
            states = [np.array([np.cos(theta0), np.sin(theta0), theta_dot0], dtype=np.float32)]

            for _ in range(max_steps):
                theta, theta_dot = env.unwrapped.state  # type: ignore[union-attr]
                action = _spin_action(theta_dot)
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


def _seeds_reaching_targets(
    targets: np.ndarray,
    max_steps: int,
    damping: float = 0.0,
) -> np.ndarray:
    """Find zero-action seeds that land at ``targets`` after ``max_steps`` steps.

    Pendulum-v1's zero-action step is semi-implicit ("symplectic") Euler:
        newthdot = clip(thdot + (3g/2l) sin(th) dt, -max_speed, max_speed)
        newth    = th + newthdot dt
        newthdot *= exp(-damping dt)                    (damping, applied after)
    The position update uses the *new* velocity, so this is not reversible by
    simply negating velocity and re-running the same forward step — that
    would evaluate the sin(th) acceleration at the wrong angle and drift
    (confirmed empirically: multi-step error of >1 rad). Instead the map is
    inverted in closed form, one step at a time, walking backward from each
    target: undo damping, recover th from th = newth - newthdot dt, then
    recover thdot from the acceleration evaluated at that *recovered* th
    (matching what the forward step actually used). Exact up to float
    precision, provided no step along the way saturates the velocity clip —
    true for the achievable phase-space region this is used for.

    Returns an (N, 2) array of (theta0, theta_dot0) seeds, same shape as
    ``targets``.
    """
    theta = targets[:, 0].copy()
    theta_dot = targets[:, 1].copy()
    for _ in range(max_steps):
        theta_dot = theta_dot * np.exp(damping * _DT)          # undo damping
        theta_prev = theta - theta_dot * _DT                    # invert position update
        theta_dot_prev = theta_dot - (3 * _G / 2) * np.sin(theta_prev) * _DT
        theta, theta_dot = theta_prev, theta_dot_prev
    return np.stack([theta, theta_dot], axis=-1)


def _collect_zero_episodes(
    n_episodes: int,
    img_size: int,
    max_steps: int,
    damping: float = 0.0,
    seeds: np.ndarray | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Collect episodes with the action fixed at zero (uncontrolled dynamics).

    ``seeds``: explicit (N, 2) array of (theta0, theta_dot0) to start from;
    defaults to the standard covering grid (``_grid_seeds``).
    """
    env = PendulumPixelEnv(img_size=img_size, damping=damping)
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

    ``max_vel`` defaults to Pendulum-v1's own ``max_speed`` clip (8 rad/s):
    the env can never actually reach angular velocities beyond that, so
    targeting a wider range wouldn't converge — the backward solve would find
    a seed for an unreachable target and the forward rollout would just
    saturate at the clip instead of arriving where asked.
    """
    targets = _grid_seeds(n_episodes, min_theta_dot=-max_vel, max_theta_dot=max_vel)
    seeds = _seeds_reaching_targets(targets, max_steps=max_steps, damping=damping)
    return _collect_zero_episodes(
        n_episodes=n_episodes,
        img_size=img_size,
        max_steps=max_steps,
        damping=damping,
        seeds=seeds,
    )


# ── State-only data collection (no pixel rendering) ──────────────────────────


def _collect_state_only_episodes(
    n_episodes: int,
    epsilon: float,
    max_steps: int,
    energy_k: float,
    desc: str,
    damping: float = 0.0,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Collect (states, actions) without rendering frames.

    Returns a list of (states, actions) tuples:
        states  : (T+1, 2) float32 — (theta, theta_dot)
        actions : (T,)  float32
    """
    env = gym.make("Pendulum-v1", render_mode=None)
    episodes = []
    seeds = _grid_seeds(n_episodes)

    for i in tqdm(range(n_episodes), desc=desc):
        kp = random.uniform(2.0, 15.0)
        kd = random.uniform(0.5, 5.0)

        env.reset()
        theta0, theta_dot0 = seeds[i]
        env.unwrapped.state = np.array([theta0, theta_dot0])  # type: ignore[union-attr]
        states = [np.array([theta0, theta_dot0], dtype=np.float32)]
        actions = []

        for _ in range(max_steps):
            theta, theta_dot = env.unwrapped.state  # type: ignore[union-attr]

            if random.random() < epsilon:
                action = float(np.random.uniform(-2.0, 2.0))
            elif abs(theta) < _UPRIGHT_THRESHOLD:
                action = _pd_action(theta, theta_dot, kp, kd)
            else:
                action = _energy_pumping_action(theta, theta_dot, energy_k)

            env.step(np.array([action], dtype=np.float32))
            theta_new, theta_dot_new = env.unwrapped.state  # type: ignore[union-attr]
            if damping != 0.0:
                theta_dot_new *= np.exp(-damping * _DT)
                env.unwrapped.state = np.array([theta_new, theta_dot_new])  # type: ignore[union-attr]
            actions.append(action)
            states.append(np.array([theta_new, theta_dot_new], dtype=np.float32))

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
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Collect state-only Pendulum episodes (no pixel rendering)."""
    episodes = _collect_state_only_episodes(
        n_episodes=n_episodes,
        epsilon=epsilon,
        max_steps=max_steps,
        energy_k=energy_k,
        desc="Collecting state data",
        damping=damping,
    )
    print(f"  Collected {n_episodes} episodes ({max_steps} steps each).")
    return episodes


def collect_state_val_trajectories(
    n_episodes: int,
    max_steps: int = 200,
    energy_k: float = 1.0,
    damping: float = 0.0,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return _collect_state_only_episodes(
        n_episodes=n_episodes,
        epsilon=0.0,
        max_steps=max_steps,
        energy_k=energy_k,
        desc="Val trajectories (energy-pump)",
        damping=damping,
    )


def collect_state_random_trajectories(
    n_episodes: int,
    max_steps: int = 200,
    damping: float = 0.0,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return _collect_state_only_episodes(
        n_episodes=n_episodes,
        epsilon=1.0,
        max_steps=max_steps,
        energy_k=1.0,
        desc="Val trajectories (random)",
        damping=damping,
    )


def _collect_state_spin_episodes(
    n_episodes: int,
    max_steps: int,
    damping: float = 0.0,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    env = gym.make("Pendulum-v1", render_mode=None)
    episodes = []
    seeds = _grid_seeds(n_episodes)

    for i in tqdm(range(n_episodes), desc="Val trajectories (spin)"):
        env.reset()
        theta0, theta_dot0 = seeds[i]
        env.unwrapped.state = np.array([theta0, theta_dot0])  # type: ignore[union-attr]
        states = [np.array([theta0, theta_dot0], dtype=np.float32)]
        actions = []

        for _ in range(max_steps):
            _, theta_dot = env.unwrapped.state  # type: ignore[union-attr]
            action = _spin_action(theta_dot)
            env.step(np.array([action], dtype=np.float32))
            theta_new, theta_dot_new = env.unwrapped.state  # type: ignore[union-attr]
            if damping != 0.0:
                theta_dot_new *= np.exp(-damping * _DT)
                env.unwrapped.state = np.array([theta_new, theta_dot_new])  # type: ignore[union-attr]
            actions.append(action)
            states.append(np.array([theta_new, theta_dot_new], dtype=np.float32))

        episodes.append((
            torch.from_numpy(np.stack(states)),
            torch.tensor(actions, dtype=torch.float32),
        ))

    return episodes


def collect_state_spin_trajectories(
    n_episodes: int,
    max_steps: int = 200,
    damping: float = 0.0,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return _collect_state_spin_episodes(
        n_episodes=n_episodes,
        max_steps=max_steps,
        damping=damping,
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


class PendulumDataset(Dataset):
    """Episode-level dataset. Each item is a full episode.

    frames  : (T+1, 3, img_size, img_size) float32 [0,1]
    actions : (T,)  float32
    states  : (T+1, 3) float32 — (cos(theta), sin(theta), theta_dot)
    """

    def __init__(
        self, episodes: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ):
        self.frames = torch.stack([e[0] for e in episodes])  # (N, T+1, 3, H, W)
        self.actions = torch.stack([e[1] for e in episodes])  # (N, T)
        self.states = torch.stack([e[2] for e in episodes])  # (N, T+1, 3)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx], self.actions[idx], self.states[idx]
