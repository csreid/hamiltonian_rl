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
        frame = self.env.render()  # (H, W, 3) uint8
        t = preprocess_frame(frame, self.img_size)
        return (t * 255).byte().numpy()

    def _apply_damping(self) -> None:
        if self.damping != 0.0:
            theta, theta_dot = self.env.unwrapped.state  # type: ignore[union-attr]
            theta_dot *= np.exp(-self.damping * _DT)
            self.env.unwrapped.state = np.array([theta, theta_dot])  # type: ignore[union-attr]

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        return self._obs(), {}

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

    for _ in tqdm(range(n_episodes), desc=desc):
        kp = random.uniform(2.0, 15.0)
        kd = random.uniform(0.5, 5.0)

        obs, _ = env.reset()
        theta0, theta_dot0 = env.unwrapped.state  # type: ignore[union-attr]
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
                torch.from_numpy(np.stack(states)),  # (T+1, 2)
            )
        )

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

    for _ in tqdm(range(n_episodes), desc="Val trajectories (spin)"):
        obs, _ = env.reset()
        theta0, theta_dot0 = env.unwrapped.state  # type: ignore[union-attr]
        frames = [torch.from_numpy(obs).float() / 255.0]
        actions = []
        states = [np.array([np.cos(theta0), np.sin(theta0), theta_dot0], dtype=np.float32)]

        for _ in range(max_steps):
            _, theta_dot = env.unwrapped.state  # type: ignore[union-attr]
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
                torch.from_numpy(np.stack(states)),
            )
        )

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
        states  : (T+1, 3) float32 — (cos(theta), sin(theta), theta_dot)
        actions : (T,)  float32
    """
    env = gym.make("Pendulum-v1", render_mode=None)
    episodes = []

    for _ in tqdm(range(n_episodes), desc=desc):
        kp = random.uniform(2.0, 15.0)
        kd = random.uniform(0.5, 5.0)

        obs, _ = env.reset()
        theta0, theta_dot0 = env.unwrapped.state  # type: ignore[union-attr]
        states = [np.array([np.cos(theta0), np.sin(theta0), theta_dot0], dtype=np.float32)]
        actions = []

        for _ in range(max_steps):
            theta, theta_dot = env.unwrapped.state  # type: ignore[union-attr]

            if random.random() < epsilon:
                action = float(np.random.uniform(-2.0, 2.0))
            elif abs(theta) < _UPRIGHT_THRESHOLD:
                action = _pd_action(theta, theta_dot, kp, kd)
            else:
                action = _energy_pumping_action(theta, theta_dot, energy_k)

            obs, _, _, _, _ = env.step(np.array([action], dtype=np.float32))
            if damping != 0.0:
                theta_new, theta_dot_new = env.unwrapped.state  # type: ignore[union-attr]
                theta_dot_new *= np.exp(-damping * _DT)
                env.unwrapped.state = np.array([theta_new, theta_dot_new])  # type: ignore[union-attr]
                obs = np.array([np.cos(theta_new), np.sin(theta_new), theta_dot_new], dtype=np.float32)
            actions.append(action)
            states.append(obs.astype(np.float32))

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

    for _ in tqdm(range(n_episodes), desc="Val trajectories (spin)"):
        obs, _ = env.reset()
        theta0, theta_dot0 = env.unwrapped.state  # type: ignore[union-attr]
        states = [np.array([np.cos(theta0), np.sin(theta0), theta_dot0], dtype=np.float32)]
        actions = []

        for _ in range(max_steps):
            _, theta_dot = env.unwrapped.state  # type: ignore[union-attr]
            action = _spin_action(theta_dot)
            obs, _, _, _, _ = env.step(np.array([action], dtype=np.float32))
            if damping != 0.0:
                theta_new, theta_dot_new = env.unwrapped.state  # type: ignore[union-attr]
                theta_dot_new *= np.exp(-damping * _DT)
                env.unwrapped.state = np.array([theta_new, theta_dot_new])  # type: ignore[union-attr]
                obs = np.array([np.cos(theta_new), np.sin(theta_new), theta_dot_new], dtype=np.float32)
            actions.append(action)
            states.append(obs.astype(np.float32))

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

    states  : (T+1, 3) float32 — (cos(theta), sin(theta), theta_dot)
    actions : (T,)  float32
    """

    def __init__(self, episodes: list[tuple[torch.Tensor, torch.Tensor]]):
        self.states = torch.stack([e[0] for e in episodes])   # (N, T+1, 3)
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
