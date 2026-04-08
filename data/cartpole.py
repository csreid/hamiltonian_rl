"""CartPole environment utilities, data collection, and preprocessing."""

from __future__ import annotations

import random
from collections import deque

import gymnasium as gym
import gymnasium.envs.classic_control.cartpole as _cartpole_mod
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from tqdm import tqdm
from torch.utils.data import Dataset

# CartPole failure thresholds (same as gym CartPole-v1)
_CART_X_LIMIT = 2.4
_POLE_THETA_LIMIT = 0.2094  # 12 degrees in radians


# ── Image preprocessing ──────────────────────────────────────────────────────


def preprocess_frame(frame: np.ndarray, img_size: int = 32) -> torch.Tensor:
    """Convert a raw CartPole RGB frame to a normalised (3, H, W) tensor.

    Args:
        frame:    (H, W, 3) uint8 numpy array from env.render()
        img_size: target spatial size (square)

    Returns:
        (3, img_size, img_size) float32 in [0, 1]
    """
    t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    return TF.resize(t, [img_size, img_size], antialias=True)


class FrameBuffer:
    """Fixed-length deque of preprocessed frames.

    Initialised by repeating the first frame n_frames times so that
    the context is always full even at episode start.
    """

    def __init__(self, n_frames: int, img_size: int):
        self.n_frames = n_frames
        self.img_size = img_size
        self._buf: deque[torch.Tensor] = deque(maxlen=n_frames)

    def reset(self, frame: np.ndarray) -> None:
        processed = preprocess_frame(frame, self.img_size)
        self._buf.clear()
        for _ in range(self.n_frames):
            self._buf.append(processed)

    def push(self, frame: np.ndarray) -> None:
        self._buf.append(preprocess_frame(frame, self.img_size))

    def get(self) -> torch.Tensor:
        """Return (n_frames, 3, img_size, img_size) stacked tensor."""
        return torch.stack(list(self._buf))


# ── Continuous-force CartPole ─────────────────────────────────────────────────


class ContinuousCartPoleEnv(_cartpole_mod.CartPoleEnv):
    """CartPole-v1 with a continuous action in [-1, 1].

    action=1 applies full force rightward; action=-1 applies full force
    leftward; intermediate values scale linearly.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

    def step(self, action):
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        force = float(np.asarray(action).flat[0]) * self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (
            force + self.polemass_length * np.square(theta_dot) * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (
                4.0 / 3.0
                - self.masspole * np.square(costheta) / self.total_mass
            )
        )
        xacc = (
            temp - self.polemass_length * thetaacc * costheta / self.total_mass
        )
        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        self.state = np.array((x, x_dot, theta, theta_dot), dtype=np.float64)
        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        if not terminated:
            reward = 0.0 if self._sutton_barto_reward else 1.0
        elif self.steps_beyond_terminated is None:
            self.steps_beyond_terminated = 0
            reward = -1.0 if self._sutton_barto_reward else 1.0
        else:
            self.steps_beyond_terminated += 1
            reward = -1.0 if self._sutton_barto_reward else 0.0
        if self.render_mode == "human":
            self.render()
        return (
            np.array(self.state, dtype=np.float32),
            reward,
            terminated,
            False,
            {},
        )


# ── CartPole pixel environment wrapper ───────────────────────────────────────


class CartPolePixelEnv(gym.Wrapper):
    """CartPole-v1 with (3, img_size, img_size) uint8 pixel observations.

    SB3 expects channels-first (C, H, W) uint8 images in the observation
    space.  The custom SmallCNN feature extractor below handles 32×32
    inputs (the default NatureCNN requires ≥84px).
    """

    def __init__(self, img_size: int = 32):
        env = ContinuousCartPoleEnv(render_mode="rgb_array")
        super().__init__(env)
        self.img_size = img_size
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(3, img_size, img_size),
            dtype=np.uint8,
        )

    def _obs(self) -> np.ndarray:
        frame = self.env.render()  # (H, W, 3) uint8
        t = preprocess_frame(frame, self.img_size)
        return (t * 255).byte().numpy()  # (3, H, W) uint8

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        return self._obs(), {}

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        return self._obs(), reward, terminated, truncated, info


class SmallCNN(BaseFeaturesExtractor):
    """Minimal CNN feature extractor for 32×32 pixel input.

    Three stride-2 convolutions: 32→16→8→4, then flatten + linear.
    Compatible with SB3 CnnPolicy when passed as features_extractor_class.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        in_ch = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, *observation_space.shape)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flat, features_dim), nn.ReLU())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(obs.float() / 255.0))


# ── Data collection ──────────────────────────────────────────────────────────


def _pd_action(state: np.ndarray) -> int:
    """Simple pole-angle + angular-velocity PD controller."""
    _, _, theta, theta_dot = state
    return 1 if (theta + 0.3 * theta_dot) > 0 else 0


def collect_data(
    n_episodes: int,
    n_frames: int,
    img_size: int,
    scripted_fraction: float = 0.5,
    max_steps: int = 500,
) -> list[tuple]:
    """Collect CartPole transitions for world-model training.

    Each transition is a tuple:
        (context_frames, action_float, next_frame, true_state)

    context_frames : (n_frames, 3, img_size, img_size) float32 [0,1]
    action_float   : float — encoded as −1.0 (left) or +1.0 (right)
    next_frame     : (3, img_size, img_size) float32 [0,1]
    true_state     : (4,) float32 — (x, x_dot, theta, theta_dot)

    Half the episodes use the PD controller; the rest use a random policy.
    This provides a mix of long (balanced) and short (falling) trajectories.
    """
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    buf = FrameBuffer(n_frames, img_size)
    transitions: list[tuple] = []

    for ep in tqdm(range(n_episodes), desc="Collecting data"):
        use_pd = ep < int(n_episodes * scripted_fraction)
        _, _ = env.reset()
        frame = env.render()
        buf.reset(frame)

        for _ in range(max_steps):
            true_state = np.array(env.unwrapped.state, dtype=np.float32)
            context = buf.get()  # (n_frames, 3, H, W)

            action = (
                _pd_action(true_state) if use_pd else env.action_space.sample()
            )
            action_float = float(2 * action - 1)  # 0 → -1.0, 1 → +1.0

            _, _, terminated, truncated, _ = env.step(action)
            next_frame_raw = env.render()
            next_frame = preprocess_frame(next_frame_raw, img_size)
            next_state = np.array(env.unwrapped.state, dtype=np.float32)

            transitions.append((context, action_float, next_frame, next_state))
            buf.push(next_frame_raw)

            if terminated or truncated:
                break

    env.close()
    print(
        f"  Collected {len(transitions)} transitions from {n_episodes} episodes."
    )
    return transitions


def collect_val_trajectories(
    n_episodes: int,
    n_frames: int,
    img_size: int,
    max_steps: int = 500,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Collect full episode trajectories for visualisation.

    Returns a list of (frames, actions) tuples, one per episode:
        frames  : (T+1, 3, img_size, img_size) float32 [0,1] — all frames
                  including the initial context frames and every subsequent frame
        actions : (T,)  float32 — action taken at each step (−1.0 or +1.0)

    Uses the PD controller so episodes are long enough to be interesting.
    """
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    trajectories = []

    for _ in tqdm(range(n_episodes), desc="Val trajectories"):
        _, _ = env.reset()
        frame = env.render()
        frames = [preprocess_frame(frame, img_size)]
        actions = []

        for _ in range(max_steps):
            true_state = np.array(env.unwrapped.state, dtype=np.float32)
            action = _pd_action(true_state)
            action_float = float(2 * action - 1)

            _, _, terminated, truncated, _ = env.step(action)
            frames.append(preprocess_frame(env.render(), img_size))
            actions.append(action_float)

            if terminated or truncated:
                break

        trajectories.append(
            (
                torch.stack(frames),  # (T+1, 3, H, W)
                torch.tensor(actions, dtype=torch.float32),  # (T,)
            )
        )

    env.close()
    return trajectories


# ── Dataset ──────────────────────────────────────────────────────────────────


class CartPoleDataset(Dataset):
    def __init__(self, transitions: list[tuple]):
        ctx = torch.stack([t[0] for t in transitions])  # (N, T, 3, H, W)
        acts = torch.tensor(
            [t[1] for t in transitions], dtype=torch.float32
        ).unsqueeze(-1)  # (N, 1)
        nxt = torch.stack([t[2] for t in transitions])  # (N, 3, H, W)
        states = torch.from_numpy(
            np.stack([t[3] for t in transitions])
        )  # (N, 4)
        self.ctx = ctx
        self.acts = acts
        self.nxt = nxt
        self.states = states

    def __len__(self):
        return len(self.ctx)

    def __getitem__(self, idx):
        return self.ctx[idx], self.acts[idx], self.nxt[idx], self.states[idx]


# ── PPO periodic eval callback ───────────────────────────────────────────────


class SampleEfficiencyCallback(BaseCallback):
    """Evaluate PPO at regular intervals and log mean reward vs env steps.

    Logs to the shared ``comparison/mean_reward`` scalar so that the PPO
    learning curve and the MPPI single data-point appear on the same
    TensorBoard panel with env steps on the x-axis.
    """

    def __init__(
        self,
        eval_env: CartPolePixelEnv,
        writer,
        eval_freq: int = 5_000,
        n_eval_episodes: int = 10,
    ):
        super().__init__(verbose=0)
        self.eval_env = eval_env
        self.writer = writer
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            rewards = []
            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                total_r = 0.0
                while True:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, r, done, _ = self.eval_env.step(action)
                    total_r += float(r[0])
                    if done[0]:
                        break
                rewards.append(total_r)
            self.writer.add_scalar(
                "comparison/mean_reward",
                float(np.mean(rewards)),
                self.num_timesteps,  # x-axis = env steps
            )
        return True


# ── PID episode collection ────────────────────────────────────────────────────


def collect_pid_episode(
    kp_theta: float,
    kd_theta: float,
    kp_x: float,
    kd_x: float,
    epsilon: float,
    img_size: int,
    seq_len: int,
    max_steps: int,
):
    """Collect one CartPole episode driven by a PD controller.

    The control law is:
        u = kp_theta * theta + kd_theta * theta_dot + kp_x * x + kd_x * x_dot

    With probability ``epsilon`` a random action is taken instead.
    Returns an Episode (frames, actions, states).
    """
    from replay_buffer import Episode

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    buf = FrameBuffer(seq_len, img_size)

    _, _ = env.reset()
    first_frame = env.render()
    buf.reset(first_frame)

    frames = [preprocess_frame(first_frame, img_size)]
    actions: list[float] = []
    states: list[np.ndarray] = []

    for _ in range(max_steps):
        true_state = np.array(env.unwrapped.state, dtype=np.float32)
        states.append(true_state)

        x, x_dot, theta, theta_dot = true_state

        if random.random() < epsilon:
            action_float = float(random.choice([-1.0, 1.0]))
        else:
            u = (
                kp_theta * theta
                + kd_theta * theta_dot
                + kp_x * x
                + kd_x * x_dot
            )
            action_float = 1.0 if u >= 0 else -1.0

        gym_action = 1 if action_float > 0 else 0
        _, _, terminated, truncated, _ = env.step(gym_action)

        frame = env.render()
        buf.push(frame)
        frames.append(preprocess_frame(frame, img_size))
        actions.append(action_float)

        if terminated or truncated:
            break

    env.close()
    states.append(states[-1])  # pad so len(states) == len(frames)

    return Episode(
        frames=torch.stack(frames),
        actions=torch.tensor(actions, dtype=torch.float32).unsqueeze(-1),
        states=torch.from_numpy(np.stack(states)),
    )
