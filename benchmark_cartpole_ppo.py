"""Pure PPO baseline on pixel CartPole.

Trains SB3 PPO with NatureCNN directly on rendered frames (no world model).
Logs reward curve and episode videos to TensorBoard for comparison with MPPI.
"""

from __future__ import annotations

import click
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from benchmark_cartpole import CartPolePixelEnv, SampleEfficiencyCallback
from benchmark_cartpole_lstm import PPOVideoCallback


@click.command()
@click.option(
    "--img-size",
    type=int,
    default=128,
    show_default=True,
    help="Frame resolution (H=W).",
)
@click.option(
    "--n-frames",
    type=int,
    default=4,
    show_default=True,
    help="Frames stacked per observation.",
)
@click.option("--n-train-steps", type=int, default=500_000, show_default=True)
@click.option("--n-eval-episodes", type=int, default=20, show_default=True)
@click.option(
    "--eval-freq",
    type=int,
    default=10_000,
    show_default=True,
    help="Evaluate and log video every N env steps.",
)
@click.option(
    "--n-steps",
    type=int,
    default=512,
    show_default=True,
    help="PPO rollout length.",
)
@click.option("--batch-size", type=int, default=64, show_default=True)
@click.option("--lr", type=float, default=3e-4, show_default=True)
@click.option("--ent-coef", type=float, default=0.01, show_default=True)
@click.option("--device", default="auto", show_default=True)
@click.option(
    "--run-name",
    default="ppo",
    show_default=True,
    help="TensorBoard run name suffix.",
)
def main(
    img_size,
    n_frames,
    n_train_steps,
    n_eval_episodes,
    eval_freq,
    n_steps,
    batch_size,
    lr,
    ent_coef,
    device,
    run_name,
):
    writer = SummaryWriter(comment=f"_{run_name}_img{img_size}")

    env_fn = lambda: CartPolePixelEnv(img_size=img_size)
    vec_env = VecFrameStack(DummyVecEnv([env_fn]), n_stack=n_frames)
    eval_env = VecFrameStack(DummyVecEnv([env_fn]), n_stack=n_frames)

    ppo = PPO(
        "CnnPolicy",
        vec_env,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=4,
        learning_rate=lr,
        ent_coef=ent_coef,
        verbose=1,
        device=device,
    )

    callbacks = CallbackList(
        [
            SampleEfficiencyCallback(
                eval_env=eval_env,
                writer=writer,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
            ),
            PPOVideoCallback(
                img_size=img_size,
                n_frames=n_frames,
                writer=writer,
                log_freq=eval_freq,
            ),
        ]
    )

    print(
        f"Training PPO for {n_train_steps:,} steps on {img_size}×{img_size} CartPole..."
    )
    ppo.learn(total_timesteps=n_train_steps, callback=callbacks)

    # Final evaluation
    rewards = []
    for _ in tqdm(range(n_eval_episodes), desc="Final eval"):
        obs = eval_env.reset()
        total_r = 0.0
        while True:
            action, _ = ppo.predict(obs, deterministic=True)
            obs, r, done, _ = eval_env.step(action)
            total_r += float(r[0])
            if done[0]:
                break
        rewards.append(total_r)

    eval_env.close()
    vec_env.close()

    mean_r = float(np.mean(rewards))
    std_r = float(np.std(rewards))
    print(
        f"\nFinal: mean={mean_r:.1f}  std={std_r:.1f}  "
        f"min={min(rewards):.0f}  max={max(rewards):.0f}"
    )
    writer.add_scalar("comparison/mean_reward", mean_r, n_train_steps)
    writer.close()


if __name__ == "__main__":
    main()
