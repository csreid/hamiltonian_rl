"""
CartPole pixel benchmark: ControlledDissipativeHGN + MPPI vs PPO.

Pipeline
--------
1. Data collection
   Run CartPole-v1 (rgb_array) for N episodes using a 50/50 mix of random
   and a simple PD controller that keeps the pole up longer.  Record
   (context_frames, action_float, next_frame, true_cartpole_state).

2. World model training (ControlledDissipativeHGN / PHGN)
   For each transition, the model encodes the context frames → (q, p),
   applies one controlled_step with the recorded action, decodes the result,
   and minimises:
       L = recon_weight · MSE(pred_frame, next_frame)
         + kl_weight   · KL(posterior || N(0,I))
         + state_weight· MSE(decode_state(q1,p1), true_state)

3. MPPI evaluation
   At each environment step, encode the last n_frames pixel frames via
   encode_mean → (q0, p0), then call MPPI.plan to obtain a float action.
   Threshold: action > 0 → gym action 1 (push right), else 0 (push left).

4. PPO baseline
   Train Stable-Baselines3 PPO with a custom small CNN feature extractor
   on the same pixel CartPole environment for comparison.

Both agents are evaluated on the same CartPole-v1 environment with RGB
pixel observations resized to 32×32.  Results are logged to TensorBoard.
"""

from __future__ import annotations

import random

import click
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.cartpole import (  # noqa: F401 — re-exported for backward compat
    CartPoleDataset,
    CartPolePixelEnv,
    ContinuousCartPoleEnv,
    FrameBuffer,
    SampleEfficiencyCallback,
    SmallCNN,
    _pd_action,
    collect_data,
    collect_val_trajectories,
    preprocess_frame,
)
from diag_common import log_gt_pred_video
from mppi import MPPI
from phgn import ControlledDissipativeHGN

# CartPole failure thresholds (same as gym CartPole-v1)
_CART_X_LIMIT = 2.4
_POLE_THETA_LIMIT = 0.2094  # 12 degrees in radians


# ── World model training ──────────────────────────────────────────────────────


def train_world_model(
    model: ControlledDissipativeHGN,
    dataset: CartPoleDataset,
    writer: SummaryWriter,
    n_epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-4,
    kl_weight: float = 1e-3,
    recon_weight: float = 1.0,
    state_weight: float = 0.1,
    free_bits: float = 0.5,
    grad_clip: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> None:
    """Train the PHGN world model on collected CartPole transitions.

    One-step prediction loss:
        L = recon_weight  · MSE(pred_frame, actual_next_frame)
          + kl_weight     · KL(posterior ‖ N(0,I))
          + state_weight  · MSE(decoded_state, true_cartpole_state)
    """
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    global_step = 0

    for epoch in (bar := tqdm(range(n_epochs), desc="World model")):
        ep_recon = ep_kl = ep_state = ep_total = 0.0

        for ctx, acts, nxt, states in loader:
            ctx = ctx.to(device)  # (B, T, 3, H, W)
            acts = acts.to(device)  # (B, 1)
            nxt = nxt.to(device)  # (B, 3, H, W)
            states = states.to(device)  # (B, 4)

            # Encode context → (q0, p0)
            q0, p0, kl, mu, log_var = model(ctx)
            kl_loss = kl.clamp(min=free_bits).mean()

            # One controlled step with recorded action
            q1, p1 = model.controlled_step(q0, p0, acts)

            # Pixel reconstruction
            pred_frame = model.decoder(q1)
            recon_loss = F.mse_loss(pred_frame, nxt)

            # State prediction (if decoder present)
            if model.state_decoder is not None:
                pred_state = model.decode_state(q1, p1)
                state_loss = F.mse_loss(pred_state, states)
            else:
                state_loss = torch.tensor(0.0, device=device)

            loss = (
                recon_weight * recon_loss
                + kl_weight * kl_loss
                + state_weight * state_loss
            )

            optimiser.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimiser.step()

            writer.add_scalar("wm/recon_loss", recon_loss.item(), global_step)
            writer.add_scalar("wm/kl_loss", kl_loss.item(), global_step)
            writer.add_scalar("wm/state_loss", state_loss.item(), global_step)
            writer.add_scalar("wm/total_loss", loss.item(), global_step)
            global_step += 1

            ep_recon += recon_loss.item()
            ep_kl += kl_loss.item()
            ep_state += state_loss.item()
            ep_total += loss.item()

        n = len(loader)
        bar.set_postfix(
            loss=f"{ep_total / n:.4f}",
            recon=f"{ep_recon / n:.4f}",
            state=f"{ep_state / n:.4f}",
        )


# ── MPPI cost function ────────────────────────────────────────────────────────


def make_cartpole_cost(model: ControlledDissipativeHGN, device: torch.device):
    """Return an MPPI cost function for CartPole balancing.

    Uses model.decode_state to convert each imagined latent state to
    (x, ẋ, θ, θ̇) and accumulates:
        • soft cost: θ² + 0.1 x²   (penalise leaning and off-centre)
        • hard cost: 100 · 1[failed]  (penalise predicted failure)

    Args:
        model:  ControlledDissipativeHGN with obs_state_dim=4
        device: torch device

    Returns:
        cost_fn: callable(qs, ps) → (K,) accumulated costs
    """

    def cost_fn(qs, ps):
        K = qs[0].shape[0]
        costs = torch.zeros(K, device=device)
        for q, p in zip(qs[1:], ps[1:]):  # skip the initial (t=0) state
            state = model.decode_state(q, p)  # (K, 4)
            x = state[:, 0]
            theta = state[:, 2]
            step_cost = theta.pow(2) + 0.1 * x.pow(2)
            failed = (x.abs() > _CART_X_LIMIT) | (
                theta.abs() > _POLE_THETA_LIMIT
            )
            costs += step_cost + 100.0 * failed.float()
        return costs

    return cost_fn


# ── MPPI evaluation ───────────────────────────────────────────────────────────


def run_mppi_episode(
    model: ControlledDissipativeHGN,
    planner: MPPI,
    cost_fn,
    n_frames: int,
    img_size: int,
    max_steps: int = 500,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Run one CartPole episode under MPPI. Returns total reward."""
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    buf = FrameBuffer(n_frames, img_size)

    _, _ = env.reset()
    buf.reset(env.render())
    planner.reset()
    model.eval()

    total_reward = 0.0
    with torch.no_grad():
        for _ in range(max_steps):
            ctx = buf.get().unsqueeze(0).to(device)  # (1, T, 3, H, W)
            q0, p0 = model.encode_mean(ctx)

            action_float = planner.plan(q0, p0, cost_fn)
            gym_action = 1 if action_float.item() > 0 else 0

            _, reward, terminated, truncated, _ = env.step(gym_action)
            buf.push(env.render())
            total_reward += reward
            if terminated or truncated:
                break

    env.close()
    return total_reward


def evaluate_mppi(
    model: ControlledDissipativeHGN,
    planner: MPPI,
    cost_fn,
    writer: SummaryWriter,
    n_episodes: int,
    n_frames: int,
    img_size: int,
    device: torch.device,
    env_step_budget: int = 0,
) -> float:
    """Evaluate MPPI for n_episodes episodes. Logs to TensorBoard.

    Logs ``comparison/mean_reward`` at x = env_step_budget so it appears
    on the same panel as PPO's learning curve with env steps on the x-axis.
    """
    rewards = []
    for _ in tqdm(range(n_episodes), desc="MPPI eval"):
        r = run_mppi_episode(
            model, planner, cost_fn, n_frames, img_size, device=device
        )
        rewards.append(r)

    mean_r = float(np.mean(rewards))
    std_r = float(np.std(rewards))

    # Single data point at x = env steps used for data collection.
    writer.add_scalar("comparison/mean_reward", mean_r, env_step_budget)

    writer.add_text(
        "eval/mppi_summary",
        f"mean={mean_r:.1f}  std={std_r:.1f}  "
        f"min={min(rewards):.0f}  max={max(rewards):.0f}  "
        f"env_steps={env_step_budget:,}",
    )
    print(
        f"\nMPPI:  mean reward = {mean_r:.1f} ± {std_r:.1f}"
        f"  (over {n_episodes} episodes, {env_step_budget:,} env steps used)"
    )
    return mean_r


# ── Rollout visualisation ─────────────────────────────────────────────────────


def log_rollout_reconstructions(
    model: ControlledDissipativeHGN,
    trajectories: list[tuple[torch.Tensor, torch.Tensor]],
    writer: SummaryWriter,
    epoch: int,
    n_frames: int,
    device: torch.device,
    n_trajs: int = 4,
    rollout_steps: int = 20,
) -> None:
    """Roll out the world model over a recorded action sequence and log GT vs predicted frames.

    For each trajectory:
      1. Encode the first n_frames frames → (q0, p0) via encode_mean
      2. Apply controlled_step for each recorded action, accumulating decoded frames
      3. Log a side-by-side GT | predicted video using log_gt_pred_video

    Args:
        trajectories:  output of collect_val_trajectories
        n_trajs:       how many trajectories to visualise (uses the shortest ones
                       so the video length is consistent)
        rollout_steps: how many steps to roll out (capped to trajectory length)
    """
    model.eval()
    n_trajs = min(n_trajs, len(trajectories))

    # Pick the n_trajs longest trajectories up to rollout_steps
    trajs = sorted(trajectories, key=lambda t: len(t[1]), reverse=True)[
        :n_trajs
    ]
    max_steps = min(rollout_steps, min(len(a) for _, a in trajs))

    gt_vids = []  # each (T, 3, H, W)
    pred_vids = []

    with torch.no_grad():
        for frames, actions in trajs:
            # Context: first n_frames frames → latent state
            ctx = frames[:n_frames].unsqueeze(0).to(device)  # (1, T, 3, H, W)
            q, p = model.encode_mean(ctx)

            gt_seq = []
            pred_seq = []

            for t in range(max_steps):
                u = actions[t].reshape(1, 1).to(device)  # (1, 1)
                q, p = model.controlled_step(q, p, u)
                pred_frame = model.decoder(q).clamp(0, 1)  # (1, 3, H, W)

                gt_seq.append(frames[n_frames + t])  # (3, H, W)
                pred_seq.append(pred_frame.squeeze(0).cpu())  # (3, H, W)

            gt_vids.append(torch.stack(gt_seq))  # (T, 3, H, W)
            pred_vids.append(torch.stack(pred_seq))  # (T, 3, H, W)

    # Stack into (N, T, 3, H, W) for log_gt_pred_video
    gt_tensor = torch.stack(gt_vids)  # (N, T, 3, H, W)
    pred_tensor = torch.stack(pred_vids)

    log_gt_pred_video(
        writer, "val/gt_vs_pred_rollout", gt_tensor, pred_tensor, epoch
    )


# ── PPO baseline ─────────────────────────────────────────────────────────────


def train_and_eval_ppo(
    writer: SummaryWriter,
    img_size: int,
    n_train_steps: int,
    n_eval_episodes: int,
    n_frames: int = 4,
    ppo_eval_freq: int = 5_000,
    device: str = "auto",
) -> float:
    """Train SB3 PPO on pixel CartPole and evaluate.

    Logs ``comparison/mean_reward`` vs env steps every ``ppo_eval_freq``
    steps via SampleEfficiencyCallback, so the learning curve lands on the
    same TensorBoard panel as the MPPI data point.
    """
    env_fn = lambda: CartPolePixelEnv(img_size=img_size)
    vec_env = VecFrameStack(DummyVecEnv([env_fn]), n_stack=n_frames)
    eval_env = VecFrameStack(DummyVecEnv([env_fn]), n_stack=n_frames)

    policy_kwargs = dict(
        features_extractor_class=SmallCNN,
        features_extractor_kwargs=dict(features_dim=256),
    )

    ppo = PPO(
        "CnnPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        n_steps=512,
        batch_size=64,
        n_epochs=4,
        learning_rate=3e-4,
        ent_coef=0.01,
        verbose=0,
        device=device,
    )

    callback = SampleEfficiencyCallback(
        eval_env=eval_env,
        writer=writer,
        eval_freq=ppo_eval_freq,
        n_eval_episodes=n_eval_episodes,
    )

    print(
        f"\nTraining PPO for {n_train_steps:,} env steps "
        f"(eval every {ppo_eval_freq:,} steps)..."
    )
    ppo.learn(total_timesteps=n_train_steps, callback=callback)

    # Final evaluation point at exactly n_train_steps.
    rewards = []
    for _ in tqdm(range(n_eval_episodes), desc="PPO final eval"):
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
    mean_r = float(np.mean(rewards))
    std_r = float(np.std(rewards))
    writer.add_scalar("comparison/mean_reward", mean_r, n_train_steps)
    writer.add_text(
        "eval/ppo_summary",
        f"mean={mean_r:.1f}  std={std_r:.1f}  "
        f"min={min(rewards):.0f}  max={max(rewards):.0f}",
    )
    print(
        f"PPO:   mean reward = {mean_r:.1f} ± {std_r:.1f}"
        f"  (over {n_eval_episodes} episodes)"
    )
    return mean_r


# ── CLI ──────────────────────────────────────────────────────────────────────


@click.command()
@click.option("--img-size", type=int, default=32, show_default=True)
@click.option(
    "--n-frames",
    type=int,
    default=4,
    show_default=True,
    help="Context frames stacked for the encoder.",
)
@click.option(
    "--pos-ch",
    type=int,
    default=4,
    show_default=True,
    help="Position channels in the latent state.",
)
@click.option("--dt", type=float, default=0.05, show_default=True)
# ── data collection ──
@click.option(
    "--n-collect",
    type=int,
    default=200,
    show_default=True,
    help="Episodes to collect for world-model training.",
)
@click.option(
    "--scripted-fraction",
    type=float,
    default=0.5,
    show_default=True,
    help="Fraction of collection episodes using the PD controller.",
)
# ── world model ──
@click.option("--wm-epochs", type=int, default=30, show_default=True)
@click.option("--wm-batch", type=int, default=32, show_default=True)
@click.option("--wm-lr", type=float, default=1e-4, show_default=True)
@click.option("--kl-weight", type=float, default=1e-3, show_default=True)
@click.option("--recon-weight", type=float, default=1.0, show_default=True)
@click.option("--state-weight", type=float, default=0.5, show_default=True)
@click.option("--free-bits", type=float, default=0.5, show_default=True)
@click.option("--grad-clip", type=float, default=1.0, show_default=True)
# ── MPPI ──
@click.option("--mppi-horizon", type=int, default=20, show_default=True)
@click.option("--mppi-samples", type=int, default=256, show_default=True)
@click.option("--mppi-temperature", type=float, default=0.05, show_default=True)
@click.option("--mppi-sigma", type=float, default=0.5, show_default=True)
@click.option(
    "--n-eval-mppi",
    type=int,
    default=20,
    show_default=True,
    help="MPPI evaluation episodes.",
)
# ── PPO ──
@click.option(
    "--n-ppo-steps",
    type=int,
    default=200_000,
    show_default=True,
    help="PPO training env steps.",
)
@click.option(
    "--n-eval-ppo",
    type=int,
    default=20,
    show_default=True,
    help="PPO evaluation episodes.",
)
@click.option(
    "--ppo-eval-freq",
    type=int,
    default=5_000,
    show_default=True,
    help="Log PPO comparison/mean_reward every this many env steps.",
)
@click.option(
    "--skip-ppo",
    is_flag=True,
    default=False,
    help="Skip the PPO baseline (useful for quick iteration).",
)
def main(
    img_size,
    n_frames,
    pos_ch,
    dt,
    n_collect,
    scripted_fraction,
    wm_epochs,
    wm_batch,
    wm_lr,
    kl_weight,
    recon_weight,
    state_weight,
    free_bits,
    grad_clip,
    mppi_horizon,
    mppi_samples,
    mppi_temperature,
    mppi_sigma,
    n_eval_mppi,
    n_ppo_steps,
    n_eval_ppo,
    ppo_eval_freq,
    skip_ppo,
):
    assert img_size == 32, (
        "PHGN encoder/decoder expects 32×32; set --img-size 32"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    writer = SummaryWriter(comment="_phgn_cartpole")

    hparam_text = (
        "| Hyperparameter | Value |\n|---|---|\n"
        f"| img_size | {img_size} |\n"
        f"| n_frames | {n_frames} |\n"
        f"| pos_ch | {pos_ch} |\n"
        f"| dt | {dt} |\n"
        f"| n_collect | {n_collect} |\n"
        f"| scripted_fraction | {scripted_fraction} |\n"
        f"| wm_epochs | {wm_epochs} |\n"
        f"| wm_lr | {wm_lr} |\n"
        f"| kl_weight | {kl_weight} |\n"
        f"| recon_weight | {recon_weight} |\n"
        f"| state_weight | {state_weight} |\n"
        f"| mppi_horizon | {mppi_horizon} |\n"
        f"| mppi_samples | {mppi_samples} |\n"
        f"| mppi_temperature | {mppi_temperature} |\n"
        f"| mppi_sigma | {mppi_sigma} |\n"
        f"| n_ppo_steps | {n_ppo_steps} |\n"
        f"| ppo_eval_freq | {ppo_eval_freq} |\n"
    )
    writer.add_text("hparams", hparam_text, 0)

    # ── Phase 1: Data collection ──────────────────────────────────────────
    print("=== Phase 1: Data collection ===")
    transitions = collect_data(
        n_collect,
        n_frames,
        img_size,
        scripted_fraction=scripted_fraction,
    )
    dataset = CartPoleDataset(transitions)

    print("\nCollecting validation trajectories for visualisation...")
    val_trajectories = collect_val_trajectories(
        n_episodes=8, n_frames=n_frames, img_size=img_size
    )

    # ── Phase 2: World model training ────────────────────────────────────
    print("\n=== Phase 2: World model training ===")
    model = ControlledDissipativeHGN(
        n_frames=n_frames,
        pos_ch=pos_ch,
        img_ch=3,
        dt=dt,
        control_dim=1,
        obs_state_dim=4,  # CartPole: (x, x_dot, theta, theta_dot)
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"State dim D = {model.state_dim}  (J,R are {model.state_dim}×{model.state_dim})\n"
    )

    train_world_model(
        model,
        dataset,
        writer,
        n_epochs=wm_epochs,
        batch_size=wm_batch,
        lr=wm_lr,
        kl_weight=kl_weight,
        recon_weight=recon_weight,
        state_weight=state_weight,
        free_bits=free_bits,
        grad_clip=grad_clip,
        device=device,
    )

    # ── Phase 2b: Rollout visualisation ──────────────────────────────────
    print("\n=== Phase 2b: Logging rollout reconstructions ===")
    log_rollout_reconstructions(
        model,
        val_trajectories,
        writer,
        epoch=wm_epochs - 1,
        n_frames=n_frames,
        device=device,
    )

    # ── Phase 3: MPPI evaluation ──────────────────────────────────────────
    print("\n=== Phase 3: MPPI evaluation ===")
    cost_fn = make_cartpole_cost(model, device)
    planner = MPPI(
        model=model,
        horizon=mppi_horizon,
        n_samples=mppi_samples,
        temperature=mppi_temperature,
        noise_sigma=mppi_sigma,
        control_dim=1,
        control_min=-1.0,
        control_max=1.0,
        device=device,
    )
    mppi_mean = evaluate_mppi(
        model,
        planner,
        cost_fn,
        writer,
        n_eval_mppi,
        n_frames,
        img_size,
        device,
        env_step_budget=len(transitions),
    )

    # ── Phase 4: PPO baseline ─────────────────────────────────────────────
    if not skip_ppo:
        print("\n=== Phase 4: PPO baseline ===")
        ppo_mean = train_and_eval_ppo(
            writer,
            img_size=img_size,
            n_train_steps=n_ppo_steps,
            n_eval_episodes=n_eval_ppo,
            n_frames=n_frames,
            ppo_eval_freq=ppo_eval_freq,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        # Comparison summary
        summary = (
            f"| Method | Mean reward |\n|---|---|\n"
            f"| PHGN + MPPI | {mppi_mean:.1f} |\n"
            f"| PPO (SB3) | {ppo_mean:.1f} |\n"
        )
        writer.add_text("eval/comparison", summary)
        print(f"\n{'=' * 40}")
        print(f"PHGN + MPPI  →  {mppi_mean:.1f}")
        print(f"PPO (SB3)    →  {ppo_mean:.1f}")
        print(f"{'=' * 40}")

    writer.close()
    print("\nDone. Run: tensorboard --logdir runs")


if __name__ == "__main__":
    main()
