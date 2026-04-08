"""Pixel-to-state benchmark: seq-to-seq supervised state estimation.

A middle step between the DHGN family and the full online PHGN-LSTM loop.
No Hamiltonian rollout — just:

    pixel frames (B, T, C, H, W)
        → LSTM encoder  (forward_all)
        → latent (q_t, p_t) at every timestep
        → state_decoder
        → predicted CartPole state (x, ẋ, θ, θ̇)  for every t

Loss: MSE against env.unwrapped.state ground truth (+ optional KL).

Use this to verify the encoder + state_decoder can learn a meaningful
latent representation before plugging in the Hamiltonian dynamics.
"""

from __future__ import annotations

import random

import click
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from benchmark_cartpole import FrameBuffer, preprocess_frame
from checkpoint_common import make_run_dir, save_checkpoint
from diag_common import (
    ActivationMonitor,
    log_gradient_stats,
    log_histograms,
    log_latent_stats,
    log_weight_norms,
)
from phgn_lstm import ControlledDHGN_LSTM
from replay_buffer import Episode, EpisodeReplayBuffer

_STATE_LABELS = [
    "cart_pos (x)",
    "cart_vel (ẋ)",
    "pole_angle (θ)",
    "pole_vel (θ̇)",
]


# ── Episode collection ────────────────────────────────────────────────────────


def collect_episode(img_size: int, max_steps: int) -> Episode:
    """Run one CartPole episode with random ±1 actions."""
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    _, _ = env.reset()

    first_frame = env.render()
    frames = [preprocess_frame(first_frame, img_size)]
    actions: list[float] = []
    states: list[np.ndarray] = []

    for _ in range(max_steps):
        states.append(np.array(env.unwrapped.state, dtype=np.float32))
        action_float = float(random.choice([-1.0, 1.0]))
        gym_action = 1 if action_float > 0 else 0
        _, _, terminated, truncated, _ = env.step(gym_action)
        frame = env.render()
        frames.append(preprocess_frame(frame, img_size))
        actions.append(action_float)
        if terminated or truncated:
            break

    env.close()
    states.append(states[-1])  # pad terminal state

    return Episode(
        frames=torch.stack(frames),  # (T+1, 3, H, W)
        actions=torch.tensor(actions, dtype=torch.float32).unsqueeze(
            -1
        ),  # (T, 1)
        states=torch.from_numpy(np.stack(states)),  # (T+1, 4)
    )


# ── Training step ─────────────────────────────────────────────────────────────


def train_step(
    model: ControlledDHGN_LSTM,
    buffer: EpisodeReplayBuffer,
    optimizer: torch.optim.Optimizer,
    seq_len: int,
    batch_size: int,
    kl_weight: float,
    free_bits: float,
    grad_clip: float,
    device: torch.device,
    writer: SummaryWriter,
    step: int,
    act_monitor: ActivationMonitor,
    do_diag: bool = False,
) -> dict[str, float]:
    """One gradient step: encode seq_len frames, decode per-timestep state.

    forward_all() returns the LSTM hidden state after each frame prefix.
    We decode every hidden state to a predicted CartPole state and supervise
    with MSE against the ground truth at the corresponding timestep.

    When do_diag is True, logs gradient norms, activation stats, latent
    stats, J/R matrix norms, and weight norms to TensorBoard.
    """
    frames, _actions, states = buffer.sample_sequences(
        batch_size, seq_len=seq_len
    )
    frames = frames.to(device)  # (B, seq_len+1, 3, H, W)
    if states is not None:
        states = states.to(device)  # (B, seq_len+1, 4)

    model.train()

    # Encode all seq_len context frames in one pass.
    # all_mu[:, t] is the posterior mean after seeing frames 0..t (inclusive).
    all_mu, all_logvar = model.encoder.forward_all(frames[:, :seq_len])
    all_logvar = all_logvar.clamp(-10, 10)

    # Reparameterise: (B, seq_len, latent_ch, 4, 4)
    eps = torch.randn_like(all_mu)
    z_all = all_mu + eps * (0.5 * all_logvar).exp()

    B, T, ch, h, w = z_all.shape
    z_flat = z_all.reshape(B * T, ch, h, w)

    # f_psi: latent_ch → 2*pos_ch  (position + momentum)
    s_flat = model.f_psi(z_flat)  # (B*T, 2*pos_ch, 4, 4)
    q_flat, p_flat = model._split(s_flat)  # (B*T, pos_ch, 4, 4) each

    # Decode predicted states
    pred_states_flat = model.decode_state(q_flat, p_flat)  # (B*T, 4)
    pred_states = pred_states_flat.reshape(B, T, -1)  # (B, seq_len, 4)

    # Ground truth: states[:, 0:seq_len] aligns with frame prefixes 0..seq_len-1
    target_states = states[:, :seq_len]  # (B, seq_len, 4)
    state_loss = F.mse_loss(pred_states, target_states)

    # Optional KL regularisation
    kl_per = (-0.5 * (1 + all_logvar - all_mu.pow(2) - all_logvar.exp())).sum(
        dim=[2, 3, 4]
    )  # (B, seq_len)
    kl_loss = kl_per.clamp(min=free_bits).mean()

    loss = state_loss + kl_weight * kl_loss

    optimizer.zero_grad()
    loss.backward()

    if do_diag:
        log_gradient_stats(writer, model, step)
        act_monitor.log(writer, step)
        act_monitor.check_flags(step)
        log_latent_stats(
            writer,
            {"z": (all_mu.detach(), all_logvar.detach())},
            step,
            free_bits,
        )
        log_weight_norms(writer, model, step)
        with torch.no_grad():
            writer.add_scalar(
                "diag/J_frobenius", model.get_J().norm(p="fro").item(), step
            )
            writer.add_scalar(
                "diag/R_frobenius", model.get_R().norm(p="fro").item(), step
            )
            J = model.get_J().cpu()
            R = model.get_R().cpu()
            writer.add_scalar(
                "diag/J_skew_residual", (J + J.T).norm().item(), step
            )
            writer.add_scalar(
                "diag/R_min_eigenvalue",
                torch.linalg.eigvalsh(R).min().item(),
                step,
            )

    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    return {
        "loss/total": loss.item(),
        "loss/state": state_loss.item(),
        "loss/kl": kl_loss.item(),
    }


# ── Evaluation ────────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(
    model: ControlledDHGN_LSTM,
    seq_len: int,
    img_size: int,
    max_steps: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one episode, return (true_states, pred_states) as numpy arrays."""
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    buf = FrameBuffer(seq_len, img_size)

    _, _ = env.reset()
    first_frame = env.render()
    buf.reset(first_frame)

    true_states: list[np.ndarray] = []
    pred_states: list[np.ndarray] = []

    model.eval()
    for _ in range(max_steps):
        true_states.append(np.array(env.unwrapped.state, dtype=np.float32))

        ctx = buf.get().unsqueeze(0).to(device)  # (1, seq_len, 3, H, W)
        mu, _ = model.encoder(ctx)  # (1, latent_ch, 4, 4)
        s = model.f_psi(mu)
        q, p = model._split(s)
        pred = model.decode_state(q, p).squeeze(0).cpu().numpy()  # (4,)
        pred_states.append(pred)

        gym_action = random.choice([0, 1])
        _, _, terminated, truncated, _ = env.step(gym_action)
        frame = env.render()
        buf.push(frame)
        if terminated or truncated:
            break

    env.close()
    return np.stack(true_states), np.stack(pred_states)


def plot_predictions(
    true: np.ndarray,
    pred: np.ndarray,
    step: int,
    writer: SummaryWriter,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    t = np.arange(len(true))
    for i, (ax, label) in enumerate(zip(axes.flat, _STATE_LABELS)):
        ax.plot(t, true[:, i], label="ground truth", color="steelblue")
        ax.plot(
            t, pred[:, i], label="predicted", color="darkorange", linestyle="--"
        )
        ax.set_title(label)
        ax.set_xlabel("step")
        ax.legend(fontsize=8)
    fig.suptitle(f"Pixel → state predictions (step {step})")
    fig.tight_layout()
    writer.add_figure("eval/state_predictions", fig, global_step=step)
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────


@click.command()
@click.option(
    "--img-size", default=64, show_default=True, help="Frame resolution"
)
@click.option(
    "--pos-ch", default=8, show_default=True, help="Latent position channels"
)
@click.option(
    "--seq-len", default=8, show_default=True, help="LSTM context length"
)
@click.option(
    "--feat-dim",
    default=256,
    show_default=True,
    help="Per-frame CNN feature dim",
)
@click.option(
    "--n-collect",
    default=200,
    show_default=True,
    help="Episodes to collect before training",
)
@click.option(
    "--n-iters", default=5000, show_default=True, help="Gradient steps"
)
@click.option("--batch-size", default=32, show_default=True, help="Batch size")
@click.option("--lr", default=3e-4, show_default=True, help="Learning rate")
@click.option(
    "--kl-weight", default=1e-3, show_default=True, help="KL loss weight"
)
@click.option(
    "--free-bits", default=0.5, show_default=True, help="KL free-bits floor"
)
@click.option(
    "--grad-clip",
    default=10.0,
    show_default=True,
    help="Gradient clip norm (0=off)",
)
@click.option(
    "--max-steps", default=500, show_default=True, help="Max steps per episode"
)
@click.option(
    "--eval-every",
    default=500,
    show_default=True,
    help="Eval + checkpoint interval",
)
@click.option(
    "--diag-every",
    default=100,
    show_default=True,
    help="Gradient/activation/latent diag interval",
)
@click.option(
    "--hist-every",
    default=1000,
    show_default=True,
    help="Weight/gradient histogram interval",
)
@click.option(
    "--buffer-cap",
    default=1000,
    show_default=True,
    help="Replay buffer capacity",
)
@click.option(
    "--device", default="cuda", show_default=True, help="torch device"
)
def main(
    img_size: int,
    pos_ch: int,
    seq_len: int,
    feat_dim: int,
    n_collect: int,
    n_iters: int,
    batch_size: int,
    lr: float,
    kl_weight: float,
    free_bits: float,
    grad_clip: float,
    max_steps: int,
    eval_every: int,
    diag_every: int,
    hist_every: int,
    buffer_cap: int,
    device: str,
) -> None:
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    model = ControlledDHGN_LSTM(
        pos_ch=pos_ch,
        img_ch=3,
        feat_dim=feat_dim,
        img_size=img_size,
        control_dim=1,
        obs_state_dim=4,
    ).to(dev)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    buffer = EpisodeReplayBuffer(capacity=buffer_cap, min_seq_len=seq_len)
    act_monitor = ActivationMonitor(model)

    run_dir = make_run_dir("pixels_to_state")
    writer = SummaryWriter(comment="_pixels_to_state")

    hparams = dict(
        img_size=img_size,
        pos_ch=pos_ch,
        seq_len=seq_len,
        feat_dim=feat_dim,
        n_collect=n_collect,
        n_iters=n_iters,
        batch_size=batch_size,
        lr=lr,
        kl_weight=kl_weight,
        free_bits=free_bits,
        grad_clip=grad_clip,
    )
    hparam_text = "| param | value |\n| --- | --- |\n" + "".join(
        f"| {k} | {v} |\n" for k, v in hparams.items()
    )
    writer.add_text("hparams", hparam_text, 0)

    # ── Data collection ───────────────────────────────────────────────────
    print(f"Collecting {n_collect} episodes …")
    for _ in tqdm(range(n_collect), desc="collect"):
        ep = collect_episode(img_size=img_size, max_steps=max_steps)
        buffer.push(ep)

    print(f"Buffer: {len(buffer)} episodes, {buffer.num_steps()} steps")
    writer.add_scalar("collect/episodes", len(buffer), 0)
    writer.add_scalar("collect/total_steps", buffer.num_steps(), 0)

    # ── Training ──────────────────────────────────────────────────────────
    for step in tqdm(range(1, n_iters + 1), desc="train"):
        do_diag = (diag_every > 0) and (step % diag_every == 0)

        losses = train_step(
            model=model,
            buffer=buffer,
            optimizer=optimizer,
            seq_len=seq_len,
            batch_size=batch_size,
            kl_weight=kl_weight,
            free_bits=free_bits,
            grad_clip=grad_clip,
            device=dev,
            writer=writer,
            step=step,
            act_monitor=act_monitor,
            do_diag=do_diag,
        )
        for k, v in losses.items():
            writer.add_scalar(k, v, step)

        if hist_every > 0 and step % hist_every == 0:
            log_histograms(writer, model, step)

        if step % eval_every == 0:
            true_s, pred_s = evaluate(
                model=model,
                seq_len=seq_len,
                img_size=img_size,
                max_steps=max_steps,
                device=dev,
            )
            mse_per_dim = ((true_s - pred_s) ** 2).mean(axis=0)
            for i, label in enumerate(_STATE_LABELS):
                writer.add_scalar(f"eval/mse_{label}", mse_per_dim[i], step)
            writer.add_scalar("eval/mse_mean", mse_per_dim.mean(), step)

            plot_predictions(true_s, pred_s, step=step, writer=writer)

            save_checkpoint(
                run_dir=run_dir,
                epoch=step,
                model=model,
                hparams=hparams,
                metrics={k: v for k, v in losses.items()},
            )
            tqdm.write(
                f"step {step:5d}  state_loss={losses['loss/state']:.4f}"
                f"  eval_mse={mse_per_dim.mean():.4f}"
            )

    writer.close()
    print(f"Run saved to {run_dir}")


if __name__ == "__main__":
    main()
