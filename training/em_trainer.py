"""EM-style trainer for ControlledDHGN_LSTM on CartPole.

Alternates between:
  E-step — train encoder + state_decoder with per-timestep state supervision
  M-step — train dynamics (H, J/R, B, decoder) with frozen encoder
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from checkpoint_common import make_run_dir, save_checkpoint
from data.cartpole import FrameBuffer, preprocess_frame
from diag_common import ActivationMonitor
from training.logging_mixin import LoggingMixin
from training.losses import LossConfig

_STATE_LABELS = [
    "cart_pos (x)",
    "cart_vel (ẋ)",
    "pole_angle (θ)",
    "pole_vel (θ̇)",
]


@dataclass
class EMConfig:
    """All hyperparameters for the EM training loop."""

    # Model / sequence
    seq_len: int = 8
    dt: float = 0.05
    img_size: int = 64

    # Online loop
    n_iterations: int = 300
    collect_per_iter: int = 5
    em_e_steps: int = 25
    em_m_steps: int = 25
    n_warmup: int = 20
    min_buffer: int = 20
    buffer_capacity: int = 2000
    max_episode_steps: int = 500

    # Training
    batch_size: int = 32
    lr: float = 1e-4
    lr_dynamics: float = 1e-4
    kl_weight: float = 1e-3
    recon_weight: float = 1.0
    state_weight: float = 0.5
    fwd_weight: float = 0.5
    anchor_context: int = 3
    free_bits: float = 0.5
    grad_clip: float = 1.0

    # PER
    per_alpha: float = 0.6
    per_beta: float = 0.4
    per_beta_annealing: float = 0.0

    # MPPI
    mppi_horizon: int = 20
    mppi_samples: int = 256
    mppi_temperature: float = 0.05
    mppi_sigma: float = 0.5

    # Eval / logging
    eval_every: int = 10
    n_eval_episodes: int = 10
    log_every: int = 1
    checkpoint_every: int = 50
    diag_every: int = -1  # disabled by default for EM (expensive)

    loss: LossConfig = field(default_factory=LossConfig)

    @classmethod
    def from_kwargs(cls, kwargs: dict) -> "EMConfig":
        return cls(
            seq_len=kwargs.get("seq_len", 8),
            dt=kwargs.get("dt", 0.05),
            img_size=kwargs.get("img_size", 64),
            n_iterations=kwargs.get("n_iterations", 300),
            collect_per_iter=kwargs.get("collect_per_iter", 5),
            em_e_steps=kwargs.get("em_e_steps", 25),
            em_m_steps=kwargs.get("em_m_steps", 25),
            n_warmup=kwargs.get("n_warmup", 20),
            min_buffer=kwargs.get("min_buffer", 20),
            buffer_capacity=kwargs.get("buffer_capacity", 2000),
            max_episode_steps=kwargs.get("max_episode_steps", 500),
            batch_size=kwargs.get("batch_size", 32),
            lr=kwargs.get("lr", 1e-4),
            lr_dynamics=kwargs.get("lr_dynamics", 1e-4),
            kl_weight=kwargs.get("kl_weight", 1e-3),
            recon_weight=kwargs.get("recon_weight", 1.0),
            state_weight=kwargs.get("state_weight", 0.5),
            fwd_weight=kwargs.get("fwd_weight", 0.5),
            anchor_context=kwargs.get("anchor_context", 3),
            free_bits=kwargs.get("free_bits", 0.5),
            grad_clip=kwargs.get("grad_clip", 1.0),
            per_alpha=kwargs.get("per_alpha", 0.6),
            per_beta=kwargs.get("per_beta", 0.4),
            per_beta_annealing=kwargs.get("per_beta_annealing", 0.0),
            mppi_horizon=kwargs.get("mppi_horizon", 20),
            mppi_samples=kwargs.get("mppi_samples", 256),
            mppi_temperature=kwargs.get("mppi_temperature", 0.05),
            mppi_sigma=kwargs.get("mppi_sigma", 0.5),
            eval_every=kwargs.get("eval_every", 10),
            n_eval_episodes=kwargs.get("n_eval_episodes", 10),
            log_every=kwargs.get("log_every", 1),
            checkpoint_every=kwargs.get("checkpoint_every", 50),
            diag_every=kwargs.get("diag_every", -1),
        )


@dataclass
class CartPoleEMConfig(EMConfig):
    """Extends EMConfig with model architecture parameters for CartPole."""

    pos_ch: int = 8
    feat_dim: int = 256
    obs_state_dim: int = 4
    separable: bool = True

    @classmethod
    def from_kwargs(cls, kwargs: dict) -> "CartPoleEMConfig":
        em_cfg = EMConfig.from_kwargs(kwargs)
        em_fields = {
            f: getattr(em_cfg, f) for f in EMConfig.__dataclass_fields__
        }
        return cls(
            **em_fields,
            pos_ch=kwargs.get("pos_ch", 8),
            feat_dim=kwargs.get("feat_dim", 256),
            obs_state_dim=kwargs.get("obs_state_dim", 4),
            separable=kwargs.get("separable", True),
        )


class EMTrainer(LoggingMixin):
    """EM-style trainer for ControlledDHGN_LSTM on CartPole.

    Does NOT inherit BaseTrainer — the outer loop (collect → E-steps → M-steps)
    is fundamentally different from the epoch→batch loop.  Shares LoggingMixin
    for TensorBoard logging utilities.

    Usage:
        trainer = EMTrainer(cfg, model, writer, run_dir, device, buffer, planner, cost_fn, model_hparams)
        trainer.fit()
    """

    def __init__(
        self,
        cfg: EMConfig,
        model,
        writer: SummaryWriter,
        run_dir: Path,
        device: torch.device,
        buffer,
        planner,
        cost_fn,
        model_hparams: dict,
    ):
        self.cfg = cfg
        self.model = model
        self.writer = writer
        self.run_dir = run_dir
        self.device = device
        self.buffer = buffer
        self.planner = planner
        self.cost_fn = cost_fn
        self.model_hparams = model_hparams
        self.global_step = 0
        self._act_monitor = ActivationMonitor(model)

        self.enc_optimizer = torch.optim.Adam(self._encoder_params(), lr=cfg.lr)
        self.dyn_optimizer = torch.optim.Adam(
            self._dynamics_params(), lr=cfg.lr_dynamics
        )

    # ── Main loop ─────────────────────────────────────────────────────────────

    def fit(self) -> None:
        cfg = self.cfg
        use_mppi = len(self.buffer) >= cfg.min_buffer
        best_mean_reward = 0.0
        total_env_steps = self.buffer.num_steps()

        for iteration in (
            pbar := tqdm(range(cfg.n_iterations), desc="Iterations")
        ):
            self.cost_fn = self._make_cost_fn()

            # ── Collect ────────────────────────────────────────────────────
            ep_lens = self._collect_phase(
                use_mppi=use_mppi, total_env_steps_ref=[total_env_steps]
            )
            total_env_steps += sum(ep_lens)

            if not use_mppi and len(self.buffer) >= cfg.min_buffer:
                use_mppi = True
                tqdm.write(
                    f"\n[iter {iteration}] Buffer ready — switching to MPPI"
                )

            self.log_scalar(
                "collect/mean_episode_len",
                float(np.mean(ep_lens)),
                step=iteration,
            )
            self.log_scalar(
                "collect/buffer_episodes", len(self.buffer), step=iteration
            )
            self.log_scalar(
                "collect/total_env_steps", total_env_steps, step=iteration
            )

            if len(self.buffer) == 0:
                continue

            # ── E-step ─────────────────────────────────────────────────────
            e_totals: dict[str, float] = {}
            for _ in range(cfg.em_e_steps):
                losses = self._e_step()
                for k, v in losses.items():
                    e_totals[k] = e_totals.get(k, 0.0) + v
                self.global_step += 1

            # ── M-step ─────────────────────────────────────────────────────
            m_totals: dict[str, float] = {}
            for _ in range(cfg.em_m_steps):
                losses = self._m_step()
                for k, v in losses.items():
                    m_totals[k] = m_totals.get(k, 0.0) + v
                self.global_step += 1

            # ── Logging ────────────────────────────────────────────────────
            if cfg.log_every > 0 and iteration % cfg.log_every == 0:
                if cfg.em_e_steps > 0:
                    self.log_scalars(
                        {
                            f"train/{k}": v / cfg.em_e_steps
                            for k, v in e_totals.items()
                        },
                        step=iteration,
                    )
                if cfg.em_m_steps > 0:
                    self.log_scalars(
                        {
                            f"train/{k}": v / cfg.em_m_steps
                            for k, v in m_totals.items()
                        },
                        step=iteration,
                    )
                pri_stats = self.buffer.priority_stats()
                self.log_scalar(
                    "per/priority_mean", pri_stats["mean"], step=iteration
                )
                self.log_scalar(
                    "per/priority_max", pri_stats["max"], step=iteration
                )
                self.log_scalar(
                    "per/priority_min", pri_stats["min"], step=iteration
                )
                self.log_scalar("per/beta", self.buffer.beta, step=iteration)

            # ── Periodic checkpoint ────────────────────────────────────────
            if (
                cfg.checkpoint_every > 0
                and (iteration + 1) % cfg.checkpoint_every == 0
            ):
                save_checkpoint(
                    self.run_dir,
                    iteration,
                    self.model,
                    self.model_hparams,
                    {"iteration": iteration},
                    stem=f"iter_{iteration + 1}",
                )

            # ── Evaluate ───────────────────────────────────────────────────
            if (iteration + 1) % cfg.eval_every == 0:
                mean_r, all_rewards, video = self._evaluate()
                self.log_scalar("eval/mean_reward", mean_r, step=iteration)
                self.log_scalar(
                    "eval/mean_reward_vs_env_steps",
                    mean_r,
                    step=total_env_steps,
                )
                self.writer.add_histogram(
                    "eval/reward_dist",
                    np.array(all_rewards, dtype=np.float32),
                    iteration,
                )
                self.writer.add_video("eval/rollout", video, iteration, fps=30)

                if mean_r > best_mean_reward:
                    best_mean_reward = mean_r
                    save_checkpoint(
                        self.run_dir,
                        iteration,
                        self.model,
                        self.model_hparams,
                        {"mean_reward": mean_r},
                        stem="best",
                    )

                pbar.set_postfix(
                    mean_r=f"{mean_r:.1f}",
                    best=f"{best_mean_reward:.1f}",
                    buf=len(self.buffer),
                )

        self._act_monitor.remove()
        self.writer.close()

    # ── E-step ────────────────────────────────────────────────────────────────

    def _e_step(self) -> dict[str, float]:
        """Train encoder + state_decoder with per-timestep state supervision."""
        cfg = self.cfg
        batch = self.buffer.sample(cfg.batch_size, seq_len=cfg.seq_len)
        frames = batch.frames.to(self.device)
        is_weights = batch.is_weights.to(self.device)
        lengths = batch.lengths.to(self.device)
        states = (
            batch.states.to(self.device) if batch.states is not None else None
        )

        B = frames.shape[0]
        self.model.train()

        all_mu, all_logvar = self.model.encoder.forward_all(
            frames[:, : cfg.seq_len]
        )
        all_logvar = all_logvar.clamp(-10, 10)

        eps = torch.randn_like(all_mu)
        z_all = all_mu + eps * (0.5 * all_logvar).exp()

        BT, ch, h, w = B * cfg.seq_len, *z_all.shape[2:]
        z_flat = z_all.reshape(BT, ch, h, w)
        s_flat = self.model.f_psi(z_flat)
        q_flat, p_flat = self.model._split(s_flat)

        pred_states = self.model.decode_state(q_flat, p_flat).reshape(
            B, cfg.seq_len, -1
        )
        target_states = states[:, : cfg.seq_len]

        t_idx = torch.arange(cfg.seq_len, device=self.device)
        valid_mask = (t_idx >= (cfg.seq_len - lengths.unsqueeze(1))).float()

        state_elem = F.mse_loss(
            pred_states, target_states, reduction="none"
        ).mean(dim=2)
        state_per = (state_elem * valid_mask).sum(dim=1) / lengths.float()

        kl_elem = (
            -0.5 * (1 + all_logvar - all_mu.pow(2) - all_logvar.exp())
        ).sum(dim=[2, 3, 4])
        kl_per = (kl_elem * valid_mask).sum(dim=1) / lengths.float()
        kl_per = kl_per.clamp(min=cfg.free_bits)

        per_sample = state_per + cfg.kl_weight * kl_per
        loss = (is_weights * per_sample).mean()

        self.enc_optimizer.zero_grad()
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self._encoder_params(), cfg.grad_clip
            )
        self.enc_optimizer.step()

        batch.commit(per_sample.detach().cpu().numpy())
        return {
            "e_loss": loss.item(),
            "e_state": state_per.mean().item(),
            "e_kl": kl_per.mean().item(),
        }

    # ── M-step ────────────────────────────────────────────────────────────────

    def _m_step(self) -> dict[str, float]:
        """Train dynamics (H, J, R, B, decoder) with encoder frozen."""
        cfg = self.cfg
        batch = self.buffer.sample(cfg.batch_size, seq_len=cfg.seq_len)
        frames = batch.frames.to(self.device)
        actions = batch.actions.to(self.device)
        is_weights = batch.is_weights.to(self.device)
        lengths = batch.lengths.to(self.device)
        states = (
            batch.states.to(self.device) if batch.states is not None else None
        )

        B = frames.shape[0]
        rollout_steps = cfg.seq_len - 1
        self.model.train()

        do_state = (
            cfg.state_weight > 0
            and states is not None
            and self.model.state_decoder is not None
        )
        use_anchor = cfg.fwd_weight > 0 and cfg.anchor_context < cfg.seq_len - 1

        with torch.no_grad():
            if use_anchor:
                all_mu, all_logvar = self.model.encoder.forward_all(
                    frames[:, : cfg.seq_len]
                )
                all_logvar = all_logvar.clamp(-10, 10)
                mu = all_mu[:, -1]
                log_var = all_logvar[:, -1]
            else:
                mu, log_var = self.model.encoder(frames[:, : cfg.seq_len])
                log_var = log_var.clamp(-10, 10)

        z = mu + torch.randn_like(mu) * (0.5 * log_var).exp()
        s0 = self.model.f_psi(z)
        q_T, p_T = self.model._split(s0)
        kl_per = (
            (-0.5 * (1 + log_var - mu.pow(2) - log_var.exp()))
            .sum(dim=[1, 2, 3])
            .clamp(min=cfg.free_bits)
        )
        if use_anchor:
            mu_k = all_mu[:, cfg.anchor_context].detach()

        q, p = q_T, p_T
        pred_frames_list = [self.model.decoder(q)]
        state_preds = [self.model.decode_state(q, p)] if do_state else []
        for i in range(rollout_steps):
            u_rev = actions[:, cfg.seq_len - 2 - i]
            q, p = self.model.controlled_step(q, p, u_rev, dt=-cfg.dt)
            pred_frames_list.append(self.model.decoder(q))
            if do_state:
                state_preds.append(self.model.decode_state(q, p))

        pred_frames = torch.stack(pred_frames_list, dim=1)
        target_frames = frames[:, : cfg.seq_len].flip(dims=[1])

        t_bwd = torch.arange(cfg.seq_len, device=self.device)
        valid_mask = (t_bwd >= (cfg.seq_len - lengths.unsqueeze(1))).float()

        recon_elem = F.mse_loss(
            pred_frames, target_frames, reduction="none"
        ).mean(dim=[2, 3, 4])
        recon_per = (recon_elem * valid_mask).sum(dim=1) / lengths.float()

        if do_state:
            pred_states = torch.stack(state_preds, dim=1)
            target_states_bwd = states[:, : cfg.seq_len].flip(dims=[1])
            state_elem = F.mse_loss(
                pred_states, target_states_bwd, reduction="none"
            ).mean(dim=2)
            state_per = (state_elem * valid_mask).sum(dim=1) / lengths.float()
        else:
            state_per = torch.zeros(B, device=self.device)

        if use_anchor:
            k = cfg.anchor_context
            s_k = self.model.f_psi(mu_k)
            q_k, p_k = self.model._split(s_k)
            n_fwd = cfg.seq_len - 1 - k
            fwd_pred_list = [self.model.decoder(q_k)]
            fwd_state_list = (
                [self.model.decode_state(q_k, p_k)] if do_state else []
            )
            q, p = q_k, p_k
            for i in range(n_fwd):
                q, p = self.model.controlled_step(
                    q, p, actions[:, k + i], dt=cfg.dt
                )
                fwd_pred_list.append(self.model.decoder(q))
                if do_state:
                    fwd_state_list.append(self.model.decode_state(q, p))

            fwd_pred = torch.stack(fwd_pred_list, dim=1)
            fwd_target = frames[:, k : cfg.seq_len]
            t_fwd = torch.arange(n_fwd + 1, device=self.device)
            fwd_valid = (k + t_fwd < lengths.unsqueeze(1)).float()
            denom_fwd = fwd_valid.sum(dim=1).clamp(min=1)
            anchor_recon_per = (
                F.mse_loss(fwd_pred, fwd_target, reduction="none").mean(
                    dim=[2, 3, 4]
                )
                * fwd_valid
            ).sum(dim=1) / denom_fwd

            if do_state:
                fwd_st_pred = torch.stack(fwd_state_list, dim=1)
                fwd_st_tgt = states[:, k : cfg.seq_len]
                anchor_state_per = (
                    F.mse_loss(fwd_st_pred, fwd_st_tgt, reduction="none").mean(
                        dim=2
                    )
                    * fwd_valid
                ).sum(dim=1) / denom_fwd
            else:
                anchor_state_per = torch.zeros(B, device=self.device)
        else:
            anchor_recon_per = torch.zeros(B, device=self.device)
            anchor_state_per = torch.zeros(B, device=self.device)

        per_sample = (
            cfg.recon_weight * recon_per
            + cfg.kl_weight * kl_per
            + cfg.state_weight * state_per
            + cfg.fwd_weight
            * (
                cfg.recon_weight * anchor_recon_per
                + cfg.state_weight * anchor_state_per
            )
        )
        loss = (is_weights * per_sample).mean()

        self.dyn_optimizer.zero_grad()
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self._dynamics_params(), cfg.grad_clip
            )
        self.dyn_optimizer.step()

        batch.commit(per_sample.detach().cpu().numpy())
        return {
            "m_loss": loss.item(),
            "m_recon": recon_per.mean().item(),
            "m_kl": kl_per.mean().item(),
            "m_state": state_per.mean().item(),
            "m_anchor_recon": anchor_recon_per.mean().item(),
            "m_anchor_state": anchor_state_per.mean().item(),
        }

    # ── Collection ────────────────────────────────────────────────────────────

    def _collect_phase(self, use_mppi: bool, total_env_steps_ref) -> list[int]:
        import gymnasium as gym
        from replay_buffer import Episode

        cfg = self.cfg
        ep_lens = []
        for _ in range(cfg.collect_per_iter):
            ep = self._collect_episode(use_mppi=use_mppi)
            self.buffer.push(ep)
            ep_lens.append(len(ep))
        return ep_lens

    def _collect_episode(self, use_mppi: bool):
        import gymnasium as gym
        from replay_buffer import Episode

        cfg = self.cfg
        env = gym.make("CartPole-v1", render_mode="rgb_array")
        buf = FrameBuffer(cfg.seq_len, cfg.img_size)

        _, _ = env.reset()
        first_frame = env.render()
        buf.reset(first_frame)

        frames = [preprocess_frame(first_frame, cfg.img_size)]
        actions: list[float] = []
        states: list[np.ndarray] = []

        if self.planner is not None:
            self.planner.reset()

        self.model.eval()
        cost_fn = self.cost_fn
        with torch.no_grad():
            for _ in range(cfg.max_episode_steps):
                true_state = np.array(env.unwrapped.state, dtype=np.float32)
                states.append(true_state)

                if (
                    use_mppi
                    and self.planner is not None
                    and cost_fn is not None
                ):
                    ctx = buf.get().unsqueeze(0).to(self.device)
                    q0, p0 = self.model.encode_mean(ctx)
                    action_t = self.planner.plan(q0, p0, cost_fn)
                    action_float = float(action_t.item())
                else:
                    action_float = float(random.choice([-1.0, 1.0]))

                gym_action = 1 if action_float > 0 else 0
                _, _, terminated, truncated, _ = env.step(gym_action)

                frame = env.render()
                buf.push(frame)
                frames.append(preprocess_frame(frame, cfg.img_size))
                actions.append(action_float)

                if terminated or truncated:
                    break

        env.close()
        states.append(states[-1])

        return Episode(
            frames=torch.stack(frames),
            actions=torch.tensor(actions, dtype=torch.float32).unsqueeze(-1),
            states=torch.from_numpy(np.stack(states)),
        )

    # ── Evaluation ────────────────────────────────────────────────────────────

    def _evaluate(self) -> tuple[float, list[int], torch.Tensor]:
        cfg = self.cfg
        rewards: list[int] = []
        sample_frames = None
        cost_fn = self._make_cost_fn()
        for i in range(cfg.n_eval_episodes):
            ep = self._collect_episode(use_mppi=True)
            rewards.append(len(ep))
            if i == 0:
                sample_frames = ep.frames
        video = (sample_frames.unsqueeze(0).clamp(0, 1) * 255).to(torch.uint8)
        return float(np.mean(rewards)), rewards, video

    # ── Parameter groups ──────────────────────────────────────────────────────

    def _encoder_params(self) -> list:
        """E-step parameters: LSTM encoder, f_psi, state_decoder."""
        params = list(self.model.encoder.parameters()) + list(
            self.model.f_psi.parameters()
        )
        if self.model.state_decoder is not None:
            params += list(self.model.state_decoder.parameters())
        return params

    def _dynamics_params(self) -> list:
        """M-step parameters: Hamiltonian, J/R matrices, B, decoder, coord_head."""
        return (
            list(self.model.hamiltonian.parameters())
            + list(self.model.decoder.parameters())
            + list(self.model.coord_head.parameters())
            + [self.model.A, self.model.L_param, self.model.B]
        )

    def _make_cost_fn(self):
        """Return the CartPole MPPI cost function for the current model."""
        _CART_X_LIMIT = 2.4
        _POLE_THETA_LIMIT = 0.2094
        model = self.model
        device = self.device

        def cost_fn(qs, ps):
            K = qs[0].shape[0]
            costs = torch.zeros(K, device=device)
            for q, p in zip(qs[1:], ps[1:]):
                state = model.decode_state(q, p)
                x, theta = state[:, 0], state[:, 2]
                costs += theta.pow(2) + 0.1 * x.pow(2)
                failed = (x.abs() > _CART_X_LIMIT) | (
                    theta.abs() > _POLE_THETA_LIMIT
                )
                costs += 100.0 * failed.float()
            return costs

        return cost_fn

    # ── Visualisation helpers (available to subclasses) ───────────────────────

    def _log_backward_rollout(self, buffer, step: int) -> None:
        """Log ground-truth vs backward-rollout frames and state trajectories."""
        if not buffer.can_sample(self.cfg.seq_len):
            return

        self.model.eval()
        with torch.no_grad():
            batch = buffer.sample(1, seq_len=self.cfg.seq_len)
            frames = batch.frames.to(self.device)
            actions = batch.actions.to(self.device)
            states = (
                batch.states.to(self.device)
                if batch.states is not None
                else None
            )

            q_T, p_T, *_ = self.model(frames[:, : self.cfg.seq_len])
            q, p = q_T, p_T
            pred_frames_list = [self.model.decoder(q)]
            pred_states_list = []
            if self.model.state_decoder is not None:
                pred_states_list.append(self.model.decode_state(q, p))

            for i in range(self.cfg.seq_len - 1):
                u_rev = actions[:, self.cfg.seq_len - 2 - i]
                q, p = self.model.controlled_step(q, p, u_rev, dt=-self.cfg.dt)
                pred_frames_list.append(self.model.decoder(q))
                if self.model.state_decoder is not None:
                    pred_states_list.append(self.model.decode_state(q, p))

        pred_t = torch.stack(pred_frames_list, dim=1)
        gt = frames[:, : self.cfg.seq_len].flip(dims=[1])
        self.writer.add_video(
            "eval/bwd_gt_rollout",
            (gt.clamp(0, 1) * 255).to(torch.uint8),
            step,
            fps=15,
        )
        self.writer.add_video(
            "eval/bwd_pred_rollout",
            (pred_t.clamp(0, 1) * 255).to(torch.uint8),
            step,
            fps=15,
        )

        if states is not None and pred_states_list:
            gt_s = states[0, : self.cfg.seq_len].flip(dims=[0]).cpu().numpy()
            pr_s = torch.cat(pred_states_list, dim=0).cpu().numpy()
            t = np.arange(self.cfg.seq_len)
            fig, axes = plt.subplots(2, 2, figsize=(10, 7))
            for i, (ax, label) in enumerate(zip(axes.flat, _STATE_LABELS)):
                ax.plot(t, gt_s[:, i], label="gt", color="steelblue")
                ax.plot(
                    t,
                    pr_s[:, i],
                    label="pred",
                    color="darkorange",
                    linestyle="--",
                )
                ax.set_title(label)
                ax.legend(fontsize=8)
            fig.suptitle(f"Backward rollout (iter {step})")
            fig.tight_layout()
            self.writer.add_figure("eval/bwd_state_rollout", fig, step)
            plt.close(fig)

    def _log_forward_validation(self, buffer, step: int) -> None:
        """Validate forward extrapolation beyond the encoding context."""
        long_seq = 2 * self.cfg.seq_len
        seq_len = self.cfg.seq_len

        if buffer.can_sample(long_seq):
            batch = buffer.sample(1, seq_len=long_seq)
            frames = batch.frames.to(self.device)
            actions = batch.actions.to(self.device)
            states = (
                batch.states.to(self.device)
                if batch.states is not None
                else None
            )
        elif buffer.can_sample(seq_len):
            eligible = [ep for ep in buffer._episodes if len(ep) >= seq_len]
            ep = random.choice(eligible)
            end = min(long_seq, len(ep))
            frames = ep.frames[: end + 1].unsqueeze(0).to(self.device)
            actions = ep.actions[:end].unsqueeze(0).to(self.device)
            states = (
                ep.states[: end + 1].unsqueeze(0).to(self.device)
                if ep.states is not None
                else None
            )
        else:
            return

        total_frames = frames.shape[1]
        self.model.eval()
        with torch.no_grad():
            q_T, p_T, *_ = self.model(frames[:, :seq_len])
            q, p = q_T, p_T
            for i in range(seq_len - 1):
                u_rev = actions[:, seq_len - 2 - i]
                q, p = self.model.controlled_step(q, p, u_rev, dt=-self.cfg.dt)

            pred_states_list = []
            if self.model.state_decoder is not None:
                pred_states_list.append(self.model.decode_state(q, p))
            for t_step in range(total_frames - 1):
                q, p = self.model.controlled_step(
                    q, p, actions[:, t_step], dt=self.cfg.dt
                )
                if self.model.state_decoder is not None:
                    pred_states_list.append(self.model.decode_state(q, p))

        if states is not None and pred_states_list:
            n = len(pred_states_list)
            gt_s = states[0, :n].cpu().numpy()
            pr_s = torch.cat(pred_states_list, dim=0).cpu().numpy()
            t = np.arange(n)
            fig, axes = plt.subplots(2, 2, figsize=(10, 7))
            for i, (ax, label) in enumerate(zip(axes.flat, _STATE_LABELS)):
                ax.plot(t, gt_s[:, i], label="gt", color="steelblue")
                ax.plot(
                    t,
                    pr_s[:, i],
                    label="pred",
                    color="darkorange",
                    linestyle="--",
                )
                ax.axvline(seq_len - 1, color="gray", linestyle=":", alpha=0.7)
                ax.set_title(label)
                ax.legend(fontsize=8)
            fig.suptitle(f"Forward validation (iter {step})")
            fig.tight_layout()
            self.writer.add_figure("eval/fwd_state_rollout", fig, step)
            plt.close(fig)


# ── Offline EM ────────────────────────────────────────────────────────────────


@dataclass
class OfflineEMConfig(CartPoleEMConfig):
    """Extends CartPoleEMConfig with offline-specific hyperparameters."""

    # Data collection
    n_transitions: int = 100_000
    test_fraction: float = 0.2
    n_pretrain_steps: int = 500

    # PID gain ranges (low, high)
    pid_kp_theta_range: tuple = field(default_factory=lambda: (5.0, 25.0))
    pid_kd_theta_range: tuple = field(default_factory=lambda: (0.5, 5.0))
    pid_kp_x_range: tuple = field(default_factory=lambda: (-2.0, 2.0))
    pid_kd_x_range: tuple = field(default_factory=lambda: (-1.0, 1.0))
    pid_epsilon: float = 0.05

    @classmethod
    def from_kwargs(cls, kwargs: dict) -> "OfflineEMConfig":
        cp_cfg = CartPoleEMConfig.from_kwargs(kwargs)
        cp_fields = {
            f: getattr(cp_cfg, f) for f in CartPoleEMConfig.__dataclass_fields__
        }
        return cls(
            **cp_fields,
            n_transitions=kwargs.get("n_transitions", 100_000),
            test_fraction=kwargs.get("test_fraction", 0.2),
            n_pretrain_steps=kwargs.get("n_pretrain_steps", 500),
            pid_kp_theta_range=tuple(
                kwargs.get("pid_kp_theta_range", (5.0, 25.0))
            ),
            pid_kd_theta_range=tuple(
                kwargs.get("pid_kd_theta_range", (0.5, 5.0))
            ),
            pid_kp_x_range=tuple(kwargs.get("pid_kp_x_range", (-2.0, 2.0))),
            pid_kd_x_range=tuple(kwargs.get("pid_kd_x_range", (-1.0, 1.0))),
            pid_epsilon=kwargs.get("pid_epsilon", 0.05),
        )


class OfflineEMTrainer(EMTrainer):
    """EM trainer for the offline setting: pre-collected data, no online collection.

    Adds:
    - Encoder pre-training phase (E-step only) before EM begins
    - Held-out test buffer for periodic test-loss logging
    - Backward rollout and forward validation visualisations at eval steps
    """

    def __init__(
        self,
        cfg: OfflineEMConfig,
        model,
        writer: SummaryWriter,
        run_dir: Path,
        device: torch.device,
        buffer,
        test_buffer,
        planner,
        cost_fn,
        model_hparams: dict,
    ):
        super().__init__(
            cfg,
            model,
            writer,
            run_dir,
            device,
            buffer,
            planner,
            cost_fn,
            model_hparams,
        )
        self.test_buffer = test_buffer

    # ── Main loop ─────────────────────────────────────────────────────────────

    def fit(self) -> None:
        cfg = self.cfg
        best_mean_reward = 0.0

        # ── Encoder pre-training ──────────────────────────────────────────────
        if cfg.n_pretrain_steps > 0:
            print(f"\nPre-training encoder for {cfg.n_pretrain_steps} steps...")
            eval_freq = max(1, cfg.n_pretrain_steps // 20)
            for step in tqdm(range(cfg.n_pretrain_steps), desc="Pre-train"):
                losses = self._e_step()
                self.log_scalars(
                    {f"pretrain/{k}": v for k, v in losses.items()}, step=step
                )
                if (
                    self.test_buffer is not None
                    and len(self.test_buffer) > 0
                    and (step + 1) % eval_freq == 0
                ):
                    test_losses = self._compute_test_losses()
                    self.log_scalars(
                        {
                            f"pretrain/test_{k}": v
                            for k, v in test_losses.items()
                        },
                        step=step,
                    )
            print("Pre-training done.")

        # ── Offline EM loop ────────────────────────────────────────────────────
        print(f"\nTraining for {cfg.n_iterations} iterations (offline EM)...")
        for iteration in (
            pbar := tqdm(range(cfg.n_iterations), desc="Training")
        ):
            e_totals: dict[str, float] = {}
            for _ in range(cfg.em_e_steps):
                losses = self._e_step()
                for k, v in losses.items():
                    e_totals[k] = e_totals.get(k, 0.0) + v
                self.global_step += 1

            m_totals: dict[str, float] = {}
            for _ in range(cfg.em_m_steps):
                losses = self._m_step()
                for k, v in losses.items():
                    m_totals[k] = m_totals.get(k, 0.0) + v
                self.global_step += 1

            if cfg.log_every > 0 and iteration % cfg.log_every == 0:
                if cfg.em_e_steps > 0:
                    self.log_scalars(
                        {
                            f"train/{k}": v / cfg.em_e_steps
                            for k, v in e_totals.items()
                        },
                        step=iteration,
                    )
                if cfg.em_m_steps > 0:
                    self.log_scalars(
                        {
                            f"train/{k}": v / cfg.em_m_steps
                            for k, v in m_totals.items()
                        },
                        step=iteration,
                    )
                pri_stats = self.buffer.priority_stats()
                self.log_scalars(
                    {
                        "per/priority_mean": pri_stats["mean"],
                        "per/priority_max": pri_stats["max"],
                        "per/priority_min": pri_stats["min"],
                        "per/beta": self.buffer.beta,
                    },
                    step=iteration,
                )
                if self.test_buffer is not None and len(self.test_buffer) > 0:
                    test_losses = self._compute_test_losses()
                    self.log_scalars(
                        {f"test/{k}": v for k, v in test_losses.items()},
                        step=iteration,
                    )

            if (
                cfg.checkpoint_every > 0
                and (iteration + 1) % cfg.checkpoint_every == 0
            ):
                save_checkpoint(
                    self.run_dir,
                    iteration,
                    self.model,
                    self.model_hparams,
                    {"iteration": iteration},
                    stem=f"iter_{iteration + 1}",
                )

            if (iteration + 1) % cfg.eval_every == 0:
                self.cost_fn = self._make_cost_fn()
                mean_r, all_rewards, video = self._evaluate()
                self.log_scalar("eval/mean_reward", mean_r, step=iteration)
                self.writer.add_histogram(
                    "eval/reward_dist",
                    np.array(all_rewards, dtype=np.float32),
                    iteration,
                )
                self.writer.add_video("eval/rollout", video, iteration, fps=30)
                self._log_backward_rollout(self.buffer, iteration)
                self._log_forward_validation(self.buffer, iteration)

                if mean_r > best_mean_reward:
                    best_mean_reward = mean_r
                    save_checkpoint(
                        self.run_dir,
                        iteration,
                        self.model,
                        self.model_hparams,
                        {"mean_reward": mean_r},
                        stem="best",
                    )

                pbar.set_postfix(
                    mean_r=f"{mean_r:.1f}", best=f"{best_mean_reward:.1f}"
                )

        save_checkpoint(
            self.run_dir,
            cfg.n_iterations - 1,
            self.model,
            self.model_hparams,
            {"best_mean_reward": best_mean_reward},
            stem="last",
        )
        self._act_monitor.remove()
        self.writer.close()

    # ── Test-set loss evaluation ───────────────────────────────────────────────

    @torch.no_grad()
    def _compute_test_losses(self) -> dict[str, float]:
        """Compute E-step and M-step losses on the held-out test buffer."""
        cfg = self.cfg
        self.model.eval()

        # E-step losses
        batch = self.test_buffer.sample(cfg.batch_size, seq_len=cfg.seq_len)
        frames = batch.frames.to(self.device)
        lengths = batch.lengths.to(self.device)
        states = (
            batch.states.to(self.device) if batch.states is not None else None
        )

        B = frames.shape[0]
        all_mu, all_logvar = self.model.encoder.forward_all(
            frames[:, : cfg.seq_len]
        )
        all_logvar = all_logvar.clamp(-10, 10)
        z_all = all_mu + torch.randn_like(all_mu) * (0.5 * all_logvar).exp()
        z_flat = z_all.reshape(B * cfg.seq_len, *z_all.shape[2:])
        s_flat = self.model.f_psi(z_flat)
        q_flat, p_flat = self.model._split(s_flat)

        t_idx = torch.arange(cfg.seq_len, device=self.device)
        valid_mask = (t_idx >= (cfg.seq_len - lengths.unsqueeze(1))).float()

        kl_elem = (
            -0.5 * (1 + all_logvar - all_mu.pow(2) - all_logvar.exp())
        ).sum(dim=[2, 3, 4])
        e_kl = ((kl_elem * valid_mask).sum(dim=1) / lengths.float()).clamp(
            min=cfg.free_bits
        )

        if states is not None:
            pred_states = self.model.decode_state(q_flat, p_flat).reshape(
                B, cfg.seq_len, -1
            )
            state_elem = F.mse_loss(
                pred_states, states[:, : cfg.seq_len], reduction="none"
            ).mean(dim=2)
            e_state = (state_elem * valid_mask).sum(dim=1) / lengths.float()
        else:
            e_state = torch.zeros(B, device=self.device)

        e_loss = (e_state + cfg.kl_weight * e_kl).mean()

        # M-step losses
        batch2 = self.test_buffer.sample(cfg.batch_size, seq_len=cfg.seq_len)
        frames2 = batch2.frames.to(self.device)
        actions2 = batch2.actions.to(self.device)
        lengths2 = batch2.lengths.to(self.device)
        states2 = (
            batch2.states.to(self.device) if batch2.states is not None else None
        )

        B2 = frames2.shape[0]
        rollout_steps = cfg.seq_len - 1
        do_state = (
            cfg.state_weight > 0
            and states2 is not None
            and self.model.state_decoder is not None
        )
        use_anchor = cfg.fwd_weight > 0 and cfg.anchor_context < cfg.seq_len - 1

        if use_anchor:
            all_mu2, all_logvar2 = self.model.encoder.forward_all(
                frames2[:, : cfg.seq_len]
            )
            all_logvar2 = all_logvar2.clamp(-10, 10)
            mu2, log_var2 = all_mu2[:, -1], all_logvar2[:, -1]
            mu_k = all_mu2[:, cfg.anchor_context]
        else:
            mu2, log_var2 = self.model.encoder(frames2[:, : cfg.seq_len])
            log_var2 = log_var2.clamp(-10, 10)

        z2 = mu2 + torch.randn_like(mu2) * (0.5 * log_var2).exp()
        s0 = self.model.f_psi(z2)
        q_T, p_T = self.model._split(s0)
        kl2_per = (
            (-0.5 * (1 + log_var2 - mu2.pow(2) - log_var2.exp()))
            .sum(dim=[1, 2, 3])
            .clamp(min=cfg.free_bits)
        )

        q, p = q_T, p_T
        pred_frames_list = [self.model.decoder(q)]
        state_preds2 = [self.model.decode_state(q, p)] if do_state else []
        for i in range(rollout_steps):
            u_rev = actions2[:, cfg.seq_len - 2 - i]
            q, p = self.model.controlled_step(q, p, u_rev, dt=-cfg.dt)
            pred_frames_list.append(self.model.decoder(q))
            if do_state:
                state_preds2.append(self.model.decode_state(q, p))

        pred_frames2 = torch.stack(pred_frames_list, dim=1)
        target_frames2 = frames2[:, : cfg.seq_len].flip(dims=[1])
        t_bwd = torch.arange(cfg.seq_len, device=self.device)
        valid_mask2 = (t_bwd >= (cfg.seq_len - lengths2.unsqueeze(1))).float()

        recon_elem2 = F.mse_loss(
            pred_frames2, target_frames2, reduction="none"
        ).mean(dim=[2, 3, 4])
        m_recon = (recon_elem2 * valid_mask2).sum(dim=1) / lengths2.float()

        if do_state:
            pred_states2 = torch.stack(state_preds2, dim=1)
            state_elem2 = F.mse_loss(
                pred_states2,
                states2[:, : cfg.seq_len].flip(dims=[1]),
                reduction="none",
            ).mean(dim=2)
            m_state = (state_elem2 * valid_mask2).sum(dim=1) / lengths2.float()
        else:
            m_state = torch.zeros(B2, device=self.device)

        if use_anchor:
            k = cfg.anchor_context
            s_k = self.model.f_psi(mu_k)
            q_k, p_k = self.model._split(s_k)
            n_fwd = cfg.seq_len - 1 - k
            fwd_pred_list = [self.model.decoder(q_k)]
            fwd_state_list = (
                [self.model.decode_state(q_k, p_k)] if do_state else []
            )
            q, p = q_k, p_k
            for i in range(n_fwd):
                q, p = self.model.controlled_step(
                    q, p, actions2[:, k + i], dt=cfg.dt
                )
                fwd_pred_list.append(self.model.decoder(q))
                if do_state:
                    fwd_state_list.append(self.model.decode_state(q, p))

            fwd_pred = torch.stack(fwd_pred_list, dim=1)
            fwd_target = frames2[:, k : cfg.seq_len]
            t_fwd = torch.arange(n_fwd + 1, device=self.device)
            fwd_valid = (k + t_fwd < lengths2.unsqueeze(1)).float()
            denom_fwd = fwd_valid.sum(dim=1).clamp(min=1)
            anchor_recon = (
                F.mse_loss(fwd_pred, fwd_target, reduction="none").mean(
                    dim=[2, 3, 4]
                )
                * fwd_valid
            ).sum(dim=1) / denom_fwd

            if do_state:
                fwd_st_pred = torch.stack(fwd_state_list, dim=1)
                anchor_state = (
                    F.mse_loss(
                        fwd_st_pred,
                        states2[:, k : cfg.seq_len],
                        reduction="none",
                    ).mean(dim=2)
                    * fwd_valid
                ).sum(dim=1) / denom_fwd
            else:
                anchor_state = torch.zeros(B2, device=self.device)
        else:
            anchor_recon = torch.zeros(B2, device=self.device)
            anchor_state = torch.zeros(B2, device=self.device)

        m_per = (
            cfg.recon_weight * m_recon
            + cfg.kl_weight * kl2_per
            + cfg.state_weight * m_state
            + cfg.fwd_weight
            * (
                cfg.recon_weight * anchor_recon
                + cfg.state_weight * anchor_state
            )
        )

        self.model.train()
        return {
            "e_loss": e_loss.item(),
            "e_state": e_state.mean().item(),
            "e_kl": e_kl.mean().item(),
            "m_loss": m_per.mean().item(),
            "m_recon": m_recon.mean().item(),
            "m_kl": kl2_per.mean().item(),
            "m_state": m_state.mean().item(),
            "m_anchor_recon": anchor_recon.mean().item(),
            "m_anchor_state": anchor_state.mean().item(),
        }
