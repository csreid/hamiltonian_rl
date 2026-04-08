"""PHGN-LSTM online EM experiment on CartPole."""

from __future__ import annotations

import click
from tqdm import tqdm

from experiments.base import Experiment
from mppi import MPPI
from phgn_lstm import ControlledDHGN_LSTM
from prioritized_replay_buffer import PrioritizedEpisodeReplayBuffer
from training.em_trainer import CartPoleEMConfig, EMTrainer


class CartPoleOnlineExperiment(Experiment):
    def tb_comment(self) -> str:
        return "_phgn_lstm_online"

    def build_model(self, cfg: CartPoleEMConfig) -> ControlledDHGN_LSTM:
        return ControlledDHGN_LSTM(
            pos_ch=cfg.pos_ch,
            img_ch=3,
            dt=cfg.dt,
            feat_dim=cfg.feat_dim,
            img_size=cfg.img_size,
            control_dim=1,
            obs_state_dim=cfg.obs_state_dim,
            separable=cfg.separable,
        )

    def run(self, cfg: CartPoleEMConfig) -> None:
        import torch

        device, writer, run_dir = self._setup()

        model = self.build_model(cfg).to(device)
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        buffer = PrioritizedEpisodeReplayBuffer(
            capacity=cfg.buffer_capacity,
            min_seq_len=1,
            alpha=cfg.per_alpha,
            beta=cfg.per_beta,
            beta_annealing=cfg.per_beta_annealing,
        )

        planner = MPPI(
            model=model,
            horizon=cfg.mppi_horizon,
            n_samples=cfg.mppi_samples,
            temperature=cfg.mppi_temperature,
            noise_sigma=cfg.mppi_sigma,
            control_dim=1,
            control_min=-1.0,
            control_max=1.0,
            device=device,
        )

        model_hparams = {
            "img_size": cfg.img_size,
            "pos_ch": cfg.pos_ch,
            "feat_dim": cfg.feat_dim,
            "dt": cfg.dt,
            "seq_len": cfg.seq_len,
            "separable": cfg.separable,
            "obs_state_dim": cfg.obs_state_dim,
        }

        trainer = EMTrainer(
            cfg=cfg,
            model=model,
            writer=writer,
            run_dir=run_dir,
            device=device,
            buffer=buffer,
            planner=planner,
            cost_fn=None,
            model_hparams=model_hparams,
        )

        print(f"\nWarmup: collecting {cfg.n_warmup} random episodes...")
        for i in tqdm(range(cfg.n_warmup), desc="Warmup"):
            ep = trainer._collect_episode(use_mppi=False)
            buffer.push(ep)
            writer.add_scalar("collect/episode_len", len(ep), i)

        print(f"Buffer: {len(buffer)} episodes, {buffer.num_steps()} steps\n")
        trainer.fit()
        print("\nDone. Run: tensorboard --logdir runs")


@click.command()
# model
@click.option("--img-size", type=int, default=64, show_default=True)
@click.option("--pos-ch", type=int, default=8, show_default=True)
@click.option("--feat-dim", type=int, default=256, show_default=True)
@click.option("--seq-len", type=int, default=8, show_default=True)
@click.option("--dt", type=float, default=0.05, show_default=True)
@click.option("--no-separable", "separable", default=True, flag_value=False)
@click.option("--obs-state-dim", type=int, default=4, show_default=True)
# online loop
@click.option("--n-iterations", type=int, default=300, show_default=True)
@click.option("--collect-per-iter", type=int, default=5, show_default=True)
@click.option("--em-e-steps", type=int, default=25, show_default=True)
@click.option("--em-m-steps", type=int, default=25, show_default=True)
@click.option("--n-warmup", type=int, default=20, show_default=True)
@click.option("--min-buffer", type=int, default=20, show_default=True)
@click.option("--buffer-capacity", type=int, default=2000, show_default=True)
@click.option("--max-episode-steps", type=int, default=500, show_default=True)
# training
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option("--lr", type=float, default=1e-4, show_default=True)
@click.option("--lr-dynamics", type=float, default=1e-4, show_default=True)
@click.option("--kl-weight", type=float, default=1e-3, show_default=True)
@click.option("--recon-weight", type=float, default=1.0, show_default=True)
@click.option("--state-weight", type=float, default=0.5, show_default=True)
@click.option("--fwd-weight", type=float, default=0.5, show_default=True)
@click.option("--anchor-context", type=int, default=3, show_default=True)
@click.option("--free-bits", type=float, default=0.5, show_default=True)
@click.option("--grad-clip", type=float, default=1.0, show_default=True)
# PER
@click.option("--per-alpha", type=float, default=0.6, show_default=True)
@click.option("--per-beta", type=float, default=0.4, show_default=True)
@click.option(
    "--per-beta-annealing", type=float, default=0.0, show_default=True
)
# MPPI
@click.option("--mppi-horizon", type=int, default=20, show_default=True)
@click.option("--mppi-samples", type=int, default=256, show_default=True)
@click.option("--mppi-temperature", type=float, default=0.05, show_default=True)
@click.option("--mppi-sigma", type=float, default=0.5, show_default=True)
# eval / logging
@click.option("--eval-every", type=int, default=10, show_default=True)
@click.option("--n-eval-episodes", type=int, default=10, show_default=True)
@click.option("--log-every", type=int, default=1, show_default=True)
@click.option("--checkpoint-every", type=int, default=50, show_default=True)
def main(**kwargs):
    assert kwargs["seq_len"] >= 2
    assert kwargs["img_size"] % 8 == 0
    CartPoleOnlineExperiment().run(CartPoleEMConfig.from_kwargs(kwargs))


if __name__ == "__main__":
    main()
