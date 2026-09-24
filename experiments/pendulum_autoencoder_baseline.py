"""Autoencoder-only baseline — no port-Hamiltonian structure at all.

This is a standalone comparison point against ``experiments/pendulum_offline.py``'s
Phase 2/3/joint models: it trains *only* the LSTM autoencoder (encoder + f_psi +
decoder + next_frame_decoder), identically to ``pendulum_offline.py phase1``, and
saves a ``WorldModel(autoencoder, dynamics=None)`` checkpoint — no
``HamiltonianFlowModel`` is ever constructed or trained.

Everything (data collection, architecture, training loop, ablation knobs,
logging/diagnostics, checkpoint format) is reused directly from
``pendulum_offline.py``'s Phase 1 implementation; this file only gives that
training regimen its own CLI entry point and run directory so it reads as a
deliberate baseline experiment rather than one subcommand nested in a
multi-phase file.

Since there is no dynamics model, "planning"/"dreaming" from a checkpoint
produced here can't roll forward through Hamiltonian phase space. Instead it
must use ``WorldModel.dream_autoencoder_only`` (``hamilton_rl/models.py``),
which iteratively predicts frame_{t+1} from (h_t, a_t) via the autoencoder's
own ``next_frame_decoder`` and re-encodes each predicted frame to get h_{t+1}
— i.e. dynamics-free rollout, purely from what the autoencoder already
learned during reconstruction training.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import click
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.pendulum import (
    PendulumMultiRolloutDataset,
    collect_random_trajectories,
    collect_seeded_random_rollouts,
    collect_spin_trajectories,
    collect_val_trajectories,
    _DRAG_COEFF,
)
from hamilton_rl.checkpoint import load_world_model, make_run_dir
from hamilton_rl.cli_config import config_option
from hamilton_rl.models import TemporalAutoencoder, WorldModel
from experiments.pendulum_offline import (
    _eval_loss_phase1,
    _log_cnn_feature_distribution_phase1,
    _log_cnn_feature_fold_probe_phase1,
    _log_cnn_feature_regression_phase1,
    _log_context_prediction_phase1,
    _log_h_state_regression_coeffs_phase1,
    _log_half_latent_probes_phase1,
    _log_hparams_table,
    _log_hparams_text,
    _log_latent_distribution_phase1,
    _log_latent_scatter_phase1,
    _log_markov_pairwise_probe_phase1,
    _log_reconstruction_lstm_video,
    _log_training_rollout,
    _plot_phase_space_coverage,
    _train_epoch_phase1,
)


@click.command()
@config_option
@click.option("--resume-from", type=str, default=None,
              help="Path to a checkpoint (.pt) whose autoencoder weights to warm-start "
                   "from; training still writes to a fresh run dir, and the optimizer "
                   "and epoch count both restart from scratch")
# data
@click.option("--n-windows", type=int, default=200, show_default=True,
              help="Random rollout windows sampled per training epoch")
@click.option("--img-size", type=int, default=64, show_default=True)
@click.option("--energy-k", type=float, default=1.0, show_default=True,
              help="Gain for energy-pumping controller (val episodes only)")
@click.option("--max-steps", type=int, default=200, show_default=True,
              help="Steps per training window (val episodes default to 2x this)")
@click.option("--n-samples", type=int, default=2000, show_default=True,
              help="Total env-step budget for training data collection, split "
                   "across many short random rollouts seeded across phase "
                   "space (see collect_seeded_random_rollouts); "
                   "n_seeds = n_samples // rollout_len")
@click.option("--rollout-len", type=int, default=0, show_default=True,
              help="Steps per seeded rollout (0 = 2x --max-steps); must be "
                   ">= --max-steps")
@click.option("--damping", type=float, default=0.0, show_default=True,
              help="Linear viscous damping coefficient")
@click.option("--drag", type=float, default=_DRAG_COEFF, show_default=True,
              help="Quadratic (Rayleigh) drag coefficient")
@click.option("--zero-action", is_flag=True, default=False, show_default=True,
              help="Diagnostic: collect training rollouts with a constant 0 "
                   "torque instead of uniform-random actions, degenerating "
                   "to the uncontrolled pendulum.")
# model architecture
@click.option("--pos-ch", type=int, default=8, show_default=True)
@click.option("--feat-dim", type=int, default=256, show_default=True)
@click.option("--latent-dim", type=int, default=32, show_default=True)
@click.option("--lstm-layers", type=int, default=1, show_default=True,
              help="Number of stacked LSTM layers in the encoder (lstm encoder only)")
@click.option("--encoder-type", type=click.Choice(["lstm", "framestack"]), default="lstm",
              show_default=True,
              help="How h_t is built from frames: 'lstm' (causal LSTM hidden state over "
                   "the whole context) or 'framestack' (memoryless function of the two "
                   "most recent frames)")
# training
@click.option("--epochs", type=int, default=3000, show_default=True)
@click.option("--batch-size", type=int, default=8, show_default=True)
@click.option("--lr", type=float, default=1e-4, show_default=True)
@click.option("--kl-weight", type=float, default=1e-3, show_default=True)
@click.option("--free-bits", type=float, default=0.5, show_default=True)
@click.option("--deterministic", is_flag=True, default=False, show_default=True,
              help="Ablation: skip VAE reparameterization/KL entirely and train h "
                   "as a plain deterministic autoencoder latent (kl-weight/free-bits "
                   "are ignored when set).")
@click.option("--grad-clip", type=float, default=1.0, show_default=True)
@click.option("--max-context-len", type=int, default=0, show_default=True,
              help="Max frames fed to LSTM per batch step (0 = full sequence). "
                   "Sampled uniformly from [2, max-context-len] each step.")
@click.option("--diag-context-frames", type=int, default=3, show_default=True,
              help="Context length (frames before 'frame t') for the "
                   "val/context_prediction diagnostic image.")
@click.option("--diag-n-samples", type=int, default=4, show_default=True,
              help="Number of sampled windows shown in the "
                   "val/context_prediction diagnostic image.")
@click.option("--temporal-reg-weight", type=float, default=0.1, show_default=True,
              help="Temporal metric regulariser weight (0 to disable)")
@click.option("--temporal-scale", type=float, default=0.01, show_default=True,
              help="Expected h-space distance per timestep")
@click.option("--sparsity-weight", type=float, default=0.0, show_default=True,
              help="L1 penalty on latent mean, pushes irrelevant dims to 0 (0 to disable)")
@click.option("--time-reversal-weight", type=float, default=0.0, show_default=True,
              help="Weight on the time-reversal q/p consistency term only")
@click.option("--mi-weight", type=float, default=0.0, show_default=True,
              help="Weight on an HSIC dependence penalty between the q-half "
                   "and p-half of the latent (0 to disable)")
@click.option("--use-gate", is_flag=True, default=False, show_default=True,
              help="Replace/augment L1 sparsity with a learned per-dim L0 "
                   "hard-concrete gate on the latent mean")
@click.option("--gate-weight", type=float, default=0.0, show_default=True,
              help="L0 penalty weight on the expected number of active gate "
                   "dims (0 to disable; requires --use-gate)")
@click.option("--gate-warmup-epochs", type=int, default=200, show_default=True,
              help="Epochs over which --gate-weight ramps linearly from 0 to its target value.")
@click.option("--gate-lr-mult", type=float, default=0.1, show_default=True,
              help="Multiplier on --lr for the gate's log_alpha parameters")
@click.option("--ema-alpha", type=float, default=0.99, show_default=True)
@click.option("--convergence-patience", type=int, default=0, show_default=True,
              help="Epochs of stable EMA before stopping; 0 disables")
@click.option("--convergence-threshold", type=float, default=1e-4, show_default=True)
# logging
@click.option("--log-every", type=int, default=5, show_default=True)
@click.option("--val-every", type=int, default=10, show_default=True,
              help="Epochs between validation plots (0 to disable)")
@click.option("--n-val-episodes", type=int, default=-1, show_default=True,
              help="Val episodes per type (-1 = n_windows // 2)")
@click.option("--val-max-steps", type=int, default=0, show_default=True,
              help="Steps per val episode (0 = 2x --max-steps)")
@click.option("--checkpoint-every", type=int, default=10, show_default=True)
def main(**kwargs):
    """Train the autoencoder-only baseline (no port-Hamiltonian dynamics).

    Identical training regimen to ``pendulum_offline.py phase1`` — see that
    file's module docstring and ``_train_epoch_phase1`` for the loss and data
    pipeline. This command exists as its own entry point so baseline runs
    land in their own run/model directories, distinct from Phase 1 runs
    staged toward a full port-Hamiltonian model.
    """
    assert kwargs["img_size"] % 8 == 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")

    writer = SummaryWriter(comment="_pendulum_autoencoder_baseline")
    run_dir = make_run_dir("pendulum_autoencoder_baseline")

    n_val_episodes = kwargs["n_val_episodes"]
    if n_val_episodes < 0:
        n_val_episodes = kwargs["n_windows"] // 2
    n_val = n_val_episodes if kwargs["val_every"] > 0 else 0
    val_steps = kwargs["val_max_steps"] or kwargs["max_steps"] * 2
    rollout_len = kwargs["rollout_len"] or kwargs["max_steps"] * 2

    n_seeds = max(1, kwargs["n_samples"] // rollout_len)
    print(f"\nCollecting {n_seeds} seeded random rollouts of {rollout_len} steps each...")
    rollouts = collect_seeded_random_rollouts(
        n_samples=kwargs["n_samples"],
        rollout_len=rollout_len,
        img_size=kwargs["img_size"],
        damping=kwargs["damping"],
        drag=kwargs["drag"],
        zero_action=kwargs["zero_action"],
    )

    rollout_cache_path = run_dir / "rollout_cache.pt"
    torch.save(rollouts, rollout_cache_path)
    print(f"Saved rollout cache ({len(rollouts)} rollouts x {rollout_len} steps) to {rollout_cache_path}")

    _log_training_rollout(rollouts, writer)

    val_energy, val_random, val_spin = [], [], []
    if n_val > 0:
        print(f"Collecting {n_val} val episodes per type ({val_steps} steps each)...")
        val_energy = collect_val_trajectories(
            n_episodes=n_val, img_size=kwargs["img_size"],
            max_steps=val_steps, energy_k=kwargs["energy_k"], damping=kwargs["damping"],
        )
        val_random = collect_random_trajectories(
            n_episodes=n_val, img_size=kwargs["img_size"],
            max_steps=val_steps, damping=kwargs["damping"],
        )
        val_spin = collect_spin_trajectories(
            n_episodes=n_val, img_size=kwargs["img_size"],
            max_steps=val_steps, damping=kwargs["damping"],
        )

    coverage_fig = _plot_phase_space_coverage(rollouts)
    writer.add_figure("data/phase_space_coverage", coverage_fig, 0)
    plt.close(coverage_fig)

    dataset = PendulumMultiRolloutDataset(
        rollouts, window_len=kwargs["max_steps"], n_windows=kwargs["n_windows"],
    )
    loader = DataLoader(
        dataset, batch_size=kwargs["batch_size"], shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    print(f"Dataset: {len(dataset)} windows/epoch of {dataset.window_len} steps")

    model = TemporalAutoencoder(
        latent_dim=kwargs["latent_dim"],
        feat_dim=kwargs["feat_dim"],
        pos_ch=kwargs["pos_ch"],
        img_size=kwargs["img_size"],
        control_dim=1,
        num_layers=kwargs["lstm_layers"],
        encoder_type=kwargs["encoder_type"],
        use_gate=kwargs["use_gate"],
    ).to(device)
    print(f"Autoencoder baseline model parameters: {sum(p.numel() for p in model.parameters()):,}")

    if kwargs["resume_from"]:
        print(f"Resuming autoencoder weights from {kwargs['resume_from']}...")
        resume_model = load_world_model(kwargs["resume_from"], device)
        model.load_state_dict(resume_model.autoencoder.state_dict())
        del resume_model

    if model.encoder.gate is not None:
        gate_params = list(model.encoder.gate.parameters())
        gate_param_ids = {id(p) for p in gate_params}
        other_params = [p for p in model.parameters() if id(p) not in gate_param_ids]
        optimizer = torch.optim.Adam([
            {"params": other_params, "lr": kwargs["lr"]},
            {"params": gate_params, "lr": kwargs["lr"] * kwargs["gate_lr_mult"]},
        ])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=kwargs["lr"])

    data_config = {k: kwargs[k] for k in (
        "n_windows", "n_samples", "img_size", "energy_k",
        "max_steps", "damping", "drag",
    )}
    data_config["rollout_len"] = rollout_len
    world_model = WorldModel(model, dynamics=None, data_config=data_config)

    hparams = {k: v for k, v in kwargs.items()}
    _log_hparams_text(writer, hparams)
    _log_hparams_table(writer, hparams, {})
    best_loss = float("inf")
    ema_loss = None
    converge_streak = 0

    print("\n=== Autoencoder baseline: reconstruction training (no dynamics) ===")
    for epoch in tqdm(range(kwargs["epochs"]), desc="AE baseline", dynamic_ncols=True):
        gate_warmup = kwargs["gate_warmup_epochs"]
        gate_weight_epoch = kwargs["gate_weight"] * (
            min(1.0, epoch / gate_warmup) if gate_warmup > 0 else 1.0
        )
        metrics = _train_epoch_phase1(
            model=model,
            loader=loader,
            optimizer=optimizer,
            kl_weight=kwargs["kl_weight"],
            free_bits=kwargs["free_bits"],
            grad_clip=kwargs["grad_clip"],
            device=device,
            temporal_reg_weight=kwargs["temporal_reg_weight"],
            temporal_scale=kwargs["temporal_scale"],
            sparsity_weight=kwargs["sparsity_weight"],
            gate_weight=gate_weight_epoch,
            max_context_len=kwargs["max_context_len"],
            deterministic=kwargs["deterministic"],
            time_reversal_weight=kwargs["time_reversal_weight"],
            mi_weight=kwargs["mi_weight"],
        )
        metrics["phase1/gate_weight_effective"] = gate_weight_epoch

        alpha = kwargs["ema_alpha"]
        prev_ema = ema_loss
        ema_loss = (
            metrics["phase1/loss"]
            if ema_loss is None
            else alpha * ema_loss + (1.0 - alpha) * metrics["phase1/loss"]
        )

        if prev_ema is not None and kwargs["convergence_patience"] > 0:
            rel_change = abs(ema_loss - prev_ema) / (abs(prev_ema) + 1e-8)
            if rel_change < kwargs["convergence_threshold"]:
                converge_streak += 1
                if converge_streak >= kwargs["convergence_patience"]:
                    tqdm.write(
                        f"  Converged at epoch {epoch + 1}"
                        f" (EMA Δ={rel_change:.2e}, streak={converge_streak})"
                    )
                    break
            else:
                converge_streak = 0

        if (epoch + 1) % kwargs["log_every"] == 0:
            for k, v in metrics.items():
                writer.add_scalar(k, v, epoch)
            writer.add_scalar("phase1/ema_loss", ema_loss, epoch)
            tqdm.write(
                f"  epoch {epoch + 1:4d}"
                f"  loss={metrics['phase1/loss']:.4f}"
                f"  ema={ema_loss:.4f}"
                f"  recon={metrics['phase1/recon']:.4f}"
                f"  next={metrics['phase1/recon_next']:.4f}"
                f"  kl={metrics['phase1/kl']:.4f}"
                f"  tc={metrics['phase1/temporal_reg']:.4f}"
            )

        if kwargs["val_every"] > 0 and (epoch + 1) % kwargs["val_every"] == 0:
            policy_val_trajs = (
                (val_energy, "energy_pump"),
                (val_random, "random"),
                (val_spin, "spin"),
            )
            for val_trajs, label in policy_val_trajs:
                if not val_trajs:
                    continue
                val_metrics = _eval_loss_phase1(model, val_trajs, device)
                for k, v in val_metrics.items():
                    writer.add_scalar(f"{k}/{label}", v, epoch)
                _log_reconstruction_lstm_video(
                    model=model, val_traj=val_trajs[0],
                    device=device, writer=writer, epoch=epoch,
                    tag=f"val/reconstruction_lstm/{label}",
                )
                if len(val_trajs) >= 2:
                    _log_h_state_regression_coeffs_phase1(
                        model=model, val_trajs=val_trajs,
                        device=device, writer=writer, epoch=epoch,
                        tag=f"val/h_state_regression_coeffs/{label}",
                    )
            scatter_sets = [(vt, label) for vt, label in policy_val_trajs if len(vt) >= 2]
            if scatter_sets:
                _log_latent_scatter_phase1(
                    model=model, val_traj_sets=scatter_sets,
                    device=device, writer=writer, epoch=epoch,
                )
                _log_half_latent_probes_phase1(
                    model=model, val_traj_sets=scatter_sets,
                    device=device, writer=writer, epoch=epoch,
                )
                _log_latent_distribution_phase1(
                    model=model, val_traj_sets=scatter_sets,
                    device=device, writer=writer, epoch=epoch,
                )
                _log_cnn_feature_distribution_phase1(
                    model=model, val_traj_sets=scatter_sets,
                    device=device, writer=writer, epoch=epoch,
                )
                _log_cnn_feature_regression_phase1(
                    model=model, val_traj_sets=scatter_sets,
                    device=device, writer=writer, epoch=epoch,
                )
                _log_cnn_feature_fold_probe_phase1(
                    model=model, val_traj_sets=scatter_sets,
                    device=device, writer=writer, epoch=epoch,
                )
            _log_markov_pairwise_probe_phase1(
                model=model, val_traj_sets=policy_val_trajs,
                device=device, writer=writer, epoch=epoch,
            )
            _log_reconstruction_lstm_video(
                model=model, val_traj=dataset[0],
                device=device, writer=writer, epoch=epoch,
                tag="train/reconstruction_lstm",
            )
            _log_context_prediction_phase1(
                model=model, dataset=dataset,
                device=device, writer=writer, epoch=epoch,
                context_frames=kwargs["diag_context_frames"],
                n_samples=kwargs["diag_n_samples"],
                tag="train/context_prediction",
            )

        if (
            kwargs["checkpoint_every"] > 0
            and (epoch + 1) % kwargs["checkpoint_every"] == 0
            and metrics["phase1/loss"] < best_loss
        ):
            world_model.save(run_dir, "best", hparams, metrics, epoch)
            best_loss = metrics["phase1/loss"]

    world_model.save(run_dir, "final", hparams, metrics, epoch)

    print(
        f"\nBaseline checkpoint saved to {run_dir} (dynamics=None). "
        "Load it with WorldModel.dream_autoencoder_only(...) for dynamics-free "
        "rollout/planning comparisons."
    )

    writer.close()
    print("\nDone. Run: tensorboard --logdir runs")
    os._exit(0)


if __name__ == "__main__":
    main()
