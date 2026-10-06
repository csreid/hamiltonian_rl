"""Render ladder — how hard can the observation be before phase-space learning breaks?

Phase 1 of ``pendulum_offline.py`` (causal encoder + f_psi + decoders, with
the q-half of the latent feeding the current-frame decoder and the full latent
feeding the next-frame decoder) run on *synthetic* observations of the true
pendulum state (q, p) = (theta, theta_dot), instead of pixels. The training
loop is ``_train_epoch_phase1`` itself — only the model's input/output layers
(MLPs over vectors instead of CNNs over images) and the data differ.

Observation axes (see ``Observation``). Every render is *implied-p*: p is
not visible in a single observation, only through the change between frames
(the pixel pendulum's situation). Renders where p is directly visible were
removed — the current-frame decoder only sees z_q, so such a render forces p
into z_q and makes the q/p split meaningless.
  render        what the observation is a function of
    q_only      g(theta/pi) — a random map of the scalar angle
    xy          (sin theta, -cos theta) — the pendulum tip; periodic, 2 channels
    xy_lift     g(sin theta, -cos theta) — periodic *and* lifted to obs_dim
                channels by a random map; closest synthetic stand-in for pixels
  nonlinearity  alpha in [0, 1]: g = (1-alpha)*linear + alpha*random_MLP.
                0 = linear, 1 = fully nonlinear. Ignored by ``xy``.
  nl_gain       curvature of the random MLP (first-layer weight scale).
  obs_dim       number of observation channels (ignored by ``xy``).
  obs_noise     iid Gaussian noise std, in units of per-channel obs std.

Observations are standardised per channel (statistics fit on the training
rollouts), so reconstruction MSE of 1.0 = "predict the mean" on every rung.

Recovery is measured by probes on held-out rollouts, not by reconstruction
(which stays low long after phase-space structure has broken): linear and
MLP probes of z_q, z_p and full z onto q-features (cos theta, sin theta) and
p (theta_dot). A healthy split has high z_q->q and z_p->p and low z_q->p.
Probes skip the first ``--probe-skip`` timesteps, where a causal encoder has
no motion evidence yet.

Subcommands:
  train  one run (plain flags or --config YAML).
  sweep  a grid of renders x nonlinearities x --vary options x seeds, one
         summary CSV.
"""

from __future__ import annotations

import csv
import itertools
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import click
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.pendulum import (
    PendulumMultiRolloutDataset,
    _DRAG_COEFF,
    _MAX_SPEED,
    _grid_seeds,
    analytic_pendulum_step,
    angle_normalize,
)
from experiments.pendulum_offline import (
    _eval_loss_phase1,
    _train_epoch_phase1,
)
from hamilton_rl.checkpoint import make_run_dir
from hamilton_rl.cli_config import config_option
from hamilton_rl.models import NormalizingFlow

RENDERS = ("q_only", "xy", "xy_lift")


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


class _RandomMLP:
    """Fixed (never trained) random tanh MLP, in_dim -> out_dim.

    ``gain`` scales the first layer's weights, i.e. how many "wiggles" the
    map has across the input range — the curvature knob for the nonlinear
    renders.
    """

    def __init__(self, in_dim, out_dim, gen, gain, width=32):
        def w(i, o, scale=1.0):
            return torch.randn(i, o, generator=gen) * scale / i**0.5

        self.w1, self.b1 = w(in_dim, width, gain), torch.randn(width, generator=gen) * 0.5
        self.w2, self.b2 = w(width, width), torch.randn(width, generator=gen) * 0.5
        self.w3 = w(width, out_dim)

    def __call__(self, x):
        h = torch.tanh(x @ self.w1 + self.b1)
        h = torch.tanh(h @ self.w2 + self.b2)
        return h @ self.w3


class _Map:
    """alpha-blend of a random linear map and a random MLP, in_dim -> out_dim."""

    def __init__(self, in_dim, out_dim, alpha, gain, gen):
        self.alpha = alpha
        self.lin = torch.randn(in_dim, out_dim, generator=gen) / in_dim**0.5
        self.mlp = _RandomMLP(in_dim, out_dim, gen, gain) if alpha > 0 else None

    def __call__(self, x):
        out = (1 - self.alpha) * (x @ self.lin)
        if self.mlp is not None:
            out = out + self.alpha * self.mlp(x)
        return out


class Observation:
    """O(q, p) -> standardised observation vector (independent of p). See the module docstring."""

    def __init__(
        self,
        render: str,
        obs_dim: int = 8,
        nonlinearity: float = 0.0,
        nl_gain: float = 2.0,
        noise: float = 0.0,
        seed: int = 0,
    ):
        if render not in RENDERS:
            raise ValueError(f"render must be one of {RENDERS}, got {render!r}")
        if not 0.0 <= nonlinearity <= 1.0:
            raise ValueError("nonlinearity must be in [0, 1]")
        self.render, self.noise = render, noise
        gen = torch.Generator().manual_seed(seed)
        if render == "q_only":
            self.g = _Map(1, obs_dim, nonlinearity, nl_gain, gen)
        elif render == "xy_lift":
            self.g = _Map(2, obs_dim, nonlinearity, nl_gain, gen)
        self.obs_dim = 2 if render == "xy" else obs_dim
        self.mean = torch.zeros(self.obs_dim)
        self.std = torch.ones(self.obs_dim)

    def _raw(self, theta, theta_dot):
        # theta is wrapped to [-pi, pi): for q_only the seam at +-pi is part
        # of the problem. theta_dot is deliberately unused — p is implied.
        if self.render == "q_only":
            return self.g((theta / torch.pi).unsqueeze(-1))
        xy = torch.stack([torch.sin(theta), -torch.cos(theta)], dim=-1)
        return xy if self.render == "xy" else self.g(xy)

    @torch.no_grad()
    def fit(self, theta, theta_dot):
        raw = self._raw(theta, theta_dot).reshape(-1, self.obs_dim)
        self.mean, self.std = raw.mean(0), raw.std(0).clamp_min(1e-6)

    @torch.no_grad()
    def __call__(self, theta, theta_dot):
        obs = (self._raw(theta, theta_dot) - self.mean) / self.std
        if self.noise > 0:
            obs = obs + self.noise * torch.randn_like(obs)
        return obs


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def collect_state_rollouts(
    n_rollouts: int,
    rollout_len: int,
    damping: float,
    drag: float,
    zero_action: bool,
):
    """Random-action rollouts seeded on a covering (theta, theta_dot) grid.

    Same scheme as ``collect_seeded_random_rollouts`` but without rendering,
    and batched over rollouts. Returns theta (wrapped), theta_dot, actions:
    (N, T+1), (N, T+1), (N, T).
    """
    seeds = _grid_seeds(
        n_rollouts, min_theta_dot=-_MAX_SPEED, max_theta_dot=_MAX_SPEED
    )
    theta = torch.tensor(seeds[:, 0], dtype=torch.float64)
    theta_dot = torch.tensor(seeds[:, 1], dtype=torch.float64)
    if zero_action:
        actions = torch.zeros(n_rollouts, rollout_len, dtype=torch.float64)
    else:
        actions = torch.rand(n_rollouts, rollout_len, dtype=torch.float64) * 4 - 2
    thetas, theta_dots = [theta], [theta_dot]
    for t in range(rollout_len):
        theta, theta_dot = analytic_pendulum_step(
            theta, theta_dot, actions[:, t], damping=damping, drag=drag
        )
        thetas.append(theta)
        theta_dots.append(theta_dot)
    theta = angle_normalize(torch.stack(thetas, dim=1))
    return theta.float(), torch.stack(theta_dots, dim=1).float(), actions.float()


def build_rollouts(obs_fn, theta, theta_dot, actions):
    """-> list of (obs, actions, states) with states = (cos, sin, theta_dot)."""
    obs = obs_fn(theta, theta_dot)
    states = torch.stack(
        [torch.cos(theta), torch.sin(theta), theta_dot], dim=-1
    )
    return [(obs[i], actions[i], states[i]) for i in range(obs.shape[0])]


# ---------------------------------------------------------------------------
# Model: TemporalAutoencoder's interface over vectors instead of images
# ---------------------------------------------------------------------------


def _mlp(i, h, o, n_hidden=2):
    layers, d = [], i
    for _ in range(n_hidden):
        layers += [nn.Linear(d, h), nn.LeakyReLU()]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, o))


class _VecLSTMEncoder(nn.Module):
    def __init__(self, obs_dim, feat_dim, latent_dim, num_layers):
        super().__init__()
        self.embed = _mlp(obs_dim, feat_dim, feat_dim, 1)
        self.lstm = nn.LSTM(feat_dim, feat_dim, num_layers, batch_first=True)
        self.mu_head = nn.Linear(feat_dim, latent_dim)
        self.logvar_head = nn.Linear(feat_dim, latent_dim)
        self.gate = None

    def forward_all(self, x):
        out, _ = self.lstm(self.embed(x))
        return self.mu_head(out), self.logvar_head(out)


class _VecFrameStackEncoder(nn.Module):
    def __init__(self, obs_dim, feat_dim, latent_dim):
        super().__init__()
        self.embed = _mlp(obs_dim, feat_dim, feat_dim, 1)
        self.fuse = nn.Sequential(
            nn.Linear(2 * feat_dim, feat_dim),
            nn.LeakyReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.LeakyReLU(),
        )
        self.mu_head = nn.Linear(feat_dim, latent_dim)
        self.logvar_head = nn.Linear(feat_dim, latent_dim)
        self.gate = None

    def forward_all(self, x):
        f = self.embed(x)
        prev = torch.cat([f[:, :1], f[:, :-1]], dim=1)
        out = self.fuse(torch.cat([prev, f], dim=-1))
        return self.mu_head(out), self.logvar_head(out)


class _NextDecoder(nn.Module):
    def __init__(self, latent_dim, control_dim, feat_dim, obs_dim):
        super().__init__()
        self.net = _mlp(latent_dim + control_dim, feat_dim, obs_dim)

    def forward(self, h, a):
        if a.dim() == 1:
            a = a.unsqueeze(-1)
        return self.net(torch.cat([h, a], dim=-1))


class VecAutoencoder(nn.Module):
    """Same attribute surface as TemporalAutoencoder (what
    ``_train_epoch_phase1``/``_eval_loss_phase1`` touch): encoder.forward_all,
    f_psi, decoder, next_frame_decoder, latent_dim."""

    def __init__(
        self,
        obs_dim,
        latent_dim=32,
        feat_dim=128,
        control_dim=1,
        num_layers=1,
        encoder_type="lstm",
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.config = dict(
            obs_dim=obs_dim,
            latent_dim=latent_dim,
            feat_dim=feat_dim,
            control_dim=control_dim,
            num_layers=num_layers,
            encoder_type=encoder_type,
        )
        q_dim = latent_dim // 2
        if encoder_type == "lstm":
            self.encoder = _VecLSTMEncoder(obs_dim, feat_dim, latent_dim, num_layers)
        elif encoder_type == "framestack":
            self.encoder = _VecFrameStackEncoder(obs_dim, feat_dim, latent_dim)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type!r}")
        self.f_psi = NormalizingFlow(q_dim)
        self.decoder = _mlp(q_dim, feat_dim, obs_dim)
        self.next_frame_decoder = _NextDecoder(
            latent_dim, control_dim, feat_dim, obs_dim
        )

    @property
    def q_dim(self):
        return self.latent_dim // 2


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def _r2(pred, true):
    """Mean per-column R²."""
    ss_res = ((true - pred) ** 2).sum(0)
    ss_tot = ((true - true.mean(0)) ** 2).sum(0).clamp_min(1e-12)
    return float((1 - ss_res / ss_tot).mean())


def _lin_probe(xtr, ytr, xva, yva):
    def ones(x):
        return torch.cat([x, torch.ones(len(x), 1, dtype=x.dtype)], 1)

    w = torch.linalg.lstsq(ones(xtr), ytr, driver="gelsd").solution
    return _r2(ones(xva) @ w, yva)


def _mlp_probe(xtr, ytr, xva, yva, steps, device):
    mx, sx = xtr.mean(0), xtr.std(0).clamp_min(1e-6)
    my, sy = ytr.mean(0), ytr.std(0).clamp_min(1e-6)
    xtr = ((xtr - mx) / sx).float().to(device)
    xva = ((xva - mx) / sx).float().to(device)
    ytr_n = ((ytr - my) / sy).float().to(device)
    net = _mlp(xtr.shape[1], 64, ytr.shape[1]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(steps):
        opt.zero_grad()
        nn.functional.mse_loss(net(xtr), ytr_n).backward()
        opt.step()
    with torch.no_grad():
        pred = net(xva).cpu().double() * sy + my
    return _r2(pred, yva)


@torch.no_grad()
def _encode(model, rollouts, n_steps, skip, device):
    obs = torch.stack([r[0][: n_steps + 1] for r in rollouts]).to(device)
    states = torch.stack([r[2][: n_steps + 1] for r in rollouts])
    model.eval()
    mu, _ = model.encoder.forward_all(obs)
    mu = mu[:, skip:].cpu().double().reshape(-1, mu.shape[-1])
    states = states[:, skip:].double().reshape(-1, 3)
    return mu, states[:, :2], states[:, 2:]


def run_probes(
    model, train_rollouts, val_rollouts, n_steps, skip, device, mlp_steps
):
    """R² of {z_q, z_p, z} -> {q, p} on held-out rollouts.

    q-target = (cos theta, sin theta) (periodic-safe), p-target = theta_dot.
    mlp_steps = 0 skips the MLP probes.
    """
    z_tr, q_tr, p_tr = _encode(model, train_rollouts, n_steps, skip, device)
    z_va, q_va, p_va = _encode(model, val_rollouts, n_steps, skip, device)
    qd = model.q_dim
    inputs = {
        "zq": (z_tr[:, :qd], z_va[:, :qd]),
        "zp": (z_tr[:, qd:], z_va[:, qd:]),
        "z": (z_tr, z_va),
    }
    targets = {"q": (q_tr, q_va), "p": (p_tr, p_va)}
    out = {}
    for iname, (xtr, xva) in inputs.items():
        for tname, (ytr, yva) in targets.items():
            out[f"lin/{iname}->{tname}"] = _lin_probe(xtr, ytr, xva, yva)
            if mlp_steps > 0:
                out[f"mlp/{iname}->{tname}"] = _mlp_probe(
                    xtr, ytr, xva, yva, mlp_steps, device
                )
    for kind in ("lin", "mlp"):
        if f"{kind}/zq->q" in out:
            out[f"{kind}/split_score"] = min(
                out[f"{kind}/zq->q"], out[f"{kind}/zp->p"]
            )
    return out


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run_experiment(cfg: dict) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    alpha_tag = "" if cfg["render"] == "xy" else f"_a{cfg['nonlinearity']:g}"
    tag = f"{cfg['render']}{alpha_tag}{cfg.get('tag_suffix', '')}_s{cfg['seed']}"
    run_dir = make_run_dir(f"render_ladder/{tag}")
    writer = SummaryWriter(comment=f"_render_ladder_{tag}")
    print(f"\n=== {tag} -> {run_dir} ===")

    obs_fn = Observation(
        cfg["render"],
        obs_dim=cfg["obs_dim"],
        nonlinearity=cfg["nonlinearity"],
        nl_gain=cfg["nl_gain"],
        noise=cfg["obs_noise"],
        seed=cfg["render_seed"],
    )
    collect = dict(
        rollout_len=cfg["rollout_len"],
        damping=cfg["damping"],
        drag=cfg["drag"],
        zero_action=cfg["zero_action"],
    )
    tr_theta, tr_thd, tr_act = collect_state_rollouts(
        cfg["n_rollouts"], **collect
    )
    obs_fn.fit(tr_theta, tr_thd)  # standardise on train; reused for val
    train_rollouts = build_rollouts(obs_fn, tr_theta, tr_thd, tr_act)
    val_rollouts = build_rollouts(
        obs_fn, *collect_state_rollouts(cfg["n_val_rollouts"], **collect)
    )

    dataset = PendulumMultiRolloutDataset(
        train_rollouts,
        window_len=cfg["window_len"],
        n_windows=cfg["n_windows"],
    )
    loader = DataLoader(
        dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=0
    )
    model = VecAutoencoder(
        obs_dim=obs_fn.obs_dim,
        latent_dim=cfg["latent_dim"],
        feat_dim=cfg["feat_dim"],
        num_layers=cfg["lstm_layers"],
        encoder_type=cfg["encoder_type"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    print(
        f"obs_dim={obs_fn.obs_dim}  "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )

    probe_args = dict(
        n_steps=cfg["window_len"],
        skip=cfg["probe_skip"],
        device=device,
    )
    metrics = {}
    for epoch in tqdm(range(cfg["epochs"]), desc=tag, dynamic_ncols=True):
        metrics = _train_epoch_phase1(
            model=model,
            loader=loader,
            optimizer=optimizer,
            kl_weight=cfg["kl_weight"],
            free_bits=cfg["free_bits"],
            grad_clip=cfg["grad_clip"],
            device=device,
            temporal_reg_weight=cfg["temporal_reg_weight"],
            temporal_scale=cfg["temporal_scale"],
            max_context_len=cfg["max_context_len"],
            deterministic=cfg["deterministic"],
            time_reversal_weight=cfg["time_reversal_weight"],
            mi_weight=cfg["mi_weight"],
        )
        if (epoch + 1) % cfg["log_every"] == 0:
            for k, v in metrics.items():
                writer.add_scalar(k, v, epoch)
        if cfg["val_every"] > 0 and (epoch + 1) % cfg["val_every"] == 0:
            val = _eval_loss_phase1(model, val_rollouts, device)
            for k, v in val.items():
                writer.add_scalar(k, v, epoch)
            probes = run_probes(
                model, train_rollouts, val_rollouts, mlp_steps=0, **probe_args
            )
            for k, v in probes.items():
                writer.add_scalar(f"probe/{k}", v, epoch)
            tqdm.write(
                f"  epoch {epoch + 1:4d}"
                f"  recon={metrics['phase1/recon']:.4f}"
                f"  val={val['phase1/val_recon']:.4f}"
                f"  lin zq->q={probes['lin/zq->q']:.3f}"
                f"  zp->p={probes['lin/zp->p']:.3f}"
                f"  zq->p={probes['lin/zq->p']:.3f}"
            )

    results = {
        **_eval_loss_phase1(model, val_rollouts, device),
        **{
            f"train/{k}": v
            for k, v in metrics.items()
            if k in ("phase1/recon", "phase1/recon_next")
        },
        **run_probes(
            model,
            train_rollouts,
            val_rollouts,
            mlp_steps=cfg["mlp_probe_steps"],
            **probe_args,
        ),
    }
    for k, v in results.items():
        writer.add_scalar(f"final/{k}", v, cfg["epochs"])
    writer.close()

    torch.save(
        {"config": cfg, "model_config": model.config, "model": model.state_dict()},
        run_dir / "model.pt",
    )
    with open(run_dir / "results.json", "w") as f:
        json.dump({"config": cfg, "results": results}, f, indent=2)
    print(json.dumps(results, indent=2))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.group()
def cli():
    """Render-ladder experiment: phase-1 learning on synthetic observations."""


@cli.command("train")
@config_option
@click.option("--render", type=click.Choice(RENDERS), default="xy_lift", show_default=True)
@click.option("--nonlinearity", type=click.FloatRange(0, 1), default=1.0, show_default=True,
              help="Blend of random MLP into the render: 0 = linear, 1 = fully nonlinear.")
@click.option("--nl-gain", type=float, default=2.0, show_default=True,
              help="Curvature of the random MLP (first-layer weight scale).")
@click.option("--obs-dim", type=int, default=8, show_default=True,
              help="Observation channels (ignored by xy, which is 2).")
@click.option("--obs-noise", type=float, default=0.0, show_default=True,
              help="Observation noise std, in units of per-channel obs std.")
@click.option("--render-seed", type=int, default=0, show_default=True,
              help="Seed for the render's random weights.")
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--n-rollouts", type=int, default=200, show_default=True)
@click.option("--n-val-rollouts", type=int, default=50, show_default=True)
@click.option("--rollout-len", type=int, default=100, show_default=True)
@click.option("--window-len", type=int, default=50, show_default=True,
              help="Training window length; probes also encode this many steps.")
@click.option("--n-windows", type=int, default=200, show_default=True)
@click.option("--damping", type=float, default=0.0, show_default=True)
@click.option("--drag", type=float, default=_DRAG_COEFF, show_default=True)
@click.option("--zero-action", is_flag=True, default=False, show_default=True)
@click.option("--feat-dim", type=int, default=128, show_default=True)
@click.option("--latent-dim", type=int, default=32, show_default=True)
@click.option("--lstm-layers", type=int, default=1, show_default=True)
@click.option("--encoder-type", type=click.Choice(["lstm", "framestack"]), default="lstm", show_default=True)
@click.option("--epochs", type=int, default=1000, show_default=True)
@click.option("--batch-size", type=int, default=16, show_default=True)
@click.option("--lr", type=float, default=3e-4, show_default=True)
@click.option("--kl-weight", type=float, default=1e-3, show_default=True)
@click.option("--free-bits", type=float, default=0.5, show_default=True)
@click.option("--deterministic", is_flag=True, default=False, show_default=True)
@click.option("--grad-clip", type=float, default=1.0, show_default=True)
@click.option("--max-context-len", type=int, default=0, show_default=True)
@click.option("--temporal-reg-weight", type=float, default=0.1, show_default=True)
@click.option("--temporal-scale", type=float, default=0.01, show_default=True)
@click.option("--time-reversal-weight", type=float, default=0.0, show_default=True)
@click.option("--mi-weight", type=float, default=0.0, show_default=True)
@click.option("--log-every", type=int, default=5, show_default=True)
@click.option("--val-every", type=int, default=50, show_default=True,
              help="Val recon + linear probes every N epochs (0 = final only).")
@click.option("--probe-skip", type=int, default=2, show_default=True,
              help="Leading timesteps excluded from probes.")
@click.option("--mlp-probe-steps", type=int, default=1500, show_default=True,
              help="Adam steps per MLP probe, final eval only (0 = skip).")
def train_cmd(**kwargs):
    """One run."""
    run_experiment(kwargs)


@cli.command("sweep")
@click.option("--base-config", type=click.Path(exists=True, dir_okay=False), default=None,
              help="YAML of train options shared by every run.")
@click.option("--renders", type=str, default="q_only,xy_lift", show_default=True,
              help="Comma-separated renders.")
@click.option("--alphas", type=str, default="1", show_default=True,
              help="Comma-separated nonlinearities (xy ignores this).")
@click.option("--vary", "vary", multiple=True,
              help="KEY=v1,v2,... — cross any train option (e.g. nl_gain=2,5,10,20; "
                   "repeatable). Default: nl_gain=2,5,10,20.")
@click.option("--seeds", type=int, default=3, show_default=True,
              help="Seeds per cell; each seed also reseeds the render's weights.")
def sweep_cmd(base_config, renders, alphas, vary, seeds):
    """Grid of renders x nonlinearities x --vary options x seeds -> one summary CSV."""
    params = {p.name: p for p in train_cmd.params if p.name != "config"}
    base = {name: p.default for name, p in params.items()}
    if base_config:
        with open(base_config) as f:
            over = yaml.safe_load(f) or {}
        unknown = sorted(set(over) - set(base))
        if unknown:
            raise click.BadParameter(f"unknown option(s): {', '.join(unknown)}")
        base.update(over)

    vary_axes = {}
    for spec in vary or ("nl_gain=2,5,10,20",):
        key, _, vals = spec.partition("=")
        if key not in params or key in ("render", "nonlinearity", "seed", "render_seed"):
            raise click.BadParameter(f"cannot vary {key!r}", param_hint="'--vary'")
        vary_axes[key] = [params[key].type.convert(v, params[key], None) for v in vals.split(",")]

    # xy has no obs_dim / nonlinearity / nl_gain: collapse those axes so it runs once per remaining combo.
    xy_inert = ("nl_gain", "obs_dim")
    cells, seen = [], set()
    for render in (r.strip() for r in renders.split(",")):
        for alpha in ([0.0] if render == "xy" else [float(a) for a in alphas.split(",")]):
            for combo in itertools.product(*vary_axes.values()):
                over = dict(zip(vary_axes, combo))
                if render == "xy":
                    over = {k: base[k] if k in xy_inert else v for k, v in over.items()}
                cell = (render, alpha, tuple(over.items()))
                if cell not in seen:
                    seen.add(cell)
                    cells.append(cell)
    print(f"{len(cells)} cells x {seeds} seeds = {len(cells) * seeds} runs")

    rows = []
    for render, alpha, over in cells:
        suffix = "".join(f"_{k}{v:g}" if isinstance(v, (int, float)) else f"_{k}{v}" for k, v in over)
        for seed in range(seeds):
            cfg = {
                **base,
                **dict(over),
                "render": render,
                "nonlinearity": alpha,
                "seed": seed,
                "render_seed": seed,
                "tag_suffix": suffix,
            }
            results = run_experiment(cfg)
            rows.append({
                "render": render, "nonlinearity": alpha, **dict(over),
                "seed": seed, **results,
            })

    out = (
        Path("models/render_ladder")
        / f"sweep_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(k for r in rows for k in r))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    cols = [
        "phase1/val_recon", "mlp/z->q", "mlp/z->p",
        "mlp/zq->q", "mlp/zp->p", "mlp/zq->p", "mlp/split_score",
    ]
    names = ["recon", "z->q", "z->p", "zq->q", "zp->p", "zq->p", "split"]
    print(f"\nSweep summary (mean over {seeds} seeds) -> {out}")
    print(f"{'render':<8} {'a':>4} {'varied':<22} " + " ".join(f"{n:>7}" for n in names))
    for render, alpha, over in cells:
        sel = [
            r for r in rows
            if r["render"] == render and r["nonlinearity"] == alpha
            and all(r[k] == v for k, v in over)
        ]
        label = " ".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in over)
        vals = [np.mean([r[c] for r in sel]) for c in cols]
        print(f"{render:<8} {alpha:>4g} {label:<22} " + " ".join(f"{v:>7.3f}" for v in vals))


if __name__ == "__main__":
    cli()
