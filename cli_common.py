"""Shared CLI options for all HGN benchmark scripts.

Usage:
    from cli_common import shared_options

    @click.command()
    @shared_options
    @click.option('--script-specific', ...)
    def main(seq_len, dt, ..., script_specific):
        ...

All defaults match the paper (Toth et al., ICLR 2020) mass-spring system.
Scripts may accept but ignore options that don't apply to them.
"""

import click
import yaml


def load_config(ctx, param, value):
	if value:
		with open(value) as f:
			ctx.default_map = yaml.safe_load(f)
	return value


def shared_options(func):
	"""Decorator: adds all common CLI options to a click command."""
	options = [
		click.option(
			"--config",
			default=None,
			is_eager=True,
			expose_value=False,
			callback=load_config,
			type=click.Path(exists=True),
			help="Path to YAML config file.",
		),
		# ---- sequence / physics ----
		click.option(
			"--seq-len",
			type=int,
			default=31,
			show_default=True,
			help="Total frames per trajectory.",
		),
		click.option(
			"--dt",
			type=float,
			default=0.125,
			show_default=True,
			help="Leapfrog step size (paper: 0.125).",
		),
		click.option(
			"--train-rollout",
			type=int,
			default=30,
			show_default=True,
			help="Steps to roll out during training.",
		),
		click.option(
			"--spring-constant", type=float, default=2.0, show_default=True
		),
		click.option("--mass", type=float, default=0.5, show_default=True),
		click.option(
			"--max-amplitude", type=float, default=1.0, show_default=True
		),
		# ---- dataset ----
		click.option("--n-train", type=int, default=50000, show_default=True),
		click.option("--n-val", type=int, default=10000, show_default=True),
		# ---- training ----
		click.option("--batch-size", type=int, default=16, show_default=True),
		click.option("--n-epochs", type=int, default=5, show_default=True),
		click.option("--lr", type=float, default=1.5e-4, show_default=True),
		click.option(
			"--kl-weight", type=float, default=1e-3, show_default=True
		),
		click.option("--free-bits", type=float, default=1.0, show_default=True),
		click.option(
			"--grad-clip",
			type=float,
			default=1.0,
			show_default=True,
			help="Max gradient norm; 0 disables.",
		),
		click.option("--log-every", type=int, default=10, show_default=True),
		# ---- model ----
		click.option(
			"--n-frames",
			type=int,
			default=31,
			show_default=True,
			help="Context frames fed to the encoder (paper: set equal to seq-len).",
		),
		click.option(
			"--recon-weight", type=float, default=1.0, show_default=True
		),
		click.option(
			"--diag-every",
			type=int,
			default=1,
			show_default=True,
			help="Log diagnostics every N steps. -1 disables all diagnostics.",
		),
		# ---- rendering (image-based scripts only) ----
		click.option("--img-size", type=int, default=32, show_default=True),
		click.option(
			"--blob-sigma", type=float, default=2.0, show_default=True
		),
	]
	for opt in reversed(options):
		func = opt(func)
	return func
