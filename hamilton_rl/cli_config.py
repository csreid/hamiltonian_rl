"""Shared ``--config`` option for experiment CLIs.

Adds a YAML-config-file option to a click command that behaves like a bulk
``default=`` override: values loaded from the file populate ``ctx.default_map``
before click resolves option values, so an explicit CLI flag still wins over
the file, which still wins over the option's own hardcoded default. Because
commands take ``**kwargs`` and the click options are the single source of
truth for what's valid, a YAML file automatically keeps pace with new/renamed
flags — the only check needed is rejecting keys that *don't* match any option,
so a typo'd key fails loudly instead of being silently ignored.
"""

from __future__ import annotations

import click
import yaml


def _load_config(ctx: click.Context, param: click.Parameter, value: str | None) -> None:
    if value is None:
        return
    with open(value) as f:
        data = yaml.safe_load(f) or {}

    valid = {p.name for p in ctx.command.params}
    unknown = sorted(set(data) - valid)
    if unknown:
        raise click.BadParameter(
            f"unknown option(s) in config file: {', '.join(unknown)}",
            ctx=ctx,
            param_hint="'--config'",
        )

    ctx.default_map = {**(ctx.default_map or {}), **data}


def config_option(f):
    """Decorator adding ``--config PATH`` (YAML) as defaults for all other options.

    Apply directly below the ``@click.command``/``@cli.command`` decorator, so
    it's evaluated eagerly before other options are resolved.
    """
    return click.option(
        "--config",
        type=click.Path(exists=True, dir_okay=False),
        default=None,
        is_eager=True,
        expose_value=False,
        callback=_load_config,
        help="YAML file of option defaults (CLI flags still override).",
    )(f)
