"""Shared Streamlit-app helpers for the pendulum world-model dashboards.

Checkpoint picking (``pick_checkpoint``) and pixel-rollout GIF assembly
(``build_sidebyside_frames`` / ``frames_to_gif``) are used by both
``pendulum_dreamer.py`` and ``pendulum_planner.py``. Kept out of either app
file since importing a Streamlit script module runs its top-level UI code as
a side effect.
"""

from __future__ import annotations

import io
import re
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageDraw

# ── Checkpoint picking ──────────────────────────────────────────────────────

_NON_CHECKPOINT_STEMS = {"h_cache", "episodes_cache"}
_TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})$")


def list_checkpoint_tree(
    models_root: Path,
) -> tuple[dict[str, dict[date, dict[str, list[Path]]]], list[Path]]:
    """Walk ``models_root`` into a nested identifier→date→time→files tree.

    Expected layout: ``models/<identifier>/<YYYY-MM-DD>_<HH-MM-SS>/<stem>.pt``.
    Returns ``(tree, unstructured)`` where ``unstructured`` holds any ``.pt``
    files that don't match that layout.
    """
    pt_files = sorted(
        f for f in models_root.rglob("*.pt") if f.stem not in _NON_CHECKPOINT_STEMS
    ) if models_root.exists() else []

    tree: dict[str, dict[date, dict[str, list[Path]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    unstructured: list[Path] = []

    for f in pt_files:
        rel = f.relative_to(models_root)
        parts = rel.parts
        if len(parts) >= 3:
            *id_parts, ts_part, _ = parts
            m = _TIMESTAMP_RE.match(ts_part)
            if m:
                identifier = "/".join(id_parts)
                run_date = datetime.strptime(m.group(1), "%Y-%m-%d").date()
                run_time = m.group(2)
                tree[identifier][run_date][run_time].append(f)
                continue
        unstructured.append(f)

    return tree, unstructured


def pick_checkpoint(models_root: Path, label: str, key_prefix: str) -> Path | None:
    """Render a nested identifier → date → time → checkpoint sidebar selector.

    Returns the chosen checkpoint path, or None (after showing a warning and
    calling ``st.stop()``) if no checkpoints are found under ``models_root``.
    """
    tree, _unstructured = list_checkpoint_tree(models_root)

    if not tree:
        st.warning(f"No `.pt` checkpoints found under `{models_root}/` (excluding data caches).")
        st.stop()

    identifiers = sorted(tree.keys())
    identifier = st.selectbox(f"{label} — model", identifiers, key=f"{key_prefix}_id")
    dates = sorted(tree[identifier].keys(), reverse=True)
    chosen_date = st.selectbox(
        f"{label} — date",
        dates,
        format_func=lambda d: d.strftime("%A, %B %-d %Y"),
        key=f"{key_prefix}_date",
    )
    times = sorted(tree[identifier][chosen_date].keys(), reverse=True)
    chosen_time = st.selectbox(
        f"{label} — run",
        times,
        format_func=lambda t: t.replace("-", ":"),
        key=f"{key_prefix}_time",
    )
    files = tree[identifier][chosen_date][chosen_time]
    file_names = [f.name for f in files]
    chosen_name = st.selectbox(
        f"{label} — checkpoint",
        file_names,
        index=len(file_names) - 1,  # default to last (highest epoch)
        key=f"{key_prefix}_file",
    )
    return models_root / identifier / f"{chosen_date.strftime('%Y-%m-%d')}_{chosen_time}" / chosen_name


# ── Frame / GIF helpers ─────────────────────────────────────────────────────


def to_uint8(frames: torch.Tensor) -> list[np.ndarray]:
    """(N, C, H, W) float [0,1] → list of (H, W, C) uint8 arrays."""
    return [
        (f.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        for f in frames
    ]


def label_frame(img: Image.Image, text: str, color: tuple) -> Image.Image:
    img = img.copy()
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (img.width - 1, 14)], fill=(0, 0, 0, 180))
    draw.text((3, 2), text, fill=color)
    return img


def build_sidebyside_frames(
    left_frames: list[np.ndarray],
    right_frames: list[np.ndarray],
    display_size: int,
    left_label: str = "GT",
    right_label: str = "Model",
    left_color: tuple = (100, 200, 255),
    right_color: tuple = (255, 160, 60),
    gap: int = 4,
) -> list[Image.Image]:
    """Combine two matching frame sequences (left/right) into one wide frame each."""
    out: list[Image.Image] = []
    total_w = display_size * 2 + gap
    for i, (l_arr, r_arr) in enumerate(zip(left_frames, right_frames)):
        l_pil = Image.fromarray(l_arr).resize((display_size, display_size), Image.BILINEAR)
        r_pil = Image.fromarray(r_arr).resize((display_size, display_size), Image.BILINEAR)
        l_pil = label_frame(l_pil, f"{left_label}  t={i}", color=left_color)
        r_pil = label_frame(r_pil, f"{right_label}  t={i}", color=right_color)
        canvas = Image.new("RGB", (total_w, display_size), (40, 40, 40))
        canvas.paste(l_pil, (0, 0))
        canvas.paste(r_pil, (display_size + gap, 0))
        out.append(canvas)
    return out


def frames_to_gif(frames: list[Image.Image], fps: float) -> bytes:
    duration_ms = max(20, int(1000 / fps))
    buf = io.BytesIO()
    frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return buf.getvalue()
