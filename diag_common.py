"""Shared diagnostic utilities for HGN benchmark scripts."""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from tqdm import tqdm


# ---- Activation monitoring ----


class ActivationMonitor:
    """Forward hooks tracking neuron saturation and dead units.

    Tracks:
      - Tanh:       fraction of outputs with |x| > 0.97  (gradient ≈ 0)
      - Sigmoid:    fraction of outputs with x < 0.03 or x > 0.97
      - ReLU:       fraction of dead units (output == 0)
      - LSTM/GRU:   hidden-state mean and std (collapse detection)
    """

    TANH_SAT_THRESH = 0.97
    SIGMOID_SAT_THRESH = 0.03
    RED_FLAG_SAT = 0.50
    RED_FLAG_DEAD = 0.50

    def __init__(self, model: nn.Module):
        self.stats: dict[str, float] = {}
        self._hooks = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Tanh):
                self._hooks.append(
                    module.register_forward_hook(self._tanh_hook(name))
                )
            elif isinstance(module, nn.Sigmoid):
                self._hooks.append(
                    module.register_forward_hook(self._sigmoid_hook(name))
                )
            elif isinstance(module, nn.ReLU):
                self._hooks.append(
                    module.register_forward_hook(self._relu_hook(name))
                )
            elif isinstance(module, (nn.LSTM, nn.GRU)):
                self._hooks.append(
                    module.register_forward_hook(self._rnn_hook(name))
                )

    def _tanh_hook(self, name):
        def hook(_, __, output):
            sat = (
                (output.detach().abs() > self.TANH_SAT_THRESH)
                .float()
                .mean()
                .item()
            )
            self.stats[f"act/tanh_sat/{name}"] = sat

        return hook

    def _sigmoid_hook(self, name):
        def hook(_, __, output):
            o = output.detach()
            sat = (
                (
                    (o < self.SIGMOID_SAT_THRESH)
                    | (o > 1 - self.SIGMOID_SAT_THRESH)
                )
                .float()
                .mean()
                .item()
            )
            self.stats[f"act/sigmoid_sat/{name}"] = sat

        return hook

    def _relu_hook(self, name):
        def hook(_, __, output):
            dead = (output.detach() == 0).float().mean().item()
            self.stats[f"act/relu_dead/{name}"] = dead

        return hook

    def _rnn_hook(self, name):
        # output[0] is the full sequence output (seq, batch, hidden) for LSTM and GRU.
        def hook(_, __, output):
            h = output[0].detach()
            self.stats[f"act/rnn_h_mean/{name}"] = h.mean().item()
            self.stats[f"act/rnn_h_std/{name}"] = h.std().item()

        return hook

    def log(self, writer, step):
        for key, val in self.stats.items():
            writer.add_scalar(f"diag/{key}", val, step)

    def check_flags(self, step):
        for key, val in self.stats.items():
            if "tanh_sat" in key and val > self.RED_FLAG_SAT:
                tqdm.write(
                    f"  [RED FLAG step {step}] Tanh saturation {val:.1%} in {key}"
                )
            if "sigmoid_sat" in key and val > self.RED_FLAG_SAT:
                tqdm.write(
                    f"  [RED FLAG step {step}] Sigmoid saturation {val:.1%} in {key}"
                )
            if "relu_dead" in key and val > self.RED_FLAG_DEAD:
                tqdm.write(
                    f"  [RED FLAG step {step}] Dead ReLUs {val:.1%} in {key}"
                )
            if "rnn_h_std" in key and val < 1e-4:
                tqdm.write(
                    f"  [RED FLAG step {step}] RNN hidden state collapsed "
                    f"(std={val:.2e}) in {key}"
                )

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---- Gradient / weight diagnostics ----


def log_gradient_stats(writer, model, step) -> float:
    """Log per-layer and total gradient norms. Flags vanishing / exploding."""
    total_sq = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            norm = param.grad.data.norm(2).item()
            writer.add_scalar(f"diag/grad/layer/{name}", norm, step)
            total_sq += norm**2
    total_norm = total_sq**0.5
    writer.add_scalar("diag/grad/total_norm", total_norm, step)
    if total_norm < 1e-6:
        tqdm.write(
            f"  [RED FLAG step {step}] Vanishing gradients: total norm = {total_norm:.2e}"
        )
    elif total_norm < 1e-4:
        tqdm.write(
            f"  [WARNING  step {step}] Very small gradients: total norm = {total_norm:.2e}"
        )
    if total_norm > 100.0:
        tqdm.write(
            f"  [RED FLAG step {step}] Exploding gradients: total norm = {total_norm:.2e}"
        )
    return total_norm


def log_weight_norms(writer, model, step):
    """Log L2 norm of every parameter tensor."""
    for name, param in model.named_parameters():
        writer.add_scalar(
            f"diag/weight/{name}", param.data.norm(2).item(), step
        )


def log_histograms(writer, model, step):
    """Log weight and gradient histograms (expensive — call infrequently)."""
    for name, param in model.named_parameters():
        if param.data.numel() == 0:
            continue
        writer.add_histogram(f"hist/weight/{name}", param.data, step)
        if param.grad is not None and param.grad.numel() > 0:
            writer.add_histogram(f"hist/grad/{name}", param.grad.data, step)


# ---- Latent / Hamiltonian diagnostics ----


def log_latent_stats(writer, latents: dict, step: int, free_bits: float):
    """Log VAE latent statistics for one or more named (mu, log_var) pairs.

    Args:
        latents:   {name: (mu, log_var)}, e.g. ``{"z": (mu, log_var)}``
                   or ``{"p": (mu_p, lv_p), "q": (mu_q, lv_q)}``.
        step:      global training step.
        free_bits: KL floor used for posterior-collapse detection.
    """
    kl_parts: list[torch.Tensor] = []
    lv_maxes: list[float] = []
    lv_mins: list[float] = []
    mu_stds: list[float] = []

    for name, (mu, log_var) in latents.items():
        writer.add_scalar(f"diag/latent/{name}/mu_mean", mu.mean().item(), step)
        writer.add_scalar(f"diag/latent/{name}/mu_std", mu.std().item(), step)
        writer.add_scalar(
            f"diag/latent/{name}/log_var_mean", log_var.mean().item(), step
        )
        writer.add_scalar(
            f"diag/latent/{name}/log_var_std", log_var.std().item(), step
        )
        kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
        kl_parts.append(kl.flatten())
        lv_maxes.append(log_var.max().item())
        lv_mins.append(log_var.min().item())
        mu_stds.append(mu.std().item())

    kl_all = torch.cat(kl_parts)
    collapse_frac = (kl_all < free_bits).float().mean().item()
    writer.add_scalar("diag/latent/kl_collapse_frac", collapse_frac, step)
    if collapse_frac > 0.9:
        tqdm.write(
            f"  [RED FLAG step {step}] Posterior collapse: "
            f"{collapse_frac:.1%} of latent dims at free-bits floor"
        )

    lv_max = max(lv_maxes)
    lv_min = min(lv_mins)
    writer.add_scalar("diag/latent/log_var_max", lv_max, step)
    writer.add_scalar("diag/latent/log_var_min", lv_min, step)
    if lv_max > 10:
        tqdm.write(
            f"  [RED FLAG step {step}] log_var very large ({lv_max:.2f})"
            " -> numerical instability"
        )
    if lv_min < -10:
        tqdm.write(
            f"  [RED FLAG step {step}] log_var very small ({lv_min:.2f})"
            " -> near-zero variance"
        )

    mu_std_max = max(mu_stds)
    if mu_std_max < 0.01:
        tqdm.write(
            f"  [RED FLAG step {step}] Encoder mu nearly constant"
            f" (std={mu_std_max:.2e}) -- encoder may be ignoring the latent"
        )


def log_hamiltonian_grad_stats(writer, H_fn, q0, p0, step):
    """Log dH/dq and dH/dp norms. Near-zero means no motion during rollout.

    Args:
        H_fn: callable ``H_fn(q, p) -> (B,)`` Tensor.  Both scripts should
              wrap their model's H method to match this (q, p) convention.
    """
    with torch.enable_grad():
        q_ = q0.detach().requires_grad_(True)
        p_ = p0.detach().requires_grad_(True)
        H_val = H_fn(q_, p_).sum()
        dH_dq = torch.autograd.grad(H_val, q_, retain_graph=True)[0]
        dH_dp = torch.autograd.grad(H_val, p_)[0]

    dHdq_norm = dH_dq.norm().item()
    dHdp_norm = dH_dp.norm().item()
    writer.add_scalar("diag/hamiltonian/dH_dq_norm", dHdq_norm, step)
    writer.add_scalar("diag/hamiltonian/dH_dp_norm", dHdp_norm, step)

    with torch.no_grad():
        H_vals = H_fn(q0, p0)
    writer.add_scalar("diag/hamiltonian/H_mean", H_vals.mean().item(), step)
    writer.add_scalar("diag/hamiltonian/H_std", H_vals.std().item(), step)

    if dHdq_norm < 1e-6 or dHdp_norm < 1e-6:
        tqdm.write(
            f"  [RED FLAG step {step}] Hamiltonian gradients near zero: "
            f"dH/dq={dHdq_norm:.2e}  dH/dp={dHdp_norm:.2e}"
            " -- rollout will produce no motion"
        )


# ---- Image utilities ----


def image_centroid(imgs):
    """Intensity-weighted centroid. imgs: (..., 3, H, W) -> (..., 2) as (x, y) in [0, 1]."""
    intensity = imgs[..., 0, :, :]  # (..., H, W)
    H, W = intensity.shape[-2:]
    xs = torch.linspace(0, 1, W, device=imgs.device)
    ys = torch.linspace(0, 1, H, device=imgs.device)
    total = intensity.sum(dim=(-2, -1)).clamp(min=1e-6)
    x_cent = (intensity * xs).sum(dim=(-2, -1)) / total
    y_cent = (intensity * ys.unsqueeze(-1)).sum(dim=(-2, -1)) / total
    return torch.stack([x_cent, y_cent], dim=-1)


def draw_x(frames, positions, color, size=6, width=2):
    """Draw an X marker on each frame at the given normalised (x, y) position.

    frames:    (N, T, 3, H, W) float [0, 1]
    positions: (N, T, 2)       float [0, 1]
    color:     (r, g, b) tuple in [0, 255]
    Returns a new float tensor with markers drawn.
    """
    N, T, C, H, W = frames.shape
    uint8 = (frames.clamp(0, 1) * 255).to(torch.uint8).numpy()
    out = np.empty_like(uint8)
    for n in range(N):
        for t in range(T):
            img = Image.fromarray(uint8[n, t].transpose(1, 2, 0))
            draw = ImageDraw.Draw(img)
            px = int(positions[n, t, 0].item() * W)
            py = int(positions[n, t, 1].item() * H)
            draw.line(
                [(px - size, py - size), (px + size, py + size)],
                fill=color,
                width=width,
            )
            draw.line(
                [(px + size, py - size), (px - size, py + size)],
                fill=color,
                width=width,
            )
            out[n, t] = np.array(img).transpose(2, 0, 1)
    return torch.from_numpy(out).float() / 255.0


# ---- Validation video helpers ----


def log_marker_video(
    writer, tag, frames, gt_pos, pred_pos, epoch, size=6, fps=4
):
    """Log a video with green X markers at GT positions and red X at predicted positions.

    Args:
        frames:   (N, T, 3, H, W) float [0, 1]
        gt_pos:   (N, T, 2) normalised (x, y) in [0, 1]
        pred_pos: (N, T, 2) normalised (x, y) in [0, 1]
    """
    video = draw_x(frames, gt_pos, color=(0, 255, 0), size=size)
    video = draw_x(video, pred_pos, color=(255, 0, 0), size=size)
    writer.add_video(
        tag, (video.clamp(0, 1) * 255).to(torch.uint8), epoch, fps=fps
    )


def log_gt_pred_video(writer, tag, gt_frames, pred_frames, epoch, fps=4):
    """Log a side-by-side GT | pred video separated by a 1-pixel white divider.

    Args:
        gt_frames:   (N, T, C, H, W) float [0, 1]
        pred_frames: (N, T, C, H, W) float [0, 1]
    """
    N, T, C, H, W = gt_frames.shape
    divider = torch.ones(N, T, C, 1, W, device=gt_frames.device)
    combined = torch.cat([gt_frames, divider, pred_frames], dim=3)
    writer.add_video(
        tag, (combined.clamp(0, 1) * 255).to(torch.uint8), epoch, fps=fps
    )


def log_hamiltonian_conservation(
    writer, H_fn, qs, ps, N_VID, epoch, context_len=None
):
    """Plot H(q, p) over the rollout and log to TensorBoard.

    Args:
        H_fn: callable ``H_fn(q, p) -> (B,)`` following the shared (q, p) convention.
        qs, ps: either a list of (B, ...) tensors (one per rollout step) or a single
                (B, T, ...) tensor.  Both conventions are accepted.
        context_len: if given, draw a vertical line marking the end of the training
                     context window (i.e. the model has seen steps 0..context_len-1
                     during training; steps beyond this are extrapolation).
    """
    import matplotlib.pyplot as plt

    with torch.no_grad():
        if isinstance(qs, torch.Tensor):
            T = qs.shape[1]
            H_traj = torch.stack(
                [H_fn(qs[:, t], ps[:, t]) for t in range(T)], dim=1
            ).cpu()
        else:
            H_traj = torch.stack(
                [H_fn(q, p) for q, p in zip(qs, ps)], dim=1
            ).cpu()

    N_plot = min(N_VID, H_traj.shape[0])
    fig, ax = plt.subplots(figsize=(8, 4))
    for i in range(N_plot):
        h_i = H_traj[i].numpy()
        ax.plot(h_i, label=f"sample {i}", alpha=0.8)
        ax.axhline(h_i[0], color=f"C{i}", linestyle="--", alpha=0.4)
    if context_len is not None:
        ax.axvline(
            context_len - 1,
            color="black",
            linestyle=":",
            linewidth=1.5,
            label=f"context end ({context_len})",
        )
    ax.set_xlabel("Rollout step")
    ax.set_ylabel("H(q, p)")
    ax.set_title(f"Hamiltonian conservation (epoch {epoch + 1})")
    ax.legend(fontsize=8)
    plt.tight_layout()
    writer.add_figure("val/hamiltonian_over_time", fig, epoch)
    plt.close(fig)


# ---- Dataset generation (re-exported from data.sho) ----

from data.sho import render_frame, generate_dataset  # noqa: F401, E402
