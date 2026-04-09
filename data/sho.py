"""Synthetic Simple Harmonic Oscillator dataset generation."""

import numpy as np
import torch
from tqdm import tqdm


def render_frame(q_val, img_size, sigma, max_amp, margin=4):
	"""Render a Gaussian blob at horizontal position proportional to q."""
	x_center = (q_val / max_amp) * (img_size / 2 - margin) + img_size / 2
	y_center = float(img_size // 2)
	ys = torch.arange(img_size).float()
	xs = torch.arange(img_size).float()
	yy, xx = torch.meshgrid(ys, xs, indexing="ij")
	img = torch.exp(
		-((xx - x_center) ** 2 + (yy - y_center) ** 2) / (2 * sigma**2)
	)
	return img.unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)


def generate_dataset(
	n,
	seq_len,
	dt,
	img_size,
	blob_sigma,
	max_amplitude,
	spring_constant=1.0,
	mass=1.0,
	margin=4,
):
	"""Generate N synthetic SHO trajectories rendered as image sequences.

	Args:
	    margin: pixel margin used in render_frame; must match any coordinate
	            conversion done at inference time.
	"""
	omega = np.sqrt(spring_constant / mass)
	all_p, all_q, all_frames = [], [], []
	for _ in tqdm(range(n), desc="Datasets"):
		amplitude = torch.rand(1).item() * (max_amplitude - 0.5) + 0.5
		phase = torch.rand(1).item() * 2 * np.pi
		ts = torch.arange(seq_len).float() * dt
		q = amplitude * torch.cos(omega * ts + phase)
		p = -mass * omega * amplitude * torch.sin(omega * ts + phase)
		frames = torch.stack(
			[
				render_frame(
					q[i].item(), img_size, blob_sigma, max_amplitude, margin
				)
				for i in range(seq_len)
			]
		)
		all_p.append(p)
		all_q.append(q)
		all_frames.append(frames)
	return (
		torch.stack(all_p),  # (N, T)
		torch.stack(all_q),  # (N, T)
		torch.stack(all_frames),  # (N, T, 3, H, W)
	)


def generate_damped_dataset(
	n,
	seq_len,
	dt,
	img_size,
	blob_sigma,
	max_amplitude,
	spring_constant=1.0,
	mass=1.0,
	damping=0.5,
	margin=4,
):
	"""Generate N underdamped SHO image sequences with viscous damping γ.

	Analytical solution (underdamped regime, ζ < 1):
	    ω₀ = √(k/m)
	    ζ  = γ / (2√(km))          damping ratio
	    ωd = ω₀ √(1 − ζ²)         damped natural frequency
	    q(t) = A exp(−ζ ω₀ t) cos(ωd t + φ)
	    p(t) = m dq/dt

	Args:
	    damping: viscous damping coefficient γ

	Raises:
	    AssertionError if the system is overdamped (ζ ≥ 1).
	"""
	omega_0 = np.sqrt(spring_constant / mass)
	zeta = damping / (2.0 * np.sqrt(spring_constant * mass))
	assert zeta < 1.0, (
		f"System is overdamped (ζ={zeta:.3f} ≥ 1); "
		"reduce --damping or increase --spring-constant / --mass."
	)
	omega_d = omega_0 * np.sqrt(1.0 - zeta**2)

	all_p, all_q, all_frames = [], [], []
	for _ in tqdm(range(n), desc="Damped dataset", leave=False):
		amplitude = torch.rand(1).item() * (max_amplitude - 0.5) + 0.5
		phase = torch.rand(1).item() * 2.0 * np.pi
		t_np = np.arange(seq_len, dtype=np.float32) * dt

		decay = np.exp(-zeta * omega_0 * t_np)
		cos_part = np.cos(omega_d * t_np + phase)
		sin_part = np.sin(omega_d * t_np + phase)

		q_np = amplitude * decay * cos_part
		dq_dt = (
			amplitude
			* decay
			* (-zeta * omega_0 * cos_part - omega_d * sin_part)
		)
		p_np = mass * dq_dt

		q = torch.from_numpy(q_np)
		p = torch.from_numpy(p_np)
		frames = torch.stack(
			[
				render_frame(
					q[i].item(), img_size, blob_sigma, max_amplitude, margin
				)
				for i in range(seq_len)
			]
		)
		all_p.append(p)
		all_q.append(q)
		all_frames.append(frames)

	return (
		torch.stack(all_p),  # (N, T)
		torch.stack(all_q),  # (N, T)
		torch.stack(all_frames),  # (N, T, 3, H, W)
	)
