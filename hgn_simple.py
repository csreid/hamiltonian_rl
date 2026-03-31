"""
Simplified HGN on scalar SHO data (x-coordinate only).

Architecture (per Toth et al. 2020):
  - Inference net: GRU over scalar x sequence -> VAE posterior z
  - f_psi: z -> (p, q) in small flat phase space
  - Hamiltonian: MLP(p, q) -> scalar energy
  - Decoder: linear(q) -> scalar x  (momentum excluded, per paper)
  - Rollout: leapfrog integration

Intentionally tiny: phase_size=4, hidden=8.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image, ImageDraw
import click
from cli_common import shared_options


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class HGNSimple(nn.Module):
	def __init__(self, phase_size: int = 4, hidden: int = 8):
		super().__init__()
		self.phase_size = phase_size

		# Inference: GRU over scalar observations -> posterior over z
		self.gru = nn.GRU(1, hidden, batch_first=True)
		self.vae_head = nn.Linear(hidden, phase_size * 2)  # -> mu, logvar

		# f_psi: maps sampled z to initial (p, q)
		self.f_psi = nn.Sequential(
			nn.Linear(phase_size, hidden),
			nn.Tanh(),
			nn.Linear(hidden, phase_size * 2),
		)

		# Hamiltonian: H(p, q) -> scalar
		self.hamiltonian = nn.Sequential(
			nn.Linear(phase_size * 2, hidden),
			nn.Softplus(),
			nn.Linear(hidden, hidden),
			nn.Softplus(),
			nn.Linear(hidden, 1),
		)

		# Decoder: q -> scalar x  (paper: decoder uses q only, not p)
		self.decoder = nn.Linear(phase_size, 1)

	# ------------------------------------------------------------------
	def encode(self, x_seq):
		"""
		x_seq: (batch, T, 1)
		Returns z (batch, T, phase_size), kl (batch, T)
		"""
		h, _ = self.gru(x_seq)  # (batch, T, hidden)
		out = self.vae_head(h)  # (batch, T, 2*phase_size)
		mu, logvar = out.chunk(2, dim=-1)
		eps = torch.randn_like(mu)
		z = mu + eps * (0.5 * logvar).exp()
		kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1)
		return z, kl

	def H(self, p, q):
		"""Scalar Hamiltonian. (..., phase_size) -> (...)"""
		return self.hamiltonian(torch.cat([p, q], dim=-1)).squeeze(-1)

	def decode(self, q):
		"""q (..., phase_size) -> scalar x (..., 1)"""
		return self.decoder(q)

	@torch.enable_grad()
	def leapfrog_step(self, p, q, dt):
		"""Störmer-Verlet leapfrog step."""
		# half-step p
		q_ = q.detach().requires_grad_(True)
		p_ = p.detach().requires_grad_(True)
		dH_dq = torch.autograd.grad(
			self.H(p_, q_).sum(), q_, create_graph=self.training
		)[0]
		p_half = p - 0.5 * dt * dH_dq

		# full-step q
		q_ = q.detach().requires_grad_(True)
		ph_ = p_half.detach().requires_grad_(True)
		dH_dp = torch.autograd.grad(
			self.H(ph_, q_).sum(), ph_, create_graph=self.training
		)[0]
		q_next = q + dt * dH_dp

		# half-step p
		qn_ = q_next.detach().requires_grad_(True)
		ph_ = p_half.detach().requires_grad_(True)
		dH_dq = torch.autograd.grad(
			self.H(ph_, qn_).sum(), qn_, create_graph=self.training
		)[0]
		p_next = p_half - 0.5 * dt * dH_dq

		return p_next, q_next

	def forward(self, x_seq):
		"""
		x_seq: (batch, T, 1)
		Returns p (batch, T, phase_size), q (batch, T, phase_size), kl (batch, T)
		"""
		z, kl = self.encode(x_seq)
		out = self.f_psi(z)
		p, q = out.chunk(2, dim=-1)
		return p, q, kl


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


def rollout(model, p0, q0, steps, dt):
	"""
	Roll out `steps` leapfrog steps from (p0, q0).
	Returns ps, qs each (batch, steps, phase_size).
	"""
	ps, qs = [p0], [q0]
	p, q = p0, q0
	for _ in range(steps - 1):
		p, q = model.leapfrog_step(p, q, dt)
		ps.append(p)
		qs.append(q)
	return torch.stack(ps, dim=1), torch.stack(qs, dim=1)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def generate_sho(n, seq_len, dt, spring_constant=1.0, mass=1.0, max_amp=2.0):
	"""
	Generate scalar SHO sequences (x only).
	Returns x (N, T), p_gt (N, T), q_gt (N, T).
	"""
	omega = np.sqrt(spring_constant / mass)
	xs, ps, qs = [], [], []
	for _ in range(n):
		amp = np.random.uniform(0.5, max_amp)
		phase = np.random.uniform(0, 2 * np.pi)
		ts = np.arange(seq_len) * dt
		q = amp * np.cos(omega * ts + phase)
		p = -mass * omega * amp * np.sin(omega * ts + phase)
		xs.append(q)  # observable = position = q
		qs.append(q)
		ps.append(p)
	to_t = lambda a: torch.tensor(np.array(a), dtype=torch.float32)
	return to_t(xs), to_t(ps), to_t(qs)


# ---------------------------------------------------------------------------
# Visual helpers
# ---------------------------------------------------------------------------

IMG_SIZE = 64
BLOB_SIGMA = 2.5


def render_frame(q_val, max_amp=2.0, img_size=IMG_SIZE, sigma=BLOB_SIGMA):
	"""Render a Gaussian blob at horizontal position proportional to q_val."""
	margin = 8
	x_center = (q_val / max_amp) * (img_size / 2 - margin) + img_size / 2
	y_center = img_size / 2
	xs = torch.arange(img_size).float()
	ys = torch.arange(img_size).float()
	yy, xx = torch.meshgrid(ys, xs, indexing="ij")
	img = torch.exp(
		-((xx - x_center) ** 2 + (yy - y_center) ** 2) / (2 * sigma**2)
	)
	return img.unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)


def draw_x(frames, positions, color, size=5, width=2):
	"""
	Draw X markers on frames at normalised (x, y) positions.
	frames:    (N, T, 3, H, W) float [0,1]
	positions: (N, T, 2)       float [0,1]
	"""
	N, T, C, H, W = frames.shape
	uint8 = (frames.clamp(0, 1) * 255).to(torch.uint8).numpy()
	import numpy as np_local

	out = np_local.empty_like(uint8)
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
			out[n, t] = np_local.array(img).transpose(2, 0, 1)
	return torch.from_numpy(out).float() / 255.0


def log_trajectory(writer, pred_x, true_x, epoch, n_init, dt, n_show=4):
	"""Predicted vs true x over time for n_show sequences."""
	n_show = min(n_show, pred_x.shape[0])
	T = pred_x.shape[1]
	t_axis = (np.arange(T) + n_init - 1) * dt

	fig, axes = plt.subplots(n_show, 1, figsize=(10, 2.5 * n_show), sharex=True)
	if n_show == 1:
		axes = [axes]
	for i, ax in enumerate(axes):
		ax.plot(t_axis, true_x[i].numpy(), label="True x", color="tab:blue")
		ax.plot(
			t_axis,
			pred_x[i].numpy(),
			label="Predicted x",
			linestyle="--",
			color="tab:orange",
		)
		ax.set_ylabel("x")
		if i == 0:
			ax.legend(fontsize=8)
	axes[-1].set_xlabel("Time")
	plt.suptitle(f"Trajectory  (epoch {epoch + 1})")
	plt.tight_layout()
	writer.add_figure("val/trajectory", fig, epoch)
	plt.close(fig)


def log_phase_portrait(writer, ps, qs, p_gt, q_gt, epoch, n_show=4):
	"""
	Overlay learned latent orbit (dim 0) against true (p, q) phase portrait.
	ps, qs: (N, T, phase_size)  on any device
	p_gt, q_gt: (N, T) cpu tensors
	"""
	n_show = min(n_show, ps.shape[0])
	fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 4))
	if n_show == 1:
		axes = [axes]
	ps_cpu = ps.cpu()
	qs_cpu = qs.cpu()
	for i, ax in enumerate(axes):
		ax.plot(
			q_gt[i].numpy(),
			p_gt[i].numpy(),
			label="True",
			color="tab:blue",
			alpha=0.7,
		)
		ax.plot(
			qs_cpu[i, :, 0].numpy(),
			ps_cpu[i, :, 0].numpy(),
			label="Latent[0]",
			linestyle="--",
			color="tab:orange",
			alpha=0.7,
		)
		ax.set_xlabel("q")
		ax.set_ylabel("p")
		if i == 0:
			ax.legend(fontsize=8)
		ax.set_title(f"Seq {i}")
	plt.suptitle(f"Phase portrait  (epoch {epoch + 1})")
	plt.tight_layout()
	writer.add_figure("val/phase_portrait", fig, epoch)
	plt.close(fig)


def log_video(writer, pred_x, true_x, epoch, max_amp=2.0, fps=4):
	"""
	Render Gaussian blob video.
	true position = green X, predicted position = red X.
	pred_x, true_x: (N, T) cpu float tensors
	"""
	N, T = pred_x.shape
	img_size = IMG_SIZE

	# Build ground-truth frames from true_x
	frames = torch.stack(
		[
			torch.stack(
				[render_frame(true_x[n, t].item(), max_amp) for t in range(T)]
			)
			for n in range(N)
		]
	)  # (N, T, 3, H, W)

	margin = 8
	# Normalised x positions for markers
	true_xn = true_x / max_amp * (img_size / 2 - margin) / img_size + 0.5
	pred_xn = pred_x / max_amp * (img_size / 2 - margin) / img_size + 0.5
	half = torch.full((N, T, 1), 0.5)

	true_pos = torch.cat([true_xn.unsqueeze(-1), half], dim=-1)  # (N, T, 2)
	pred_pos = torch.cat([pred_xn.unsqueeze(-1), half], dim=-1)

	frames = draw_x(frames, true_pos, color=(0, 255, 0))  # green = true
	frames = draw_x(frames, pred_pos, color=(255, 0, 0))  # red   = pred

	video = (frames.clamp(0, 1) * 255).to(torch.uint8)
	writer.add_video("val/reconstruction", video, epoch, fps=fps)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@click.command()
@shared_options
@click.option("--phase-size", type=int, default=4, show_default=True)
@click.option("--hidden", type=int, default=8, show_default=True)
@click.option(
	"--n-init",
	type=int,
	default=10,
	show_default=True,
	help="Encoder context frames before rollout.",
)
@click.option("--energy-weight", type=float, default=1e-2, show_default=True)
@click.option("--nondeg-weight", type=float, default=1e-3, show_default=True)
def main(
	phase_size,
	hidden,
	seq_len,
	n_init,
	train_rollout,
	dt,
	n_train,
	n_val,
	batch_size,
	n_epochs,
	lr,
	kl_weight,
	free_bits,
	energy_weight,
	nondeg_weight,
	grad_clip,
	log_every,
	# shared but unused by this script:
	img_size,
	blob_sigma,
	spring_constant,
	mass,
	max_amplitude,
):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}")

	# Data
	print("Generating data...")
	x_tr, p_tr, q_tr = generate_sho(n_train, seq_len, dt)
	x_val, p_val, q_val = generate_sho(n_val, seq_len, dt)
	train_loader = DataLoader(
		TensorDataset(x_tr, q_tr, p_tr),
		batch_size=batch_size,
		shuffle=True,
	)

	# Model
	model = HGNSimple(phase_size, hidden).to(device)
	n_params = sum(p.numel() for p in model.parameters())
	print(f"Parameters: {n_params}")

	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	writer = SummaryWriter(comment="_hgn_simple_sho")
	global_step = 0

	for epoch in (pbar := tqdm(range(n_epochs), desc="Epochs")):
		model.train()
		ep_loss = ep_recon = ep_kl = 0.0

		for x_batch, q_true, _ in tqdm(
			train_loader, desc=f"  e{epoch}", leave=False
		):
			x_batch = x_batch.to(device)  # (B, T)
			q_true = q_true.to(device)  # (B, T)

			x_in = x_batch.unsqueeze(-1)  # (B, T, 1) - scalar input

			# Encode context frames to get initial phase-space state
			p_enc, q_enc, kl = model(x_in)  # each (B, T, phase_size)
			p0 = p_enc[:, n_init - 1].requires_grad_(True)
			q0 = q_enc[:, n_init - 1].requires_grad_(True)

			kl_loss = kl[:, :n_init].clamp(min=free_bits).mean()

			# Leapfrog rollout
			ps, qs = rollout(model, p0, q0, steps=train_rollout, dt=dt)
			# ps, qs: (B, train_rollout, phase_size)

			# Reconstruction: decode q -> x, compare to true x
			pred_x = model.decode(qs).squeeze(-1)  # (B, train_rollout)
			target = q_true[:, n_init - 1 : n_init - 1 + train_rollout]
			recon_loss = F.mse_loss(pred_x, target)

			# Energy conservation: H should be constant along rollout
			H_traj = model.H(ps, qs)  # (B, train_rollout)
			energy_loss = H_traj.var(dim=1).mean()

			# Non-degeneracy: penalise flat Hamiltonian (kills leapfrog motion)
			p_nd = p0.detach().requires_grad_(True)
			q_nd = q0.detach().requires_grad_(True)
			H_nd = model.H(p_nd, q_nd).sum()
			dH_dp, dH_dq = torch.autograd.grad(
				H_nd, [p_nd, q_nd], create_graph=True
			)
			nondeg_loss = 1.0 / (
				dH_dp.pow(2).mean() + dH_dq.pow(2).mean() + 1e-6
			)

			loss = (
				recon_loss
				+ kl_weight * kl_loss
				+ energy_weight * energy_loss
				+ nondeg_weight * nondeg_loss
			)

			optimizer.zero_grad()
			loss.backward()
			if grad_clip > 0:
				nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
			optimizer.step()

			writer.add_scalar("train/loss", loss.item(), global_step)
			writer.add_scalar("train/recon", recon_loss.item(), global_step)
			writer.add_scalar("train/kl", kl_loss.item(), global_step)
			writer.add_scalar(
				"train/energy_cons", energy_loss.item(), global_step
			)
			writer.add_scalar("train/nondeg", nondeg_loss.item(), global_step)
			global_step += 1
			ep_loss += loss.item()
			ep_recon += recon_loss.item()
			ep_kl += kl_loss.item()

		n = len(train_loader)
		pbar.set_postfix(
			loss=f"{ep_loss / n:.4f}",
			recon=f"{ep_recon / n:.4f}",
			kl=f"{ep_kl / n:.4f}",
		)

		# Validation
		if (epoch + 1) % log_every == 0:
			model.eval()
			x_v = x_val.unsqueeze(-1).to(device)  # (N_val, T, 1)
			q_v = q_val.to(device)
			p_v = p_val.to(device)
			val_rollout_len = seq_len - n_init + 1

			with torch.no_grad():
				p_enc_v, q_enc_v, _ = model(x_v)
				p0_v = p_enc_v[:, n_init - 1]
				q0_v = q_enc_v[:, n_init - 1]

			ps_v, qs_v = rollout(
				model, p0_v, q0_v, steps=val_rollout_len, dt=dt
			)
			pred_x_v = (
				model.decode(qs_v).squeeze(-1).detach().cpu()
			)  # (N_val, rollout_len)
			target_v = q_v[:, n_init - 1 :].cpu()

			val_mse = F.mse_loss(pred_x_v, target_v).item()
			writer.add_scalar("val/mse", val_mse, epoch)

			# Pearson r between predicted and true x
			p_flat = pred_x_v.reshape(-1).numpy()
			gt_flat = target_v.reshape(-1).numpy()
			r = float(np.corrcoef(p_flat, gt_flat)[0, 1])
			writer.add_scalar("val/corr_x", abs(r), epoch)

			# Latent space correlations
			ps_np = ps_v[:, :, 0].cpu().numpy().reshape(-1)
			qs_np = qs_v[:, :, 0].cpu().numpy().reshape(-1)
			gt_q = q_v[:, n_init - 1 :].cpu().numpy().reshape(-1)
			gt_p = p_v[:, n_init - 1 :].cpu().numpy().reshape(-1)
			rq = float(np.corrcoef(qs_np, gt_q)[0, 1])
			rp = float(np.corrcoef(ps_np, gt_p)[0, 1])
			writer.add_scalar("val/corr_q_latent", abs(rq), epoch)
			writer.add_scalar("val/corr_p_latent", abs(rp), epoch)

			# --- Visuals ---
			N_VIS = 4
			log_trajectory(
				writer,
				pred_x_v[:N_VIS],
				target_v[:N_VIS],
				epoch,
				n_init,
				dt,
			)
			log_phase_portrait(
				writer,
				ps_v[:N_VIS],
				qs_v[:N_VIS],
				p_v[:N_VIS, n_init - 1 :].cpu(),
				q_v[:N_VIS, n_init - 1 :].cpu(),
				epoch,
			)
			log_video(writer, pred_x_v[:N_VIS], target_v[:N_VIS], epoch)

			tqdm.write(
				f"  epoch {epoch + 1:3d}  val_mse={val_mse:.5f}"
				f"  r_x={r:.3f}  r_q={rq:.3f}  r_p={rp:.3f}"
			)

	writer.close()
	print("Done. Run: tensorboard --logdir runs")


if __name__ == "__main__":
	main()
