"""
diagnose_phgn.py  –  surgical diagnostics for ControlledDHGN_LSTM white-box failure

Run from the same directory as pendulum_offline.py:
    python diagnose_phgn.py

Each section is independent; read the PASS/FAIL lines and the notes after each block.
"""

from __future__ import annotations
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
import signal


def _hard_exit(sig=None, frame=None):
	print("\n[interrupted] hard exit to avoid CUDA context hang")
	os._exit(1)


signal.signal(signal.SIGINT, _hard_exit)
signal.signal(signal.SIGTERM, _hard_exit)


def _excepthook(exc_type, exc_value, exc_tb):
	import traceback

	traceback.print_exception(exc_type, exc_value, exc_tb)
	os._exit(1)


sys.excepthook = _excepthook

from data.pendulum import PendulumDataset, collect_data
from phgn_lstm import ControlledDHGN_LSTM

# ──────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 64
POS_CH = 8
FEAT_DIM = 256
DT = 0.05


def sep(title=""):
	print(f"\n{'=' * 60}")
	if title:
		print(f"  {title}")
		print(f"{'=' * 60}")


def ok(msg):
	print(f"  [PASS] {msg}")


def bad(msg):
	print(f"  [FAIL] {msg}")


def info(msg):
	print(f"  [INFO] {msg}")


# ──────────────────────────────────────────────────────────────────────────────
sep("0. COLLECT A FEW EPISODES")

episodes = collect_data(
	n_episodes=6, img_size=IMG_SIZE, epsilon=0.1, energy_k=1.0
)
dataset = PendulumDataset(episodes[:4])
val_ep = episodes[4]

frames_np, actions_np, states_np = val_ep
info(f"frames dtype={frames_np.dtype}  shape={frames_np.shape}")
info(
	f"frames min={frames_np.min():.3f}  max={frames_np.max():.3f}  mean={frames_np.mean():.3f}"
)

if frames_np.dtype != np.float32 and frames_np.max() > 1.5:
	bad("frames look like uint8 [0-255] — divide by 255 in your dataset!")
elif frames_np.min() < 0 or frames_np.max() > 1.01:
	bad(
		f"frames outside [0,1]: min={frames_np.min():.3f} max={frames_np.max():.3f}"
	)
else:
	ok("frames in [0,1] float32")

# fraction of the image that is background (near-white)
bg_frac = (frames_np > 0.9).float().mean().item()
info(f"Background fraction (pixels > 0.9): {bg_frac:.3f}")
if bg_frac > 0.95:
	bad(
		"Image is >95% background — MSE with a white prediction will be tiny (~{:.4f}). "
		"Consider a focal/foreground-weighted loss or a darker background.".format(
			(1 - bg_frac) ** 2
		)
	)
else:
	ok("Sufficient foreground content")

# save a sample frame
frames_t = torch.as_tensor(frames_np)
plt.imsave(
	"/tmp/diag_frame0.png", frames_t[0].permute(1, 2, 0).numpy().clip(0, 1)
)
info("Saved sample frame to /tmp/diag_frame0.png")

# ──────────────────────────────────────────────────────────────────────────────
sep("1. BUILD MODEL")

model = ControlledDHGN_LSTM(
	pos_ch=POS_CH,
	img_ch=3,
	dt=DT,
	feat_dim=FEAT_DIM,
	img_size=IMG_SIZE,
	control_dim=1,
	separable=True,
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
info(f"Parameters: {n_params:,}")

# Check that all expected sub-modules exist
for attr in [
	"encoder",
	"decoder",
	"f_psi",
	"_split",
	"controlled_step",
	"encode_mean",
	"H",
]:
	if hasattr(model, attr):
		ok(f"model.{attr} exists")
	else:
		bad(f"model.{attr} MISSING")

# ──────────────────────────────────────────────────────────────────────────────
sep("2. ENCODER OUTPUT SANITY")

frames_b = frames_t.unsqueeze(0).to(DEVICE)  # (1, T+1, C, H, W)

with torch.no_grad():
	mu, logvar = model.encoder(frames_b)

info(
	f"mu     shape={tuple(mu.shape)}  min={mu.min():.3f}  max={mu.max():.3f}  mean={mu.abs().mean():.4f}"
)
info(
	f"logvar shape={tuple(logvar.shape)}  min={logvar.min():.3f}  max={logvar.max():.3f}"
)

if mu.abs().mean() < 1e-4:
	bad("mu is essentially all zeros — encoder has collapsed")
else:
	ok("mu is non-trivial")

if logvar.min() < -9 or logvar.max() > 3:
	bad(
		f"logvar out of reasonable range: [{logvar.min():.2f}, {logvar.max():.2f}]"
	)
else:
	ok(
		f"logvar range looks reasonable: [{logvar.min():.2f}, {logvar.max():.2f}]"
	)

sigma = (0.5 * logvar).exp()
info(
	f"sigma  min={sigma.min():.4f}  max={sigma.max():.4f}  mean={sigma.mean():.4f}"
)
if sigma.mean() < 0.01:
	bad("sigma near zero — reparameterization gradient will vanish")
else:
	ok("sigma looks non-trivial")

# ──────────────────────────────────────────────────────────────────────────────
sep("3. f_psi / _split")

with torch.no_grad():
	z = mu  # no noise for this test
	qp = model.f_psi(z)
	q, p = model._split(qp)

info(f"z   shape={tuple(z.shape)}")
info(f"qp  shape={tuple(qp.shape)}")
info(
	f"q   shape={tuple(q.shape)}  min={q.min():.3f}  max={q.max():.3f}  mean={q.abs().mean():.4f}"
)
info(
	f"p   shape={tuple(p.shape)}  min={p.min():.3f}  max={p.max():.3f}  mean={p.abs().mean():.4f}"
)

if q.abs().mean() < 1e-4 and p.abs().mean() < 1e-4:
	bad(
		"q and p both near zero after f_psi/_split — linear layer may be zeroed or mu is dead"
	)
else:
	ok("q/p are non-trivial after f_psi/_split")

# ──────────────────────────────────────────────────────────────────────────────
sep("4. DECODER OUTPUT SANITY (no dynamics)")

with torch.no_grad():
	pred = model.decoder(q)

info(
	f"pred shape={tuple(pred.shape)}  min={pred.min():.4f}  max={pred.max():.4f}  mean={pred.mean():.4f}"
)

target = frames_b[:, 0]  # (1, C, H, W)
mse_init = F.mse_loss(pred, target).item()
info(f"MSE(decoder(q0), frame_0) at init = {mse_init:.6f}")

if pred.max() - pred.min() < 0.01:
	bad(
		"Decoder output has near-zero variance — it's outputting a constant (collapsed)"
	)
else:
	ok("Decoder output has non-trivial variance")

# Save side-by-side
gt_img = target.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1)
pred_img = pred.squeeze(0).permute(1, 2, 0).cpu().detach().numpy().clip(0, 1)
fig, axes = plt.subplots(1, 2, figsize=(6, 3))
axes[0].imshow(gt_img)
axes[0].set_title("Ground truth")
axes[0].axis("off")
axes[1].imshow(pred_img)
axes[1].set_title("Decoder(q0)")
axes[1].axis("off")
fig.savefig("/tmp/diag_decoder_init.png", dpi=80, bbox_inches="tight")
plt.close(fig)
info("Saved init reconstruction to /tmp/diag_decoder_init.png")

# ──────────────────────────────────────────────────────────────────────────────
sep("5. GRADIENT FLOW: single-step reconstruction, no dynamics")

model.train()
mu2, logvar2 = model.encoder(frames_b)
z2 = mu2  # no sampling — cleaner gradient signal for this test
q2, p2 = model._split(model.f_psi(z2))
pred2 = model.decoder(q2)
loss2 = F.mse_loss(pred2, frames_b[:, 0])
loss2.backward()

print()
dead_enc = []
for name, param in model.encoder.named_parameters():
	if param.grad is None:
		dead_enc.append(name)
	elif param.grad.abs().max() == 0:
		dead_enc.append(name + " (zero grad)")

if dead_enc:
	bad(f"Encoder params with no/zero gradient ({len(dead_enc)} of them):")
	for n in dead_enc[:10]:
		print(f"    {n}")
else:
	ok("All encoder parameters have non-zero gradients")

dead_dec = []
for name, param in model.decoder.named_parameters():
	if param.grad is None:
		dead_dec.append(name)
	elif param.grad.abs().max() == 0:
		dead_dec.append(name + " (zero grad)")

if dead_dec:
	bad(f"Decoder params with no/zero gradient ({len(dead_dec)} of them):")
	for n in dead_dec[:10]:
		print(f"    {n}")
else:
	ok("All decoder parameters have non-zero gradients")

# Print a representative sample of grad magnitudes
print()
info("Sample gradient magnitudes (encoder):")
for name, param in list(model.encoder.named_parameters())[:6]:
	g = param.grad
	if g is not None:
		print(
			f"    {name:40s}  abs_mean={g.abs().mean():.2e}  abs_max={g.abs().max():.2e}"
		)
	else:
		print(f"    {name:40s}  NONE")

model.zero_grad()

# ──────────────────────────────────────────────────────────────────────────────
sep("6. GRADIENT FLOW THROUGH controlled_step")

model.train()
mu3, logvar3 = model.encoder(frames_b)
z3 = mu3
q3, p3 = model._split(model.f_psi(z3))

actions_t = torch.as_tensor(actions_np).unsqueeze(0).to(DEVICE)  # (1, T)
u = actions_t[:, 0:1]  # (1, 1)
q3_next, p3_next = model.controlled_step(q3, p3, u)
pred3 = model.decoder(q3_next)
loss3 = F.mse_loss(pred3, frames_b[:, 1])
loss3.backward()

enc_grads_through_step = []
for name, param in model.encoder.named_parameters():
	if param.grad is not None and param.grad.abs().max() > 0:
		enc_grads_through_step.append((name, param.grad.abs().mean().item()))

if enc_grads_through_step:
	ok(
		f"Encoder receives gradients through controlled_step ({len(enc_grads_through_step)} params)"
	)
else:
	bad(
		"NO encoder gradients survive through controlled_step — dynamics may be detaching the graph"
	)
	info("Check: does controlled_step or RK4 call .detach() anywhere?")
	info("Check: are J/R/B matrices computed with torch.no_grad()?")

model.zero_grad()

# ──────────────────────────────────────────────────────────────────────────────
sep("7. HAMILTONIAN FUNCTION SANITY")

with torch.no_grad():
	H_val = model.H(q3, p3)
	info(f"H(q,p) = {H_val.item():.4f}")
	if not torch.isfinite(H_val):
		bad("H(q,p) is NaN or Inf!")
	else:
		ok("H(q,p) is finite")

	# After one step, H should change only by the dissipation/control work
	q_s, p_s = model.controlled_step(q3.detach(), p3.detach(), u)
	H_next = model.H(q_s, p_s)
	info(
		f"H after one step = {H_next.item():.4f}  (delta = {H_next.item() - H_val.item():.4f})"
	)
	if not torch.isfinite(H_next):
		bad("H diverges after one step!")
	else:
		ok("H is finite after one step")

# ──────────────────────────────────────────────────────────────────────────────
sep("8. MEMORIZATION TEST  (decoder-only, no dynamics)")

info(
	"Overfitting decoder(f_psi(mu)) → frame_0 for 200 steps (no KL, no dynamics)"
)
info(
	"If this doesn't converge, the encoder→f_psi→decoder path is broken at the architecture level."
)

model_mem = ControlledDHGN_LSTM(
	pos_ch=POS_CH,
	img_ch=3,
	dt=DT,
	feat_dim=FEAT_DIM,
	img_size=IMG_SIZE,
	control_dim=1,
	separable=True,
).to(DEVICE)
opt_mem = torch.optim.Adam(model_mem.parameters(), lr=1e-3)

losses_mem = []
for step in range(200):
	mu_m, _ = model_mem.encoder(frames_b)
	q_m, _ = model_mem._split(model_mem.f_psi(mu_m))
	pred_m = model_mem.decoder(q_m)
	loss_m = F.mse_loss(pred_m, frames_b[:, 0])
	opt_mem.zero_grad()
	loss_m.backward()
	opt_mem.step()
	losses_mem.append(loss_m.item())

info(f"Loss at step   0: {losses_mem[0]:.6f}")
info(f"Loss at step  50: {losses_mem[49]:.6f}")
info(f"Loss at step 199: {losses_mem[-1]:.6f}")

if losses_mem[-1] < losses_mem[0] * 0.1:
	ok(
		"Decoder-only memorization converges — encoder/decoder pipeline is functional"
	)
else:
	bad("Decoder-only memorization FAILED — fundamental architecture issue")
	info(
		"Possible causes: dead activations, wrong tensor shapes, missing connection between encoder and decoder"
	)

fig, ax = plt.subplots(figsize=(6, 3))
ax.semilogy(losses_mem)
ax.set_xlabel("Step")
ax.set_ylabel("MSE")
ax.set_title("Memorization test (no dynamics)")
fig.savefig("/tmp/diag_memorization.png", dpi=80, bbox_inches="tight")
plt.close(fig)
info("Saved memorization curve to /tmp/diag_memorization.png")

# Save final reconstruction from memorization test
with torch.no_grad():
	mu_f, _ = model_mem.encoder(frames_b)
	q_f, _ = model_mem._split(model_mem.f_psi(mu_f))
	pred_f = model_mem.decoder(q_f)

gt_img = frames_b[0, 0].permute(1, 2, 0).cpu().numpy().clip(0, 1)
pred_img = pred_f.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1)
fig, axes = plt.subplots(1, 2, figsize=(6, 3))
axes[0].imshow(gt_img)
axes[0].set_title("Ground truth")
axes[0].axis("off")
axes[1].imshow(pred_img)
axes[1].set_title("After 200 steps")
axes[1].axis("off")
fig.savefig("/tmp/diag_memorization_recon.png", dpi=80, bbox_inches="tight")
plt.close(fig)
info("Saved memorization reconstruction to /tmp/diag_memorization_recon.png")

# ──────────────────────────────────────────────────────────────────────────────
sep("9. MEMORIZATION TEST WITH T=1 ROLLOUT")

info("Overfitting through one controlled_step for 300 steps (no KL)")

model_t1 = ControlledDHGN_LSTM(
	pos_ch=POS_CH,
	img_ch=3,
	dt=DT,
	feat_dim=FEAT_DIM,
	img_size=IMG_SIZE,
	control_dim=1,
	separable=True,
).to(DEVICE)
opt_t1 = torch.optim.Adam(model_t1.parameters(), lr=1e-3)

losses_t1 = []
for step in range(300):
	mu_t, _ = model_t1.encoder(frames_b)
	q_t, p_t = model_t1._split(model_t1.f_psi(mu_t))
	# frame_0 reconstruction
	loss_t1 = F.mse_loss(model_t1.decoder(q_t), frames_b[:, 0])
	# one step reconstruction
	q_t1, p_t1 = model_t1.controlled_step(q_t, p_t, actions_t[:, 0:1])
	loss_t1 = loss_t1 + F.mse_loss(model_t1.decoder(q_t1), frames_b[:, 1])
	opt_t1.zero_grad()
	loss_t1.backward()
	opt_t1.step()
	losses_t1.append(loss_t1.item())

info(f"Loss at step   0: {losses_t1[0]:.6f}")
info(f"Loss at step 100: {losses_t1[99]:.6f}")
info(f"Loss at step 299: {losses_t1[-1]:.6f}")

if losses_t1[-1] < losses_t1[0] * 0.1:
	ok("T=1 rollout memorization converges — dynamics path has gradients")
else:
	bad("T=1 rollout memorization FAILED")
	if losses_mem[-1] < losses_mem[0] * 0.1:
		info(
			"→ Encoder/decoder alone works (test 8 passed) but gradient dies in controlled_step"
		)
		info(
			"  This confirms the issue is in the dynamics graph, not the encoder/decoder"
		)
	else:
		info("→ Both tests fail: the encoder/decoder path itself is broken")

fig, ax = plt.subplots(figsize=(6, 3))
ax.semilogy(losses_t1)
ax.set_xlabel("Step")
ax.set_ylabel("MSE")
ax.set_title("T=1 rollout memorization test")
fig.savefig("/tmp/diag_t1_memorization.png", dpi=80, bbox_inches="tight")
plt.close(fig)
info("Saved T=1 memorization curve to /tmp/diag_t1_memorization.png")

# ──────────────────────────────────────────────────────────────────────────────
sep("10. NaN/INF PATROL")

model_nan = ControlledDHGN_LSTM(
	pos_ch=POS_CH,
	img_ch=3,
	dt=DT,
	feat_dim=FEAT_DIM,
	img_size=IMG_SIZE,
	control_dim=1,
	separable=True,
).to(DEVICE)

mu_n, logvar_n = model_nan.encoder(frames_b)
z_n = mu_n + torch.randn_like(mu_n) * (0.5 * logvar_n).exp()
q_n, p_n = model_nan._split(model_nan.f_psi(z_n))

has_nan = False
checks = {
	"mu": mu_n,
	"logvar": logvar_n,
	"z": z_n,
	"q": q_n,
	"p": p_n,
}
for name, tensor in checks.items():
	if not torch.isfinite(tensor).all():
		bad(f"{name} contains NaN/Inf")
		has_nan = True
	else:
		ok(f"{name} is finite")

# Roll out 20 steps and watch for explosion
q_r, p_r = q_n.detach(), p_n.detach()
exploded_at = None
for t in range(20):
	u_r = actions_t[
		:, min(t, actions_t.shape[1] - 1) : min(t, actions_t.shape[1] - 1) + 1
	]
	with torch.no_grad():
		q_r, p_r = model_nan.controlled_step(q_r, p_r, u_r)
	if not torch.isfinite(q_r).all() or not torch.isfinite(p_r).all():
		exploded_at = t
		break
	if q_r.abs().max() > 1e4 or p_r.abs().max() > 1e4:
		bad(
			f"q/p exploding at step {t}: max_q={q_r.abs().max():.1e}  max_p={p_r.abs().max():.1e}"
		)
		exploded_at = t
		break

if exploded_at is None:
	ok("20-step rollout stays finite and bounded")
else:
	bad(f"Rollout diverged at step {exploded_at}")
	info(
		"Check your RK4 step size (dt), R matrix positivity, and J skew-symmetry"
	)

# ──────────────────────────────────────────────────────────────────────────────
sep("SUMMARY")
print("""
Interpret results in order:
  Test 8 FAIL → encoder/decoder architecture is broken (shapes, dead activations, detach)
  Test 8 PASS, Test 9 FAIL → gradient dies in controlled_step/RK4 (likely .detach() in dynamics)
  Test 8 PASS, Test 9 PASS → the issue is in training loop bookkeeping (KL, loss scale, lr)
  Test 5 dead grads → .detach() between encoder and decoder somewhere
  Test 6 dead grads → .detach() in controlled_step or the J/R/B computation
  Test 10 explosion → numerics unstable, reduce dt or check R positive-definiteness
  Section 0 bg_frac FAIL → trivial MSE floor, model learns mean background
  
Diagnostic images saved to /tmp/diag_*.png
""")
