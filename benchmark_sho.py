"""
Benchmark the learned Hamiltonian on a simple harmonic oscillator (SHO).

H = p^2/2 + q^2/2
dq/dt =  p
dp/dt = -q

Trajectories in phase space are closed ellipses, and H is conserved exactly.
We train model.hamiltonian to approximate H, then verify the leapfrog integrator
conserves energy and produces correct phase portraits.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from hgn import RecurrentHGN

# size=1: scalar p and q (1D SHO)
model = RecurrentHGN(size=1)

# --- Train the Hamiltonian network ---
# Sample random phase-space points and regress against the true H
N = 10_000
p_data = torch.randn(N, 1)
q_data = torch.randn(N, 1)
H_true = 0.5 * p_data**2 + 0.5 * q_data**2  # (N, 1)

optimizer = torch.optim.Adam(model.hamiltonian.parameters(), lr=1e-3)
writer = SummaryWriter(comment="_sho_benchmark")

hparam_text = (
	"| Hyperparameter | Value |\n"
	"|---|---|\n"
	f"| N (training points) | {N} |\n"
	"| lr | 1e-3 |\n"
	"| train_steps | 3000 |\n"
	"| STEP_SIZE | 0.1 |\n"
	"| N_STEPS | 200 |\n"
)
writer.add_text("hparams", hparam_text, 0)

print("Training Hamiltonian network on SHO data...")
for step in (pbar := tqdm(range(3000))):
	H_pred = model.H(p_data, q_data).unsqueeze(-1)
	loss = nn.functional.mse_loss(H_pred, H_true)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	writer.add_scalar("train/loss", loss.item(), step)
	pbar.set_postfix(loss=f"{loss.item():.6f}")

print("Done.\n")

# --- Roll out trajectories from several initial conditions ---
model.eval()

STEP_SIZE = 0.1
N_STEPS = 200

# Initial conditions: points on the unit circle in phase space
initial_conditions = [
	(torch.tensor([[1.0]]), torch.tensor([[0.0]])),
	(torch.tensor([[0.0]]), torch.tensor([[1.0]])),
	(torch.tensor([[0.7]]), torch.tensor([[0.7]])),
	(torch.tensor([[2.0]]), torch.tensor([[0.0]])),
]

fig, (ax_phase, ax_energy) = plt.subplots(1, 2, figsize=(12, 5))

for p0, q0 in initial_conditions:
	ps, qs = [p0.detach()], [q0.detach()]
	p, q = p0, q0
	for _ in range(N_STEPS):
		p, q = model.hamiltonian_step(p, q, step_size=STEP_SIZE)
		ps.append(p.detach())
		qs.append(q.detach())

	ps = torch.cat(ps).numpy().squeeze()
	qs = torch.cat(qs).numpy().squeeze()
	Hs = 0.5 * ps**2 + 0.5 * qs**2

	ax_phase.plot(qs, ps)
	ax_energy.plot(Hs, label=f"p0={p0.item():.1f}, q0={q0.item():.1f}")

ax_phase.set_xlabel("q")
ax_phase.set_ylabel("p")
ax_phase.set_title("Phase portrait (should be closed ellipses)")
ax_phase.set_aspect("equal")

ax_energy.set_xlabel("Step")
ax_energy.set_ylabel("H (true)")
ax_energy.set_title("Energy conservation (should be flat)")
ax_energy.legend()

plt.tight_layout()
writer.add_figure("sho/phase_portrait_and_energy", fig)
writer.close()
plt.savefig("sho_benchmark.png", dpi=150)
print("Saved sho_benchmark.png")
