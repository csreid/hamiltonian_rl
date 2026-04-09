"""Streamlit dashboard for exploring Hamiltonian dynamics models.

Usage:
    streamlit run dashboard.py

Point the sidebar at a directory of saved .pt files.  Each checkpoint can
optionally have a JSON sidecar (same stem, .json extension) that specifies
the model class and constructor kwargs; without one the dashboard infers them
from the state-dict keys.

Sidecar format example:
    {
        "model_class": "ControlledDissipativeHGN",
        "model_kwargs": {"pos_ch": 4, "n_frames": 4, "control_dim": 1}
    }
"""

import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import yaml

# Make sure project modules are importable when the dashboard is run from any
# working directory.
sys.path.insert(0, str(Path(__file__).parent))

from benchmark_dhgn import generate_damped_dataset
from dhgn import DissipativeHGN
from dhgn_lstm import DHGN_LSTM
from diag_common import generate_dataset
from hgn import RecurrentHGN
from hgn_lstm import HGN_LSTM
from hgn_org import HGN
from phgn import ControlledDissipativeHGN

MODEL_CLASSES = {
	"HGN": HGN,
	"HGN_LSTM": HGN_LSTM,
	"DissipativeHGN": DissipativeHGN,
	"DHGN_LSTM": DHGN_LSTM,
	"ControlledDissipativeHGN": ControlledDissipativeHGN,
	"RecurrentHGN": RecurrentHGN,
}

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _detect_model_type(sd: dict) -> str:
	keys = set(sd.keys())
	if any("img_encoder" in k for k in keys):
		return "RecurrentHGN"
	if "B" in keys:
		return "ControlledDissipativeHGN"
	lstm_encoder = any("encoder.frame_cnn" in k for k in keys)
	if lstm_encoder:
		return "DHGN_LSTM" if "A" in keys else "HGN_LSTM"
	if "A" in keys:
		return "DissipativeHGN"
	return "HGN"


def _infer_kwargs(model_type: str, sd: dict) -> dict:
	kw: dict = {}
	if model_type == "RecurrentHGN":
		# latent_encoder.fc.weight shape: (size*2, hidden)
		if "latent_encoder.fc.weight" in sd:
			kw["size"] = sd["latent_encoder.fc.weight"].shape[0] // 2
		return kw
	if model_type in ("HGN_LSTM", "DHGN_LSTM"):
		# decoder.block1.conv1.weight: (64, pos_ch, 3, 3)
		if "decoder.block1.conv1.weight" in sd:
			kw["pos_ch"] = sd["decoder.block1.conv1.weight"].shape[1]
		# encoder.frame_cnn.net.9.weight: (feat_dim, 1024) — the Linear layer
		if "encoder.frame_cnn.net.9.weight" in sd:
			kw["feat_dim"] = sd["encoder.frame_cnn.net.9.weight"].shape[0]
		return kw
	# Spatial models (HGN / DissipativeHGN / ControlledDissipativeHGN)
	# decoder.block1.conv1.weight: (64, pos_ch, 3, 3)
	if "decoder.block1.conv1.weight" in sd:
		kw["pos_ch"] = sd["decoder.block1.conv1.weight"].shape[1]
	# encoder.body.0.weight: (32, n_frames*3, 3, 3)
	if "encoder.body.0.weight" in sd:
		kw["n_frames"] = sd["encoder.body.0.weight"].shape[1] // 3
	if model_type == "ControlledDissipativeHGN":
		if "B" in sd:
			kw["control_dim"] = sd["B"].shape[1]
		# state_decoder is a 5-layer Sequential: Linear SiLU Linear SiLU Linear
		if "state_decoder.4.weight" in sd:
			kw["obs_state_dim"] = sd["state_decoder.4.weight"].shape[0]
	return kw


@st.cache_resource(show_spinner="Loading model…")
def _load_model(path_str: str):
	path = Path(path_str)

	# Try full model object first (torch.save(model, path))
	try:
		obj = torch.load(path, map_location="cpu", weights_only=False)
		if isinstance(obj, torch.nn.Module):
			obj.eval()
			return obj, type(obj).__name__, {}
	except Exception:
		pass

	# Fall back to state-dict + optional JSON sidecar
	sd = torch.load(path, map_location="cpu", weights_only=True)
	meta_path = path.with_suffix(".json")
	if meta_path.exists():
		with open(meta_path) as fh:
			meta = json.load(fh)
		model_type = meta.get("model_class", _detect_model_type(sd))
		kw = meta.get("model_kwargs", _infer_kwargs(model_type, sd))
	else:
		model_type = _detect_model_type(sd)
		kw = _infer_kwargs(model_type, sd)

	model = MODEL_CLASSES[model_type](**kw)
	model.load_state_dict(sd)
	model.eval()

	# Load hparams from YAML sidecar if present (written by save_checkpoint).
	hparams = {}
	yaml_path = path.with_suffix(".yaml")
	if yaml_path.exists():
		with open(yaml_path) as fh:
			doc = yaml.safe_load(fh)
			hparams = doc.get("hparams", {})

	return model, model_type, kw, hparams


# ---------------------------------------------------------------------------
# Rollout helpers
# ---------------------------------------------------------------------------


def _to_rgb(t: torch.Tensor) -> np.ndarray:
	"""(1, C, H, W) or (C, H, W) → (H, W, C) uint8."""
	if t.dim() == 4:
		t = t[0]
	arr = t.detach().cpu().clamp(0, 1).numpy()
	return (np.transpose(arr, (1, 2, 0)) * 255).astype(np.uint8)


def _rollout_image(
	model,
	model_type: str,
	pos_ch: int,
	n_steps: int,
	dt: float,
	seed: int,
	hparams: dict,
):
	torch.manual_seed(seed)

	is_lstm = model_type in ("HGN_LSTM", "DHGN_LSTM")
	if is_lstm and hparams:
		# Generate a ground-truth synthetic trajectory using the saved hparams,
		# encode it to get an on-manifold (q, p), then roll forward.
		context_len = hparams.get("context_len", 32)
		train_dt = hparams.get("dt", dt)
		img_size = hparams.get("img_size", 32)
		blob_sigma = hparams.get("blob_sigma", 2.0)
		max_amplitude = hparams.get("max_amplitude", 1.0)
		spring_k = hparams.get("spring_constant", 1.0)
		mass = hparams.get("mass", 1.0)

		if model_type == "DHGN_LSTM":
			damping = hparams.get("damping", 0.3)
			_, _, ctx_frames = generate_damped_dataset(
				1,
				context_len,
				train_dt,
				img_size,
				blob_sigma,
				max_amplitude,
				spring_k,
				mass,
				damping,
				margin=4,
			)
		else:
			_, _, ctx_frames = generate_dataset(
				1,
				context_len,
				train_dt,
				img_size,
				blob_sigma,
				max_amplitude,
				spring_k,
				mass,
				margin=4,
			)

		with torch.no_grad():
			q0, p0, *_ = model(ctx_frames)  # (1, pos_ch, 4, 4)
	else:
		q0 = torch.randn(1, pos_ch, 4, 4) * 0.1
		p0 = torch.randn(1, pos_ch, 4, 4) * 0.3

	with torch.no_grad():
		frames_t, _coords, qs, ps = model.rollout(
			q0, p0, n_steps=n_steps, dt=dt, return_states=True
		)
	frames = [_to_rgb(f) for f in frames_t]
	energies = []
	for q, p in zip(qs, ps):
		with torch.no_grad():
			energies.append(model.H(q, p).item())
	return frames, np.array(energies)


def _rollout_sho(model, size: int, n_steps: int, dt: float, seed: int):
	torch.manual_seed(seed)
	p = torch.randn(1, size) * 0.5
	q = torch.randn(1, size) * 0.5

	qs = [q[0].detach().numpy().copy()]
	ps = [p[0].detach().numpy().copy()]
	with torch.no_grad():
		energies = [model.H(p, q).item()]

	for _ in range(n_steps):
		with torch.no_grad():
			p, q = model.hamiltonian_step(p, q, step_size=dt)
		qs.append(q[0].detach().numpy().copy())
		ps.append(p[0].detach().numpy().copy())
		with torch.no_grad():
			energies.append(model.H(p, q).item())

	return np.array(qs), np.array(ps), np.array(energies)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

st.set_page_config(page_title="HGN Explorer", layout="wide")
st.title("Hamiltonian Dynamics Explorer")

# ── Sidebar: model selection ─────────────────────────────────────────────────
with st.sidebar:
	st.header("Model")
	models_dir = st.text_input("Models directory", value="./models")
	models_path = Path(models_dir)

	if not models_path.exists():
		st.warning(f"Directory not found: `{models_dir}`")
		st.stop()

	pt_files = sorted(models_path.rglob("*.pt"))
	if not pt_files:
		st.warning("No `.pt` files found.")
		st.stop()

	names = [str(f.relative_to(models_path)) for f in pt_files]
	selected_name = st.selectbox("Checkpoint", names)
	model_path = models_path / selected_name

	st.divider()
	st.header("Rollout")
	dt = st.slider(
		"dt (step size)",
		min_value=0.001,
		max_value=2.0,
		value=0.1,
		step=0.001,
		format="%.3f",
	)
	n_steps = st.slider(
		"Steps", min_value=10, max_value=1000, value=100, step=10
	)
	seed = st.number_input(
		"Random seed", min_value=0, max_value=99999, value=42, step=1
	)
	run_btn = st.button(
		"▶  Run rollout", type="primary", use_container_width=True
	)

# ── Load model ────────────────────────────────────────────────────────────────
try:
	model, model_type, kw, hparams = _load_model(str(model_path))
except Exception as exc:
	st.error(f"Failed to load `{model_path.name}`:\n\n```\n{exc}\n```")
	st.stop()

with st.sidebar:
	st.success(f"**{model_type}**")
	if kw:
		with st.expander("Inferred kwargs"):
			st.json(kw)

	# Warn if params differ from the last run
	last_params = st.session_state.get("rollout_params")
	current_params = (str(model_path), dt, n_steps, seed)
	if last_params and last_params != current_params:
		st.warning("Parameters changed — press ▶ Run rollout to update.")

# ── Trigger rollout ───────────────────────────────────────────────────────────
model_changed = st.session_state.get("rollout_model") != str(model_path)
need_rollout = model_changed or run_btn

if need_rollout:
	with st.spinner("Running rollout…"):
		try:
			if model_type == "RecurrentHGN":
				size = kw.get("size", getattr(model, "size", 1))
				qs, ps, energies = _rollout_sho(model, size, n_steps, dt, seed)
				st.session_state["rollout"] = {
					"type": "sho",
					"qs": qs,
					"ps": ps,
					"energies": energies,
				}
			else:
				pos_ch = kw.get("pos_ch", getattr(model, "pos_ch", 16))
				frames, energies = _rollout_image(
					model, model_type, pos_ch, n_steps, dt, seed, hparams
				)
				st.session_state["rollout"] = {
					"type": "image",
					"frames": frames,
					"energies": energies,
				}
			st.session_state["rollout_model"] = str(model_path)
			st.session_state["rollout_params"] = current_params
		except Exception as exc:
			st.error(f"Rollout failed:\n\n```\n{exc}\n```")
			st.stop()

rollout = st.session_state.get("rollout")
if rollout is None:
	st.info("Press **▶ Run rollout** in the sidebar to start.")
	st.stop()

# ── Display: image model ──────────────────────────────────────────────────────
if rollout["type"] == "image":
	frames: list = rollout["frames"]
	energies: np.ndarray = rollout["energies"]
	n = len(frames)

	stored_params = st.session_state.get("rollout_params", ())
	if len(stored_params) >= 4:
		st.caption(
			f"Rollout: **{stored_params[1]:.3f}** dt · **{n - 1}** steps · seed **{stored_params[3]}**"
		)

	t = st.slider("Time step", 0, n - 1, 0, key="img_t")

	col_frame, col_energy = st.columns([1, 2], gap="large")

	with col_frame:
		st.subheader("Decoded frame")
		# Upscale tiny 32×32 frames for visibility
		frame_np = frames[t]
		from PIL import Image

		pil_img = Image.fromarray(frame_np).resize((256, 256), Image.NEAREST)
		st.image(np.array(pil_img), caption=f"Step {t} / {n - 1}")

	with col_energy:
		st.subheader("Hamiltonian H(q, p) over time")
		steps = np.arange(n)
		# Use matplotlib (always available) for plotting
		import matplotlib.pyplot as plt

		fig, ax = plt.subplots(figsize=(8, 3))
		ax.plot(steps, energies, linewidth=1.5, label="H(t)")
		ax.axvline(
			t, color="red", linewidth=1.2, linestyle="--", label=f"t={t}"
		)
		ax.set_xlabel("step")
		ax.set_ylabel("H")
		ax.legend(fontsize=9)
		ax.set_title("Energy conservation")
		fig.tight_layout()
		st.pyplot(fig)
		plt.close(fig)

		H0 = energies[0]
		dH = np.max(np.abs(energies - H0))
		col_a, col_b = st.columns(2)
		col_a.metric("H at t=0", f"{H0:.4f}")
		col_b.metric("Max |ΔH|", f"{dH:.4f}")

# ── Display: SHO / RecurrentHGN ──────────────────────────────────────────────
elif rollout["type"] == "sho":
	qs: np.ndarray = rollout["qs"]  # (n+1, size)
	ps: np.ndarray = rollout["ps"]
	energies: np.ndarray = rollout["energies"]
	n = len(qs)
	size = qs.shape[1]

	stored_params = st.session_state.get("rollout_params", ())
	if len(stored_params) >= 4:
		st.caption(
			f"Rollout: **{stored_params[1]:.3f}** dt · **{n - 1}** steps · seed **{stored_params[3]}**"
		)

	t = st.slider("Time step", 0, n - 1, 0, key="sho_t")

	import matplotlib.pyplot as plt

	col_phase, col_time, col_energy = st.columns(3)

	with col_phase:
		st.subheader("Phase portrait")
		fig, ax = plt.subplots(figsize=(4, 4))
		ax.plot(qs[:, 0], ps[:, 0], linewidth=1, alpha=0.6, color="steelblue")
		ax.scatter(
			qs[t, 0],
			ps[t, 0],
			color="red",
			zorder=5,
			s=60,
			label=f"t={t}",
		)
		ax.set_xlabel("q")
		ax.set_ylabel("p")
		ax.legend(fontsize=9)
		ax.set_aspect("equal", adjustable="datalim")
		fig.tight_layout()
		st.pyplot(fig)
		plt.close(fig)

	with col_time:
		st.subheader("q(t) and p(t)")
		steps = np.arange(n)
		fig, ax = plt.subplots(figsize=(4, 3))
		ax.plot(steps, qs[:, 0], label="q", linewidth=1.2)
		ax.plot(steps, ps[:, 0], label="p", linewidth=1.2, linestyle="--")
		ax.axvline(t, color="red", linewidth=1, linestyle=":")
		ax.set_xlabel("step")
		ax.legend(fontsize=9)
		fig.tight_layout()
		st.pyplot(fig)
		plt.close(fig)

		if size > 1:
			st.caption(f"Showing component 0 of {size}.")

	with col_energy:
		st.subheader("Hamiltonian H(t)")
		fig, ax = plt.subplots(figsize=(4, 3))
		ax.plot(steps, energies, linewidth=1.2, color="darkorange")
		ax.axvline(t, color="red", linewidth=1, linestyle=":")
		ax.set_xlabel("step")
		ax.set_ylabel("H")
		fig.tight_layout()
		st.pyplot(fig)
		plt.close(fig)

		H0 = energies[0]
		dH = np.max(np.abs(energies - H0))
		st.metric("Max |ΔH|", f"{dH:.4f}")
