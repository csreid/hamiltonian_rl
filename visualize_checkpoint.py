"""Streamlit checkpoint visualiser for PHGN-LSTM.

Usage:
    streamlit run visualize_checkpoint.py

Select a checkpoint, generate a PID-controlled episode, and inspect the
overlaid ground-truth / model-prediction video.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import yaml
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))

from data.cartpole import FrameBuffer, preprocess_frame
from phgn_lstm import ControlledDHGN_LSTM


# ── Model loading ─────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading model…")
def load_model(pt_path_str: str) -> tuple[ControlledDHGN_LSTM, dict]:
	pt_path = Path(pt_path_str)
	yaml_path = pt_path.with_suffix(".yaml")

	hparams: dict = {}
	if yaml_path.exists():
		with open(yaml_path) as fh:
			doc = yaml.safe_load(fh)
			hparams = doc.get("hparams", {})

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model = ControlledDHGN_LSTM(
		pos_ch=hparams.get("pos_ch", 8),
		img_ch=3,
		dt=hparams.get("dt", 0.05),
		feat_dim=hparams.get("feat_dim", 256),
		img_size=hparams.get("img_size", 64),
		control_dim=1,
		separable=hparams.get("separable", True),
	).to(device)

	sd = torch.load(pt_path, map_location=device, weights_only=True)
	model.load_state_dict(sd)
	model.eval()

	return model, hparams


# ── PID controller ────────────────────────────────────────────────────────────


class PIDController:
	def __init__(self, kp: float, ki: float, kd: float):
		self.kp = kp
		self.ki = ki
		self.kd = kd
		self._integral = 0.0
		self._prev_error = 0.0

	def reset(self) -> None:
		self._integral = 0.0
		self._prev_error = 0.0

	def compute(
		self, theta: float, theta_dot: float, dt: float = 0.02
	) -> float:
		"""Return a continuous action in [-1, 1] based on pole angle."""
		error = theta + 0.1 * theta_dot
		self._integral += error * dt
		derivative = (error - self._prev_error) / dt
		self._prev_error = error
		u = self.kp * error + self.ki * self._integral + self.kd * derivative
		return float(np.clip(u, -1.0, 1.0))


# ── Episode collection ────────────────────────────────────────────────────────


def collect_episode(
	img_size: int,
	anchor_context: int,
	max_steps: int,
	pid_kp: float = 10.0,
	pid_ki: float = 0.1,
	pid_kd: float = 2.0,
) -> dict:
	"""Collect one CartPole episode under PID control.

	Returns a dict with:
	    gt_frames  – list of (H, W, 3) uint8 numpy arrays (T+1 frames)
	    actions    – list of float (T actions)
	    gt_states  – list of (4,) float32 arrays (T+1 states)
	"""
	env = gym.make("CartPole-v1", render_mode="rgb_array")
	buf = FrameBuffer(anchor_context, img_size)

	_, _ = env.reset()
	first_frame = env.render()
	buf.reset(first_frame)

	gt_frames: list[np.ndarray] = [first_frame]
	actions: list[float] = []
	gt_states: list[np.ndarray] = []

	pid = PIDController(pid_kp, pid_ki, pid_kd)
	pid.reset()

	for _ in range(max_steps):
		state = np.array(env.unwrapped.state, dtype=np.float32)
		gt_states.append(state)

		action_float = pid.compute(state[2], state[3])
		gym_action = 1 if action_float > 0 else 0
		_, _, terminated, truncated, _ = env.step(gym_action)

		frame = env.render()
		buf.push(frame)
		gt_frames.append(frame)
		actions.append(action_float)

		if terminated or truncated:
			break

	gt_states.append(gt_states[-1])  # pad so len matches gt_frames
	env.close()

	return {"gt_frames": gt_frames, "actions": actions, "gt_states": gt_states}


# ── Model rollout ─────────────────────────────────────────────────────────────


def _tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
	"""(3, H, W) float32 [0,1] → (H, W, 3) uint8."""
	return (t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(
		np.uint8
	)


def run_model_rollout(
	model: ControlledDHGN_LSTM,
	gt_frames: list[np.ndarray],
	actions: list[float],
	anchor_context: int,
	img_size: int,
	dt: float,
	device: torch.device,
) -> dict:
	"""Encode the first `anchor_context` ground-truth frames, then roll forward
	using the recorded actions to generate model predictions.

	Returns a dict with:
	    model_pixel_frames  – decoded pixel frames, length = T - anchor_context + 2
	    rollout_start       – index into gt_frames where model frames start
	"""
	frames_tensor = torch.stack(
		[preprocess_frame(f, img_size) for f in gt_frames]
	)  # (T+1, 3, H, W)

	T = len(actions)
	ctx = (
		frames_tensor[:anchor_context].unsqueeze(0).to(device)
	)  # (1, k, 3, H, W)

	model_pixel_frames: list[np.ndarray] = []

	model.eval()
	with torch.no_grad():
		q, p = model.encode_mean(ctx)

		# Step 0: reconstruction of frame[anchor_context-1]
		model_pixel_frames.append(_tensor_to_uint8(model.decoder(q)[0]))

		# Roll forward for the rest of the episode
		for t in range(anchor_context - 1, T):
			u = torch.tensor([[actions[t]]], dtype=torch.float32, device=device)
			q, p = model.controlled_step(q, p, u, dt=dt)
			model_pixel_frames.append(_tensor_to_uint8(model.decoder(q)[0]))

	return {
		"model_pixel_frames": model_pixel_frames,
		"rollout_start": anchor_context - 1,
	}


# ── Frame compositing ─────────────────────────────────────────────────────────

_ARROW_RIGHT = (60, 200, 60)
_ARROW_LEFT = (220, 60, 60)
_CONTEXT_BORDER = (180, 180, 60)


def _tint_blue(arr: np.ndarray) -> np.ndarray:
	f = arr.astype(np.float32)
	f[:, :, 0] *= 0.75
	f[:, :, 1] *= 0.82
	f[:, :, 2] = np.clip(f[:, :, 2] + 35, 0, 255)
	return f.clip(0, 255).astype(np.uint8)


def _tint_orange(arr: np.ndarray) -> np.ndarray:
	f = arr.astype(np.float32)
	f[:, :, 0] = np.clip(f[:, :, 0] + 35, 0, 255)
	f[:, :, 1] *= 0.82
	f[:, :, 2] *= 0.60
	return f.clip(0, 255).astype(np.uint8)


def _draw_arrow(
	img: Image.Image, action: float, action_max: float = 1.0
) -> Image.Image:
	if action == 0.0:
		return img
	img = img.copy()
	draw = ImageDraw.Draw(img)
	W, H = img.size
	cy = H - 14
	cx = W // 2
	max_half = W // 5
	magnitude = min(abs(action) / max(action_max, 1e-6), 1.0)
	half_len = max(4, int(max_half * magnitude))
	color = _ARROW_RIGHT if action > 0 else _ARROW_LEFT
	tip_x = cx + half_len if action > 0 else cx - half_len
	tail_x = cx - half_len if action > 0 else cx + half_len
	shaft_width = max(2, int(4 * magnitude))
	head = max(5, int(10 * magnitude))
	draw.line([(tail_x, cy), (tip_x, cy)], fill=color, width=shaft_width)
	if action > 0:
		draw.polygon(
			[
				(tip_x, cy),
				(tip_x - head, cy - head // 2),
				(tip_x - head, cy + head // 2),
			],
			fill=color,
		)
	else:
		draw.polygon(
			[
				(tip_x, cy),
				(tip_x + head, cy - head // 2),
				(tip_x + head, cy + head // 2),
			],
			fill=color,
		)
	return img


def _draw_context_border(img: Image.Image) -> Image.Image:
	img = img.copy()
	draw = ImageDraw.Draw(img)
	W, H = img.size
	draw.rectangle([(0, 0), (W - 1, H - 1)], outline=_CONTEXT_BORDER, width=4)
	return img


def build_composite_frames(
	gt_frames: list[np.ndarray],
	model_frames: list[np.ndarray],
	actions: list[float],
	rollout_start: int,
	display_size: int,
	alpha: float,
) -> list[Image.Image]:
	action_max = max(abs(a) for a in actions) if actions else 1.0
	composite: list[Image.Image] = []

	for t, gt_raw in enumerate(gt_frames):
		gt_pil = Image.fromarray(gt_raw).resize(
			(display_size, display_size), Image.BILINEAR
		)

		if t < rollout_start:
			frame = _draw_context_border(gt_pil)
		else:
			model_idx = t - rollout_start
			if model_idx < len(model_frames):
				model_pil = Image.fromarray(model_frames[model_idx]).resize(
					(display_size, display_size), Image.BILINEAR
				)
				gt_tinted = Image.fromarray(_tint_blue(np.array(gt_pil)))
				model_tinted = Image.fromarray(
					_tint_orange(np.array(model_pil))
				)
				frame = Image.blend(gt_tinted, model_tinted, alpha)
			else:
				frame = gt_pil

		action_idx = min(t, len(actions) - 1)
		if actions:
			frame = _draw_arrow(
				frame, actions[action_idx], action_max=action_max
			)

		composite.append(frame)

	return composite


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


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="PHGN-LSTM Checkpoint Visualiser", layout="wide")
st.title("PHGN-LSTM Checkpoint Visualiser")

with st.sidebar:
	st.header("Checkpoint")
	models_root = Path("models")
	pt_files: list[Path] = []
	if models_root.exists():
		pt_files = sorted(models_root.rglob("*.pt"))

	if not pt_files:
		st.warning("No `.pt` checkpoints found under `models/`.")
		st.stop()

	rel_names = [str(f.relative_to(models_root.parent)) for f in pt_files]
	selected_name = st.selectbox("Checkpoint file", rel_names)
	pt_path = models_root.parent / selected_name

	st.divider()
	st.header("Episode settings")
	anchor_context = st.number_input(
		"Context frames (k)", min_value=1, max_value=20, value=3, step=1
	)
	max_steps = st.number_input(
		"Max episode steps", min_value=10, max_value=2000, value=500, step=10
	)

	st.subheader("PID tuning")
	pid_kp = st.slider("Kp", 0.0, 50.0, 10.0, step=0.5)
	pid_ki = st.slider("Ki", 0.0, 5.0, 0.1, step=0.05)
	pid_kd = st.slider("Kd", 0.0, 20.0, 2.0, step=0.1)

	st.divider()
	generate_btn = st.button(
		"▶ Generate episode", type="primary", use_container_width=True
	)

# ── Load model ────────────────────────────────────────────────────────────────
try:
	model, hparams = load_model(str(pt_path))
	device = next(model.parameters()).device
except Exception as exc:
	st.error(f"Failed to load checkpoint:\n\n```\n{exc}\n```")
	st.stop()

img_size = hparams.get("img_size", 64)
dt = hparams.get("dt", 0.05)

with st.sidebar:
	st.caption(
		f"img_size={img_size}  dt={dt}  "
		f"pos_ch={hparams.get('pos_ch', '?')}  "
		f"feat_dim={hparams.get('feat_dim', '?')}"
	)

# ── Episode generation ────────────────────────────────────────────────────────
if generate_btn:
	with st.spinner("Collecting episode…"):
		try:
			ep_data = collect_episode(
				img_size=img_size,
				anchor_context=int(anchor_context),
				max_steps=int(max_steps),
				pid_kp=pid_kp,
				pid_ki=pid_ki,
				pid_kd=pid_kd,
			)
		except Exception as exc:
			st.error(f"Episode collection failed:\n\n```\n{exc}\n```")
			st.stop()

	with st.spinner("Running model rollout…"):
		try:
			rollout = run_model_rollout(
				model=model,
				gt_frames=ep_data["gt_frames"],
				actions=ep_data["actions"],
				anchor_context=int(anchor_context),
				img_size=img_size,
				dt=dt,
				device=device,
			)
		except Exception as exc:
			st.error(f"Model rollout failed:\n\n```\n{exc}\n```")
			st.stop()

	st.session_state.update(
		ep_data=ep_data,
		rollout=rollout,
		anchor_context=int(anchor_context),
		checkpoint=str(pt_path),
	)

# ── Check we have data to show ────────────────────────────────────────────────
ep_data = st.session_state.get("ep_data")
rollout = st.session_state.get("rollout")

if ep_data is None:
	st.info(
		"Configure settings in the sidebar and press **▶ Generate episode**."
	)
	st.stop()

gt_frames = ep_data["gt_frames"]
actions = ep_data["actions"]
anchor = st.session_state["anchor_context"]
rollout_start = rollout["rollout_start"]
overlay_frames = rollout["model_pixel_frames"]

episode_len = len(actions)

st.success(
	f"Episode: **{episode_len}** steps  |  Checkpoint: `{st.session_state['checkpoint']}`"
)

# ── Video controls ────────────────────────────────────────────────────────────
col_fps, col_alpha, col_size = st.columns(3)
with col_fps:
	fps = st.slider("Playback FPS", min_value=1, max_value=60, value=15, step=1)
with col_alpha:
	blend_alpha = st.slider(
		"Blend α  (0 = GT only · 1 = model only)", 0.0, 1.0, 0.5, step=0.05
	)
with col_size:
	display_size = st.select_slider(
		"Display size (px)", [128, 192, 256, 384, 512], value=256
	)

# ── Render video ──────────────────────────────────────────────────────────────
with st.spinner("Rendering video…"):
	composite = build_composite_frames(
		gt_frames=gt_frames,
		model_frames=overlay_frames,
		actions=actions,
		rollout_start=rollout_start,
		display_size=display_size,
		alpha=blend_alpha,
	)
	gif_bytes = frames_to_gif(composite, fps)

st.subheader("Episode video")
legend_cols = st.columns(4)
legend_cols[0].markdown("🟡 **Yellow border** = context frames")
legend_cols[1].markdown("🔵 **Blue tint** = ground truth")
legend_cols[2].markdown("🟠 **Orange tint** = model prediction")
legend_cols[3].markdown("🟢 Arrow right / 🔴 Arrow left = action")

st.image(gif_bytes, use_container_width=False)
