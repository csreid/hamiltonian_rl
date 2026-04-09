import pytest
import torch
from hgn import RecurrentHGN


def test_instantiation():
	model = RecurrentHGN(size=32)
	assert isinstance(model, RecurrentHGN)


def test_forward_output_shapes():
	model = RecurrentHGN(size=32)
	model.eval()

	batch, seq, C, H, W = 2, 5, 3, 64, 64
	imgs = torch.randint(0, 256, (batch, seq, C, H, W), dtype=torch.uint8)

	with torch.no_grad():
		p, q, kl = model(imgs)

	assert p.shape == (batch, seq, 32), (
		f"Expected p shape (2, 5, 32), got {p.shape}"
	)
	assert q.shape == (batch, seq, 32), (
		f"Expected q shape (2, 5, 32), got {q.shape}"
	)


def test_forward_float_input():
	model = RecurrentHGN(size=16)
	model.eval()

	batch, seq, C, H, W = 1, 3, 3, 84, 84
	imgs = torch.rand(batch, seq, C, H, W)

	with torch.no_grad():
		p, q, kl = model(imgs)

	assert p.shape == (batch, seq, 16)
	assert q.shape == (batch, seq, 16)
	assert kl.shape == (batch, seq)


def test_forward_single_step():
	model = RecurrentHGN(size=8)
	model.eval()

	imgs = torch.rand(1, 1, 3, 32, 32)

	with torch.no_grad():
		p, q, kl = model(imgs)

	assert p.shape == (1, 1, 8)
	assert q.shape == (1, 1, 8)
	assert kl.shape == (1, 1)
