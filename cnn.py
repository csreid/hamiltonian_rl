import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
	def __init__(self, channels: int):
		super().__init__()
		self.net = nn.Sequential(
			nn.LeakyReLU(),
			nn.Conv2d(channels, channels, kernel_size=3, padding=1),
			nn.LeakyReLU(),
			nn.Conv2d(channels, channels, kernel_size=3, padding=1),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return x + self.net(x)


class ConvSequence(nn.Module):
	def __init__(self, in_channels: int, out_channels: int):
		super().__init__()
		self.net = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
			ResidualBlock(out_channels),
			ResidualBlock(out_channels),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)


class ImpalaCNN(nn.Module):
	"""
	Visual encoder from the IMPALA paper (Espeholt et al., 2018).

	A stack of conv sequences, each consisting of a convolution, max-pool, and
	two residual blocks. Designed to feed into a recurrent module.

	Expects input shape: (batch, 3, 84, 84)

	Args:
	    features_dim: size of the output feature vector
	    channels: output channels for each conv sequence stage
	"""

	def __init__(
		self,
		features_dim: int = 256,
		channels: tuple[int, ...] = (16, 32, 32),
	):
		super().__init__()

		stages = []
		in_ch = 3
		for out_ch in channels:
			stages.append(ConvSequence(in_ch, out_ch))
			in_ch = out_ch

		self.conv_stages = nn.Sequential(*stages)
		self.flatten = nn.Flatten()

		conv_out_dim = self._conv_out_dim()
		self.fc = nn.Sequential(
			nn.LeakyReLU(),
			nn.Linear(conv_out_dim, features_dim),
			nn.LeakyReLU(),
		)

	def _conv_out_dim(self) -> int:
		with torch.no_grad():
			dummy = torch.zeros(1, 3, 84, 84)
			return self.flatten(self.conv_stages(dummy)).shape[1]

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.dtype == torch.uint8:
			x = x.float() / 255.0
		return self.fc(self.flatten(self.conv_stages(x)))
