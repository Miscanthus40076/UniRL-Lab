from __future__ import annotations

import torch
from torch import nn

from .transforms import symlog

def _activation(name: str) -> type[nn.Module]:
    if name == "silu":
        return nn.SiLU
    if name == "elu":
        return nn.ELU
    if name == "relu":
        return nn.ReLU
    raise ValueError(f"Unsupported activation: {name}")


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int = 256,
    num_layers: int = 2,
    activation: str = "silu",
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = int(input_dim)
    act_cls = _activation(activation)
    for _ in range(int(num_layers)):
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(act_cls())
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class MLPEncoder(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        activation: str = "silu",
        use_symlog: bool = True,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.embed_dim = int(embed_dim)
        self.use_symlog = bool(use_symlog)
        self.net = build_mlp(self.obs_dim, self.embed_dim, hidden_dim, num_layers, activation)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.use_symlog:
            obs = symlog(obs)
        return self.net(obs)


class CNNEncoder(nn.Module):
    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        embed_dim: int = 128,
        hidden_dim: int = 256,
        activation: str = "silu",
    ):
        super().__init__()
        channels, height, width = (int(x) for x in obs_shape)
        act_cls = _activation(activation)
        self.obs_shape = (channels, height, width)
        self.embed_dim = int(embed_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1),
            act_cls(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            act_cls(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            act_cls(),
            nn.Flatten(),
        )
        with torch.no_grad():
            conv_dim = int(self.conv(torch.zeros(1, channels, height, width)).shape[-1])
        self.proj = nn.Sequential(
            nn.Linear(conv_dim, hidden_dim),
            act_cls(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.float()
        if obs.max() > 2.0:
            obs = obs / 255.0
        return self.proj(self.conv(obs))
