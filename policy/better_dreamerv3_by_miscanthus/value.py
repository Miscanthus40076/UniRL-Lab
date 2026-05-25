from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from .encoder import build_mlp


@dataclass(slots=True)
class DreamerValueConfig:
    feat_dim: int
    hidden_dim: int = 256
    num_layers: int = 2


class DreamerValue(nn.Module):
    def __init__(self, config: DreamerValueConfig):
        super().__init__()
        self.config = config
        self.net = build_mlp(config.feat_dim, 1, config.hidden_dim, config.num_layers)

    def forward(self, feat):
        return self.net(feat).squeeze(-1)
