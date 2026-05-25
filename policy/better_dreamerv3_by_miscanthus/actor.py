from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from .encoder import build_mlp


class TanhNormal:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std
        self.normal = torch.distributions.Normal(mean, std)

    def rsample(self) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.normal.rsample()
        action = torch.tanh(raw)
        log_prob = self.normal.log_prob(raw) - torch.log(1.0 - action.square() + 1e-6)
        return action, log_prob.sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        return self.normal.entropy().sum(dim=-1)

    def mode(self) -> torch.Tensor:
        return torch.tanh(self.mean)


@dataclass(slots=True)
class DreamerActorConfig:
    feat_dim: int
    action_dim: int = 4
    hidden_dim: int = 256
    num_layers: int = 2
    min_std: float = 0.1
    max_std: float = 1.0
    init_std: float = 1.0


class DreamerActor(nn.Module):
    def __init__(self, config: DreamerActorConfig):
        super().__init__()
        self.config = config
        self.net = build_mlp(config.feat_dim, 2 * config.action_dim, config.hidden_dim, config.num_layers)
        if config.max_std <= config.min_std:
            raise ValueError("max_std must be greater than min_std")
        init_std = min(max(config.init_std, config.min_std), config.max_std)
        frac = (init_std - config.min_std) / (config.max_std - config.min_std)
        frac = min(max(frac, 1e-6), 1.0 - 1e-6)
        init_raw = math.log(frac / (1.0 - frac)) - 2.0
        with torch.no_grad():
            last = self.net[-1]
            if isinstance(last, nn.Linear):
                last.bias[config.action_dim:].fill_(init_raw)

    def forward(self, feat: torch.Tensor) -> TanhNormal:
        raw = self.net(feat)
        mean, std_raw = torch.chunk(raw, 2, dim=-1)
        std = (self.config.max_std - self.config.min_std) * torch.sigmoid(std_raw + 2.0) + self.config.min_std
        return TanhNormal(mean, std)

    def sample(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self(feat)
        action, log_prob = dist.rsample()
        return action, log_prob, dist.entropy()

    def mode(self, feat: torch.Tensor) -> torch.Tensor:
        return self(feat).mode()
