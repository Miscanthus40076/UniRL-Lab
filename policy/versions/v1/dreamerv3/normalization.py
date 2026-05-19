from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class RunningNormConfig:
    rate: float = 0.01
    eps: float = 1e-8


class RunningNormalizer:
    def __init__(self, config: RunningNormConfig | None = None):
        self.config = config or RunningNormConfig()
        self.mean = 0.0
        self.var = 1.0
        self.initialized = False

    def update(self, values: torch.Tensor):
        flat = values.detach().float().reshape(-1)
        if flat.numel() == 0:
            return
        batch_mean = float(flat.mean().cpu())
        batch_var = float(flat.var(unbiased=False).cpu())
        if not self.initialized:
            self.mean = batch_mean
            self.var = max(batch_var, self.config.eps)
            self.initialized = True
            return
        rate = float(self.config.rate)
        self.mean = (1.0 - rate) * self.mean + rate * batch_mean
        self.var = (1.0 - rate) * self.var + rate * max(batch_var, self.config.eps)

    def normalize(self, values: torch.Tensor, update: bool = False) -> torch.Tensor:
        if update:
            self.update(values)
        mean = torch.as_tensor(self.mean, device=values.device, dtype=values.dtype)
        std = torch.as_tensor(self.std, device=values.device, dtype=values.dtype)
        return (values - mean) / std

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, device=values.device, dtype=values.dtype)
        std = torch.as_tensor(self.std, device=values.device, dtype=values.dtype)
        return values * std + mean

    @property
    def std(self) -> float:
        return max(self.var, self.config.eps) ** 0.5

    def state_dict(self) -> dict:
        return {
            "config": {
                "rate": self.config.rate,
                "eps": self.config.eps,
            },
            "mean": self.mean,
            "var": self.var,
            "initialized": self.initialized,
        }

    @classmethod
    def from_state_dict(cls, payload: dict):
        norm = cls(RunningNormConfig(**payload["config"]))
        norm.mean = float(payload["mean"])
        norm.var = float(payload["var"])
        norm.initialized = bool(payload["initialized"])
        return norm
