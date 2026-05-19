from __future__ import annotations

import torch
import torch.nn.functional as F


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))


def twohot_encode(
    values: torch.Tensor,
    num_bins: int = 255,
    low: float = -20.0,
    high: float = 20.0,
) -> torch.Tensor:
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2")
    clipped = values.clamp(low, high)
    bins = torch.linspace(low, high, num_bins, device=values.device, dtype=values.dtype)
    scaled = (clipped - low) / (high - low) * (num_bins - 1)
    lower = torch.floor(scaled).long()
    upper = torch.clamp(lower + 1, max=num_bins - 1)
    upper_weight = scaled - lower.float()
    lower_weight = 1.0 - upper_weight
    target = torch.zeros(*values.shape, num_bins, device=values.device, dtype=values.dtype)
    target.scatter_add_(-1, lower.unsqueeze(-1), lower_weight.unsqueeze(-1))
    target.scatter_add_(-1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
    return target


def twohot_logprob(
    logits: torch.Tensor,
    values: torch.Tensor,
    num_bins: int = 255,
    low: float = -20.0,
    high: float = 20.0,
) -> torch.Tensor:
    target = twohot_encode(values, num_bins=num_bins, low=low, high=high)
    log_probs = F.log_softmax(logits, dim=-1)
    return (target * log_probs).sum(dim=-1)


def twohot_mean(
    logits: torch.Tensor,
    num_bins: int = 255,
    low: float = -20.0,
    high: float = 20.0,
) -> torch.Tensor:
    bins = torch.linspace(low, high, num_bins, device=logits.device, dtype=logits.dtype)
    probs = torch.softmax(logits, dim=-1)
    return (probs * bins).sum(dim=-1)
