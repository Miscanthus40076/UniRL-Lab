from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(slots=True)
class OperatorTokenConfig:
    enabled: bool = True
    num_tokens: int = 8
    token_dim: int = 32
    hidden_dim: int = 128
    gumbel_temperature: float = 1.0
    straight_through: bool = True
    train_on_capacity_only: bool = True
    effect_loss_scale: float = 1.0
    inverse_loss_scale: float = 0.5
    prior_loss_scale: float = 0.1
    usage_entropy_scale: float = 0.01
    detach_features: bool = True

    def validate(self):
        if self.num_tokens <= 1:
            raise ValueError("operator_token.num_tokens must be > 1")
        if self.token_dim <= 0:
            raise ValueError("operator_token.token_dim must be > 0")
        if self.hidden_dim <= 0:
            raise ValueError("operator_token.hidden_dim must be > 0")
        if self.gumbel_temperature <= 0.0:
            raise ValueError("operator_token.gumbel_temperature must be > 0")
        if self.effect_loss_scale < 0.0:
            raise ValueError("operator_token.effect_loss_scale must be >= 0")
        if self.inverse_loss_scale < 0.0:
            raise ValueError("operator_token.inverse_loss_scale must be >= 0")
        if self.prior_loss_scale < 0.0:
            raise ValueError("operator_token.prior_loss_scale must be >= 0")
        if self.usage_entropy_scale < 0.0:
            raise ValueError("operator_token.usage_entropy_scale must be >= 0")


def _build_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class OperatorTokenizerPosterior(nn.Module):
    def __init__(self, feat_dim: int, action_dim: int, config: OperatorTokenConfig):
        super().__init__()
        self.config = config
        input_dim = feat_dim * 3 + action_dim
        self.net = _build_mlp(input_dim, config.hidden_dim, config.num_tokens)

    def forward(self, x_t, x_tp1, delta_x, action_t):
        logits = self.net(torch.cat([x_t, x_tp1, delta_x, action_t], dim=-1))
        probs = torch.softmax(logits, dim=-1)
        if self.training:
            onehot = F.gumbel_softmax(
                logits,
                tau=self.config.gumbel_temperature,
                hard=self.config.straight_through,
                dim=-1,
            )
        else:
            token_id = torch.argmax(logits, dim=-1)
            onehot = F.one_hot(token_id, num_classes=self.config.num_tokens).float()
        token_id = torch.argmax(onehot, dim=-1)
        return {
            "token_logits": logits,
            "token_probs": probs,
            "token_onehot_st": onehot,
            "token_id": token_id,
        }


class OperatorTokenPrior(nn.Module):
    def __init__(self, feat_dim: int, action_dim: int, config: OperatorTokenConfig):
        super().__init__()
        self.net = _build_mlp(feat_dim + action_dim, config.hidden_dim, config.num_tokens)

    def forward(self, x_t, action_t):
        return self.net(torch.cat([x_t, action_t], dim=-1))


class OperatorEffectModel(nn.Module):
    def __init__(self, feat_dim: int, action_dim: int, token_dim: int, hidden_dim: int):
        super().__init__()
        self.net = _build_mlp(feat_dim + action_dim + token_dim, hidden_dim, feat_dim)

    def forward(self, x_t, action_t, token_embedding):
        return self.net(torch.cat([x_t, action_t, token_embedding], dim=-1))


class NoTokenEffectModel(nn.Module):
    def __init__(self, feat_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = _build_mlp(feat_dim + action_dim, hidden_dim, feat_dim)

    def forward(self, x_t, action_t):
        return self.net(torch.cat([x_t, action_t], dim=-1))


class InverseActionProbe(nn.Module):
    def __init__(self, feat_dim: int, action_dim: int, token_dim: int, hidden_dim: int):
        super().__init__()
        self.net = _build_mlp(feat_dim * 2 + token_dim, hidden_dim, action_dim)

    def forward(self, x_t, delta_x, token_embedding):
        return self.net(torch.cat([x_t, delta_x, token_embedding], dim=-1))


class NoTokenInverseActionProbe(nn.Module):
    def __init__(self, feat_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = _build_mlp(feat_dim * 2, hidden_dim, action_dim)

    def forward(self, x_t, delta_x):
        return self.net(torch.cat([x_t, delta_x], dim=-1))


def masked_mean(values: torch.Tensor, mask: torch.Tensor | None, eps: float = 1e-8) -> torch.Tensor:
    if mask is None:
        return values.mean()
    masked = values * mask
    denom = torch.clamp(mask.sum(), min=eps)
    return masked.sum() / denom


def masked_mean_per_step(values: torch.Tensor, mask: torch.Tensor | None, eps: float = 1e-8) -> torch.Tensor:
    if mask is None:
        return values.mean(dim=-1)
    denom = torch.clamp(mask.sum(dim=-1), min=eps)
    return (values * mask).sum(dim=-1) / denom


def masked_entropy(token_usage: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    probs = token_usage / torch.clamp(token_usage.sum(), min=eps)
    probs = torch.clamp(probs, min=eps)
    return -(probs * torch.log(probs)).sum()


def compute_token_usage_metrics(token_probs: torch.Tensor, mask: torch.Tensor | None, eps: float = 1e-8) -> dict[str, torch.Tensor]:
    flat_probs = token_probs.reshape(-1, token_probs.shape[-1])
    if mask is None:
        usage = flat_probs.mean(dim=0)
    else:
        flat_mask = mask.reshape(-1, 1).float()
        if torch.count_nonzero(flat_mask) <= 0:
            raise ValueError("capacity mask contains no active transitions")
        usage = (flat_probs * flat_mask).sum(dim=0) / torch.clamp(flat_mask.sum(), min=eps)
    entropy = masked_entropy(usage, eps=eps)
    perplexity = torch.exp(entropy)
    active = (usage > 0.01).float().sum()
    return {
        "token_usage": usage,
        "token_entropy": entropy,
        "token_perplexity": perplexity,
        "num_active_tokens": active,
    }


def random_accuracy_baseline(num_tokens: int) -> float:
    return 1.0 / float(max(1, num_tokens))


def safe_relative_improvement(baseline: torch.Tensor, improved: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (baseline - improved) / torch.clamp(baseline, min=eps)


def bad_numeric_count(*tensors: torch.Tensor | None) -> int:
    count = 0
    for tensor in tensors:
        if tensor is None:
            continue
        count += int((~torch.isfinite(tensor)).sum().item())
    return count
