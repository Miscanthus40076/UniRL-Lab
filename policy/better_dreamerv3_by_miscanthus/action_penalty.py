from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class ActionPenaltyConfig:
    enabled: bool = False
    action_norm_weight: float = 0.02
    action_norm_threshold: float = 3.6
    action_delta_weight: float = 0.01
    action_delta_threshold: float = 2.4
    clip_max: float = 1.0

    def validate(self):
        if self.action_norm_weight < 0.0:
            raise ValueError("action_penalty.action_norm_weight must be >= 0")
        if self.action_delta_weight < 0.0:
            raise ValueError("action_penalty.action_delta_weight must be >= 0")
        if self.action_norm_threshold < 0.0:
            raise ValueError("action_penalty.action_norm_threshold must be >= 0")
        if self.action_delta_threshold < 0.0:
            raise ValueError("action_penalty.action_delta_threshold must be >= 0")
        if self.clip_max <= 0.0:
            raise ValueError("action_penalty.clip_max must be > 0")

    def asdict(self) -> dict:
        return asdict(self)


def compute_action_penalty(
    action: torch.Tensor,
    *,
    prev_action: torch.Tensor | None = None,
    config: ActionPenaltyConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if action.ndim < 1:
        raise ValueError("action must have at least one dimension")
    if not config.enabled:
        penalty = torch.zeros(action.shape[:-1], dtype=action.dtype, device=action.device)
        return penalty, _action_penalty_info(action, penalty, None)

    action_norm = torch.linalg.vector_norm(action, dim=-1)
    norm_penalty = float(config.action_norm_weight) * F.relu(
        action_norm - float(config.action_norm_threshold)
    ).square()

    if prev_action is None:
        if action.ndim >= 3:
            first_delta = torch.zeros_like(action[:, :1])
            delta = torch.cat([first_delta, action[:, 1:] - action[:, :-1]], dim=1)
        else:
            delta = torch.zeros_like(action)
    else:
        delta = action - prev_action.to(device=action.device, dtype=action.dtype)
    delta_norm = torch.linalg.vector_norm(delta, dim=-1)
    delta_penalty = float(config.action_delta_weight) * F.relu(
        delta_norm - float(config.action_delta_threshold)
    ).square()

    penalty = (norm_penalty + delta_penalty).clamp(min=0.0, max=float(config.clip_max))
    return penalty, _action_penalty_info(action, penalty, delta_norm)


def _action_penalty_info(
    action: torch.Tensor,
    penalty: torch.Tensor,
    delta_norm: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    action_norm = torch.linalg.vector_norm(action, dim=-1)
    if delta_norm is None:
        delta_norm = torch.zeros_like(action_norm)
    return {
        "action_penalty": penalty,
        "action_norm": action_norm,
        "action_delta_norm": delta_norm,
        "action_penalty_nonzero": penalty > 0,
    }
