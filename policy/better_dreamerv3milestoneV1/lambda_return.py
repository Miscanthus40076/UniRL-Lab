from __future__ import annotations

import torch


def lambda_return(
    rewards: torch.Tensor,
    values: torch.Tensor,
    continues: torch.Tensor,
    gamma: float = 0.99,
    lambda_: float = 0.95,
) -> torch.Tensor:
    if values.shape[0] != rewards.shape[0] + 1:
        raise ValueError("values must have exactly one bootstrap step")
    returns = []
    next_return = values[-1]
    for t in reversed(range(rewards.shape[0])):
        bootstrap = (1.0 - lambda_) * values[t + 1] + lambda_ * next_return
        next_return = rewards[t] + gamma * continues[t] * bootstrap
        returns.append(next_return)
    return torch.stack(list(reversed(returns)), dim=0)
