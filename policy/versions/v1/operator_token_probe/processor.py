from __future__ import annotations

import torch


class OperatorTokenProbeProcessor:
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

    def prepare_transition_batch(self, batch: dict) -> dict:
        prepared = {}
        for key, value in batch.items():
            if key.startswith("_"):
                prepared[key] = value
                continue
            if torch.is_tensor(value):
                prepared[key] = value.to(self.device)
            else:
                prepared[key] = torch.as_tensor(value, device=self.device)
        return prepared
