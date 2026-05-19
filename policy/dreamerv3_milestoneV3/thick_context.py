from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class ThickContextConfig:
    enabled: bool = False
    context_dim: int = 64
    hidden_dim: int = 128
    update_penalty: float = 0.0005
    detach_base_feat: bool = False
    init_context_zero: bool = True
    gate_bias_init: float = -2.0
    gate_type: str = "sigmoid"
    l0_temperature: float = 0.5
    l0_noise: bool = True
    l0_straight_through: bool = True
    l0_min: float = 0.0
    l0_max: float = 1.0

    def validate(self):
        if self.context_dim <= 0:
            raise ValueError("thick_context.context_dim must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("thick_context.hidden_dim must be positive")
        if self.update_penalty < 0.0:
            raise ValueError("thick_context.update_penalty must be >= 0")
        if self.gate_type not in {"sigmoid", "l0_st"}:
            raise ValueError("thick_context.gate_type must be 'sigmoid' or 'l0_st'")
        if self.l0_temperature <= 0.0:
            raise ValueError("thick_context.l0_temperature must be > 0")
        if self.l0_min < 0.0 or self.l0_max > 1.0 or self.l0_min > self.l0_max:
            raise ValueError("thick_context.l0_min/l0_max must satisfy 0 <= min <= max <= 1")


class SlowContextModule(nn.Module):
    _noise_eps = 1e-6

    def __init__(self, base_feat_dim: int, config: ThickContextConfig):
        super().__init__()
        self.base_feat_dim = int(base_feat_dim)
        self.config = config
        self.config.validate()

        input_dim = self.base_feat_dim + int(config.context_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.candidate_net = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.context_dim),
        )
        self.init_context_param = None
        if not config.init_context_zero:
            self.init_context_param = nn.Parameter(torch.zeros(config.context_dim))
        self._init_gate_bias()

    def _init_gate_bias(self):
        last = self.gate_net[-1]
        if isinstance(last, nn.Linear):
            nn.init.constant_(last.bias, float(self.config.gate_bias_init))

    def zero_context(self, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        return torch.zeros(
            int(batch_size),
            int(self.config.context_dim),
            device=device,
            dtype=dtype,
        )

    def initial_context(self, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        if self.init_context_param is None:
            return self.zero_context(batch_size, device=device, dtype=dtype)
        init = self.init_context_param.to(device=device, dtype=dtype)
        return init.unsqueeze(0).expand(int(batch_size), -1)

    def _normalize_prev_context(
        self,
        prev_context: torch.Tensor | None,
        batch_size: int,
        device,
        dtype,
    ) -> torch.Tensor:
        if prev_context is None:
            return self.initial_context(batch_size, device=device, dtype=dtype)
        if prev_context.shape != (batch_size, self.config.context_dim):
            raise ValueError(
                f"prev_context shape must be {(batch_size, self.config.context_dim)}, got {tuple(prev_context.shape)}"
            )
        return prev_context.to(device=device, dtype=dtype)

    def _normalize_is_first_step(self, is_first: torch.Tensor | None, batch_size: int, device, dtype) -> torch.Tensor | None:
        if is_first is None:
            return None
        is_first = is_first.to(device=device, dtype=dtype)
        if is_first.ndim == 2 and is_first.shape[-1] == 1:
            is_first = is_first.squeeze(-1)
        return is_first.reshape(batch_size, 1)

    def _normalize_is_first_seq(
        self,
        is_first: torch.Tensor | None,
        batch_size: int,
        seq_len: int,
        device,
        dtype,
    ) -> torch.Tensor | None:
        if is_first is None:
            return None
        is_first = is_first.to(device=device, dtype=dtype)
        if is_first.ndim == 3 and is_first.shape[-1] == 1:
            is_first = is_first.squeeze(-1)
        return is_first.reshape(batch_size, seq_len)

    def _sample_logistic_noise(self, gate_logit: torch.Tensor) -> torch.Tensor:
        uniform = torch.rand_like(gate_logit)
        uniform = uniform.clamp_(self._noise_eps, 1.0 - self._noise_eps)
        return torch.log(uniform) - torch.log1p(-uniform)

    def _compute_gate(self, gate_logit: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.config.gate_type == "sigmoid":
            gate_soft = torch.sigmoid(gate_logit)
            gate_hard = gate_soft
            gate = gate_soft
            open_prob = gate_soft
        else:
            temperature = float(self.config.l0_temperature)
            open_prob = torch.sigmoid(gate_logit / temperature)
            if self.training:
                if self.config.l0_noise:
                    noise = self._sample_logistic_noise(gate_logit)
                    gate_soft = torch.sigmoid((gate_logit + noise) / temperature)
                else:
                    gate_soft = open_prob
                gate_hard = (gate_soft > 0.5).to(gate_soft.dtype)
                if self.config.l0_straight_through:
                    gate = gate_hard.detach() - gate_soft.detach() + gate_soft
                else:
                    gate = gate_soft
            else:
                gate_soft = open_prob
                gate_hard = (gate_soft > 0.5).to(gate_soft.dtype)
                gate = gate_hard
            gate_soft = gate_soft.clamp(min=float(self.config.l0_min), max=float(self.config.l0_max))
            gate_hard = gate_hard.clamp(min=float(self.config.l0_min), max=float(self.config.l0_max))
            gate = gate.clamp(min=float(self.config.l0_min), max=float(self.config.l0_max))
        return {
            "gate": gate,
            "gate_soft": gate_soft,
            "gate_hard": gate_hard,
            "gate_logit": gate_logit,
            "open_prob": open_prob,
        }

    def _step(self, base_feat_t: torch.Tensor, prev_context: torch.Tensor) -> dict[str, torch.Tensor]:
        gate_feat = base_feat_t.detach() if self.config.detach_base_feat else base_feat_t
        gate_input = torch.cat([gate_feat, prev_context], dim=-1)
        gate_logit_t = self.gate_net(gate_input)
        gate_outputs = self._compute_gate(gate_logit_t)
        gate_t = gate_outputs["gate"]
        candidate_t = torch.tanh(self.candidate_net(gate_input))
        context_t = (1.0 - gate_t) * prev_context + gate_t * candidate_t
        augmented_feat_t = torch.cat([base_feat_t, context_t], dim=-1)
        delta_norm_t = torch.linalg.vector_norm(context_t - prev_context, dim=-1)
        return {
            "context": context_t,
            "gate": gate_t,
            "gate_soft": gate_outputs["gate_soft"],
            "gate_hard": gate_outputs["gate_hard"],
            "gate_logit": gate_outputs["gate_logit"],
            "open_prob": gate_outputs["open_prob"],
            "candidate_context": candidate_t,
            "augmented_feat": augmented_feat_t,
            "prev_context": prev_context,
            "delta_norm": delta_norm_t,
        }

    def forward_details(
        self,
        base_feat_t: torch.Tensor,
        prev_context: torch.Tensor | None = None,
        is_first: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if base_feat_t.ndim == 2:
            batch_size = int(base_feat_t.shape[0])
            prev_context_t = self._normalize_prev_context(
                prev_context,
                batch_size=batch_size,
                device=base_feat_t.device,
                dtype=base_feat_t.dtype,
            )
            first_mask = self._normalize_is_first_step(
                is_first,
                batch_size=batch_size,
                device=base_feat_t.device,
                dtype=base_feat_t.dtype,
            )
            if first_mask is not None:
                prev_context_t = prev_context_t * (1.0 - first_mask)
            outputs = self._step(base_feat_t, prev_context_t)
            outputs["gate_type"] = self.config.gate_type
            return outputs

        if base_feat_t.ndim != 3:
            raise ValueError(f"base_feat_t must have shape [B, D] or [B, T, D], got {tuple(base_feat_t.shape)}")

        batch_size, seq_len, _ = base_feat_t.shape
        prev_context_t = self._normalize_prev_context(
            prev_context,
            batch_size=batch_size,
            device=base_feat_t.device,
            dtype=base_feat_t.dtype,
        )
        first_seq = self._normalize_is_first_seq(
            is_first,
            batch_size=batch_size,
            seq_len=seq_len,
            device=base_feat_t.device,
            dtype=base_feat_t.dtype,
        )

        contexts = []
        gates = []
        gate_softs = []
        gate_hards = []
        gate_logits = []
        open_probs = []
        candidates = []
        augmented = []
        prev_contexts = []
        delta_norms = []
        for t in range(seq_len):
            current_prev = prev_context_t
            if first_seq is not None:
                current_prev = current_prev * (1.0 - first_seq[:, t].reshape(batch_size, 1))
            step_out = self._step(base_feat_t[:, t], current_prev)
            contexts.append(step_out["context"])
            gates.append(step_out["gate"])
            gate_softs.append(step_out["gate_soft"])
            gate_hards.append(step_out["gate_hard"])
            gate_logits.append(step_out["gate_logit"])
            open_probs.append(step_out["open_prob"])
            candidates.append(step_out["candidate_context"])
            augmented.append(step_out["augmented_feat"])
            prev_contexts.append(step_out["prev_context"])
            delta_norms.append(step_out["delta_norm"])
            prev_context_t = step_out["context"]
        return {
            "gate_type": self.config.gate_type,
            "context": torch.stack(contexts, dim=1),
            "gate": torch.stack(gates, dim=1),
            "gate_soft": torch.stack(gate_softs, dim=1),
            "gate_hard": torch.stack(gate_hards, dim=1),
            "gate_logit": torch.stack(gate_logits, dim=1),
            "open_prob": torch.stack(open_probs, dim=1),
            "candidate_context": torch.stack(candidates, dim=1),
            "augmented_feat": torch.stack(augmented, dim=1),
            "prev_context": torch.stack(prev_contexts, dim=1),
            "delta_norm": torch.stack(delta_norms, dim=1),
        }

    def forward(
        self,
        base_feat_t: torch.Tensor,
        prev_context: torch.Tensor | None = None,
        is_first: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.forward_details(base_feat_t, prev_context=prev_context, is_first=is_first)
        return (
            outputs["context"],
            outputs["gate"],
            outputs["candidate_context"],
            outputs["augmented_feat"],
        )
