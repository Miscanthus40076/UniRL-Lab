from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from .encoder import build_mlp


@dataclass(slots=True)
class DreamerEventDynamicsConfig:
    enabled: bool = False
    hidden_dim: int = 128
    gate_bias_init: float = -3.0
    event_penalty: float = 0.0005
    prediction_loss_scale: float = 1.0
    action_input: bool = True
    detach_prediction_target: bool = True
    detach_base_input: bool = False
    context_mask_enabled: bool = False
    context_mask_source: str = "context_gate"
    context_mask_mode: str = "topk"
    context_mask_top_percent: float = 0.10
    context_mask_window: int = 1
    context_mask_detach: bool = True
    event_sparsity_inside_mask_only: bool = True
    capacity_enabled: bool = False
    capacity_mode: str = "soft_topk_st"
    capacity_ratio: float = 0.10
    capacity_temperature: float = 1.0
    capacity_min_k: int = 1
    capacity_eval_hard: bool = True
    capacity_use_valid_mask: bool = True
    event_sparsity_inside_capacity_only: bool = True
    capacity_v2_detach_event_input: bool = False
    capacity_v2_detach_event_target: bool = False
    capacity_v2_use_sigmoid_gate_multiplier: bool = False
    capacity_v2_freeze_backbone_for_event_loss: bool = False

    def validate(self):
        if self.hidden_dim <= 0:
            raise ValueError("event_dynamics.hidden_dim must be positive")
        if self.event_penalty < 0.0:
            raise ValueError("event_dynamics.event_penalty must be >= 0")
        if self.prediction_loss_scale < 0.0:
            raise ValueError("event_dynamics.prediction_loss_scale must be >= 0")
        if self.context_mask_source not in {"context_gate", "context_delta"}:
            raise ValueError("event_dynamics.context_mask_source must be 'context_gate' or 'context_delta'")
        if self.context_mask_mode not in {"soft", "soft_norm", "topk"}:
            raise ValueError("event_dynamics.context_mask_mode must be 'soft', 'soft_norm', or 'topk'")
        if not 0.0 < self.context_mask_top_percent <= 1.0:
            raise ValueError("event_dynamics.context_mask_top_percent must be in (0, 1]")
        if self.context_mask_window < 0:
            raise ValueError("event_dynamics.context_mask_window must be >= 0")
        if self.capacity_mode not in {"soft_topk_st"}:
            raise ValueError("event_dynamics.capacity_mode must be 'soft_topk_st'")
        if not 0.0 < self.capacity_ratio <= 1.0:
            raise ValueError("event_dynamics.capacity_ratio must be in (0, 1]")
        if self.capacity_temperature <= 0.0:
            raise ValueError("event_dynamics.capacity_temperature must be > 0")
        if self.capacity_min_k < 1:
            raise ValueError("event_dynamics.capacity_min_k must be >= 1")
        if self.context_mask_enabled and self.capacity_enabled:
            raise ValueError("context_mask_enabled and capacity_enabled are mutually exclusive in this implementation")


def _to_seq_tensor(name: str, value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    if value.ndim == 2:
        value = value.unsqueeze(-1)
    if value.ndim != 3 or value.shape[-1] != 1:
        raise ValueError(f"{name} must have shape [B, T] or [B, T, 1], got {tuple(value.shape)}")
    return value.float()


def build_context_change_mask(
    context_gate_seq,
    context_delta_norm_seq=None,
    mode="topk",
    top_percent=0.10,
    window=1,
    detach=True,
    source="context_gate",
    invalid_transition_seq=None,
):
    context_gate_seq = _to_seq_tensor("context_gate_seq", context_gate_seq)
    context_delta_norm_seq = _to_seq_tensor("context_delta_norm_seq", context_delta_norm_seq)
    if source == "context_gate":
        if context_gate_seq is None:
            raise ValueError("context_gate_seq is required when context_mask_source='context_gate'")
        source_seq = context_gate_seq
    elif source == "context_delta":
        if context_delta_norm_seq is None:
            raise ValueError("context_delta_norm_seq is required when context_mask_source='context_delta'")
        source_seq = context_delta_norm_seq
    else:
        raise ValueError(f"Unsupported context mask source: {source}")

    source_seq = torch.nan_to_num(source_seq.float(), nan=0.0, posinf=1.0, neginf=0.0)
    invalid_transition_seq = _to_seq_tensor("invalid_transition_seq", invalid_transition_seq)
    if invalid_transition_seq is None:
        invalid_transition = torch.zeros_like(source_seq, dtype=torch.bool)
    else:
        invalid_transition = invalid_transition_seq > 0.5
        if invalid_transition.shape != source_seq.shape:
            raise ValueError(
                f"invalid_transition_seq shape must match source sequence shape {tuple(source_seq.shape)}, "
                f"got {tuple(invalid_transition_seq.shape)}"
            )

    eps = 1e-8
    if mode == "soft":
        mask = source_seq.clone()
    elif mode == "soft_norm":
        denom = source_seq.mean(dim=1, keepdim=True).clamp_min(eps)
        mask = (source_seq / denom).clamp_(0.0, 1.0)
    elif mode == "topk":
        mask = torch.zeros_like(source_seq)
        values_2d = source_seq.squeeze(-1)
        invalid_2d = invalid_transition.squeeze(-1)
        batch_size, seq_len = values_2d.shape
        for batch_index in range(batch_size):
            step = 0
            while step < seq_len:
                while step < seq_len and bool(invalid_2d[batch_index, step]):
                    step += 1
                if step >= seq_len:
                    break
                segment_start = step
                while step < seq_len and not bool(invalid_2d[batch_index, step]):
                    step += 1
                segment_end = step
                segment_values = values_2d[batch_index, segment_start:segment_end]
                if segment_values.numel() == 0:
                    continue
                keep_count = max(1, int(math.ceil(float(segment_values.numel()) * float(top_percent))))
                keep_count = min(keep_count, int(segment_values.numel()))
                top_indices = torch.topk(segment_values, k=keep_count, largest=True, sorted=False).indices
                for local_index in top_indices.tolist():
                    left = max(segment_start, segment_start + int(local_index) - int(window))
                    right = min(segment_end, segment_start + int(local_index) + int(window) + 1)
                    mask[batch_index, left:right, 0] = 1.0
    else:
        raise ValueError(f"Unsupported context mask mode: {mode}")

    mask = torch.nan_to_num(mask.float(), nan=0.0, posinf=1.0, neginf=0.0)
    mask = mask.masked_fill(invalid_transition, 0.0).clamp_(0.0, 1.0)
    return mask.detach() if detach else mask


class EventCapacitySelector(nn.Module):
    def __init__(self, config: DreamerEventDynamicsConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        event_logit_seq: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        event_logit_seq = _to_seq_tensor("event_logit_seq", event_logit_seq)
        if valid_mask is None:
            valid_mask_seq = torch.ones_like(event_logit_seq)
        else:
            valid_mask_seq = _to_seq_tensor("valid_mask", valid_mask)
            if valid_mask_seq.shape != event_logit_seq.shape:
                raise ValueError(
                    f"valid_mask shape must match event_logit_seq shape {tuple(event_logit_seq.shape)}, "
                    f"got {tuple(valid_mask_seq.shape)}"
                )

        logits = torch.nan_to_num(event_logit_seq.float(), nan=0.0, posinf=0.0, neginf=0.0)
        valid = valid_mask_seq > 0.5
        batch_size, seq_len, _ = logits.shape
        hard_mask = torch.zeros_like(logits)
        soft_weight = torch.zeros_like(logits)
        k_values = torch.zeros(batch_size, 1, 1, device=logits.device, dtype=logits.dtype)
        valid_counts = torch.zeros_like(k_values)

        for batch_index in range(batch_size):
            logits_1d = logits[batch_index, :, 0]
            valid_1d = valid[batch_index, :, 0]
            valid_count = int(valid_1d.sum().item())
            valid_counts[batch_index, 0, 0] = float(valid_count)
            if valid_count <= 0:
                continue
            k = max(int(self.config.capacity_min_k), int(math.ceil(float(self.config.capacity_ratio) * valid_count)))
            k = min(k, valid_count)
            k_values[batch_index, 0, 0] = float(k)

            masked_logits = logits_1d.masked_fill(~valid_1d, -1e9)
            top_indices = torch.topk(masked_logits, k=k, largest=True, sorted=False).indices
            hard_mask[batch_index, top_indices, 0] = 1.0

            soft_scores = torch.softmax(masked_logits / float(self.config.capacity_temperature), dim=0)
            soft_scores = soft_scores * valid_1d.float() * float(k)
            soft_weight[batch_index, :, 0] = soft_scores.clamp_(0.0, 1.0)

        if self.training:
            capacity_gate = hard_mask.detach() - soft_weight.detach() + soft_weight
        else:
            capacity_gate = hard_mask if self.config.capacity_eval_hard else soft_weight
        capacity_gate = torch.nan_to_num(capacity_gate.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        soft_weight = torch.nan_to_num(soft_weight.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)

        stats = {
            "event_capacity_gate": capacity_gate,
            "event_capacity_hard": hard_mask,
            "event_capacity_soft": soft_weight,
            "event_capacity_k": k_values,
            "event_capacity_valid_count": valid_counts,
        }
        return capacity_gate, stats


class EventDynamicsModule(nn.Module):
    def __init__(self, feat_dim: int, action_dim: int, config: DreamerEventDynamicsConfig):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.action_dim = int(action_dim)
        self.config = config
        self.config.validate()

        input_dim = self.feat_dim + (self.action_dim if self.config.action_input else 0)
        self.gate_net = build_mlp(input_dim, 1, hidden_dim=self.config.hidden_dim, num_layers=2)
        self.ordinary_head = build_mlp(input_dim, self.feat_dim, hidden_dim=self.config.hidden_dim, num_layers=2)
        self.event_head = build_mlp(input_dim, self.feat_dim, hidden_dim=self.config.hidden_dim, num_layers=2)
        self.capacity_selector = EventCapacitySelector(config)
        self._init_gate_bias()

    def _init_gate_bias(self):
        last = self.gate_net[-1]
        if isinstance(last, nn.Linear):
            nn.init.constant_(last.bias, float(self.config.gate_bias_init))

    def _build_input(self, feat: torch.Tensor, action: torch.Tensor | None = None) -> torch.Tensor:
        if self.config.detach_base_input:
            feat = feat.detach()
        parts = [feat]
        if self.config.action_input:
            if action is None:
                raise ValueError("EventDynamicsModule requires action input when action_input=true")
            parts.append(action)
        return torch.cat(parts, dim=-1)

    def predict(
        self,
        feat: torch.Tensor,
        action: torch.Tensor | None = None,
        target_next_feat: torch.Tensor | None = None,
        context_change_mask: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        use_capacity_v2 = bool(self.config.capacity_enabled)
        freeze_backbone = bool(use_capacity_v2 and self.config.capacity_v2_freeze_backbone_for_event_loss)
        detach_event_input = bool(
            self.config.detach_base_input
            or (use_capacity_v2 and self.config.capacity_v2_detach_event_input)
            or freeze_backbone
        )
        detach_event_target = bool(
            self.config.detach_prediction_target
            or (use_capacity_v2 and self.config.capacity_v2_detach_event_target)
            or freeze_backbone
        )
        event_input_feat = feat.detach() if detach_event_input else feat
        model_input = self._build_input(event_input_feat, action=action)
        event_logit = self.gate_net(model_input)
        event_gate = torch.sigmoid(event_logit)
        ordinary_delta = self.ordinary_head(model_input)
        ordinary_pred = event_input_feat + ordinary_delta
        event_residual = self.event_head(model_input)
        event_pred = ordinary_pred + event_residual
        applied_mask = None
        capacity_gate = None
        capacity_stats: dict[str, torch.Tensor] = {}
        if self.config.context_mask_enabled and context_change_mask is not None:
            if context_change_mask.ndim == event_gate.ndim + 1 and context_change_mask.shape[-1] == 1:
                context_change_mask = context_change_mask.squeeze(-1)
            if context_change_mask.ndim == feat.ndim - 1:
                context_change_mask = context_change_mask.unsqueeze(-1)
            if context_change_mask.shape != event_gate.shape:
                raise ValueError(
                    f"context_change_mask shape must match event gate shape {tuple(event_gate.shape)}, "
                    f"got {tuple(context_change_mask.shape)}"
                )
            applied_mask = context_change_mask.float()
            mixed_pred = ordinary_pred + applied_mask * event_gate * event_residual
        elif self.config.capacity_enabled:
            capacity_gate, capacity_stats = self.capacity_selector(
                event_logit,
                valid_mask=valid_mask if self.config.capacity_use_valid_mask else None,
            )
            gate_multiplier = event_gate if self.config.capacity_v2_use_sigmoid_gate_multiplier else torch.ones_like(event_gate)
            mixed_pred = ordinary_pred + capacity_gate * gate_multiplier * event_residual
        else:
            mixed_pred = ordinary_pred + event_gate * (event_pred - ordinary_pred)
        outputs = {
            "event_gate": event_gate,
            "event_logit": event_logit,
            "event_logit_sigmoid": event_gate,
            "context_change_mask": applied_mask,
            "context_change_mask_source": self.config.context_mask_source if applied_mask is not None else None,
            "context_change_mask_mode": self.config.context_mask_mode if applied_mask is not None else None,
            "context_change_mask_top_percent": (
                float(self.config.context_mask_top_percent) if applied_mask is not None else None
            ),
            "event_capacity_gate": capacity_gate,
            "event_capacity_mode": self.config.capacity_mode if self.config.capacity_enabled else None,
            "event_capacity_enabled": self.config.capacity_enabled,
            "event_capacity_ratio": float(self.config.capacity_ratio) if self.config.capacity_enabled else None,
            "capacity_v2_detach_event_input": detach_event_input,
            "capacity_v2_detach_event_target": detach_event_target,
            "capacity_v2_use_sigmoid_gate_multiplier": bool(self.config.capacity_v2_use_sigmoid_gate_multiplier),
            "event_loss_updates_backbone": not (detach_event_input and detach_event_target),
            "pred_next_feat_ordinary_only": ordinary_pred,
            "pred_next_feat_event_only": event_pred,
            "pred_next_feat_event_mixed": mixed_pred,
            "event_residual": event_residual,
            "event_residual_norm": torch.linalg.vector_norm(event_residual, dim=-1),
        }
        outputs.update(capacity_stats)
        if target_next_feat is not None:
            target = target_next_feat.detach() if detach_event_target else target_next_feat
            while target.ndim < ordinary_pred.ndim:
                target = target.unsqueeze(1)
            ordinary_error = torch.nn.functional.smooth_l1_loss(ordinary_pred, target, reduction="none").mean(dim=-1)
            mixed_error = torch.nn.functional.smooth_l1_loss(mixed_pred, target, reduction="none").mean(dim=-1)
            outputs["target_next_feat"] = target
            outputs["pred_next_feat_error_ordinary_only"] = ordinary_error
            outputs["pred_next_feat_error_event_mixed"] = mixed_error
            # Use the ordinary-model miss as the eventness diagnostic baseline.
            outputs["event_prediction_error"] = ordinary_error
        return outputs
