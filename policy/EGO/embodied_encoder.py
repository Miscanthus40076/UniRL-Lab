from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from .encoder import build_mlp

EXPLICIT_ROBOT_FIELDS = (
    "proprio",
    "robot_state",
    "q",
    "joint_pos",
    "joint_vel",
    "dq",
    "gripper",
    "gripper_state",
    "ee_pose",
)


@dataclass(slots=True)
class EmbodiedEncoderConfig:
    use_embodied_encoder: bool = False
    embodied_encoder_dim: int = 64
    embodied_robot_latent_dim: int = 64
    embodied_encoder_hidden_dim: int = 128
    embodied_encoder_num_layers: int = 2
    embodied_encoder_aux_weight: float = 0.0
    freeze_embodied_encoder: bool = True
    detach_embodied_encoder: bool = True
    embodied_adapter_type: str = "mlp"
    embodied_pretrain: bool = False
    embodied_encoder_checkpoint: str | None = None
    robot_state_keys: tuple[str, ...] = (
        "proprio",
        "robot_state",
        "q",
        "joint_pos",
        "joint_vel",
        "dq",
        "gripper",
        "gripper_state",
        "ee_pose",
        "prev_action",
        "action",
    )

    def validate(self):
        if self.embodied_encoder_dim <= 0:
            raise ValueError("embodied_encoder_dim must be positive")
        if self.embodied_robot_latent_dim <= 0:
            raise ValueError("embodied_robot_latent_dim must be positive")
        if self.embodied_encoder_hidden_dim <= 0:
            raise ValueError("embodied_encoder_hidden_dim must be positive")
        if self.embodied_encoder_num_layers < 0:
            raise ValueError("embodied_encoder_num_layers must be >= 0")
        if self.embodied_encoder_aux_weight < 0.0:
            raise ValueError("embodied_encoder_aux_weight must be >= 0")
        if self.embodied_adapter_type not in {"linear", "mlp"}:
            raise ValueError("embodied_adapter_type must be 'linear' or 'mlp'")

    def asdict(self) -> dict:
        return asdict(self)


def _flatten_time_value(value: torch.Tensor, batch_shape: tuple[int, ...]) -> torch.Tensor:
    value = value.float()
    if value.shape[: len(batch_shape)] != batch_shape:
        raise ValueError(f"Robot prior field shape {tuple(value.shape)} does not start with {batch_shape}")
    return value.reshape(*batch_shape, -1)


def add_prev_action_if_missing(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if "prev_action" in batch or "action" not in batch or not torch.is_tensor(batch["action"]):
        return batch
    action = batch["action"]
    if action.ndim < 2:
        return batch
    prev_action = torch.zeros_like(action)
    if action.shape[1] > 1:
        prev_action[:, 1:] = action[:, :-1]
    is_first = batch.get("is_first")
    if torch.is_tensor(is_first) and is_first.shape[:2] == action.shape[:2]:
        reset_mask = is_first.float() > 0
        while reset_mask.ndim < prev_action.ndim:
            reset_mask = reset_mask.unsqueeze(-1)
        prev_action = torch.where(reset_mask, torch.zeros_like(prev_action), prev_action)
    enriched = dict(batch)
    enriched["prev_action"] = prev_action
    return enriched


def summarize_robot_fields(batch: dict[str, torch.Tensor], keys: Iterable[str]) -> dict[str, object]:
    available = [key for key in keys if key in batch and torch.is_tensor(batch[key])]
    explicit = [key for key in available if key in EXPLICIT_ROBOT_FIELDS]
    fallback_to_action = not explicit
    if explicit:
        used = list(explicit)
        if "prev_action" in batch:
            used.append("prev_action")
    else:
        used = ["prev_action"] if "prev_action" in batch else (["action"] if "action" in batch else [])
    missing = [key for key in EXPLICIT_ROBOT_FIELDS if key not in available]
    return {
        "available_robot_fields": available,
        "explicit_robot_fields": explicit,
        "used_robot_fields": used,
        "missing_robot_fields": missing,
        "fallback_to_action": fallback_to_action,
        "has_proprio": any(key in explicit for key in ("proprio", "robot_state")),
        "has_gripper": any(key in explicit for key in ("gripper", "gripper_state")),
        "has_joint_pos": any(key in explicit for key in ("q", "joint_pos")),
        "has_ee_pose": "ee_pose" in explicit,
    }


def extract_robot_prior_inputs(
    batch: dict[str, torch.Tensor],
    keys: Iterable[str],
    *,
    action_dim: int,
    reference: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[str]]:
    batch = add_prev_action_if_missing(batch)
    if reference is None:
        if "obs" in batch:
            reference = batch["obs"]
        elif "action" in batch:
            reference = batch["action"]
        else:
            raise ValueError("EmbodiedEncoder requires at least obs or action in batch")
    batch_shape = tuple(reference.shape[:2]) if reference.ndim >= 3 else tuple(reference.shape[:1])
    parts: list[torch.Tensor] = []
    used: list[str] = []
    summary = summarize_robot_fields(batch, keys)
    candidate_keys = list(summary["used_robot_fields"])
    for key in candidate_keys:
        if key not in batch:
            continue
        value = batch[key]
        if not torch.is_tensor(value):
            continue
        try:
            parts.append(_flatten_time_value(value, batch_shape))
        except ValueError:
            continue
        used.append(key)
    if parts:
        return torch.cat(parts, dim=-1), used
    device = reference.device
    dtype = reference.dtype if torch.is_floating_point(reference) else torch.float32
    fallback = torch.zeros(*batch_shape, int(action_dim), device=device, dtype=dtype)
    # Current replay may only store obs/action. Until explicit proprio/FK
    # fields are available, previous/current action is the only self signal.
    return fallback, []


class EmbodiedEncoder(nn.Module):
    def __init__(self, action_dim: int, config: EmbodiedEncoderConfig):
        super().__init__()
        self.action_dim = int(action_dim)
        self.config = config
        config.validate()
        self.input_proj = nn.LazyLinear(config.embodied_encoder_hidden_dim)
        self.backbone = build_mlp(
            input_dim=config.embodied_encoder_hidden_dim,
            output_dim=config.embodied_robot_latent_dim,
            hidden_dim=config.embodied_encoder_hidden_dim,
            num_layers=config.embodied_encoder_num_layers,
        )
        self.e_self_head = nn.Linear(config.embodied_robot_latent_dim, config.embodied_encoder_dim)
        self.next_robot_head = nn.Linear(config.embodied_robot_latent_dim, config.embodied_encoder_hidden_dim)
        self.next_robot_out_by_dim = nn.ModuleDict()

    def _next_robot_out(self, hidden: torch.Tensor, target_dim: int) -> torch.Tensor:
        key = str(int(target_dim))
        if key not in self.next_robot_out_by_dim:
            self.next_robot_out_by_dim[key] = nn.Linear(self.config.embodied_encoder_hidden_dim, int(target_dim)).to(
                device=hidden.device,
                dtype=hidden.dtype,
            )
        return self.next_robot_out_by_dim[key](hidden)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | list[str] | dict[str, torch.Tensor]]:
        batch = add_prev_action_if_missing(batch)
        reference = batch.get("obs", batch.get("action"))
        summary = summarize_robot_fields(batch, self.config.robot_state_keys)
        robot_input, used_keys = extract_robot_prior_inputs(
            batch,
            self.config.robot_state_keys,
            action_dim=self.action_dim,
            reference=reference,
        )
        hidden = F.silu(self.input_proj(robot_input))
        z_robot = torch.tanh(self.backbone(hidden))
        e_self = self.e_self_head(z_robot)
        return {
            "e_self": e_self,
            "z_robot": z_robot,
            "robot_input": robot_input,
            "used_keys": used_keys,
            "diagnostics": {
                **summary,
                "used_robot_fields": used_keys,
                "num_robot_fields_used": len(used_keys),
            },
            "aux_losses": {},
        }

    def aux_losses(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if "obs" not in batch or batch["obs"].ndim < 3:
            return {}
        batch = add_prev_action_if_missing(batch)
        current = {key: value[:, :-1] for key, value in batch.items() if torch.is_tensor(value) and value.ndim >= 2}
        target_batch = {key: value[:, 1:] for key, value in batch.items() if torch.is_tensor(value) and value.ndim >= 2}
        current_out = self.forward(current)
        target_input, used_target_keys = extract_robot_prior_inputs(
            target_batch,
            self.config.robot_state_keys,
            action_dim=self.action_dim,
            reference=target_batch.get("obs", target_batch.get("action")),
        )
        if not used_target_keys:
            zero = current_out["z_robot"].sum() * 0.0
            return {"aux_loss_total": zero}
        pred_hidden = F.silu(self.next_robot_head(current_out["z_robot"]))
        pred = self._next_robot_out(pred_hidden, target_input.shape[-1])
        loss = F.smooth_l1_loss(pred, target_input.detach())
        losses = {"aux_loss_total": loss}
        if any(key in used_target_keys for key in ("proprio", "robot_state", "q", "joint_pos", "joint_vel", "dq")):
            losses["proprio_pred_loss"] = loss
        if any(key in used_target_keys for key in ("gripper", "gripper_state")):
            losses["gripper_pred_loss"] = loss
        if "ee_pose" in used_target_keys:
            losses["ee_pose_pred_loss"] = loss
        return losses

    def save_checkpoint(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_type": "EGO_embodied_encoder",
                "action_dim": self.action_dim,
                "config": self.config.asdict(),
                "model_state_dict": self.state_dict(),
            },
            path,
        )


class EmbodiedAdapter(nn.Module):
    def __init__(self, base_embed_dim: int, self_embed_dim: int, output_dim: int, hidden_dim: int, adapter_type: str = "mlp"):
        super().__init__()
        input_dim = int(base_embed_dim + self_embed_dim)
        if adapter_type == "linear":
            self.net = nn.Linear(input_dim, int(output_dim))
        elif adapter_type == "mlp":
            self.net = build_mlp(input_dim, int(output_dim), int(hidden_dim), num_layers=1)
        else:
            raise ValueError("adapter_type must be 'linear' or 'mlp'")

    def forward(self, base_embed: torch.Tensor, e_self: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([base_embed, e_self], dim=-1))


def load_embodied_encoder_checkpoint(module: EmbodiedEncoder, path: str | Path, strict: bool = False):
    payload = torch.load(Path(path), map_location="cpu")
    state = payload.get("model_state_dict", payload)
    return module.load_state_dict(state, strict=strict)
