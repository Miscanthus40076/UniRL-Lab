from __future__ import annotations

from dataclasses import asdict

import torch
import torch.nn.functional as F

from policy.versions.v1.dreamerv3.operator_tokens import (
    bad_numeric_count,
    compute_token_usage_metrics,
    masked_mean,
    random_accuracy_baseline,
    safe_relative_improvement,
)

from .config import OperatorTokenProbePolicyConfig
from .operator_token_probe_model import OperatorTokenProbeModel


class OperatorTokenProbeAgent:
    def __init__(self, feat_dim: int, action_dim: int, config: OperatorTokenProbePolicyConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = OperatorTokenProbeModel(feat_dim=feat_dim, action_dim=action_dim, config=config.operator_token).to(
            self.device
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

    def _capacity_mask(self, batch: dict) -> torch.Tensor | None:
        gate = batch.get("capacity_gate")
        if gate is None:
            return None
        mask = gate.float()
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        if self.config.operator_token.train_on_capacity_only:
            mask = (mask > 0.0).float()
        return mask

    def _compute_losses(self, batch: dict, outputs: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mask = self._capacity_mask(batch)
        if self.config.operator_token.train_on_capacity_only and (
            mask is None or torch.count_nonzero(mask) <= 0
        ):
            raise ValueError("OperatorTokenProbe found no active capacity transitions in the batch")

        delta_x = batch["delta_x"].detach()
        action_t = batch["action_t"].detach()
        effect_token_per = F.smooth_l1_loss(outputs["pred_delta_x"], delta_x, reduction="none").mean(dim=-1)
        effect_no_token_per = F.smooth_l1_loss(outputs["pred_delta_x_no_token"], delta_x, reduction="none").mean(dim=-1)
        inverse_per = F.smooth_l1_loss(outputs["pred_action"], action_t, reduction="none").mean(dim=-1)
        inverse_no_token_per = F.smooth_l1_loss(outputs["pred_action_no_token"], action_t, reduction="none").mean(dim=-1)
        prior_per = F.cross_entropy(
            outputs["prior_logits"].reshape(-1, outputs["prior_logits"].shape[-1]),
            outputs["token_id"].detach().reshape(-1),
            reduction="none",
        ).reshape_as(effect_token_per)
        effect_token = masked_mean(effect_token_per, mask)
        effect_no_token = masked_mean(effect_no_token_per, mask)
        inverse_loss = masked_mean(inverse_per, mask)
        inverse_no_token = masked_mean(inverse_no_token_per, mask)
        prior_loss = masked_mean(prior_per, mask)
        usage = compute_token_usage_metrics(outputs["token_probs"], mask)
        usage_loss = -self.config.operator_token.usage_entropy_scale * usage["token_entropy"]
        total = (
            self.config.operator_token.effect_loss_scale * effect_token
            + self.config.operator_token.inverse_loss_scale * inverse_loss
            + self.config.operator_token.prior_loss_scale * prior_loss
            + usage_loss
        )
        prior_pred = torch.argmax(outputs["prior_logits"], dim=-1)
        prior_acc = ((prior_pred == outputs["token_id"].detach()).float() * (mask if mask is not None else 1.0)).sum()
        prior_acc = prior_acc / torch.clamp((mask.sum() if mask is not None else torch.tensor(prior_pred.numel(), device=self.device)), min=1.0)
        metrics = {
            "total_loss": total,
            "effect_loss_token": effect_token,
            "effect_loss_no_token": effect_no_token,
            "inverse_action_loss": inverse_loss,
            "inverse_action_loss_no_token": inverse_no_token,
            "prior_loss": prior_loss,
            "usage_entropy_loss": usage_loss,
            "token_effect_improvement": safe_relative_improvement(effect_no_token, effect_token),
            "inverse_action_improvement": safe_relative_improvement(inverse_no_token, inverse_loss),
            "prior_token_accuracy": prior_acc,
            "random_token_accuracy": torch.tensor(
                random_accuracy_baseline(self.config.operator_token.num_tokens), device=self.device
            ),
            "capacity_transition_count": (
                mask.sum() if mask is not None else torch.tensor(effect_token_per.numel(), device=self.device)
            ),
            **usage,
        }
        return total, metrics

    def update(self, batch: dict) -> tuple[dict[str, float], dict[str, object]]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.model(batch)
        total, metrics = self._compute_losses(batch, outputs)
        total.backward()
        self.optimizer.step()
        scalars = {
            key: float(value.detach().cpu())
            for key, value in metrics.items()
            if torch.is_tensor(value) and value.ndim == 0
        }
        scalars["bad_numeric_count"] = float(
            bad_numeric_count(
                total,
                outputs["token_logits"],
                outputs["token_probs"],
                outputs["pred_delta_x"],
                outputs["pred_action"],
            )
        )
        metadata = {
            "token_usage": metrics["token_usage"].detach().cpu().tolist(),
            "num_tokens": int(self.config.operator_token.num_tokens),
        }
        return scalars, metadata

    @torch.no_grad()
    def evaluate(self, batch: dict) -> dict[str, float | list[float]]:
        self.model.eval()
        outputs = self.model(batch)
        _, metrics = self._compute_losses(batch, outputs)
        result = {
            key: float(value.detach().cpu())
            for key, value in metrics.items()
            if torch.is_tensor(value) and value.ndim == 0
        }
        result["token_usage"] = metrics["token_usage"].detach().cpu().tolist()
        result["bad_numeric_count"] = float(
            bad_numeric_count(
                outputs["token_logits"],
                outputs["token_probs"],
                outputs["pred_delta_x"],
                outputs["pred_action"],
            )
        )
        result["token_id"] = outputs["token_id"].detach().cpu().tolist()
        result["effect_loss_per_step"] = (
            F.smooth_l1_loss(outputs["pred_delta_x"], batch["delta_x"].detach(), reduction="none")
            .mean(dim=-1)
            .detach()
            .cpu()
            .tolist()
        )
        result["inverse_loss_per_step"] = (
            F.smooth_l1_loss(outputs["pred_action"], batch["action_t"].detach(), reduction="none")
            .mean(dim=-1)
            .detach()
            .cpu()
            .tolist()
        )
        return result

    def state_dict(self) -> dict:
        return {
            "config": self.config.asdict(),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

    def load_state_dict(self, payload: dict):
        self.model.load_state_dict(payload["model_state_dict"])
        if "optimizer_state_dict" in payload:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
