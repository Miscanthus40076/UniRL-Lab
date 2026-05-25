from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from .event_dynamics import DreamerEventDynamicsConfig, EventDynamicsModule, build_context_change_mask
from .thick_context import SlowContextModule, ThickContextConfig

from .decoder import CNNDecoder, MLPDecoder
from .encoder import CNNEncoder, MLPEncoder
from .heads import ClassHead, ContinueHead, GraspHead, RewardHead, TwoHotSymlogHead
from .losses import WorldModelLossConfig, world_model_loss
from .rssm import RSSM, RSSMConfig


INTEREST_EMA_DECAY = 0.99
INTEREST_EPS = 1e-6
INTEREST_TOP_RATIO = 0.25


@dataclass(slots=True)
class DreamerV3WorldModelConfig:
    obs_dim: int | None = None
    obs_shape: tuple[int, ...] | None = None
    action_dim: int = 4
    encoder_type: str = "mlp"
    embed_dim: int = 128
    proprio_input_dim: int = 4
    proprio_embed_dim: int = 32
    proprio_loss_weight: float = 1.0
    deter_dim: int = 128
    stoch_dim: int = 32
    stoch_classes: int = 32
    hidden_dim: int = 256
    num_layers: int = 2
    contact_num_classes: int | None = None
    predict_grasp: bool = False
    use_symlog_obs: bool = True
    use_symlog_reward: bool = True
    use_twohot_reward: bool = True
    reward_bins: int = 255
    reward_low: float = -20.0
    reward_high: float = 20.0
    rssm_unimix: float = 0.01
    thick_context: ThickContextConfig = field(default_factory=ThickContextConfig)
    event_dynamics: DreamerEventDynamicsConfig = field(default_factory=DreamerEventDynamicsConfig)

    @property
    def base_feat_dim(self) -> int:
        return int(self.deter_dim + self.stoch_dim * self.stoch_classes)

    @property
    def augmented_feat_dim(self) -> int:
        if not self.thick_context.enabled:
            return self.base_feat_dim
        return int(self.base_feat_dim + self.thick_context.context_dim)


class DreamerV3WorldModel(nn.Module):
    def __init__(self, config: DreamerV3WorldModelConfig):
        super().__init__()
        self.config = config
        if config.encoder_type == "cnn" or config.obs_shape is not None:
            if config.obs_shape is None or len(config.obs_shape) != 3:
                raise ValueError("CNN world model requires obs_shape=(C,H,W)")
            self.encoder = CNNEncoder(tuple(config.obs_shape), config.embed_dim, config.hidden_dim)
        else:
            if config.obs_dim is None:
                raise ValueError("MLP world model requires obs_dim")
            self.encoder = MLPEncoder(
                config.obs_dim,
                config.embed_dim,
                config.hidden_dim,
                config.num_layers,
                use_symlog=config.use_symlog_obs,
            )
        self.proprio_encoder = MLPEncoder(
            int(config.proprio_input_dim),
            int(config.proprio_embed_dim),
            config.hidden_dim,
            max(1, config.num_layers),
            use_symlog=False,
        )

        self.rssm = RSSM(
            RSSMConfig(
                action_dim=config.action_dim,
                embed_dim=config.embed_dim + config.proprio_embed_dim,
                deter_dim=config.deter_dim,
                stoch_dim=config.stoch_dim,
                stoch_classes=config.stoch_classes,
                hidden_dim=config.hidden_dim,
                unimix=config.rssm_unimix,
            )
        )
        self.base_feat_dim = config.base_feat_dim
        self.augmented_feat_dim = config.augmented_feat_dim
        self.slow_context = (
            SlowContextModule(self.base_feat_dim, config.thick_context) if config.thick_context.enabled else None
        )
        self.event_dynamics = (
            EventDynamicsModule(
                feat_dim=self.augmented_feat_dim,
                action_dim=config.action_dim,
                config=config.event_dynamics,
            )
            if config.event_dynamics.enabled
            else None
        )
        if config.encoder_type == "cnn" or config.obs_shape is not None:
            self.decoder = CNNDecoder(self.augmented_feat_dim, tuple(config.obs_shape), config.hidden_dim)
        else:
            self.decoder = MLPDecoder(
                self.augmented_feat_dim,
                int(config.obs_dim),
                config.hidden_dim,
                config.num_layers,
                use_symlog=config.use_symlog_obs,
            )
        self.reward_head = (
            TwoHotSymlogHead(
                self.augmented_feat_dim,
                config.hidden_dim,
                config.num_layers,
                num_bins=config.reward_bins,
                low=config.reward_low,
                high=config.reward_high,
            )
            if config.use_twohot_reward
            else RewardHead(self.augmented_feat_dim, config.hidden_dim, config.num_layers)
        )
        self.continue_head = ContinueHead(self.augmented_feat_dim, config.hidden_dim, config.num_layers)
        self.proprio_head = MLPDecoder(
            self.base_feat_dim,
            int(config.proprio_input_dim),
            config.hidden_dim,
            max(1, config.num_layers),
            use_symlog=False,
        )
        self.grasp_head = (
            GraspHead(self.augmented_feat_dim, config.hidden_dim, config.num_layers) if config.predict_grasp else None
        )
        self.contact_head = (
            ClassHead(self.augmented_feat_dim, config.contact_num_classes, config.hidden_dim, config.num_layers)
            if config.contact_num_classes is not None
            else None
        )
        self.register_buffer("interest_ema", torch.zeros(self.augmented_feat_dim, dtype=torch.float32))

    def _seq_apply(self, module: nn.Module, x):
        batch_size, seq_len = x.shape[:2]
        out = module(x.reshape(batch_size * seq_len, *x.shape[2:]))
        return out.reshape(batch_size, seq_len, *out.shape[1:])

    def initial_context(self, batch_size: int, device=None, dtype=None):
        if self.slow_context is None:
            return None
        return self.slow_context.initial_context(batch_size, device=device, dtype=dtype)

    def get_base_feat(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.rssm.get_feat(state)

    def concat_context(self, base_feat, context=None):
        if self.slow_context is None:
            return base_feat
        if context is None:
            context = self.initial_context(base_feat.shape[0], device=base_feat.device, dtype=base_feat.dtype)
        if base_feat.ndim == 3 and context.ndim == 2:
            context = context.unsqueeze(1).expand(-1, base_feat.shape[1], -1)
        return torch.cat([base_feat, context], dim=-1)

    def build_features(self, base_feat, prev_context=None, is_first=None) -> dict:
        if self.slow_context is None:
            return {
                "base_feat": base_feat,
                "context_gate_type": "sigmoid",
                "context": None,
                "context_gate": None,
                "context_gate_soft": None,
                "context_gate_hard": None,
                "context_gate_logit": None,
                "context_l0_open_prob": None,
                "context_candidate": None,
                "context_prev": None,
                "context_delta_norm": None,
                "augmented_feat": base_feat,
            }
        outputs = self.slow_context.forward_details(base_feat, prev_context=prev_context, is_first=is_first)
        return {
            "base_feat": base_feat,
            "context_gate_type": outputs["gate_type"],
            "context": outputs["context"],
            "context_gate": outputs["gate"],
            "context_gate_soft": outputs["gate_soft"],
            "context_gate_hard": outputs["gate_hard"],
            "context_gate_logit": outputs["gate_logit"],
            "context_l0_open_prob": outputs["open_prob"],
            "context_candidate": outputs["candidate_context"],
            "context_prev": outputs["prev_context"],
            "context_delta_norm": outputs["delta_norm"],
            "augmented_feat": outputs["augmented_feat"],
        }

    def get_augmented_feat(self, state: dict, prev_context=None, is_first=None, return_details: bool = False):
        base_feat = self.get_base_feat(state)
        details = self.build_features(base_feat, prev_context=prev_context, is_first=is_first)
        if return_details:
            return details
        return details["augmented_feat"]

    def predict_event_transition(
        self,
        feat,
        action,
        target_next_feat=None,
        context_gate_seq=None,
        context_delta_norm_seq=None,
        invalid_transition_seq=None,
        context_change_mask=None,
        valid_mask=None,
        apply_capacity_constraint: bool = True,
    ):
        if self.event_dynamics is None:
            return None
        if context_change_mask is None and self.config.event_dynamics.context_mask_enabled:
            context_change_mask = build_context_change_mask(
                context_gate_seq=context_gate_seq,
                context_delta_norm_seq=context_delta_norm_seq,
                mode=self.config.event_dynamics.context_mask_mode,
                top_percent=self.config.event_dynamics.context_mask_top_percent,
                window=self.config.event_dynamics.context_mask_window,
                detach=self.config.event_dynamics.context_mask_detach,
                source=self.config.event_dynamics.context_mask_source,
                invalid_transition_seq=invalid_transition_seq,
            )
        return self.event_dynamics.predict(
            feat,
            action=action,
            target_next_feat=target_next_feat,
            context_change_mask=context_change_mask,
            valid_mask=valid_mask,
            apply_capacity_constraint=apply_capacity_constraint,
        )

    def get_interest_weight(self, device=None, dtype=None) -> torch.Tensor:
        weight = self.interest_ema.detach()
        if device is not None:
            weight = weight.to(device=device)
        if dtype is not None:
            weight = weight.to(dtype=dtype)
        total = weight.sum()
        if float(total.detach().cpu()) <= INTEREST_EPS:
            weight = torch.ones_like(weight)
            total = weight.sum()
        return weight / total.clamp_min(INTEREST_EPS)

    def _apply_interest_top_ratio(self, weight: torch.Tensor) -> torch.Tensor:
        ratio = float(INTEREST_TOP_RATIO)
        if ratio <= 0.0 or ratio >= 1.0 or weight.numel() <= 1:
            return weight
        k = max(1, int(round(weight.numel() * ratio)))
        topk = torch.topk(weight, k=k)
        mask = torch.zeros_like(weight)
        mask[topk.indices] = 1.0
        filtered = weight * mask
        total = filtered.sum()
        if float(total.detach().cpu()) <= INTEREST_EPS:
            return weight
        return filtered / total.clamp_min(INTEREST_EPS)

    @torch.no_grad()
    def update_interest_ema_from_transition(self, transition_outputs: dict[str, torch.Tensor]):
        pred_fast = transition_outputs.get("pred_next_feat_ordinary_only")
        pred_teacher = transition_outputs.get("pred_next_feat_event_mixed")
        target = transition_outputs.get("target_next_feat")
        if pred_fast is None or pred_teacher is None or target is None:
            return
        fast_err_feat = (pred_fast - target).pow(2)
        teacher_err_feat = (pred_teacher - target).pow(2)
        positive_gain_feat = torch.relu(fast_err_feat - teacher_err_feat)
        residual_feat = pred_teacher - pred_fast
        residual_mag_feat = residual_feat.abs()
        interest_evidence_feat = residual_mag_feat.detach() * positive_gain_feat.detach()
        while interest_evidence_feat.ndim > 1:
            interest_evidence_feat = interest_evidence_feat.mean(dim=0)
        self.interest_ema.mul_(float(INTEREST_EMA_DECAY)).add_(
            interest_evidence_feat.to(dtype=self.interest_ema.dtype, device=self.interest_ema.device)
            * float(1.0 - INTEREST_EMA_DECAY)
        )

    def compute_interest_reward_from_transition(
        self,
        transition_outputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor] | None:
        pred_fast = transition_outputs.get("pred_next_feat_ordinary_only")
        pred_teacher = transition_outputs.get("pred_next_feat_event_mixed")
        target = transition_outputs.get("target_next_feat")
        if pred_fast is None or pred_teacher is None or target is None:
            return None
        fast_err_feat = (pred_fast - target).pow(2)
        teacher_err_feat = (pred_teacher - target).pow(2)
        gain_feat = fast_err_feat - teacher_err_feat
        positive_gain_feat = torch.relu(gain_feat)
        residual_feat = pred_teacher - pred_fast
        residual_mag_feat = residual_feat.abs()
        interest_evidence_feat = residual_mag_feat.detach() * positive_gain_feat.detach()
        interest_weight = self._apply_interest_top_ratio(
            self.get_interest_weight(device=pred_fast.device, dtype=pred_fast.dtype)
        ).detach()
        numerator = (positive_gain_feat.detach() * interest_weight).sum(dim=-1)
        denominator = (fast_err_feat.detach() * interest_weight).sum(dim=-1).clamp_min(INTEREST_EPS)
        interest_slow_gain_reward = torch.clamp(numerator / denominator, min=0.0, max=1.0).detach()
        raw_numerator = positive_gain_feat.detach().sum(dim=-1)
        raw_denominator = fast_err_feat.detach().sum(dim=-1).clamp_min(INTEREST_EPS)
        raw_slow_gain_reward = torch.clamp(raw_numerator / raw_denominator, min=0.0, max=1.0).detach()
        local_slow_gain = (fast_err_feat - teacher_err_feat).mean(dim=-1).detach()
        return {
            "fast_err_feat": fast_err_feat.detach(),
            "teacher_err_feat": teacher_err_feat.detach(),
            "positive_gain_feat": positive_gain_feat.detach(),
            "residual_mag_feat": residual_mag_feat.detach(),
            "interest_evidence_feat": interest_evidence_feat.detach(),
            "interest_weight": interest_weight.detach(),
            "local_slow_gain": local_slow_gain,
            "raw_slow_gain_reward": raw_slow_gain_reward,
            "interest_slow_gain_reward": interest_slow_gain_reward,
        }

    def forward(self, batch: dict, loss_config: WorldModelLossConfig | None = None) -> dict:
        obs = batch["obs"]
        action = batch["action"]
        is_first = batch.get("is_first")
        embed = self._seq_apply(self.encoder, obs)
        proprio = batch.get("proprio")
        if proprio is None:
            proprio = torch.zeros(
                (*embed.shape[:2], int(self.config.proprio_input_dim)),
                dtype=embed.dtype,
                device=embed.device,
            )
        proprio = proprio.float()
        proprio_embed = self._seq_apply(self.proprio_encoder, proprio)
        combined_embed = torch.cat([embed, proprio_embed], dim=-1)
        post, prior = self.rssm.observe(combined_embed, action, is_first)
        base_feat = self.rssm.get_feat(post)
        feature_outputs = self.build_features(base_feat, prev_context=None, is_first=is_first)
        feat = feature_outputs["augmented_feat"]
        outputs = {
            "embed": embed,
            "proprio": proprio,
            "proprio_embed": proprio_embed,
            "combined_embed": combined_embed,
            "post": post,
            "prior": prior,
            "base_feat": base_feat,
            "feat": feat,
            "augmented_feat": feat,
            "context_gate_type": feature_outputs["context_gate_type"],
            "context": feature_outputs["context"],
            "context_gate": feature_outputs["context_gate"],
            "context_gate_soft": feature_outputs["context_gate_soft"],
            "context_gate_hard": feature_outputs["context_gate_hard"],
            "context_gate_logit": feature_outputs["context_gate_logit"],
            "context_l0_open_prob": feature_outputs["context_l0_open_prob"],
            "context_candidate": feature_outputs["context_candidate"],
            "context_prev": feature_outputs["context_prev"],
            "context_delta_norm": feature_outputs["context_delta_norm"],
            "obs_pred": self._seq_apply(self.decoder, feat),
            "proprio_pred": self._seq_apply(self.proprio_head, base_feat),
            "continue_logit": self._seq_apply(self.continue_head, feat),
            "grasp_logit": self._seq_apply(self.grasp_head, feat) if self.grasp_head is not None else None,
            "contact_logits": self._seq_apply(self.contact_head, feat) if self.contact_head is not None else None,
        }
        if isinstance(self.reward_head, TwoHotSymlogHead):
            outputs["reward_logits"] = self._seq_apply(self.reward_head, feat)
            outputs["reward_pred"] = self.reward_head.mean(feat.reshape(-1, feat.shape[-1])).reshape(feat.shape[:2])
        else:
            outputs["reward_pred"] = self._seq_apply(self.reward_head, feat)
        if self.slow_context is not None:
            fast_only_feat = self.concat_context(base_feat, context=None)
            outputs["fast_only_feat"] = fast_only_feat
            outputs["fast_only_obs_pred"] = self._seq_apply(self.decoder, fast_only_feat)
            outputs["fast_only_continue_logit"] = self._seq_apply(self.continue_head, fast_only_feat)
            if isinstance(self.reward_head, TwoHotSymlogHead):
                outputs["fast_only_reward_logits"] = self._seq_apply(self.reward_head, fast_only_feat)
                outputs["fast_only_reward_pred"] = self.reward_head.mean(
                    fast_only_feat.reshape(-1, fast_only_feat.shape[-1])
                ).reshape(fast_only_feat.shape[:2])
            else:
                outputs["fast_only_reward_pred"] = self._seq_apply(self.reward_head, fast_only_feat)
        if self.event_dynamics is not None and feat.shape[1] > 1:
            valid_mask = None
            if is_first is not None and self.config.event_dynamics.capacity_enabled and self.config.event_dynamics.capacity_use_valid_mask:
                valid_mask = (1.0 - is_first[:, 1:].float()).unsqueeze(-1)
            event_outputs = self.predict_event_transition(
                feat[:, :-1],
                action[:, :-1],
                target_next_feat=feat[:, 1:],
                context_gate_seq=feature_outputs["context_gate"][:, :-1]
                if feature_outputs["context_gate"] is not None
                else None,
                context_delta_norm_seq=feature_outputs["context_delta_norm"][:, :-1]
                if feature_outputs["context_delta_norm"] is not None
                else None,
                invalid_transition_seq=is_first[:, 1:] if is_first is not None else None,
                valid_mask=valid_mask,
                apply_capacity_constraint=False,
            )
            self.update_interest_ema_from_transition(event_outputs)
            outputs.update(event_outputs)
            outputs["interest_reward_components"] = self.compute_interest_reward_from_transition(event_outputs)
        if loss_config is not None:
            loss, metrics = world_model_loss(outputs, batch, loss_config)
            outputs["loss"] = loss
            outputs["metrics"] = metrics
        return outputs
