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


@dataclass(slots=True)
class DreamerV3WorldModelConfig:
    obs_dim: int | None = None
    obs_shape: tuple[int, ...] | None = None
    action_dim: int = 4
    encoder_type: str = "mlp"
    embed_dim: int = 128
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

        self.rssm = RSSM(
            RSSMConfig(
                action_dim=config.action_dim,
                embed_dim=config.embed_dim,
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
        self.grasp_head = (
            GraspHead(self.augmented_feat_dim, config.hidden_dim, config.num_layers) if config.predict_grasp else None
        )
        self.contact_head = (
            ClassHead(self.augmented_feat_dim, config.contact_num_classes, config.hidden_dim, config.num_layers)
            if config.contact_num_classes is not None
            else None
        )

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
        )

    def forward(self, batch: dict, loss_config: WorldModelLossConfig | None = None) -> dict:
        obs = batch["obs"]
        action = batch["action"]
        is_first = batch.get("is_first")
        embed = self._seq_apply(self.encoder, obs)
        post, prior = self.rssm.observe(embed, action, is_first)
        base_feat = self.rssm.get_feat(post)
        feature_outputs = self.build_features(base_feat, prev_context=None, is_first=is_first)
        feat = feature_outputs["augmented_feat"]
        outputs = {
            "embed": embed,
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
            "continue_logit": self._seq_apply(self.continue_head, feat),
            "grasp_logit": self._seq_apply(self.grasp_head, feat) if self.grasp_head is not None else None,
            "contact_logits": self._seq_apply(self.contact_head, feat) if self.contact_head is not None else None,
        }
        if isinstance(self.reward_head, TwoHotSymlogHead):
            outputs["reward_logits"] = self._seq_apply(self.reward_head, feat)
            outputs["reward_pred"] = self.reward_head.mean(feat.reshape(-1, feat.shape[-1])).reshape(feat.shape[:2])
        else:
            outputs["reward_pred"] = self._seq_apply(self.reward_head, feat)
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
            )
            outputs.update(event_outputs)
        if loss_config is not None:
            loss, metrics = world_model_loss(outputs, batch, loss_config)
            outputs["loss"] = loss
            outputs["metrics"] = metrics
        return outputs
