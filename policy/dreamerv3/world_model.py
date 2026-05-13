from __future__ import annotations

from dataclasses import dataclass

from torch import nn

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
        feat_dim = config.deter_dim + config.stoch_dim * config.stoch_classes
        if config.encoder_type == "cnn" or config.obs_shape is not None:
            self.decoder = CNNDecoder(feat_dim, tuple(config.obs_shape), config.hidden_dim)
        else:
            self.decoder = MLPDecoder(
                feat_dim,
                int(config.obs_dim),
                config.hidden_dim,
                config.num_layers,
                use_symlog=config.use_symlog_obs,
            )
        self.reward_head = (
            TwoHotSymlogHead(
                feat_dim,
                config.hidden_dim,
                config.num_layers,
                num_bins=config.reward_bins,
                low=config.reward_low,
                high=config.reward_high,
            )
            if config.use_twohot_reward
            else RewardHead(feat_dim, config.hidden_dim, config.num_layers)
        )
        self.continue_head = ContinueHead(feat_dim, config.hidden_dim, config.num_layers)
        self.grasp_head = GraspHead(feat_dim, config.hidden_dim, config.num_layers) if config.predict_grasp else None
        self.contact_head = (
            ClassHead(feat_dim, config.contact_num_classes, config.hidden_dim, config.num_layers)
            if config.contact_num_classes is not None
            else None
        )

    def _seq_apply(self, module: nn.Module, x):
        batch_size, seq_len = x.shape[:2]
        out = module(x.reshape(batch_size * seq_len, *x.shape[2:]))
        return out.reshape(batch_size, seq_len, *out.shape[1:])

    def forward(self, batch: dict, loss_config: WorldModelLossConfig | None = None) -> dict:
        obs = batch["obs"]
        action = batch["action"]
        is_first = batch.get("is_first")
        embed = self._seq_apply(self.encoder, obs)
        post, prior = self.rssm.observe(embed, action, is_first)
        feat = self.rssm.get_feat(post)
        outputs = {
            "embed": embed,
            "post": post,
            "prior": prior,
            "feat": feat,
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
        if loss_config is not None:
            loss, metrics = world_model_loss(outputs, batch, loss_config)
            outputs["loss"] = loss
            outputs["metrics"] = metrics
        return outputs
