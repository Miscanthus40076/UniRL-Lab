from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from policy.dreamerv3.decoder import CNNDecoder, MLPDecoder
from policy.dreamerv3.encoder import CNNEncoder, MLPEncoder
from policy.dreamerv3.heads import ClassHead, ContinueHead, GraspHead, RewardHead, TwoHotSymlogHead
from policy.dreamerv3.losses import WorldModelLossConfig, categorical_kl, _reduce_except_batch
from policy.dreamerv3.rssm import RSSM, RSSMConfig
from policy.dreamerv3.transforms import symlog, twohot_logprob


@dataclass(slots=True)
class DirectionWorldModelConfig:
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
    free_nats: float = 1.0
    kl_balance: float = 0.8
    rssm_unimix: float = 0.01
    separate_continue_heads: bool = False


def _head_apply(module: nn.Module | None, x: torch.Tensor) -> torch.Tensor | None:
    if module is None:
        return None
    batch_size, seq_len = x.shape[:2]
    out = module(x.reshape(batch_size * seq_len, *x.shape[2:]))
    return out.reshape(batch_size, seq_len, *out.shape[1:])


class DirectionConditionedWorldModel(nn.Module):
    def __init__(self, config: DirectionWorldModelConfig):
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
        self.feat_dim = config.deter_dim + config.stoch_dim * config.stoch_classes
        if config.encoder_type == "cnn" or config.obs_shape is not None:
            self.decoder = CNNDecoder(self.feat_dim, tuple(config.obs_shape), config.hidden_dim)
        else:
            self.decoder = MLPDecoder(
                self.feat_dim,
                int(config.obs_dim),
                config.hidden_dim,
                config.num_layers,
                use_symlog=config.use_symlog_obs,
            )
        self.forward_reward_head = self._make_reward_head()
        self.reverse_reward_head = self._make_reward_head()
        self.continue_head = None if config.separate_continue_heads else ContinueHead(
            self.feat_dim, config.hidden_dim, config.num_layers
        )
        self.forward_continue_head = ContinueHead(
            self.feat_dim, config.hidden_dim, config.num_layers
        ) if config.separate_continue_heads else None
        self.reverse_continue_head = ContinueHead(
            self.feat_dim, config.hidden_dim, config.num_layers
        ) if config.separate_continue_heads else None
        self.grasp_head = GraspHead(self.feat_dim, config.hidden_dim, config.num_layers) if config.predict_grasp else None
        self.contact_head = (
            ClassHead(self.feat_dim, config.contact_num_classes, config.hidden_dim, config.num_layers)
            if config.contact_num_classes is not None
            else None
        )

    def _make_reward_head(self) -> nn.Module:
        if self.config.use_twohot_reward:
            return TwoHotSymlogHead(
                self.feat_dim,
                self.config.hidden_dim,
                self.config.num_layers,
                num_bins=self.config.reward_bins,
                low=self.config.reward_low,
                high=self.config.reward_high,
            )
        return RewardHead(self.feat_dim, self.config.hidden_dim, self.config.num_layers)

    def _reward_outputs(self, feat: torch.Tensor, direction: str) -> dict[str, torch.Tensor]:
        reward_head = self.forward_reward_head if direction == "forward" else self.reverse_reward_head
        if isinstance(reward_head, TwoHotSymlogHead):
            logits = _head_apply(reward_head, feat)
            pred = reward_head.mean(feat.reshape(-1, feat.shape[-1])).reshape(feat.shape[:2])
            return {"reward_logits": logits, "reward_pred": pred}
        return {"reward_pred": _head_apply(reward_head, feat)}

    def reward_prediction(self, feat: torch.Tensor, direction: str) -> torch.Tensor:
        reward_head = self.forward_reward_head if direction == "forward" else self.reverse_reward_head
        if isinstance(reward_head, TwoHotSymlogHead):
            return reward_head.mean(feat)
        return reward_head(feat)

    def _continue_output(self, feat: torch.Tensor, direction: str) -> torch.Tensor:
        if self.config.separate_continue_heads:
            head = self.forward_continue_head if direction == "forward" else self.reverse_continue_head
            return _head_apply(head, feat)
        return _head_apply(self.continue_head, feat)

    def continue_prediction(self, feat: torch.Tensor, direction: str) -> torch.Tensor:
        if self.config.separate_continue_heads:
            head = self.forward_continue_head if direction == "forward" else self.reverse_continue_head
            return head(feat)
        return self.continue_head(feat)

    def posterior_outputs(self, batch: dict, direction: str) -> dict[str, torch.Tensor]:
        if direction not in {"forward", "reverse"}:
            raise ValueError(f"Unsupported direction: {direction}")
        obs = batch["obs"]
        action = batch["action"]
        is_first = batch.get("is_first")
        batch_size, seq_len = obs.shape[:2]
        embed = self.encoder(obs.reshape(batch_size * seq_len, *obs.shape[2:])).reshape(batch_size, seq_len, -1)
        post, prior = self.rssm.observe(embed, action, is_first)
        feat = self.rssm.get_feat(post)
        outputs = {
            "embed": embed,
            "post": post,
            "prior": prior,
            "feat": feat,
            "obs_pred": _head_apply(self.decoder, feat),
            "continue_logit": self._continue_output(feat, direction),
            "grasp_logit": _head_apply(self.grasp_head, feat) if self.grasp_head is not None else None,
            "contact_logits": _head_apply(self.contact_head, feat) if self.contact_head is not None else None,
        }
        outputs.update(self._reward_outputs(feat, direction))
        return outputs

    def extract_feat(self, state: dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
        z = state["probs"] if deterministic else state["z"]
        z = z.reshape(*z.shape[:-2], -1)
        return torch.cat([state["h"], z], dim=-1)

    def rollout_open_loop(self, init_state: dict[str, torch.Tensor], action_seq: torch.Tensor) -> dict[str, torch.Tensor]:
        imagined = self.rssm.imagine(action_seq, init_state)
        feat = self.rssm.get_feat(imagined)
        return {
            "state": imagined,
            "feat": feat,
            "obs_pred": _head_apply(self.decoder, feat),
        }

    def _loss_config_from_batch(self, batch: dict) -> WorldModelLossConfig:
        return WorldModelLossConfig(
            free_nats=float(batch.get("_free_nats", self.config.free_nats)),
            recon_scale=float(batch.get("_recon_scale", 1.0)),
            reward_scale=float(batch.get("_reward_scale", 1.0)),
            continue_scale=float(batch.get("_continue_scale", 1.0)),
            dynamics_scale=float(batch.get("_dynamics_scale", 1.0)),
            representation_scale=float(batch.get("_representation_scale", 0.1)),
            grasp_scale=float(batch.get("_grasp_scale", 0.2)),
            contact_scale=float(batch.get("_contact_scale", 0.2)),
            kl_balance=float(batch.get("_kl_balance", self.config.kl_balance)),
            use_symlog_obs=self.config.use_symlog_obs,
            use_symlog_reward=self.config.use_symlog_reward,
            use_twohot_reward=self.config.use_twohot_reward,
            twohot_bins=self.config.reward_bins,
            twohot_low=self.config.reward_low,
            twohot_high=self.config.reward_high,
        )

    def _loss(self, batch: dict, direction: str) -> dict[str, torch.Tensor]:
        outputs = self.posterior_outputs(batch, direction)
        config = self._loss_config_from_batch(batch)
        obs = batch["obs"].float()
        reward = batch["reward"].float()
        done = batch["done"].float()
        continue_target = 1.0 - done
        obs_target = obs / 255.0 if obs.ndim >= 4 and obs.max() > 2.0 else obs
        if obs.ndim < 4 and config.use_symlog_obs:
            obs_target = symlog(obs_target)

        recon_error = (outputs["obs_pred"] - obs_target).pow(2)
        recon_loss_per_seq = _reduce_except_batch(recon_error)
        recon_loss = recon_loss_per_seq.mean()

        reward_target = symlog(reward) if config.use_symlog_reward else reward
        if config.use_twohot_reward:
            reward_logprob = twohot_logprob(
                outputs["reward_logits"],
                reward_target,
                num_bins=config.twohot_bins,
                low=config.twohot_low,
                high=config.twohot_high,
            )
            reward_loss_per_seq = -_reduce_except_batch(reward_logprob)
            reward_loss = reward_loss_per_seq.mean()
        else:
            reward_error = (outputs["reward_pred"] - reward_target).pow(2)
            reward_loss_per_seq = _reduce_except_batch(reward_error)
            reward_loss = reward_loss_per_seq.mean()

        continue_loss_raw = F.binary_cross_entropy_with_logits(
            outputs["continue_logit"],
            continue_target,
            reduction="none",
        )
        continue_loss_per_seq = _reduce_except_batch(continue_loss_raw)
        continue_loss = continue_loss_per_seq.mean()

        post_sg = {key: value.detach() for key, value in outputs["post"].items()}
        prior_sg = {key: value.detach() for key, value in outputs["prior"].items()}
        dyn_kl_per_seq = torch.clamp(categorical_kl(post_sg, outputs["prior"]), min=float(config.free_nats)).mean(dim=1)
        rep_kl_per_seq = torch.clamp(categorical_kl(outputs["post"], prior_sg), min=float(config.free_nats)).mean(dim=1)
        dyn_kl = dyn_kl_per_seq.mean()
        rep_kl = rep_kl_per_seq.mean()
        kl_loss = config.kl_balance * dyn_kl + (1.0 - config.kl_balance) * rep_kl

        total_per_seq = (
            config.recon_scale * recon_loss_per_seq
            + config.reward_scale * reward_loss_per_seq
            + config.continue_scale * continue_loss_per_seq
            + config.dynamics_scale * dyn_kl_per_seq
            + config.representation_scale * rep_kl_per_seq
        )
        total = total_per_seq.mean()

        metrics = {
            "total_loss": total,
            "recon_loss": recon_loss,
            "reward_loss": reward_loss,
            "continue_loss": continue_loss,
            "kl_loss": kl_loss,
            "dyn_kl_loss": dyn_kl,
            "rep_kl_loss": rep_kl,
        }

        if outputs.get("grasp_logit") is not None and "is_grasping" in batch:
            grasp_loss = F.binary_cross_entropy_with_logits(outputs["grasp_logit"], batch["is_grasping"].float())
            total = total + config.grasp_scale * grasp_loss
            metrics["total_loss"] = total
            metrics["grasp_loss"] = grasp_loss
        if outputs.get("contact_logits") is not None and "contact_mode" in batch:
            contact_loss = F.cross_entropy(
                outputs["contact_logits"].reshape(-1, outputs["contact_logits"].shape[-1]),
                batch["contact_mode"].long().reshape(-1),
            )
            total = total + config.contact_scale * contact_loss
            metrics["total_loss"] = total
            metrics["contact_loss"] = contact_loss

        metrics["priority"] = total_per_seq.detach()
        metrics["direction"] = direction
        metrics["outputs"] = outputs
        return metrics

    def forward_loss(self, batch: dict) -> dict[str, torch.Tensor]:
        return self._loss(batch, "forward")

    def reverse_loss(self, batch: dict) -> dict[str, torch.Tensor]:
        return self._loss(batch, "reverse")

    def mixed_loss(self, forward_batch: dict, reverse_batch: dict) -> dict[str, torch.Tensor]:
        forward_metrics = self.forward_loss(forward_batch)
        reverse_metrics = self.reverse_loss(reverse_batch)
        forward_scale = float(forward_batch.get("_forward_loss_scale", 1.0))
        reverse_scale = float(reverse_batch.get("_reverse_loss_scale", 1.0))
        mixed_total = forward_scale * forward_metrics["total_loss"] + reverse_scale * reverse_metrics["total_loss"]
        mixed = {
            "mixed_total_loss": mixed_total,
            "forward": forward_metrics,
            "reverse": reverse_metrics,
        }
        for prefix, payload in (("forward", forward_metrics), ("reverse", reverse_metrics)):
            for key, value in payload.items():
                if key in {"direction", "outputs", "priority"}:
                    continue
                mixed[f"{prefix}_{key}"] = value
        return mixed
