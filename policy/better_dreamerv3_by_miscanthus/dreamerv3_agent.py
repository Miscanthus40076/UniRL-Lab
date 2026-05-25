from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from .action_penalty import compute_action_penalty
from .actor import DreamerActor, DreamerActorConfig
from .event_dynamics import DreamerEventDynamicsConfig, EventCapacitySelector
from .dreamerv3_model import DreamerAuxConfig, DreamerObservationSpec, DreamerV3ModelConfig
from .imagination import imagine_rollout
from .lambda_return import lambda_return
from .losses import WorldModelLossConfig
from .heads import TwoHotSymlogHead
from .normalization import RunningNormConfig, RunningNormalizer
from .operator_intrinsic_reward import OperatorIntrinsicReward, OperatorIntrinsicRewardConfig, combine_train_reward
from .thick_context import ThickContextConfig
from .transforms import symlog, symexp, twohot_logprob, twohot_mean
from .value import DreamerValue, DreamerValueConfig
from .world_model import DreamerV3WorldModel, DreamerV3WorldModelConfig

if TYPE_CHECKING:
    from .replay_buffer import EpisodeReplayBuffer


class DreamerV3Agent:
    def __init__(self, config: DreamerV3ModelConfig, loss_config: WorldModelLossConfig | None = None):
        self.config = config
        self.loss_config = loss_config or WorldModelLossConfig(
            free_nats=config.free_nats,
            kl_balance=config.kl_balance,
            use_symlog_obs=config.use_symlog_obs,
            use_symlog_reward=config.use_symlog_reward,
            use_twohot_reward=config.use_twohot_reward,
            twohot_bins=config.twohot_bins,
            twohot_low=config.twohot_low,
            twohot_high=config.twohot_high,
            context_update_penalty=(config.thick_context.update_penalty if config.thick_context.enabled else 0.0),
            event_penalty=(config.event_dynamics.event_penalty if config.event_dynamics.enabled else 0.0),
            event_prediction_scale=(
                config.event_dynamics.prediction_loss_scale if config.event_dynamics.enabled else 0.0
            ),
            event_sparsity_inside_mask_only=(
                config.event_dynamics.event_sparsity_inside_mask_only if config.event_dynamics.enabled else False
            ),
            event_sparsity_inside_capacity_only=(
                config.event_dynamics.event_sparsity_inside_capacity_only if config.event_dynamics.enabled else False
            ),
        )
        self.device = torch.device(config.device)
        self.actor_return_scale = 1.0
        self.actor_return_scale_initialized = False

        wm_config = DreamerV3WorldModelConfig(
            obs_dim=config.observation.obs_dim,
            obs_shape=config.observation.obs_shape,
            action_dim=config.action_dim,
            encoder_type=config.encoder_type,
            embed_dim=config.embed_dim,
            deter_dim=config.deter_dim,
            stoch_dim=config.stoch_dim,
            stoch_classes=config.stoch_classes,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            contact_num_classes=config.aux.contact_num_classes,
            predict_grasp=config.aux.predict_grasp,
            use_symlog_obs=config.use_symlog_obs,
            use_symlog_reward=config.use_symlog_reward,
            use_twohot_reward=config.use_twohot_reward,
            reward_bins=config.twohot_bins,
            reward_low=config.twohot_low,
            reward_high=config.twohot_high,
            rssm_unimix=config.rssm_unimix,
            thick_context=config.thick_context,
            event_dynamics=config.event_dynamics,
        )
        self.world_model = DreamerV3WorldModel(wm_config).to(self.device)
        feat_dim = config.augmented_feat_dim
        self.actor = DreamerActor(
            DreamerActorConfig(
                feat_dim=feat_dim,
                action_dim=config.action_dim,
                hidden_dim=config.actor_hidden_dim,
                num_layers=config.actor_num_layers,
                min_std=config.min_std,
                max_std=config.max_std,
                init_std=config.init_std,
            )
        ).to(self.device)
        self.value = self._make_value_model(feat_dim).to(self.device)
        self.slow_value = self._make_value_model(feat_dim).to(self.device) if config.use_slow_value else None
        if self.slow_value is not None:
            self.slow_value.load_state_dict(self.value.state_dict())
            for param in self.slow_value.parameters():
                param.requires_grad_(False)
        self.world_model_optimizer = torch.optim.Adam(
            self.world_model.parameters(),
            lr=config.world_model_lr,
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=config.value_lr)
        self.latent_state = self.world_model.rssm.init_state(1, self.device)
        self.latent_context = self.world_model.initial_context(1, self.device)
        norm_cfg = RunningNormConfig(rate=config.norm_rate, eps=config.norm_eps)
        self.return_normalizer = RunningNormalizer(norm_cfg)
        self.advantage_normalizer = RunningNormalizer(norm_cfg)
        self.operator_intrinsic_reward = (
            OperatorIntrinsicReward(
                feat_dim=feat_dim,
                action_dim=config.action_dim,
                config=config.operator_intrinsic_reward,
            ).to(self.device)
            if config.operator_intrinsic_reward.enabled
            else None
        )
        self.last_context_metrics = self._default_context_metrics()
        self.last_event_metrics = self._default_event_metrics()
        self._pending_event_transition = None
        self._operator_eval_event_logits: list[float] = []
        self._online_batch_size = 16
        self._online_seq_len = 32
        self._online_warmup_steps = 1000
        self._online_train_every = 1
        self._online_train_ratio = 1.0
        self._online_max_updates_per_step = 0
        self._online_train_calls = 0
        self._online_train_budget = 0.0

    def _default_context_metrics(self) -> dict[str, float | str]:
        if not self.config.thick_context.enabled:
            return {}
        return {
            "context_gate_type": self.config.thick_context.gate_type,
            "context_gate": 0.0,
            "context_l0_open_prob": 0.0,
            "context_gate_hard": 0.0,
            "context_gate_soft": 0.0,
            "context_gate_logit": 0.0,
            "context_gate_logit_std": 0.0,
            "context_update_loss_raw": 0.0,
            "context_update_loss_scaled": 0.0,
            "context_delta_norm": 0.0,
            "context_norm": 0.0,
        }

    def _context_metrics_from_details(self, details: dict | None) -> dict[str, float | str]:
        if not self.config.thick_context.enabled or not details or details.get("context") is None:
            return {}
        context = details["context"]
        gate = details["context_gate"]
        delta_norm = details["context_delta_norm"]
        open_prob = details.get("context_l0_open_prob")
        gate_hard = details.get("context_gate_hard")
        gate_soft = details.get("context_gate_soft")
        gate_logit = details.get("context_gate_logit")
        update_raw = open_prob if open_prob is not None else gate
        return {
            "context_gate_type": str(details.get("context_gate_type", self.config.thick_context.gate_type)),
            "context_gate": float(gate.detach().mean().cpu()),
            "context_l0_open_prob": float(update_raw.detach().mean().cpu()),
            "context_gate_hard": float((gate_hard if gate_hard is not None else gate).detach().mean().cpu()),
            "context_gate_soft": float((gate_soft if gate_soft is not None else gate).detach().mean().cpu()),
            "context_gate_logit": float((gate_logit if gate_logit is not None else torch.zeros_like(gate)).detach().mean().cpu()),
            "context_gate_logit_std": float(
                (gate_logit if gate_logit is not None else torch.zeros_like(gate)).detach().float().std(unbiased=False).cpu()
            ),
            "context_update_loss_raw": float(update_raw.detach().mean().cpu()),
            "context_update_loss_scaled": float(
                self.config.thick_context.update_penalty * update_raw.detach().mean().cpu()
            ),
            "context_delta_norm": float(delta_norm.detach().mean().cpu()),
            "context_norm": float(torch.linalg.vector_norm(context.detach(), dim=-1).mean().cpu()),
        }

    def get_context_diagnostics(self) -> dict[str, float | str]:
        return dict(self.last_context_metrics)

    def _default_event_metrics(self) -> dict[str, float | str | None]:
        if not self.config.event_dynamics.enabled:
            return {}
        metrics = {
            "event_gate": None,
            "event_logit": None,
            "event_logit_sigmoid": None,
            "event_capacity_gate": None,
            "event_capacity_enabled": float(self.config.event_dynamics.capacity_enabled),
            "event_capacity_ratio": float(self.config.event_dynamics.capacity_ratio),
            "event_capacity_k": None,
            "event_capacity_mode": self.config.event_dynamics.capacity_mode,
            "capacity_v2_detach_event_input": float(self.config.event_dynamics.capacity_v2_detach_event_input),
            "capacity_v2_detach_event_target": float(self.config.event_dynamics.capacity_v2_detach_event_target),
            "capacity_v2_use_sigmoid_gate_multiplier": float(
                self.config.event_dynamics.capacity_v2_use_sigmoid_gate_multiplier
            ),
            "event_loss_updates_backbone": float(
                not (
                    self.config.event_dynamics.capacity_enabled
                    and self.config.event_dynamics.capacity_v2_freeze_backbone_for_event_loss
                )
            ),
            "event_prediction_error": None,
            "pred_next_feat_error_ordinary_only": None,
            "pred_next_feat_error_event_mixed": None,
            "context_change_mask": None,
            "context_change_mask_source": self.config.event_dynamics.context_mask_source,
            "context_change_mask_mode": self.config.event_dynamics.context_mask_mode,
            "context_change_mask_top_percent": float(self.config.event_dynamics.context_mask_top_percent),
            "event_residual_norm": None,
        }
        if self.operator_intrinsic_reward is not None:
            metrics.update(
                {
                    "operator_reward": None,
                    "operator_token_id": None,
                    "operator_token_novelty": None,
                    "operator_rarity": None,
                    "operator_effect_gain": None,
                    "operator_controllability_gain": None,
                    "operator_valid_effect": None,
                    "operator_valid_control": None,
                    "event_intensity": None,
                    "milestone_reward": None,
                    "milestone_trigger": None,
                    "running_best_intensity": None,
                    "habituation_factor": None,
                    "token_repeat_decay": None,
                    "topk_rank": None,
                    "topk_env_bonus": None,
                    "topk_entered": None,
                    "topk_leaderboard_size": None,
                    "topk_update_count": None,
                    "operator_token_perplexity_online": None,
                    "operator_num_active_tokens_online": None,
                    "operator_token_count_min": None,
                    "operator_token_count_max": None,
                    "operator_raw_reward": None,
                    "operator_clipped_reward": None,
                    "operator_reward_saturation": None,
                    "operator_reward_nonzero_outside_capacity": None,
                    "operator_reward_mask_violation_count": None,
                    "operator_control_only_suppressed": None,
                    "operator_valid_transition": None,
                    "operator_stale_suppressed": None,
                    "operator_latent_delta_norm": None,
                    "operator_eval_capacity_window_ready": None,
                    "operator_eval_capacity_window_size": None,
                }
            )
        return metrics

    def get_event_diagnostics(self) -> dict[str, float | str | None]:
        return dict(self.last_event_metrics)

    def get_gate_diagnostics(self) -> dict[str, float | str | None]:
        diagnostics: dict[str, float | str | None] = {}
        diagnostics.update(self.get_context_diagnostics())
        diagnostics.update(self.get_event_diagnostics())
        return diagnostics

    def _operator_reward_padding(self, env_reward: torch.Tensor, transition_reward: torch.Tensor) -> torch.Tensor:
        padded = torch.zeros_like(env_reward)
        if padded.shape[1] > 1:
            padded[:, 1:] = transition_reward
        return padded

    def _transition_capacity_gate(
        self,
        world_model_outputs: dict[str, torch.Tensor] | None,
        x_t: torch.Tensor,
        x_tp1: torch.Tensor,
        action_t: torch.Tensor,
        context_gate_seq: torch.Tensor | None,
        context_delta_norm_seq: torch.Tensor | None,
        invalid_transition_seq: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if world_model_outputs is not None and world_model_outputs.get("event_capacity_gate") is not None:
            return world_model_outputs["event_capacity_gate"].detach()
        if not self.config.event_dynamics.enabled or not self.config.event_dynamics.capacity_enabled:
            return None
        with torch.no_grad():
            event_outputs = self.world_model.predict_event_transition(
                x_t,
                action_t,
                target_next_feat=x_tp1,
                context_gate_seq=context_gate_seq,
                context_delta_norm_seq=context_delta_norm_seq,
                invalid_transition_seq=invalid_transition_seq,
                valid_mask=(1.0 - invalid_transition_seq.float()).unsqueeze(-1)
                if invalid_transition_seq is not None and self.config.event_dynamics.capacity_use_valid_mask
                else None,
            )
        if event_outputs is None:
            return None
        return event_outputs.get("event_capacity_gate")

    def _augment_batch_with_operator_reward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        env_reward = batch["reward"].detach()
        action_penalty, action_penalty_info = compute_action_penalty(
            batch["action"].detach(),
            config=self.config.action_penalty,
        )
        action_penalty_metrics = {
            "action_penalty_mean": float(action_penalty.mean().detach().cpu()),
            "action_penalty_max": float(action_penalty.max().detach().cpu()),
            "action_penalty_nonzero_ratio": float((action_penalty > 0).float().mean().detach().cpu()),
            "action_norm_mean": float(action_penalty_info["action_norm"].mean().detach().cpu()),
            "action_delta_norm_mean": float(action_penalty_info["action_delta_norm"].mean().detach().cpu()),
            "actor/action_norm": float(action_penalty_info["action_norm"].mean().detach().cpu()),
            "actor/action_abs_mean": float(batch["action"].detach().float().abs().mean().cpu()),
            "actor/action_std": float(batch["action"].detach().float().std(unbiased=False).cpu()),
            "train_reward_action_penalty_mean": float((-action_penalty).mean().detach().cpu()),
        }
        if self.operator_intrinsic_reward is None:
            train_reward = env_reward - action_penalty
            train_batch = dict(batch)
            train_batch["reward"] = train_reward
            train_batch["env_reward"] = env_reward
            train_batch["train_reward"] = train_reward
            return train_batch, {
                "operator_intrinsic_reward_mean": 0.0,
                "operator_intrinsic_reward_std": 0.0,
                "operator_intrinsic_reward_max": 0.0,
                "operator_intrinsic_reward_nonzero_ratio": 0.0,
                "operator_token_novelty_mean": 0.0,
                "operator_rarity_mean": 0.0,
                "operator_effect_gain_mean": 0.0,
                "operator_controllability_gain_mean": 0.0,
                "operator_valid_effect_ratio": 0.0,
                "operator_valid_control_ratio": 0.0,
                "event_intensity_mean": 0.0,
                "event_intensity_std": 0.0,
                "event_intensity_max": 0.0,
                "milestone_reward_mean": 0.0,
                "milestone_trigger_count": 0.0,
                "running_best_intensity": 0.0,
                "habituation_factor_mean": 0.0,
                "token_repeat_decay_mean": 1.0,
                "milestone_nonzero_ratio": 0.0,
                "intrinsic/milestone_new_rate": 0.0,
                "intrinsic/milestone_reward_mean": 0.0,
                "intrinsic/milestone_reward_nonzero_rate": 0.0,
                "intrinsic/total_intrinsic_reward_mean": 0.0,
                "event_intensity_top_mean": 0.0,
                "event_intensity_normal_mean": 0.0,
                "topk_rank_mean": 0.0,
                "topk_env_bonus_mean": 0.0,
                "topk_entered_ratio": 0.0,
                "topk_leaderboard_size": 0.0,
                "topk_update_count": 0.0,
                "operator_token_perplexity_online": 0.0,
                "operator_num_active_tokens_online": 0.0,
                "operator_token_count_min": 0.0,
                "operator_token_count_max": 0.0,
                "operator_raw_reward_mean": 0.0,
                "operator_clipped_reward_mean": 0.0,
                "operator_reward_saturation_ratio": 0.0,
                "operator_reward_nonzero_outside_capacity": 0.0,
                "operator_reward_mask_violation_count": 0.0,
                "operator_control_only_suppressed_ratio": 0.0,
                "operator_valid_transition_ratio": 0.0,
                "operator_stale_suppressed_ratio": 0.0,
                "operator_latent_delta_norm_mean": 0.0,
                "train_reward_mean": float(train_reward.mean().detach().cpu()),
                "env_reward_mean": float(env_reward.mean().detach().cpu()),
                "train_reward_env_component_mean": float(env_reward.mean().detach().cpu()),
                "train_reward_operator_component_mean": 0.0,
                **action_penalty_metrics,
                "bad_numeric_count": 0.0,
            }

        with torch.no_grad():
            world_model_outputs = self.world_model(batch, None)
            feat = world_model_outputs["augmented_feat"].detach()
            x_t = feat[:, :-1]
            x_tp1 = feat[:, 1:]
            action_t = batch["action"][:, :-1].detach()
            context_gate_seq = (
                world_model_outputs["context_gate"][:, :-1].detach()
                if world_model_outputs.get("context_gate") is not None
                else None
            )
            context_delta_norm_seq = (
                world_model_outputs["context_delta_norm"][:, :-1].detach()
                if world_model_outputs.get("context_delta_norm") is not None
                else None
            )
            invalid_transition_seq = batch["is_first"][:, 1:] if "is_first" in batch else None
            capacity_gate = self._transition_capacity_gate(
                world_model_outputs,
                x_t=x_t,
                x_tp1=x_tp1,
                action_t=action_t,
                context_gate_seq=context_gate_seq,
                context_delta_norm_seq=context_delta_norm_seq,
                invalid_transition_seq=invalid_transition_seq,
            )
            intrinsic_transition_reward, reward_info = self.operator_intrinsic_reward(
                x_t=x_t,
                x_tp1=x_tp1,
                action_t=action_t,
                capacity_gate_t=capacity_gate,
                env_reward_t=env_reward[:, 1:],
                event_logit_t=world_model_outputs.get("event_logit"),
                context_delta_norm_t=context_delta_norm_seq,
                ordinary_error_t=world_model_outputs.get("pred_next_feat_error_ordinary_only"),
                mixed_error_t=world_model_outputs.get("pred_next_feat_error_event_mixed"),
                lifetime_reset_t=invalid_transition_seq,
                update_state=True,
            )

        padded_intrinsic = self._operator_reward_padding(env_reward, intrinsic_transition_reward)
        train_reward, env_component, operator_component = combine_train_reward(
            env_reward=env_reward,
            operator_reward=padded_intrinsic,
            config=self.config.operator_intrinsic_reward,
        )
        train_reward = train_reward - action_penalty
        train_batch = dict(batch)
        train_batch["reward"] = train_reward
        train_batch["env_reward"] = env_reward
        train_batch["operator_intrinsic_reward"] = padded_intrinsic
        train_batch["train_reward"] = train_reward
        metrics = {
            "operator_intrinsic_reward_mean": float(intrinsic_transition_reward.mean().detach().cpu()),
            "operator_intrinsic_reward_std": float(intrinsic_transition_reward.float().std(unbiased=False).detach().cpu()),
            "operator_intrinsic_reward_max": float(intrinsic_transition_reward.max().detach().cpu()),
            "operator_intrinsic_reward_nonzero_ratio": float((intrinsic_transition_reward > 0).float().mean().detach().cpu()),
            "operator_token_novelty_mean": float(reward_info["operator_token_novelty"].mean().detach().cpu()),
            "operator_rarity_mean": float(reward_info["operator_rarity"].mean().detach().cpu()),
            "operator_effect_gain_mean": float(reward_info["operator_effect_gain"].mean().detach().cpu()),
            "operator_controllability_gain_mean": float(
                reward_info["operator_controllability_gain"].mean().detach().cpu()
            ),
            "operator_valid_effect_ratio": float(reward_info["operator_valid_effect"].float().mean().detach().cpu()),
            "operator_valid_control_ratio": float(reward_info["operator_valid_control"].float().mean().detach().cpu()),
            "event_intensity_mean": float(reward_info["event_intensity_mean"].detach().cpu()),
            "event_intensity_std": float(reward_info["event_intensity_std"].detach().cpu()),
            "event_intensity_max": float(reward_info["event_intensity_max"].detach().cpu()),
            "milestone_reward_mean": float(reward_info["milestone_reward_mean"].detach().cpu()),
            "milestone_trigger_count": float(reward_info["milestone_trigger_count"].detach().cpu()),
            "running_best_intensity": float(reward_info["running_best_intensity"].detach().cpu()),
            "habituation_factor_mean": float(reward_info["habituation_factor_mean"].detach().cpu()),
            "token_repeat_decay_mean": float(reward_info["token_repeat_decay_mean"].detach().cpu()),
            "milestone_nonzero_ratio": float(reward_info["milestone_nonzero_ratio"].detach().cpu()),
            "intrinsic/milestone_new_rate": float(reward_info["milestone_nonzero_ratio"].detach().cpu()),
            "intrinsic/milestone_reward_mean": float(reward_info["milestone_reward_mean"].detach().cpu()),
            "intrinsic/milestone_reward_nonzero_rate": float(reward_info["milestone_nonzero_ratio"].detach().cpu()),
            "intrinsic/total_intrinsic_reward_mean": float(intrinsic_transition_reward.mean().detach().cpu()),
            "event_intensity_top_mean": float(reward_info["event_intensity_top_mean"].detach().cpu()),
            "event_intensity_normal_mean": float(reward_info["event_intensity_normal_mean"].detach().cpu()),
            "topk_rank_mean": float(reward_info["topk_rank_mean"].detach().cpu()),
            "topk_env_bonus_mean": float(reward_info["topk_env_bonus_mean"].detach().cpu()),
            "topk_entered_ratio": float(reward_info["topk_entered_ratio"].detach().cpu()),
            "topk_leaderboard_size": float(reward_info["topk_leaderboard_size"].detach().cpu()),
            "topk_update_count": float(reward_info["topk_update_count"].detach().cpu()),
            "operator_token_perplexity_online": float(reward_info["operator_token_perplexity_online"].detach().cpu()),
            "operator_num_active_tokens_online": float(reward_info["operator_num_active_tokens_online"].detach().cpu()),
            "operator_token_count_min": float(reward_info["operator_token_count_min"].detach().cpu()),
            "operator_token_count_max": float(reward_info["operator_token_count_max"].detach().cpu()),
            "operator_raw_reward_mean": float(reward_info["operator_raw_reward"].mean().detach().cpu()),
            "operator_clipped_reward_mean": float(reward_info["operator_clipped_reward"].mean().detach().cpu()),
            "operator_reward_saturation_ratio": float(
                reward_info["operator_reward_saturation"].float().mean().detach().cpu()
            ),
            "operator_reward_nonzero_outside_capacity": float(
                reward_info["operator_reward_nonzero_outside_capacity"].detach().cpu()
            ),
            "operator_reward_mask_violation_count": float(
                reward_info["operator_reward_mask_violation_count"].detach().cpu()
            ),
            "operator_control_only_suppressed_ratio": float(
                reward_info["operator_control_only_suppressed"].float().mean().detach().cpu()
            ),
            "operator_valid_transition_ratio": float(reward_info["operator_valid_transition"].float().mean().detach().cpu()),
            "operator_stale_suppressed_ratio": float(
                reward_info["operator_stale_suppressed"].float().mean().detach().cpu()
            ),
            "operator_latent_delta_norm_mean": float(reward_info["operator_latent_delta_norm"].mean().detach().cpu()),
            "train_reward_mean": float(train_reward.mean().detach().cpu()),
            "env_reward_mean": float(env_reward.mean().detach().cpu()),
            "train_reward_env_component_mean": float(env_component.mean().detach().cpu()),
            "train_reward_operator_component_mean": float(operator_component.mean().detach().cpu()),
            **action_penalty_metrics,
            "bad_numeric_count": float(reward_info["operator_bad_numeric_count"].detach().cpu()),
        }
        return train_batch, metrics

    def set_rollout_mode(self, deterministic: bool):
        modules = [self.world_model, self.actor, self.value]
        if self.slow_value is not None:
            modules.append(self.slow_value)
        for module in modules:
            if deterministic:
                module.eval()
            else:
                module.train()

    def _clone_state(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.detach().clone() for key, value in state.items()}

    def configure_online_update(
        self,
        batch_size: int,
        seq_len: int,
        warmup_steps: int,
        train_every: int = 1,
        train_ratio: float = 1.0,
        max_updates_per_step: int = 0,
    ):
        self._online_batch_size = int(batch_size)
        self._online_seq_len = int(seq_len)
        self._online_warmup_steps = int(warmup_steps)
        self._online_train_every = max(1, int(train_every))
        self._online_train_ratio = float(train_ratio)
        self._online_max_updates_per_step = int(max_updates_per_step)
        self._online_train_calls = 0
        self._online_train_budget = 0.0

    def reset_online_update_state(self):
        self._online_train_calls = 0
        self._online_train_budget = 0.0

    def _make_value_model(self, feat_dim: int):
        if self.config.use_twohot_value:
            return TwoHotSymlogHead(
                feat_dim=feat_dim,
                hidden_dim=self.config.value_hidden_dim,
                num_layers=self.config.value_num_layers,
                num_bins=self.config.twohot_bins,
                low=self.config.twohot_low,
                high=self.config.twohot_high,
            )
        return DreamerValue(
            DreamerValueConfig(
                feat_dim=feat_dim,
                hidden_dim=self.config.value_hidden_dim,
                num_layers=self.config.value_num_layers,
            )
        )

    def _value_forward(self, feat: torch.Tensor, denormalize: bool = True) -> torch.Tensor:
        return self._head_to_scalar(self.value, feat, denormalize=denormalize)

    def _slow_value_forward(self, feat: torch.Tensor, denormalize: bool = True) -> torch.Tensor:
        if self.slow_value is None:
            return self._value_forward(feat, denormalize=denormalize)
        return self._head_to_scalar(self.slow_value, feat, denormalize=denormalize)

    def _head_to_scalar(self, module, feat: torch.Tensor, denormalize: bool = True) -> torch.Tensor:
        if isinstance(module, TwoHotSymlogHead):
            logits = module(feat)
            value = symexp(twohot_mean(
                logits,
                num_bins=self.config.twohot_bins,
                low=self.config.twohot_low,
                high=self.config.twohot_high,
            ))
        else:
            value = module(feat)
        if denormalize and self.config.use_return_norm:
            return self.return_normalizer.denormalize(value)
        return value

    def _value_loss(self, feat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.reshape(-1)
        if self.config.use_return_norm:
            target = self.return_normalizer.normalize(target, update=False)
        if isinstance(self.value, TwoHotSymlogHead):
            target = symlog(target) if self.config.use_symlog_reward else target
            logits = self.value(feat)
            return -twohot_logprob(
                logits,
                target,
                num_bins=self.config.twohot_bins,
                low=self.config.twohot_low,
                high=self.config.twohot_high,
            ).mean()
        return F.mse_loss(self.value(feat), target)

    def _sync_slow_value(self):
        if self.slow_value is None:
            return
        rate = float(self.config.slow_value_rate)
        with torch.no_grad():
            for slow_param, fast_param in zip(self.slow_value.parameters(), self.value.parameters()):
                slow_param.data.lerp_(fast_param.data, rate)

    def _replay_value_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            outputs = self.world_model(batch, None)
            feat = outputs["feat"]
        flat_feat = feat.reshape(-1, feat.shape[-1])
        slow_values = self._slow_value_forward(flat_feat).view(feat.shape[0], feat.shape[1])
        current_values = self._value_forward(flat_feat).view(feat.shape[0], feat.shape[1])
        bootstrap = slow_values[:, -1]
        replay_returns = lambda_return(
            rewards=batch["reward"].transpose(0, 1),
            values=torch.cat([slow_values.transpose(0, 1), bootstrap.unsqueeze(0)], dim=0),
            continues=(1.0 - batch["done"]).transpose(0, 1),
            gamma=self.config.gamma,
            lambda_=self.config.lambda_,
        ).transpose(0, 1)
        repval_loss = self._value_loss(flat_feat, replay_returns.detach())
        slow_reg = F.mse_loss(current_values, slow_values.detach())
        return repval_loss, slow_reg

    def reset_latent(self, batch_size: int = 1):
        self.latent_state = self.world_model.rssm.init_state(batch_size, self.device)
        self.latent_context = self.world_model.initial_context(batch_size, self.device)
        self.last_context_metrics = self._default_context_metrics()
        self.last_event_metrics = self._default_event_metrics()
        self._pending_event_transition = None
        self._operator_eval_event_logits = []
        if self.operator_intrinsic_reward is not None:
            self.operator_intrinsic_reward.reset_lifetime_state()
        return self.latent_state

    def _rolling_capacity_gate_for_preview(self, event_outputs: dict[str, torch.Tensor] | None) -> tuple[torch.Tensor | None, float, int]:
        if self.operator_intrinsic_reward is None:
            return None, 0.0, 0
        cfg = self.config.operator_intrinsic_reward
        if not (cfg.capacity_only and cfg.eval_use_rolling_capacity):
            if event_outputs is None:
                return None, 0.0, 0
            return event_outputs.get("event_capacity_gate"), 1.0, 1
        if event_outputs is None or event_outputs.get("event_logit") is None:
            return None, 0.0, 0
        logit_value = float(event_outputs["event_logit"].detach().float().mean().cpu())
        self._operator_eval_event_logits.append(logit_value)
        max_window = int(cfg.rolling_capacity_window)
        if len(self._operator_eval_event_logits) > max_window:
            self._operator_eval_event_logits = self._operator_eval_event_logits[-max_window:]
        window_size = len(self._operator_eval_event_logits)
        if window_size < int(cfg.min_capacity_window):
            return torch.zeros(1, 1, 1, device=self.device, dtype=torch.float32), 0.0, window_size
        event_cfg = self.config.event_dynamics
        selector = EventCapacitySelector(event_cfg)
        selector.eval()
        logits = torch.as_tensor(self._operator_eval_event_logits, device=self.device, dtype=torch.float32).view(1, -1, 1)
        with torch.no_grad():
            capacity_gate, _ = selector(logits, valid_mask=torch.ones_like(logits))
        current_gate = capacity_gate[:, -1:, :]
        return current_gate.detach(), 1.0, window_size

    def update_latent(self, obs: np.ndarray, prev_action: np.ndarray, is_first: bool = False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if self.config.observation.mode == "vector":
            obs_t = obs_t.view(1, -1)
        action_t = torch.as_tensor(prev_action, dtype=torch.float32, device=self.device).view(1, -1)
        first_t = torch.as_tensor([float(is_first)], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            embed = self.world_model.encoder(obs_t)
            self.latent_state, _ = self.world_model.rssm.observe_step(
                state=self.latent_state,
                embed=embed,
                action=action_t,
                is_first=first_t,
            )
            feature_details = self.world_model.get_augmented_feat(
                self.latent_state,
                prev_context=self.latent_context,
                is_first=first_t,
                return_details=True,
            )
            self.latent_context = feature_details["context"].detach() if feature_details["context"] is not None else None
            self.last_context_metrics = self._context_metrics_from_details(feature_details)
        return self.latent_state

    def act(self, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            base_feat = self.world_model.get_base_feat(self.latent_state)
            feat = self.world_model.concat_context(base_feat, self.latent_context)
            action = self.actor.mode(feat) if deterministic else self.actor.sample(feat)[0]
            if self.config.event_dynamics.enabled:
                self._pending_event_transition = {
                    "state": self._clone_state(self.latent_state),
                    "context": None if self.latent_context is None else self.latent_context.detach().clone(),
                    "feat": feat.detach().clone(),
                    "action": action.detach().clone(),
                    "context_gate": self.last_context_metrics.get("context_gate"),
                    "context_delta_norm": self.last_context_metrics.get("context_delta_norm"),
                }
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def observe_event_transition(
        self,
        next_obs: np.ndarray,
        done: bool = False,
        reward: float | None = None,
        consume: bool = True,
    ) -> dict[str, float | str | None]:
        if not self.config.event_dynamics.enabled or self._pending_event_transition is None:
            self.last_event_metrics = self._default_event_metrics()
            return dict(self.last_event_metrics)

        obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if self.config.observation.mode == "vector":
            obs_t = obs_t.view(1, -1)
        pending = self._pending_event_transition
        action_t = pending["action"].to(device=self.device, dtype=torch.float32)
        state = {key: value.to(device=self.device) for key, value in pending["state"].items()}
        context = pending["context"]
        if context is not None:
            context = context.to(device=self.device, dtype=torch.float32)
        context_gate = pending.get("context_gate")
        context_delta_norm = pending.get("context_delta_norm")
        context_gate_t = None
        context_delta_norm_t = None
        if context_gate is not None:
            context_gate_t = torch.as_tensor([[[float(context_gate)]]], dtype=torch.float32, device=self.device)
        if context_delta_norm is not None:
            context_delta_norm_t = torch.as_tensor([[[float(context_delta_norm)]]], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            embed = self.world_model.encoder(obs_t)
            next_state, _ = self.world_model.rssm.observe_step(
                state=state,
                embed=embed,
                action=action_t,
                is_first=torch.zeros(1, dtype=torch.float32, device=self.device),
            )
            next_details = self.world_model.get_augmented_feat(
                next_state,
                prev_context=context,
                is_first=torch.zeros(1, dtype=torch.float32, device=self.device),
                return_details=True,
            )
            event_outputs = self.world_model.predict_event_transition(
                pending["feat"].to(device=self.device, dtype=torch.float32),
                action_t,
                target_next_feat=next_details["augmented_feat"],
                context_gate_seq=context_gate_t,
                context_delta_norm_seq=context_delta_norm_t,
            )
            operator_metrics = {}
            if self.operator_intrinsic_reward is not None:
                preview_capacity_gate, capacity_window_ready, capacity_window_size = self._rolling_capacity_gate_for_preview(
                    event_outputs
                )
                env_reward_t = None
                if reward is not None:
                    env_reward_t = torch.as_tensor(
                        [[float(reward)]],
                        dtype=torch.float32,
                        device=self.device,
                    )
                operator_reward, reward_info = self.operator_intrinsic_reward(
                    x_t=pending["feat"].to(device=self.device, dtype=torch.float32).unsqueeze(1),
                    x_tp1=next_details["augmented_feat"].unsqueeze(1),
                    action_t=action_t.unsqueeze(1),
                    capacity_gate_t=preview_capacity_gate,
                    env_reward_t=env_reward_t,
                    event_logit_t=None if event_outputs is None else event_outputs.get("event_logit"),
                    context_delta_norm_t=context_delta_norm_t,
                    ordinary_error_t=None
                    if event_outputs is None
                    else event_outputs.get("pred_next_feat_error_ordinary_only"),
                    mixed_error_t=None
                    if event_outputs is None
                    else event_outputs.get("pred_next_feat_error_event_mixed"),
                    update_state=False,
                )
                operator_metrics = {
                    "operator_reward": float(operator_reward.detach().mean().cpu()),
                    "operator_token_id": float(reward_info["operator_token_id"].detach().float().mean().cpu()),
                    "operator_token_novelty": float(reward_info["operator_token_novelty"].detach().mean().cpu()),
                    "operator_rarity": float(reward_info["operator_rarity"].detach().mean().cpu()),
                    "operator_effect_gain": float(reward_info["operator_effect_gain"].detach().mean().cpu()),
                    "operator_controllability_gain": float(
                        reward_info["operator_controllability_gain"].detach().mean().cpu()
                    ),
                    "operator_valid_effect": float(reward_info["operator_valid_effect"].detach().mean().cpu()),
                    "operator_valid_control": float(reward_info["operator_valid_control"].detach().mean().cpu()),
                    "event_intensity": float(reward_info["event_intensity"].detach().mean().cpu()),
                    "milestone_reward": float(reward_info["milestone_reward"].detach().mean().cpu()),
                    "milestone_trigger": float(reward_info["milestone_trigger_count"].detach().cpu()),
                    "running_best_intensity": float(reward_info["running_best_intensity"].detach().cpu()),
                    "habituation_factor": float(reward_info["habituation_factor"].detach().mean().cpu()),
                    "token_repeat_decay": float(reward_info["token_repeat_decay"].detach().mean().cpu()),
                    "topk_rank": float(reward_info["topk_rank_mean"].detach().cpu()),
                    "topk_env_bonus": float(reward_info["topk_env_bonus_mean"].detach().cpu()),
                    "topk_entered": float(reward_info["topk_entered_ratio"].detach().cpu()),
                    "topk_leaderboard_size": float(reward_info["topk_leaderboard_size"].detach().cpu()),
                    "topk_update_count": float(reward_info["topk_update_count"].detach().cpu()),
                    "operator_token_perplexity_online": float(
                        reward_info["operator_token_perplexity_online"].detach().cpu()
                    ),
                    "operator_num_active_tokens_online": float(
                        reward_info["operator_num_active_tokens_online"].detach().cpu()
                    ),
                    "operator_token_count_min": float(reward_info["operator_token_count_min"].detach().cpu()),
                    "operator_token_count_max": float(reward_info["operator_token_count_max"].detach().cpu()),
                    "operator_raw_reward": float(reward_info["operator_raw_reward"].detach().mean().cpu()),
                    "operator_clipped_reward": float(reward_info["operator_clipped_reward"].detach().mean().cpu()),
                    "operator_reward_saturation": float(reward_info["operator_reward_saturation"].detach().mean().cpu()),
                    "operator_reward_nonzero_outside_capacity": float(
                        reward_info["operator_reward_nonzero_outside_capacity"].detach().cpu()
                    ),
                    "operator_reward_mask_violation_count": float(
                        reward_info["operator_reward_mask_violation_count"].detach().cpu()
                    ),
                    "operator_control_only_suppressed": float(
                        reward_info["operator_control_only_suppressed"].detach().mean().cpu()
                    ),
                    "operator_valid_transition": float(reward_info["operator_valid_transition"].detach().mean().cpu()),
                    "operator_stale_suppressed": float(reward_info["operator_stale_suppressed"].detach().mean().cpu()),
                    "operator_latent_delta_norm": float(reward_info["operator_latent_delta_norm"].detach().mean().cpu()),
                    "operator_eval_capacity_window_ready": float(capacity_window_ready),
                    "operator_eval_capacity_window_size": float(capacity_window_size),
                }
            self.last_event_metrics = {
                "event_gate": float(event_outputs["event_gate"].detach().mean().cpu()),
                "event_logit": float(event_outputs["event_logit"].detach().mean().cpu()),
                "event_logit_sigmoid": float(event_outputs["event_logit_sigmoid"].detach().mean().cpu()),
                "event_capacity_gate": (
                    None
                    if event_outputs.get("event_capacity_gate") is None
                    else float(event_outputs["event_capacity_gate"].detach().mean().cpu())
                ),
                "event_capacity_enabled": float(self.config.event_dynamics.capacity_enabled),
                "event_capacity_ratio": float(self.config.event_dynamics.capacity_ratio),
                "event_capacity_k": (
                    None
                    if event_outputs.get("event_capacity_k") is None
                    else float(event_outputs["event_capacity_k"].detach().mean().cpu())
                ),
                "event_capacity_mode": self.config.event_dynamics.capacity_mode,
                "capacity_v2_detach_event_input": float(bool(event_outputs.get("capacity_v2_detach_event_input", False))),
                "capacity_v2_detach_event_target": float(bool(event_outputs.get("capacity_v2_detach_event_target", False))),
                "capacity_v2_use_sigmoid_gate_multiplier": float(
                    bool(event_outputs.get("capacity_v2_use_sigmoid_gate_multiplier", False))
                ),
                "event_loss_updates_backbone": float(bool(event_outputs.get("event_loss_updates_backbone", True))),
                "event_prediction_error": float(event_outputs["event_prediction_error"].detach().mean().cpu()),
                "pred_next_feat_error_ordinary_only": float(
                    event_outputs["pred_next_feat_error_ordinary_only"].detach().mean().cpu()
                ),
                "pred_next_feat_error_event_mixed": float(
                    event_outputs["pred_next_feat_error_event_mixed"].detach().mean().cpu()
                ),
                "context_change_mask": (
                    None
                    if event_outputs.get("context_change_mask") is None
                    else float(event_outputs["context_change_mask"].detach().mean().cpu())
                ),
                "context_change_mask_source": event_outputs.get("context_change_mask_source"),
                "context_change_mask_mode": event_outputs.get("context_change_mask_mode"),
                "context_change_mask_top_percent": event_outputs.get("context_change_mask_top_percent"),
                "event_residual_norm": float(event_outputs["event_residual_norm"].detach().mean().cpu()),
            }
            self.last_event_metrics.update(operator_metrics)
        if consume:
            self._pending_event_transition = None
        return dict(self.last_event_metrics)

    def preview_event_transition(
        self,
        next_obs: np.ndarray,
        done: bool = False,
        reward: float | None = None,
    ) -> dict[str, float | str | None]:
        return self.observe_event_transition(next_obs=next_obs, done=done, reward=reward, consume=False)

    def train_world_model(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        self.world_model.train()
        outputs = self.world_model(batch, self.loss_config)
        loss = outputs["loss"]
        self.world_model_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), self.config.grad_clip)
        self.world_model_optimizer.step()
        metrics = {}
        for key, value in outputs["metrics"].items():
            if key == "priority":
                metrics[key] = value.detach().cpu().tolist()
            else:
                metrics[key] = float(value.detach().cpu())
        return metrics

    def _set_world_model_trainable(self, trainable: bool):
        for param in self.world_model.parameters():
            param.requires_grad_(trainable)

    def _posterior_start_state(self, batch: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
        with torch.no_grad():
            outputs = self.world_model(batch, None)
            post = outputs["post"]
            context = outputs.get("context")
        batch_size, seq_len, _ = post["h"].shape
        if self.config.imag_last and self.config.imag_last > 0:
            steps = min(int(self.config.imag_last), seq_len)
            state = {
                key: value[:, -steps:].reshape(batch_size * steps, *value.shape[2:]).detach()
                for key, value in post.items()
            }
            context_state = (
                context[:, -steps:].reshape(batch_size * steps, context.shape[-1]).detach() if context is not None else None
            )
            return state, context_state
        state = {key: value.reshape(batch_size * seq_len, *value.shape[2:]).detach() for key, value in post.items()}
        context_state = context.reshape(batch_size * seq_len, context.shape[-1]).detach() if context is not None else None
        return state, context_state

    def _discount_weights(self, continues: torch.Tensor) -> torch.Tensor:
        discounts = self.config.gamma * continues.detach()
        prefix = torch.ones(1, discounts.shape[1], dtype=discounts.dtype, device=discounts.device)
        if discounts.shape[0] == 1:
            return prefix
        return torch.cat([prefix, torch.cumprod(discounts[:-1], dim=0)], dim=0)

    def train_actor_value(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        self.actor.train()
        self.value.train()
        start_state, start_context = self._posterior_start_state(batch)
        self._set_world_model_trainable(False)

        imagined = imagine_rollout(
            world_model=self.world_model,
            actor=self.actor,
            start_state=start_state,
            horizon=self.config.imagination_horizon,
            start_context=start_context,
        )
        feats = imagined["feats"]
        rewards = imagined["rewards"]
        continues = imagined["continues"]
        entropies = imagined["entropies"]
        log_probs = imagined["log_probs"]
        flat_feats = feats.reshape(-1, feats.shape[-1])
        values = self._value_forward(flat_feats).view(feats.shape[0], feats.shape[1])
        last_state = {key: value[-1] for key, value in imagined["states"].items()}
        bootstrap_feat = self.world_model.concat_context(
            self.world_model.get_base_feat(last_state),
            imagined.get("last_context"),
        )
        bootstrap = self._slow_value_forward(bootstrap_feat).unsqueeze(0)
        all_values = torch.cat([values, bootstrap], dim=0)
        returns = lambda_return(
            rewards=rewards,
            values=all_values,
            continues=continues,
            gamma=self.config.gamma,
            lambda_=self.config.lambda_,
        )
        if self.config.use_return_norm:
            self.return_normalizer.update(returns.detach())
        weights = self._discount_weights(continues)
        advantage = returns.detach() - values.detach()
        actor_return_batch_scale = 1.0
        if self.config.use_actor_return_scale:
            with torch.no_grad():
                flat_returns = returns.detach().reshape(-1)
                p_high = torch.quantile(flat_returns, float(self.config.actor_return_scale_q_high))
                p_low = torch.quantile(flat_returns, float(self.config.actor_return_scale_q_low))
                batch_scale = float((p_high - p_low).clamp_min(0.0).cpu())
                actor_return_batch_scale = batch_scale
                if not self.actor_return_scale_initialized:
                    self.actor_return_scale = batch_scale
                    self.actor_return_scale_initialized = True
                else:
                    decay = float(self.config.actor_return_scale_decay)
                    self.actor_return_scale = decay * self.actor_return_scale + (1.0 - decay) * batch_scale
            scale_value = max(self.actor_return_scale, float(self.config.actor_return_scale_min))
            norm_advantage = advantage / scale_value
        elif self.config.use_advantage_norm:
            norm_advantage = self.advantage_normalizer.normalize(advantage, update=True)
        else:
            norm_advantage = advantage
        actor_loss = -(weights * (log_probs * norm_advantage + self.config.entropy_coef * entropies)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.actor_grad_clip)
        self.actor_optimizer.step()

        value_loss = self._value_loss(flat_feats.detach(), returns.detach())
        repval_loss = torch.zeros((), device=self.device)
        slowreg_loss = torch.zeros((), device=self.device)
        if self.config.repval_loss:
            repval_loss, slowreg_loss = self._replay_value_loss(batch)
            value_loss = value_loss + self.config.repval_scale * repval_loss + self.config.slow_value_rate * slowreg_loss
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.config.value_grad_clip)
        self.value_optimizer.step()
        self._sync_slow_value()
        self._set_world_model_trainable(True)

        value_pred = self._value_forward(flat_feats.detach()).view(feats.shape[0], feats.shape[1])
        return {
            "actor_loss": float(actor_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "repval_loss": float(repval_loss.detach().cpu()),
            "slowreg_loss": float(slowreg_loss.detach().cpu()),
            "imagined_return": float(returns.detach().mean().cpu()),
            "value_pred": float(value_pred.detach().mean().cpu()),
            "return_norm_mean": float(self.return_normalizer.mean),
            "return_norm_std": float(self.return_normalizer.std),
            "actor_return_batch_scale": float(actor_return_batch_scale),
            "actor_return_scale": float(max(self.actor_return_scale, float(self.config.actor_return_scale_min))),
            "adv_norm_mean": float(self.advantage_normalizer.mean),
            "adv_norm_std": float(self.advantage_normalizer.std),
            "entropy": float(entropies.detach().mean().cpu()),
            "actor/action_entropy": float(entropies.detach().mean().cpu()),
        }

    def update_from_replay(
        self,
        replay: "EpisodeReplayBuffer",
        include_contact: bool = False,
        include_grasp: bool = False,
    ) -> tuple[dict[str, float], dict[str, object]]:
        metadata: dict[str, object] = {
            "replay_size": len(replay),
            "warmup_steps": self._online_warmup_steps,
            "train_calls": self._online_train_calls + 1,
            "device": str(self.device),
        }

        self._online_train_calls += 1
        if len(replay) < self._online_warmup_steps or self._online_train_calls % self._online_train_every != 0:
            return {}, metadata

        self._online_train_budget += self._online_train_ratio
        budget_updates = int(self._online_train_budget)
        if budget_updates <= 0:
            return {}, metadata

        num_updates = budget_updates
        if self._online_max_updates_per_step > 0:
            num_updates = min(num_updates, self._online_max_updates_per_step)
        self._online_train_budget -= num_updates

        if not replay.can_sample(self._online_batch_size, self._online_seq_len, include_contact, include_grasp):
            if include_contact or include_grasp:
                include_contact = False
                include_grasp = False
            if not replay.can_sample(self._online_batch_size, self._online_seq_len, include_contact, include_grasp):
                return {}, metadata

        aggregates: dict[str, float] = {}
        for _ in range(num_updates):
            batch = replay.sample_batch(
                batch_size=self._online_batch_size,
                seq_len=self._online_seq_len,
                device=self.device,
                include_contact=include_contact,
                include_grasp=include_grasp,
            )
            sample_refs = batch.pop("_sample_refs", [])
            train_batch, reward_metrics = self._augment_batch_with_operator_reward(batch)
            wm_metrics = self.train_world_model(train_batch)
            if "priority" in wm_metrics:
                replay.update_priorities(sample_refs, wm_metrics["priority"])
                del wm_metrics["priority"]
            actor_metrics = self.train_actor_value(train_batch)
            for metrics in (reward_metrics, wm_metrics, actor_metrics):
                for key, value in metrics.items():
                    aggregates[key] = aggregates.get(key, 0.0) + float(value)

        metrics = {key: value / num_updates for key, value in aggregates.items()}
        metrics["num_updates"] = float(num_updates)
        metrics["train_budget"] = float(self._online_train_budget)
        metadata.update(
            {
                "batch_size": self._online_batch_size,
                "seq_len": self._online_seq_len,
                "include_contact": include_contact,
                "include_grasp": include_grasp,
            }
        )
        if self.config.event_dynamics.enabled and self.config.event_dynamics.context_mask_enabled:
            metadata.update(
                {
                    "context_change_mask_source": self.config.event_dynamics.context_mask_source,
                    "context_change_mask_mode": self.config.event_dynamics.context_mask_mode,
                    "context_change_mask_top_percent": float(self.config.event_dynamics.context_mask_top_percent),
                }
            )
        if self.config.event_dynamics.enabled and self.config.event_dynamics.capacity_enabled:
            metadata.update(
                {
                    "event_capacity_enabled": True,
                    "event_capacity_mode": self.config.event_dynamics.capacity_mode,
                    "event_capacity_ratio": float(self.config.event_dynamics.capacity_ratio),
                    "capacity_v2_detach_event_input": bool(self.config.event_dynamics.capacity_v2_detach_event_input),
                    "capacity_v2_detach_event_target": bool(self.config.event_dynamics.capacity_v2_detach_event_target),
                    "capacity_v2_use_sigmoid_gate_multiplier": bool(
                        self.config.event_dynamics.capacity_v2_use_sigmoid_gate_multiplier
                    ),
                    "event_loss_updates_backbone": not bool(
                        self.config.event_dynamics.capacity_v2_freeze_backbone_for_event_loss
                    ),
                }
            )
        if self.operator_intrinsic_reward is not None:
            metadata.update(
                {
                    "operator_intrinsic_reward_enabled": True,
                    "operator_intrinsic_reward_probe_checkpoint": self.config.operator_intrinsic_reward.probe_checkpoint,
                    "operator_intrinsic_reward_capacity_only": bool(self.config.operator_intrinsic_reward.capacity_only),
                    "operator_intrinsic_reward_use_env_reward": bool(self.config.operator_intrinsic_reward.use_env_reward),
                }
            )
        return metrics, metadata

    def state_dict(self) -> dict:
        return {
            "config": self.config.asdict(),
            "loss_config": asdict(self.loss_config),
            "world_model": self.world_model.state_dict(),
            "actor": self.actor.state_dict(),
            "value": self.value.state_dict(),
            "slow_value": self.slow_value.state_dict() if self.slow_value is not None else None,
            "return_normalizer": self.return_normalizer.state_dict(),
            "advantage_normalizer": self.advantage_normalizer.state_dict(),
            "actor_return_scale": self.actor_return_scale,
            "actor_return_scale_initialized": self.actor_return_scale_initialized,
            "world_model_optimizer": self.world_model_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "operator_intrinsic_reward": (
                self.operator_intrinsic_reward.state_dict() if self.operator_intrinsic_reward is not None else None
            ),
            "online_update_state": {
                "train_calls": self._online_train_calls,
                "train_budget": self._online_train_budget,
            },
        }

    def _load_module_partial(self, module, state_dict: dict, module_name: str) -> dict[str, object]:
        current = module.state_dict()
        compatible = {}
        unexpected = []
        shape_mismatches = []
        for key, value in state_dict.items():
            if key not in current:
                unexpected.append(key)
                continue
            if current[key].shape != value.shape:
                shape_mismatches.append(
                    {
                        "key": key,
                        "expected": tuple(current[key].shape),
                        "got": tuple(value.shape),
                    }
                )
                continue
            compatible[key] = value
        load_result = module.load_state_dict(compatible, strict=False)
        return {
            "module": module_name,
            "missing": list(load_result.missing_keys),
            "unexpected": unexpected,
            "shape_mismatches": shape_mismatches,
        }

    def _load_payload(self, payload: dict, allow_partial: bool = False):
        reports = []
        modules = [
            ("world_model", self.world_model, payload["world_model"]),
            ("actor", self.actor, payload["actor"]),
            ("value", self.value, payload["value"]),
        ]
        if self.slow_value is not None and payload.get("slow_value") is not None:
            modules.append(("slow_value", self.slow_value, payload["slow_value"]))
        for module_name, module, state_dict in modules:
            if not allow_partial:
                module.load_state_dict(state_dict)
                continue
            reports.append(self._load_module_partial(module, state_dict, module_name))
        if "return_normalizer" in payload:
            self.return_normalizer = RunningNormalizer.from_state_dict(payload["return_normalizer"])
        if "advantage_normalizer" in payload:
            self.advantage_normalizer = RunningNormalizer.from_state_dict(payload["advantage_normalizer"])
        self.actor_return_scale = float(payload.get("actor_return_scale", 1.0))
        self.actor_return_scale_initialized = bool(payload.get("actor_return_scale_initialized", False))
        if "world_model_optimizer" in payload:
            try:
                self.world_model_optimizer.load_state_dict(payload["world_model_optimizer"])
            except (ValueError, RuntimeError):
                reports.append({"module": "world_model_optimizer", "warning": "optimizer state skipped"})
        if "actor_optimizer" in payload:
            try:
                self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
            except (ValueError, RuntimeError):
                reports.append({"module": "actor_optimizer", "warning": "optimizer state skipped"})
        if "value_optimizer" in payload:
            try:
                self.value_optimizer.load_state_dict(payload["value_optimizer"])
            except (ValueError, RuntimeError):
                reports.append({"module": "value_optimizer", "warning": "optimizer state skipped"})
        if self.operator_intrinsic_reward is not None and payload.get("operator_intrinsic_reward") is not None:
            try:
                self.operator_intrinsic_reward.load_state_dict(payload["operator_intrinsic_reward"], strict=False)
            except (ValueError, RuntimeError):
                reports.append({"module": "operator_intrinsic_reward", "warning": "state skipped"})
        if "online_update_state" in payload:
            state = payload["online_update_state"]
            self._online_train_calls = int(state.get("train_calls", 0))
            self._online_train_budget = float(state.get("train_budget", 0.0))
        partial = any(
            report.get("missing") or report.get("unexpected") or report.get("shape_mismatches") or report.get("warning")
            for report in reports
        )
        if partial:
            print(
                "[DreamerV3] Loaded checkpoint with partial parameter reuse. "
                "This is expected when enabling thick_context or event_dynamics on an older checkpoint."
            )
        self._last_load_report = {"partial": partial, "reports": reports}
        return self._last_load_report

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def from_state_dict(
        cls,
        payload: dict,
        device: str | None = None,
        model_config: DreamerV3ModelConfig | None = None,
    ):
        config_override = model_config is not None
        if model_config is None:
            aux = DreamerAuxConfig(**payload["config"]["aux"])
            observation = DreamerObservationSpec(**payload["config"]["observation"])
            config_data = dict(payload["config"])
            config_data["aux"] = aux
            config_data["observation"] = observation
            config_data["thick_context"] = ThickContextConfig(**payload["config"].get("thick_context", {}))
            config_data["event_dynamics"] = DreamerEventDynamicsConfig(**payload["config"].get("event_dynamics", {}))
            config_data["operator_intrinsic_reward"] = OperatorIntrinsicRewardConfig(
                **payload["config"].get("operator_intrinsic_reward", {})
            )
            if device is not None:
                config_data["device"] = device
            model_config = DreamerV3ModelConfig(**config_data)
        loss_config = None if config_override else WorldModelLossConfig(**payload["loss_config"])
        agent = cls(model_config, loss_config)
        agent._load_payload(
            payload,
            allow_partial=bool(model_config.thick_context.enabled or model_config.event_dynamics.enabled),
        )
        return agent

    @classmethod
    def load(cls, path: str | Path, device: str | None = None):
        return cls.from_state_dict(torch.load(path, map_location=device or "cpu"), device=device)
