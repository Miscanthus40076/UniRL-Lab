from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import math

import torch
import torch.nn.functional as F
from torch import nn

from .normalization import RunningNormConfig, RunningNormalizer
from .operator_tokens import OperatorTokenConfig, compute_token_usage_metrics, bad_numeric_count
from policy.operator_token_probe.operator_token_probe_model import OperatorTokenProbeModel


@dataclass(slots=True)
class OperatorIntrinsicRewardConfig:
    enabled: bool = False
    probe_checkpoint: str | None = None
    beta: float = 0.1
    reward_mode: str = "dense_gain"
    r_new_dct: float = 1.0
    use_env_reward: bool = True
    capacity_only: bool = True
    post_mask_after_normalize: bool = True
    eval_use_rolling_capacity: bool = True
    rolling_capacity_window: int = 64
    min_capacity_window: int = 16
    store_reward_in_replay: bool = False
    forbid_batch_recompute_for_milestone: bool = False
    detach_features: bool = True
    freeze_probe: bool = True
    novelty_weight: float = 1.0
    effect_gain_weight: float = 1.0
    controllability_gain_weight: float = 0.5
    min_effect_gain: float = 0.005
    min_controllability_gain: float = 0.0
    require_effect_for_control: bool = True
    min_delta_x_norm: float = 0.01
    stale_filter_enabled: bool = True
    min_latent_delta_norm: float = 0.01
    normalize_reward: bool = True
    norm_rate: float = 0.01
    clip_reward: bool = True
    reward_clip_min: float = 0.0
    reward_clip_max: float = 1.0
    update_token_counts: bool = True
    count_init: float = 1.0
    count_decay: float = 1.0
    log_token_rewards: bool = True
    intensity_event_logit_weight: float = 1.0
    intensity_context_delta_weight: float = 1.0
    intensity_effect_gain_weight: float = 1.0
    intensity_operator_rarity_weight: float = 1.0
    intensity_norm_rate: float = 0.01
    intensity_clamp_min: float = -5.0
    intensity_clamp_max: float = 5.0
    running_best_mode: str = "ema"
    running_best_rate: float = 0.01
    milestone_margin: float = 0.10
    frequency_habituation_power: float = 1.0
    repeat_decay_tau: float = 1.0
    repeat_decay_min: float = 0.05
    topk_leaderboard_capacity: int = 8
    topk_exponential_base: float = 2.0
    topk_replace_margin: float = 0.0
    topk_decay: float = 0.995
    topk_env_bonus_scale: float = 1.0
    topk_env_bonus_first_only: bool = False
    topk_env_bonus_repeat_decay: float = 0.10
    topk_env_bonus_count_decay: float = 1.0

    def validate(self):
        if self.beta < 0.0:
            raise ValueError("operator_intrinsic_reward.beta must be >= 0")
        if self.reward_mode not in {"dense_gain", "milestone", "topk_exponential", "dct_unigram_milestone"}:
            raise ValueError(
                "operator_intrinsic_reward.reward_mode must be 'dense_gain', 'milestone', "
                "'topk_exponential' or 'dct_unigram_milestone'"
            )
        if self.r_new_dct < 0.0:
            raise ValueError("operator_intrinsic_reward.r_new_dct must be >= 0")
        if self.novelty_weight < 0.0:
            raise ValueError("operator_intrinsic_reward.novelty_weight must be >= 0")
        if self.effect_gain_weight < 0.0:
            raise ValueError("operator_intrinsic_reward.effect_gain_weight must be >= 0")
        if self.controllability_gain_weight < 0.0:
            raise ValueError("operator_intrinsic_reward.controllability_gain_weight must be >= 0")
        if self.norm_rate <= 0.0:
            raise ValueError("operator_intrinsic_reward.norm_rate must be > 0")
        if self.count_init <= 0.0:
            raise ValueError("operator_intrinsic_reward.count_init must be > 0")
        if self.count_decay <= 0.0:
            raise ValueError("operator_intrinsic_reward.count_decay must be > 0")
        if self.rolling_capacity_window <= 0:
            raise ValueError("operator_intrinsic_reward.rolling_capacity_window must be > 0")
        if self.min_capacity_window <= 0:
            raise ValueError("operator_intrinsic_reward.min_capacity_window must be > 0")
        if self.min_effect_gain < 0.0:
            raise ValueError("operator_intrinsic_reward.min_effect_gain must be >= 0")
        if self.min_controllability_gain < 0.0:
            raise ValueError("operator_intrinsic_reward.min_controllability_gain must be >= 0")
        if self.min_delta_x_norm < 0.0:
            raise ValueError("operator_intrinsic_reward.min_delta_x_norm must be >= 0")
        if self.min_latent_delta_norm < 0.0:
            raise ValueError("operator_intrinsic_reward.min_latent_delta_norm must be >= 0")
        if self.intensity_norm_rate <= 0.0:
            raise ValueError("operator_intrinsic_reward.intensity_norm_rate must be > 0")
        if self.running_best_mode != "ema":
            raise ValueError("operator_intrinsic_reward.running_best_mode must be 'ema'")
        if not 0.0 < self.running_best_rate <= 1.0:
            raise ValueError("operator_intrinsic_reward.running_best_rate must be in (0, 1]")
        if self.repeat_decay_tau <= 0.0:
            raise ValueError("operator_intrinsic_reward.repeat_decay_tau must be > 0")
        if not 0.0 < self.repeat_decay_min <= 1.0:
            raise ValueError("operator_intrinsic_reward.repeat_decay_min must be in (0, 1]")
        if self.topk_leaderboard_capacity <= 0:
            raise ValueError("operator_intrinsic_reward.topk_leaderboard_capacity must be > 0")
        if self.topk_exponential_base <= 1.0:
            raise ValueError("operator_intrinsic_reward.topk_exponential_base must be > 1")
        if not 0.0 < self.topk_decay <= 1.0:
            raise ValueError("operator_intrinsic_reward.topk_decay must be in (0, 1]")
        if self.topk_env_bonus_scale < 0.0:
            raise ValueError("operator_intrinsic_reward.topk_env_bonus_scale must be >= 0")
        if not 0.0 <= self.topk_env_bonus_repeat_decay <= 1.0:
            raise ValueError("operator_intrinsic_reward.topk_env_bonus_repeat_decay must be in [0, 1]")
        if not 0.0 < self.topk_env_bonus_count_decay <= 1.0:
            raise ValueError("operator_intrinsic_reward.topk_env_bonus_count_decay must be in (0, 1]")
        if self.intensity_clamp_max < self.intensity_clamp_min:
            raise ValueError("operator_intrinsic_reward.intensity_clamp_max must be >= intensity_clamp_min")
        if self.clip_reward and self.reward_clip_max < self.reward_clip_min:
            raise ValueError("operator_intrinsic_reward.reward_clip_max must be >= reward_clip_min")
        if self.enabled and not self.probe_checkpoint:
            raise ValueError("operator_intrinsic_reward.enabled=true requires probe_checkpoint")

    def asdict(self) -> dict:
        return asdict(self)


def combine_train_reward(
    env_reward: torch.Tensor,
    operator_reward: torch.Tensor | None,
    config: OperatorIntrinsicRewardConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not config.enabled or operator_reward is None:
        zero = torch.zeros_like(env_reward)
        return env_reward, env_reward, zero
    env_component = env_reward if config.use_env_reward else torch.zeros_like(env_reward)
    operator_component = float(config.beta) * operator_reward
    return env_component + operator_component, env_component, operator_component


class OperatorIntrinsicReward(nn.Module):
    def __init__(self, feat_dim: int, action_dim: int, config: OperatorIntrinsicRewardConfig):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.action_dim = int(action_dim)
        self.config = config
        self.config.validate()
        self.running_norm = RunningNormalizer(RunningNormConfig(rate=config.norm_rate, eps=1e-8))
        self.event_logit_norm = RunningNormalizer(RunningNormConfig(rate=config.intensity_norm_rate, eps=1e-8))
        self.context_delta_norm = RunningNormalizer(RunningNormConfig(rate=config.intensity_norm_rate, eps=1e-8))
        self.effect_gain_norm = RunningNormalizer(RunningNormConfig(rate=config.intensity_norm_rate, eps=1e-8))
        self.rarity_norm = RunningNormalizer(RunningNormConfig(rate=config.intensity_norm_rate, eps=1e-8))
        self.probe_model: OperatorTokenProbeModel | None = None
        self.num_tokens = 0
        self.register_buffer("running_best_intensity", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("running_best_initialized", torch.tensor(False, dtype=torch.bool))
        self.register_buffer(
            "topk_leaderboard_scores",
            torch.full((int(self.config.topk_leaderboard_capacity),), float("-inf"), dtype=torch.float32),
        )
        self.register_buffer("topk_leaderboard_size", torch.tensor(0, dtype=torch.long))
        self.register_buffer("topk_update_count", torch.tensor(0, dtype=torch.long))
        if self.config.enabled:
            self._load_probe(Path(self.config.probe_checkpoint).resolve())
        else:
            self.register_buffer("token_counts", torch.ones(1, dtype=torch.float32))
            self.register_buffer("token_env_bonus_counts", torch.zeros(1, dtype=torch.float32))
            self.register_buffer("visited_dct_tokens", torch.zeros(1, dtype=torch.bool))

    def _load_probe(self, checkpoint_path: Path):
        payload = torch.load(checkpoint_path, map_location="cpu")
        if payload.get("policy_type") != "operator_token_probe":
            raise ValueError(f"Unsupported operator probe checkpoint: {checkpoint_path}")
        saved_feat_dim = int(payload.get("feat_dim", -1))
        saved_action_dim = int(payload.get("action_dim", -1))
        if saved_feat_dim != self.feat_dim:
            raise ValueError(
                f"Operator probe feat_dim={saved_feat_dim} does not match Dreamer augmented_feat_dim={self.feat_dim}"
            )
        if saved_action_dim != self.action_dim:
            raise ValueError(
                f"Operator probe action_dim={saved_action_dim} does not match Dreamer action_dim={self.action_dim}"
            )
        token_cfg = OperatorTokenConfig(**payload["agent"]["config"]["operator_token"])
        self.num_tokens = int(token_cfg.num_tokens)
        self.probe_model = OperatorTokenProbeModel(self.feat_dim, self.action_dim, token_cfg)
        self.probe_model.load_state_dict(payload["agent"]["model_state_dict"])
        if self.config.freeze_probe:
            self.probe_model.eval()
            for param in self.probe_model.parameters():
                param.requires_grad_(False)
        self.register_buffer(
            "token_counts",
            torch.full((self.num_tokens,), float(self.config.count_init), dtype=torch.float32),
        )
        self.register_buffer(
            "token_env_bonus_counts",
            torch.zeros((self.num_tokens,), dtype=torch.float32),
        )
        self.register_buffer("visited_dct_tokens", torch.zeros((self.num_tokens,), dtype=torch.bool))

    def reset_lifetime_state(self):
        if hasattr(self, "visited_dct_tokens"):
            self.visited_dct_tokens.zero_()

    def _masked_usage_metrics(self, token_probs: torch.Tensor, mask: torch.Tensor | None) -> dict[str, torch.Tensor]:
        if mask is not None and torch.count_nonzero(mask) <= 0:
            zero_usage = torch.zeros(token_probs.shape[-1], device=token_probs.device, dtype=token_probs.dtype)
            return {
                "token_usage": zero_usage,
                "token_entropy": torch.zeros((), device=token_probs.device, dtype=token_probs.dtype),
                "token_perplexity": torch.ones((), device=token_probs.device, dtype=token_probs.dtype),
                "num_active_tokens": torch.zeros((), device=token_probs.device, dtype=token_probs.dtype),
            }
        return compute_token_usage_metrics(token_probs, mask)

    def _token_novelty(self, token_id: torch.Tensor) -> torch.Tensor:
        counts = self.token_counts.to(device=token_id.device, dtype=torch.float32)
        gathered = counts[token_id]
        return torch.rsqrt(gathered + 1e-8)

    def _operator_rarity(self, token_id: torch.Tensor) -> torch.Tensor:
        return self._token_novelty(token_id)

    def _env_bonus_repeat_scale(self, token_id: torch.Tensor) -> torch.Tensor:
        counts = self.token_env_bonus_counts.to(device=token_id.device, dtype=torch.float32)
        gathered = counts[token_id]
        if not bool(self.config.topk_env_bonus_first_only):
            return torch.ones_like(gathered)
        decay = torch.as_tensor(
            float(self.config.topk_env_bonus_repeat_decay),
            device=gathered.device,
            dtype=gathered.dtype,
        )
        if float(decay.item()) == 0.0:
            return (gathered <= 0.0).float()
        return torch.pow(decay, gathered)

    def _update_token_counts(self, token_id: torch.Tensor, mask: torch.Tensor | None):
        if not self.config.update_token_counts:
            return
        with torch.no_grad():
            counts = self.token_counts.to(device=token_id.device, dtype=torch.float32)
            counts = counts * float(self.config.count_decay)
            token_onehot = F.one_hot(token_id.reshape(-1), num_classes=self.num_tokens).float()
            if mask is not None:
                token_mask = mask.reshape(-1, 1).float()
                token_onehot = token_onehot * token_mask
            counts = counts + token_onehot.sum(dim=0)
            self.token_counts.copy_(counts.to(device=self.token_counts.device, dtype=self.token_counts.dtype))

    def _update_env_bonus_counts(self, token_id: torch.Tensor, mask: torch.Tensor | None):
        if not self.config.update_token_counts:
            return
        with torch.no_grad():
            counts = self.token_env_bonus_counts.to(device=token_id.device, dtype=torch.float32)
            counts = counts * float(self.config.topk_env_bonus_count_decay)
            token_onehot = F.one_hot(token_id.reshape(-1), num_classes=self.num_tokens).float()
            if mask is not None:
                token_mask = mask.reshape(-1, 1).float()
                token_onehot = token_onehot * token_mask
            counts = counts + token_onehot.sum(dim=0)
            self.token_env_bonus_counts.copy_(
                counts.to(device=self.token_env_bonus_counts.device, dtype=self.token_env_bonus_counts.dtype)
            )

    def _squeeze_seq(self, value: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
        if value is None:
            return torch.zeros(reference.shape[:2], device=reference.device, dtype=reference.dtype)
        if value.ndim == 0:
            value = value.reshape(1, 1).expand(reference.shape[0], reference.shape[1])
        if value.ndim == 1:
            value = value.unsqueeze(1)
        if value.ndim == 3 and value.shape[-1] == 1:
            value = value.squeeze(-1)
        if value.ndim != 2:
            raise ValueError(f"Expected [B, T] or [B, T, 1], got {tuple(value.shape)}")
        return value.to(device=reference.device, dtype=reference.dtype)

    def _masked_mean(self, values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return values.mean()
        mask_f = mask.float()
        denom = torch.clamp(mask_f.sum(), min=1.0)
        return (values * mask_f).sum() / denom

    def _normalized_component(
        self,
        values: torch.Tensor,
        normalizer: RunningNormalizer,
        mask: torch.Tensor | None,
        update_state: bool,
    ) -> torch.Tensor:
        flat = values if mask is None else values[mask > 0]
        if update_state and flat.numel() > 0:
            normalizer.update(flat)
        normalized = normalizer.normalize(values, update=False)
        return torch.nan_to_num(normalized, nan=0.0, posinf=self.config.intensity_clamp_max, neginf=self.config.intensity_clamp_min)

    def _token_repeat_decay(self, token_id: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        repeat_steps = torch.zeros_like(token_id, dtype=torch.float32)
        if token_id.shape[1] > 1:
            same_as_prev = token_id[:, 1:] == token_id[:, :-1]
            streak = torch.zeros_like(token_id[:, 0], dtype=torch.float32)
            for time_index in range(1, token_id.shape[1]):
                streak = torch.where(same_as_prev[:, time_index - 1], streak + 1.0, torch.zeros_like(streak))
                repeat_steps[:, time_index] = streak
        decay = torch.exp(-repeat_steps / float(self.config.repeat_decay_tau))
        decay = decay.clamp(min=float(self.config.repeat_decay_min), max=1.0)
        if mask is not None:
            decay = torch.where(mask > 0, decay, torch.ones_like(decay))
        return decay

    def _habituation_factor(self, rarity: torch.Tensor) -> torch.Tensor:
        factor = torch.clamp(rarity, min=0.0, max=1.0)
        factor = factor.pow(float(self.config.frequency_habituation_power))
        return torch.nan_to_num(factor, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)

    def _safe_exp_reward(self, multiplier: float, exponent_count: int) -> float:
        try:
            value = float(multiplier) * (float(self.config.topk_exponential_base) ** float(exponent_count))
        except OverflowError:
            value = float("inf")
        if self.config.clip_reward:
            value = min(value, float(self.config.reward_clip_max))
        return max(0.0, value)

    def _milestone_reward(
        self,
        event_logit_t: torch.Tensor,
        context_delta_norm_t: torch.Tensor,
        transition_effect_gain: torch.Tensor,
        operator_rarity: torch.Tensor,
        token_id: torch.Tensor,
        capacity_mask: torch.Tensor | None,
        update_state: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        normalized_event_logit = self._normalized_component(
            event_logit_t, self.event_logit_norm, capacity_mask, update_state
        )
        normalized_context_delta = self._normalized_component(
            context_delta_norm_t, self.context_delta_norm, capacity_mask, update_state
        )
        normalized_effect_gain = self._normalized_component(
            transition_effect_gain, self.effect_gain_norm, capacity_mask, update_state
        )
        normalized_operator_rarity = self._normalized_component(
            operator_rarity, self.rarity_norm, capacity_mask, update_state
        )

        intensity = (
            float(self.config.intensity_event_logit_weight) * normalized_event_logit
            + float(self.config.intensity_context_delta_weight) * normalized_context_delta
            + float(self.config.intensity_effect_gain_weight) * normalized_effect_gain
            + float(self.config.intensity_operator_rarity_weight) * normalized_operator_rarity
        )
        intensity = torch.nan_to_num(
            intensity,
            nan=0.0,
            posinf=float(self.config.intensity_clamp_max),
            neginf=float(self.config.intensity_clamp_min),
        ).clamp_(float(self.config.intensity_clamp_min), float(self.config.intensity_clamp_max))

        running_best_value = self.running_best_intensity.to(device=intensity.device, dtype=intensity.dtype)
        margin = torch.as_tensor(float(self.config.milestone_margin), device=intensity.device, dtype=intensity.dtype)
        milestone_base = torch.relu(intensity - running_best_value - margin)
        habituation_factor = self._habituation_factor(operator_rarity)
        token_repeat_decay = self._token_repeat_decay(token_id, capacity_mask)
        milestone_reward = milestone_base * habituation_factor * token_repeat_decay

        if capacity_mask is not None:
            milestone_reward = milestone_reward * capacity_mask
            milestone_base = milestone_base * capacity_mask
            habituation_factor = torch.where(capacity_mask > 0, habituation_factor, torch.zeros_like(habituation_factor))
            token_repeat_decay = torch.where(capacity_mask > 0, token_repeat_decay, torch.ones_like(token_repeat_decay))

        if update_state:
            masked_intensity = intensity if capacity_mask is None else intensity[capacity_mask > 0]
            if masked_intensity.numel() > 0:
                batch_peak = masked_intensity.max().detach()
                if bool(self.running_best_initialized.item()):
                    rate = float(self.config.running_best_rate)
                    updated = (1.0 - rate) * self.running_best_intensity + rate * batch_peak.to(
                        device=self.running_best_intensity.device,
                        dtype=self.running_best_intensity.dtype,
                    )
                    self.running_best_intensity.copy_(updated)
                else:
                    rate = float(self.config.running_best_rate)
                    self.running_best_intensity.copy_(
                        rate
                        * batch_peak.to(
                            device=self.running_best_intensity.device,
                            dtype=self.running_best_intensity.dtype,
                        )
                    )
                    self.running_best_initialized.fill_(True)

        milestone_trigger = (milestone_reward > 0).float()
        intensity_top_mean = self._masked_mean(intensity, capacity_mask)
        intensity_normal_mean = self._masked_mean(
            intensity,
            None if capacity_mask is None else (1.0 - capacity_mask),
        )
        return milestone_reward, {
            "event_intensity": intensity.detach(),
            "milestone_reward": milestone_reward.detach(),
            "milestone_trigger": milestone_trigger.detach(),
            "habituation_factor": habituation_factor.detach(),
            "token_repeat_decay": token_repeat_decay.detach(),
            "operator_rarity": operator_rarity.detach(),
            "running_best_intensity": self.running_best_intensity.detach().to(
                device=intensity.device,
                dtype=intensity.dtype,
            ),
            "event_intensity_top_mean": intensity_top_mean.detach(),
            "event_intensity_normal_mean": intensity_normal_mean.detach(),
        }

    def _topk_exponential_reward(
        self,
        intensity: torch.Tensor,
        operator_rarity: torch.Tensor,
        token_id: torch.Tensor,
        capacity_mask: torch.Tensor | None,
        env_reward_t: torch.Tensor | None,
        update_state: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        capacity = int(self.config.topk_leaderboard_capacity)
        replace_margin = float(self.config.topk_replace_margin)

        reward = torch.zeros_like(intensity)
        trigger = torch.zeros_like(intensity)
        rank_tensor = torch.zeros_like(intensity)
        env_bonus_tensor = torch.zeros_like(intensity)
        entered_tensor = torch.zeros_like(intensity)

        habituation_factor = self._habituation_factor(operator_rarity)
        token_repeat_decay = self._token_repeat_decay(token_id, capacity_mask)

        leaderboard_scores = self.topk_leaderboard_scores.detach().to(device=intensity.device, dtype=intensity.dtype).clone()
        leaderboard_size = int(self.topk_leaderboard_size.item())
        update_count = int(self.topk_update_count.item())

        if update_state and leaderboard_size > 0:
            leaderboard_scores[:leaderboard_size] = leaderboard_scores[:leaderboard_size] * float(self.config.topk_decay)

        capacity_bool = None
        if capacity_mask is not None:
            capacity_bool = capacity_mask > 0

        env_reward_seq = None
        if env_reward_t is not None:
            env_reward_seq = self._squeeze_seq(env_reward_t, intensity)

        flat_intensity = intensity.reshape(-1)
        flat_reward = reward.reshape(-1)
        flat_trigger = trigger.reshape(-1)
        flat_rank = rank_tensor.reshape(-1)
        flat_bonus = env_bonus_tensor.reshape(-1)
        flat_entered = entered_tensor.reshape(-1)
        flat_habituation = habituation_factor.reshape(-1)
        flat_repeat_decay = token_repeat_decay.reshape(-1)
        flat_env_reward = None if env_reward_seq is None else env_reward_seq.reshape(-1)
        flat_mask = None if capacity_bool is None else capacity_bool.reshape(-1)
        env_bonus_mask = torch.zeros_like(intensity, dtype=torch.float32)
        flat_env_bonus_mask = env_bonus_mask.reshape(-1)
        env_bonus_repeat_scale = self._env_bonus_repeat_scale(token_id)
        flat_env_bonus_repeat_scale = env_bonus_repeat_scale.reshape(-1)

        for flat_index in range(flat_intensity.numel()):
            if flat_mask is not None and not bool(flat_mask[flat_index].item()):
                continue

            intensity_value = float(flat_intensity[flat_index].item())
            if not torch.isfinite(flat_intensity[flat_index]):
                continue

            entered_new = False
            rank = 0
            reward_base = 0.0

            if leaderboard_size < capacity:
                existing_scores = leaderboard_scores[:leaderboard_size]
                rank = 1 + int(torch.count_nonzero(existing_scores > intensity_value).item()) if leaderboard_size > 0 else 1
                rank = max(1, min(rank, leaderboard_size + 1))
                reward_base = float(capacity) / float(rank)
                entered_new = True
                if update_state:
                    updated_scores = torch.cat(
                        [
                            existing_scores,
                            torch.as_tensor([intensity_value], device=leaderboard_scores.device, dtype=leaderboard_scores.dtype),
                        ]
                    )
                    sorted_scores, _ = torch.sort(updated_scores, descending=True)
                    leaderboard_size = min(capacity, int(sorted_scores.numel()))
                    leaderboard_scores[:leaderboard_size] = sorted_scores[:leaderboard_size]
                    if leaderboard_size < capacity:
                        leaderboard_scores[leaderboard_size:] = float("-inf")
            elif leaderboard_size > 0:
                existing_scores = leaderboard_scores[:leaderboard_size]
                threshold = float(existing_scores[-1].item()) + replace_margin
                if intensity_value > threshold:
                    rank = 1 + int(torch.count_nonzero(existing_scores > intensity_value).item())
                    rank = max(1, min(rank, capacity))
                    next_update_count = update_count + 1
                    reward_base = self._safe_exp_reward(float(capacity) / float(rank), next_update_count)
                    entered_new = True
                    if update_state:
                        updated_scores = torch.cat(
                            [
                                existing_scores,
                                torch.as_tensor([intensity_value], device=leaderboard_scores.device, dtype=leaderboard_scores.dtype),
                            ]
                        )
                        sorted_scores, _ = torch.sort(updated_scores, descending=True)
                        leaderboard_scores[:capacity] = sorted_scores[:capacity]
                        leaderboard_size = capacity
                        update_count = next_update_count

            if not entered_new:
                continue

            scaled_reward = reward_base * float(flat_habituation[flat_index].item()) * float(flat_repeat_decay[flat_index].item())
            env_bonus = 0.0
            if flat_env_reward is not None and float(flat_env_reward[flat_index].item()) > 0.0:
                env_bonus = self._safe_exp_reward(float(self.config.topk_env_bonus_scale), update_count + 1)
                env_bonus = env_bonus * float(flat_env_bonus_repeat_scale[flat_index].item())
                if env_bonus > 0.0:
                    flat_env_bonus_mask[flat_index] = 1.0

            flat_reward[flat_index] = scaled_reward + env_bonus
            flat_trigger[flat_index] = 1.0
            flat_rank[flat_index] = float(rank)
            flat_bonus[flat_index] = float(env_bonus)
            flat_entered[flat_index] = 1.0

        if update_state:
            if torch.count_nonzero(env_bonus_mask) > 0:
                self._update_env_bonus_counts(token_id, env_bonus_mask)
            self.topk_leaderboard_scores.copy_(
                leaderboard_scores.to(
                    device=self.topk_leaderboard_scores.device,
                    dtype=self.topk_leaderboard_scores.dtype,
                )
            )
            self.topk_leaderboard_size.fill_(int(leaderboard_size))
            self.topk_update_count.fill_(int(update_count))

        if self.config.clip_reward:
            reward = reward.clamp(min=float(self.config.reward_clip_min), max=float(self.config.reward_clip_max))

        if capacity_bool is not None:
            reward = reward * capacity_mask
            trigger = torch.where(capacity_bool, trigger, torch.zeros_like(trigger))
            rank_tensor = torch.where(capacity_bool, rank_tensor, torch.zeros_like(rank_tensor))
            env_bonus_tensor = torch.where(capacity_bool, env_bonus_tensor, torch.zeros_like(env_bonus_tensor))
            entered_tensor = torch.where(capacity_bool, entered_tensor, torch.zeros_like(entered_tensor))
            habituation_factor = torch.where(capacity_bool, habituation_factor, torch.zeros_like(habituation_factor))
            token_repeat_decay = torch.where(capacity_bool, token_repeat_decay, torch.ones_like(token_repeat_decay))

        if leaderboard_size > 0:
            cutoff_value = leaderboard_scores[min(leaderboard_size - 1, capacity - 1)]
        else:
            cutoff_value = torch.zeros((), device=intensity.device, dtype=intensity.dtype)

        return reward, {
            "milestone_reward": reward.detach(),
            "milestone_trigger": trigger.detach(),
            "habituation_factor": habituation_factor.detach(),
            "token_repeat_decay": token_repeat_decay.detach(),
            "operator_rarity": operator_rarity.detach(),
            "running_best_intensity": cutoff_value.detach(),
            "event_intensity": intensity.detach(),
            "event_intensity_top_mean": self._masked_mean(intensity, capacity_mask).detach(),
            "event_intensity_normal_mean": self._masked_mean(
                intensity,
                None if capacity_mask is None else (1.0 - capacity_mask),
            ).detach(),
            "topk_rank": rank_tensor.detach(),
            "topk_env_bonus": env_bonus_tensor.detach(),
            "topk_entered": entered_tensor.detach(),
            "topk_rank_mean": rank_tensor.detach().float().mean(),
            "topk_env_bonus_mean": env_bonus_tensor.detach().float().mean(),
            "topk_entered_ratio": entered_tensor.detach().float().mean(),
            "topk_leaderboard_size": torch.as_tensor(float(leaderboard_size), device=intensity.device, dtype=intensity.dtype),
            "topk_update_count": torch.as_tensor(float(update_count), device=intensity.device, dtype=intensity.dtype),
        }

    def _dct_unigram_milestone_reward(
        self,
        token_id: torch.Tensor,
        capacity_mask: torch.Tensor | None,
        lifetime_reset_t: torch.Tensor | None,
        update_state: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        reward = torch.zeros(token_id.shape, device=token_id.device, dtype=torch.float32)
        trigger = torch.zeros_like(reward)
        rank_tensor = torch.zeros_like(reward)
        if self.num_tokens <= 0:
            visited = torch.zeros((*token_id.shape[:1], 1), device=token_id.device, dtype=torch.bool)
        else:
            visited = torch.zeros((token_id.shape[0], self.num_tokens), device=token_id.device, dtype=torch.bool)

        use_module_lifetime = bool(update_state and token_id.shape[0] == 1 and token_id.shape[1] == 1)
        if use_module_lifetime and hasattr(self, "visited_dct_tokens"):
            visited = self.visited_dct_tokens.to(device=token_id.device).view(1, -1).clone()

        if capacity_mask is None:
            capacity_bool = torch.ones(token_id.shape, device=token_id.device, dtype=torch.bool)
        else:
            capacity_bool = capacity_mask.to(device=token_id.device).reshape(token_id.shape) > 0

        if lifetime_reset_t is None:
            reset_bool = torch.zeros(token_id.shape, device=token_id.device, dtype=torch.bool)
        else:
            reset_bool = lifetime_reset_t.to(device=token_id.device).reshape(token_id.shape) > 0

        for t in range(token_id.shape[1]):
            reset_rows = reset_bool[:, t]
            if torch.any(reset_rows):
                visited[reset_rows] = False
            tokens_t = token_id[:, t].long().clamp(min=0, max=max(self.num_tokens - 1, 0))
            if self.num_tokens <= 0:
                continue
            already_seen = visited.gather(1, tokens_t.view(-1, 1)).squeeze(1)
            fire = capacity_bool[:, t] & (~already_seen)
            if torch.any(fire):
                reward[fire, t] = float(self.config.r_new_dct)
                trigger[fire, t] = 1.0
                rank_tensor[fire, t] = tokens_t[fire].float() + 1.0
                visited[fire, tokens_t[fire]] = True

        if use_module_lifetime and hasattr(self, "visited_dct_tokens"):
            self.visited_dct_tokens.copy_(visited.squeeze(0).to(device=self.visited_dct_tokens.device))

        zero = torch.zeros((), device=reward.device, dtype=reward.dtype)
        one = torch.ones_like(reward)
        return reward, {
            "milestone_reward": reward.detach(),
            "milestone_trigger": trigger.detach(),
            "habituation_factor": one.detach(),
            "token_repeat_decay": one.detach(),
            "operator_rarity": torch.zeros_like(reward).detach(),
            "running_best_intensity": zero.detach(),
            "event_intensity": torch.zeros_like(reward).detach(),
            "event_intensity_top_mean": zero.detach(),
            "event_intensity_normal_mean": zero.detach(),
            "topk_rank": rank_tensor.detach(),
            "topk_env_bonus": torch.zeros_like(reward).detach(),
            "topk_entered": trigger.detach(),
            "topk_rank_mean": rank_tensor.detach().float().mean(),
            "topk_env_bonus_mean": zero.detach(),
            "topk_entered_ratio": trigger.detach().float().mean(),
            "topk_leaderboard_size": torch.as_tensor(float(self.num_tokens), device=reward.device, dtype=reward.dtype),
            "topk_update_count": trigger.detach().sum(),
        }

    def forward(
        self,
        x_t: torch.Tensor,
        x_tp1: torch.Tensor,
        action_t: torch.Tensor,
        capacity_gate_t: torch.Tensor | None = None,
        env_reward_t: torch.Tensor | None = None,
        event_logit_t: torch.Tensor | None = None,
        context_delta_norm_t: torch.Tensor | None = None,
        ordinary_error_t: torch.Tensor | None = None,
        mixed_error_t: torch.Tensor | None = None,
        lifetime_reset_t: torch.Tensor | None = None,
        update_state: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | list[float] | int]]:
        if not self.config.enabled or self.probe_model is None:
            zeros = torch.zeros(x_t.shape[:2], device=x_t.device, dtype=x_t.dtype)
            return zeros, {
                "intrinsic_reward": zeros,
                "operator_token_id": torch.zeros_like(zeros, dtype=torch.long),
                "bad_numeric_count": torch.tensor(0.0, device=x_t.device),
            }

        if self.config.detach_features:
            x_t = x_t.detach()
            x_tp1 = x_tp1.detach()
            action_t = action_t.detach()
        delta_x = (x_tp1 - x_t).detach()

        if capacity_gate_t is not None:
            capacity_mask = capacity_gate_t.float()
            if capacity_mask.ndim == 3 and capacity_mask.shape[-1] == 1:
                capacity_mask = capacity_mask.squeeze(-1)
            capacity_mask = (capacity_mask > 0).float()
        else:
            capacity_mask = None

        if self.config.freeze_probe:
            self.probe_model.eval()
            context = torch.no_grad()
        else:
            context = torch.enable_grad()

        with context:
            outputs = self.probe_model(
                {
                    "x_t": x_t,
                    "x_tp1": x_tp1,
                    "delta_x": delta_x,
                    "action_t": action_t,
                }
            )

            effect_token_error = F.smooth_l1_loss(outputs["pred_delta_x"], delta_x, reduction="none").mean(dim=-1)
            effect_no_token_error = F.smooth_l1_loss(outputs["pred_delta_x_no_token"], delta_x, reduction="none").mean(dim=-1)
            inverse_token_error = F.smooth_l1_loss(outputs["pred_action"], action_t, reduction="none").mean(dim=-1)
            inverse_baseline_error = F.smooth_l1_loss(outputs["pred_action_no_token"], action_t, reduction="none").mean(dim=-1)

            effect_gain = torch.relu(effect_no_token_error - effect_token_error)
            controllability_gain = torch.relu(inverse_baseline_error - inverse_token_error)
            novelty = self._token_novelty(outputs["token_id"])
            operator_rarity = self._operator_rarity(outputs["token_id"])
            delta_x_norm = torch.linalg.vector_norm(delta_x, dim=-1)
            effect_ready = (effect_gain > float(self.config.min_effect_gain)) & (
                delta_x_norm > float(self.config.min_delta_x_norm)
            )
            stale_keep = torch.ones_like(effect_gain, dtype=torch.bool)
            if self.config.stale_filter_enabled:
                stale_keep = delta_x_norm >= float(self.config.min_latent_delta_norm)
            valid_effect = (effect_ready & stale_keep).float()
            valid_control_bool = controllability_gain > float(self.config.min_controllability_gain)
            if self.config.require_effect_for_control:
                valid_control_bool = valid_control_bool & effect_ready
            valid_control = (valid_control_bool & stale_keep).float()
            control_only_suppressed = (
                (controllability_gain > float(self.config.min_controllability_gain)) & (~effect_ready)
            ).float()

            transition_effect_gain = torch.relu(
                self._squeeze_seq(ordinary_error_t, x_t) - self._squeeze_seq(mixed_error_t, x_t)
            )

            milestone_info: dict[str, torch.Tensor] = {}
            if self.config.reward_mode == "milestone":
                reward, milestone_info = self._milestone_reward(
                    event_logit_t=self._squeeze_seq(event_logit_t, x_t),
                    context_delta_norm_t=self._squeeze_seq(context_delta_norm_t, x_t),
                    transition_effect_gain=transition_effect_gain,
                    operator_rarity=operator_rarity,
                    token_id=outputs["token_id"],
                    capacity_mask=capacity_mask,
                    update_state=update_state,
                )
            elif self.config.reward_mode == "dct_unigram_milestone":
                reward, milestone_info = self._dct_unigram_milestone_reward(
                    token_id=outputs["token_id"],
                    capacity_mask=capacity_mask,
                    lifetime_reset_t=lifetime_reset_t,
                    update_state=update_state,
                )
            elif self.config.reward_mode == "topk_exponential":
                intensity = (
                    float(self.config.intensity_event_logit_weight)
                    * self._normalized_component(self._squeeze_seq(event_logit_t, x_t), self.event_logit_norm, capacity_mask, update_state)
                    + float(self.config.intensity_context_delta_weight)
                    * self._normalized_component(
                        self._squeeze_seq(context_delta_norm_t, x_t), self.context_delta_norm, capacity_mask, update_state
                    )
                    + float(self.config.intensity_effect_gain_weight)
                    * self._normalized_component(transition_effect_gain, self.effect_gain_norm, capacity_mask, update_state)
                    + float(self.config.intensity_operator_rarity_weight)
                    * self._normalized_component(operator_rarity, self.rarity_norm, capacity_mask, update_state)
                )
                intensity = torch.nan_to_num(
                    intensity,
                    nan=0.0,
                    posinf=float(self.config.intensity_clamp_max),
                    neginf=float(self.config.intensity_clamp_min),
                ).clamp_(float(self.config.intensity_clamp_min), float(self.config.intensity_clamp_max))
                reward, milestone_info = self._topk_exponential_reward(
                    intensity=intensity,
                    operator_rarity=operator_rarity,
                    token_id=outputs["token_id"],
                    capacity_mask=capacity_mask,
                    env_reward_t=env_reward_t,
                    update_state=update_state,
                )
            else:
                reward = (
                    float(self.config.novelty_weight) * novelty * valid_effect
                    + float(self.config.effect_gain_weight) * effect_gain * valid_effect
                    + float(self.config.controllability_gain_weight) * controllability_gain * valid_control
                )
                milestone_info = {
                    "event_intensity": torch.zeros_like(reward),
                    "milestone_reward": reward.detach(),
                    "milestone_trigger": (reward > 0).float().detach(),
                    "habituation_factor": torch.ones_like(reward),
                    "token_repeat_decay": torch.ones_like(reward),
                    "operator_rarity": operator_rarity.detach(),
                    "running_best_intensity": self.running_best_intensity.detach().to(
                        device=reward.device,
                        dtype=reward.dtype,
                    ),
                    "event_intensity_top_mean": torch.zeros((), device=reward.device, dtype=reward.dtype),
                    "event_intensity_normal_mean": torch.zeros((), device=reward.device, dtype=reward.dtype),
                    "topk_rank": torch.zeros_like(reward),
                    "topk_env_bonus": torch.zeros_like(reward),
                    "topk_entered": torch.zeros_like(reward),
                    "topk_leaderboard_size": torch.zeros((), device=reward.device, dtype=reward.dtype),
                    "topk_update_count": torch.zeros((), device=reward.device, dtype=reward.dtype),
                }

            pre_capacity_reward = reward
            if self.config.capacity_only:
                if capacity_mask is None or torch.count_nonzero(capacity_mask) <= 0:
                    reward = torch.zeros_like(reward)
                    valid_effect = torch.zeros_like(valid_effect)
                    valid_control = torch.zeros_like(valid_control)
                    milestone_info["milestone_reward"] = torch.zeros_like(reward)
                    milestone_info["milestone_trigger"] = torch.zeros_like(reward)
                    milestone_info["habituation_factor"] = torch.zeros_like(reward)
                    milestone_info["token_repeat_decay"] = torch.ones_like(reward)
                    milestone_info["topk_rank"] = torch.zeros_like(reward)
                    milestone_info["topk_env_bonus"] = torch.zeros_like(reward)
                    milestone_info["topk_entered"] = torch.zeros_like(reward)
                else:
                    reward = reward * capacity_mask
                    valid_effect = valid_effect * capacity_mask
                    valid_control = valid_control * capacity_mask

            raw_reward = reward.detach()
            if self.config.normalize_reward:
                if update_state:
                    self.running_norm.update(raw_reward)
                reward = self.running_norm.normalize(raw_reward, update=False)
            else:
                reward = raw_reward

            if self.config.clip_reward:
                reward = reward.clamp(min=float(self.config.reward_clip_min), max=float(self.config.reward_clip_max))
            clipped_reward = reward.detach()

            if self.config.capacity_only and self.config.post_mask_after_normalize:
                if capacity_mask is None or torch.count_nonzero(capacity_mask) <= 0:
                    reward = torch.zeros_like(reward)
                else:
                    reward = reward * capacity_mask

        if update_state:
            self._update_token_counts(outputs["token_id"].detach(), capacity_mask)

        usage_metrics = self._masked_usage_metrics(outputs["token_probs"].detach(), capacity_mask)
        if capacity_mask is None:
            outside_capacity = torch.zeros_like(reward)
        else:
            outside_capacity = (1.0 - capacity_mask.float()).clamp(0.0, 1.0)
        reward_outside_capacity = ((reward.detach() > 0) & (outside_capacity > 0)).float()
        mask_violation_count = reward_outside_capacity.sum()
        if outside_capacity.sum() > 0:
            nonzero_outside_capacity = reward_outside_capacity.sum() / torch.clamp(outside_capacity.sum(), min=1.0)
        else:
            nonzero_outside_capacity = torch.zeros((), device=x_t.device, dtype=x_t.dtype)
        stale_suppressed = ((pre_capacity_reward.detach() > 0) & (~stale_keep)).float()
        saturation = torch.zeros_like(reward.detach())
        if self.config.clip_reward:
            saturation = (clipped_reward >= float(self.config.reward_clip_max)).float()
        info: dict[str, torch.Tensor | list[float] | int] = {
            "intrinsic_reward": reward.detach(),
            "operator_token_id": outputs["token_id"].detach(),
            "operator_token_novelty": novelty.detach(),
            "operator_rarity": operator_rarity.detach(),
            "operator_effect_gain": effect_gain.detach(),
            "transition_effect_gain": transition_effect_gain.detach(),
            "operator_controllability_gain": controllability_gain.detach(),
            "operator_valid_effect": valid_effect.detach(),
            "operator_valid_control": valid_control.detach(),
            "operator_raw_reward": raw_reward.detach(),
            "operator_clipped_reward": clipped_reward.detach(),
            "operator_reward_saturation": saturation.detach(),
            "operator_reward_nonzero_outside_capacity": nonzero_outside_capacity.detach(),
            "operator_reward_mask_violation_count": mask_violation_count.detach(),
            "operator_control_only_suppressed": control_only_suppressed.detach(),
            "operator_valid_transition": (valid_effect > 0).float().detach(),
            "operator_stale_suppressed": stale_suppressed.detach(),
            "operator_latent_delta_norm": delta_x_norm.detach(),
            "event_intensity": milestone_info["event_intensity"],
            "event_intensity_mean": milestone_info["event_intensity"].detach().mean(),
            "event_intensity_std": milestone_info["event_intensity"].detach().float().std(unbiased=False),
            "event_intensity_max": milestone_info["event_intensity"].detach().max(),
            "milestone_reward": milestone_info["milestone_reward"],
            "milestone_reward_mean": milestone_info["milestone_reward"].detach().mean(),
            "milestone_trigger_count": milestone_info["milestone_trigger"].detach().sum(),
            "milestone_nonzero_ratio": milestone_info["milestone_trigger"].detach().float().mean(),
            "running_best_intensity": milestone_info["running_best_intensity"],
            "habituation_factor": milestone_info["habituation_factor"],
            "habituation_factor_mean": milestone_info["habituation_factor"].detach().mean(),
            "token_repeat_decay": milestone_info["token_repeat_decay"],
            "token_repeat_decay_mean": milestone_info["token_repeat_decay"].detach().mean(),
            "event_intensity_top_mean": milestone_info["event_intensity_top_mean"],
            "event_intensity_normal_mean": milestone_info["event_intensity_normal_mean"],
            "topk_rank": milestone_info["topk_rank"].detach(),
            "topk_rank_mean": milestone_info["topk_rank"].detach().float().mean(),
            "topk_env_bonus": milestone_info["topk_env_bonus"].detach(),
            "topk_env_bonus_mean": milestone_info["topk_env_bonus"].detach().float().mean(),
            "topk_entered": milestone_info["topk_entered"].detach(),
            "topk_entered_ratio": milestone_info["topk_entered"].detach().float().mean(),
            "topk_leaderboard_size": milestone_info["topk_leaderboard_size"].detach(),
            "topk_update_count": milestone_info["topk_update_count"].detach(),
            "operator_token_perplexity_online": usage_metrics["token_perplexity"].detach(),
            "operator_num_active_tokens_online": usage_metrics["num_active_tokens"].detach(),
            "operator_token_count_min": self.token_counts.detach().float().min(),
            "operator_token_count_max": self.token_counts.detach().float().max(),
            "operator_bad_numeric_count": torch.as_tensor(
                float(
                    bad_numeric_count(
                        reward,
                        raw_reward,
                        clipped_reward,
                        novelty,
                        operator_rarity,
                        effect_gain,
                        controllability_gain,
                        delta_x_norm,
                        transition_effect_gain,
                        milestone_info["event_intensity"],
                        milestone_info["milestone_reward"],
                        milestone_info["habituation_factor"],
                        milestone_info["token_repeat_decay"],
                        outputs["pred_delta_x"],
                        outputs["pred_action"],
                    )
                ),
                device=x_t.device,
            ),
        }
        return reward.detach(), info

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        state[prefix + "_running_norm"] = {
            "config": {"rate": self.running_norm.config.rate, "eps": self.running_norm.config.eps},
            "mean": self.running_norm.mean,
            "var": self.running_norm.var,
            "initialized": self.running_norm.initialized,
        }
        for name, normalizer in (
            ("_event_logit_norm", self.event_logit_norm),
            ("_context_delta_norm", self.context_delta_norm),
            ("_effect_gain_norm", self.effect_gain_norm),
            ("_rarity_norm", self.rarity_norm),
        ):
            state[prefix + name] = normalizer.state_dict()
        return state

    def load_state_dict(self, state_dict, strict: bool = True):
        running_norm_payload = state_dict.pop("_running_norm", None)
        event_logit_norm_payload = state_dict.pop("_event_logit_norm", None)
        context_delta_norm_payload = state_dict.pop("_context_delta_norm", None)
        effect_gain_norm_payload = state_dict.pop("_effect_gain_norm", None)
        rarity_norm_payload = state_dict.pop("_rarity_norm", None)
        result = super().load_state_dict(state_dict, strict=strict)
        if running_norm_payload is not None:
            self.running_norm = RunningNormalizer.from_state_dict(running_norm_payload)
        if event_logit_norm_payload is not None:
            self.event_logit_norm = RunningNormalizer.from_state_dict(event_logit_norm_payload)
        if context_delta_norm_payload is not None:
            self.context_delta_norm = RunningNormalizer.from_state_dict(context_delta_norm_payload)
        if effect_gain_norm_payload is not None:
            self.effect_gain_norm = RunningNormalizer.from_state_dict(effect_gain_norm_payload)
        if rarity_norm_payload is not None:
            self.rarity_norm = RunningNormalizer.from_state_dict(rarity_norm_payload)
        return result
