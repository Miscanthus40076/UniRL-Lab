from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import math
from collections import deque

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
    milestone_reward: float = 1.0
    signature_source: str = "delta_x_random_projection"
    signature_dim: int = 32
    signature_seed: int = 0
    signature_normalize: bool = True
    milestone_memory_capacity: int = 128
    seen_memory_enabled: bool = True
    memory_scope: str = "run"
    reset_memory_on_env_reset: bool = False
    reset_memory_on_lifetime_reset: bool = False
    reset_memory_on_new_training_run: bool = True
    seen_similarity_threshold: float = 0.20
    active_similarity_threshold: float = 0.20
    preemption_enabled: bool = True
    preemption_margin: float = 0.05
    recent_token_window: int = 20
    recent_token_repeat_threshold: float = 0.5
    forbid_dense_gain_reward: bool = False
    compute_online_only: bool = False
    store_reward_in_replay: bool = False
    forbid_batch_recompute_for_milestone: bool = False
    use_env_reward: bool = True
    capacity_only: bool = True
    post_mask_after_normalize: bool = True
    eval_use_rolling_capacity: bool = True
    rolling_capacity_window: int = 64
    min_capacity_window: int = 16
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
    forbid_batch_recompute_for_sequence_milestone: bool = False
    enable_v2_single_event_reward: bool = False
    temporal_buffer_size: int = 64
    min_event_gap_steps: int = 3
    max_sequence_span_steps: int = 500
    use_bigram_milestone: bool = True
    use_trigram_milestone: bool = True
    bigram_milestone_reward: float = 1.0
    trigram_milestone_reward: float = 2.0
    sequence_reward_step_cap: float = 2.0
    sequence_signature_dim: int = 64
    sequence_signature_seed: int = 1
    sequence_signature_normalize: bool = True
    sequence_seen_memory_enabled: bool = True
    sequence_memory_scope: str = "run"
    bigram_memory_capacity: int = 256
    trigram_memory_capacity: int = 512
    reset_sequence_memory_on_env_reset: bool = False
    reset_sequence_memory_on_lifetime_reset: bool = False
    sequence_seen_similarity_threshold: float = 0.20
    sequence_active_similarity_threshold: float = 0.20
    sequence_preemption_enabled: bool = True
    sequence_preemption_margin: float = 0.05

    def validate(self):
        if self.beta < 0.0:
            raise ValueError("operator_intrinsic_reward.beta must be >= 0")
        if self.reward_mode not in {
            "dense_gain",
            "milestone",
            "topk_exponential",
            "dct_unigram_milestone",
            "dct_effect_milestone_v2",
            "dct_temporal_sequence_milestone_v3",
        }:
            raise ValueError(
                "operator_intrinsic_reward.reward_mode must be 'dense_gain', 'milestone', "
                "'topk_exponential', 'dct_unigram_milestone', 'dct_effect_milestone_v2' "
                "or 'dct_temporal_sequence_milestone_v3'"
            )
        if self.forbid_dense_gain_reward and self.reward_mode == "dense_gain":
            raise ValueError("operator_intrinsic_reward.forbid_dense_gain_reward=true forbids reward_mode=dense_gain")
        if self.r_new_dct < 0.0:
            raise ValueError("operator_intrinsic_reward.r_new_dct must be >= 0")
        if self.milestone_reward < 0.0:
            raise ValueError("operator_intrinsic_reward.milestone_reward must be >= 0")
        if self.signature_source != "delta_x_random_projection":
            raise ValueError("operator_intrinsic_reward.signature_source must be 'delta_x_random_projection'")
        if self.signature_dim <= 0:
            raise ValueError("operator_intrinsic_reward.signature_dim must be positive")
        if self.milestone_memory_capacity <= 0:
            raise ValueError("operator_intrinsic_reward.milestone_memory_capacity must be positive")
        if self.memory_scope != "run":
            raise ValueError("operator_intrinsic_reward.memory_scope must be 'run'")
        if self.seen_similarity_threshold < 0.0 or self.active_similarity_threshold < 0.0:
            raise ValueError("operator_intrinsic_reward similarity thresholds must be >= 0")
        if self.preemption_margin < 0.0:
            raise ValueError("operator_intrinsic_reward.preemption_margin must be >= 0")
        if self.recent_token_window <= 0:
            raise ValueError("operator_intrinsic_reward.recent_token_window must be positive")
        if not 0.0 <= self.recent_token_repeat_threshold <= 1.0:
            raise ValueError("operator_intrinsic_reward.recent_token_repeat_threshold must be in [0, 1]")
        if self.reward_mode == "dct_effect_milestone_v2":
            if not self.compute_online_only or not self.store_reward_in_replay or not self.forbid_batch_recompute_for_milestone:
                raise ValueError(
                    "dct_effect_milestone_v2 requires compute_online_only=true, "
                    "store_reward_in_replay=true and forbid_batch_recompute_for_milestone=true"
                )
            if self.normalize_reward:
                raise ValueError("dct_effect_milestone_v2 requires normalize_reward=false")
            if self.clip_reward:
                raise ValueError("dct_effect_milestone_v2 requires clip_reward=false")
        if self.reward_mode == "dct_temporal_sequence_milestone_v3":
            if not (
                self.compute_online_only
                and self.store_reward_in_replay
                and self.forbid_batch_recompute_for_milestone
                and self.forbid_batch_recompute_for_sequence_milestone
            ):
                raise ValueError(
                    "dct_temporal_sequence_milestone_v3 requires online reward storage and forbids batch recompute"
                )
            if self.normalize_reward:
                raise ValueError("dct_temporal_sequence_milestone_v3 requires normalize_reward=false")
            if self.clip_reward:
                raise ValueError("dct_temporal_sequence_milestone_v3 requires clip_reward=false")
            if self.temporal_buffer_size <= 0:
                raise ValueError("operator_intrinsic_reward.temporal_buffer_size must be positive")
            if self.min_event_gap_steps < 1:
                raise ValueError("operator_intrinsic_reward.min_event_gap_steps must be >= 1")
            if self.max_sequence_span_steps < self.min_event_gap_steps:
                raise ValueError("operator_intrinsic_reward.max_sequence_span_steps must be >= min_event_gap_steps")
            if self.sequence_signature_dim <= 0:
                raise ValueError("operator_intrinsic_reward.sequence_signature_dim must be positive")
            if self.sequence_memory_scope != "run":
                raise ValueError("operator_intrinsic_reward.sequence_memory_scope must be 'run'")
            if self.bigram_memory_capacity <= 0 or self.trigram_memory_capacity <= 0:
                raise ValueError("operator_intrinsic_reward sequence memory capacities must be positive")
            if self.sequence_reward_step_cap < 0.0:
                raise ValueError("operator_intrinsic_reward.sequence_reward_step_cap must be >= 0")
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


class DCTMilestoneMemory:
    def __init__(
        self,
        num_tokens: int,
        signature_dim: int,
        capacity: int,
        seen_enabled: bool = True,
        recent_token_window: int = 20,
        recent_token_repeat_threshold: float = 0.5,
    ):
        self.num_tokens = int(num_tokens)
        self.signature_dim = int(signature_dim)
        self.capacity = int(capacity)
        self.seen_enabled = bool(seen_enabled)
        self.recent_token_window = int(recent_token_window)
        self.recent_token_repeat_threshold = float(recent_token_repeat_threshold)
        self.active: list[dict[str, object]] = []
        self.seen: list[list[torch.Tensor]] = [[] for _ in range(max(self.num_tokens, 1))]
        self.recent_tokens: deque[int] = deque(maxlen=max(1, self.recent_token_window))
        self.eviction_count = 0
        self.preemption_count = 0
        self.new_count = 0
        self.rejected_seen_count = 0
        self.rejected_active_similar_count = 0
        self.rejected_low_priority_count = 0
        self.repeat_suppressed_count = 0
        self.reward_repeat_violation_count = 0

    def reset(self):
        self.active.clear()
        self.seen = [[] for _ in range(max(self.num_tokens, 1))]
        self.recent_tokens.clear()
        self.eviction_count = 0
        self.preemption_count = 0
        self.new_count = 0
        self.rejected_seen_count = 0
        self.rejected_active_similar_count = 0
        self.rejected_low_priority_count = 0
        self.repeat_suppressed_count = 0
        self.reward_repeat_violation_count = 0

    @staticmethod
    def _cosine_distance(signature: torch.Tensor, entries: list[torch.Tensor]) -> float:
        if not entries:
            return 1.0
        sig = signature.detach().float().cpu()
        stacked = torch.stack([entry.detach().float().cpu() for entry in entries], dim=0)
        sim = torch.matmul(stacked, sig) / (
            torch.clamp(torch.linalg.vector_norm(stacked, dim=-1), min=1e-8)
            * torch.clamp(torch.linalg.vector_norm(sig), min=1e-8)
        )
        return float((1.0 - sim.max()).clamp(min=0.0, max=2.0).item())

    def _entries_for_token(self, token_id: int) -> list[torch.Tensor]:
        return [entry["signature"] for entry in self.active if int(entry["token_id"]) == int(token_id)]

    def _recent_repeat_penalty(self, token_id: int) -> float:
        if not self.recent_tokens:
            return 0.0
        ratio = sum(1 for token in self.recent_tokens if int(token) == int(token_id)) / float(len(self.recent_tokens))
        return 0.25 if ratio > self.recent_token_repeat_threshold else 0.0

    def evaluate(
        self,
        token_id: int,
        signature: torch.Tensor,
        valid_candidate: bool,
        effect_valid: bool,
        control_valid: bool,
        seen_threshold: float,
        active_threshold: float,
        preemption_enabled: bool,
        preemption_margin: float,
        milestone_reward: float,
        update_state: bool,
    ) -> dict[str, object]:
        if not valid_candidate:
            return self._result(0.0, "invalid_candidate", token_id, 1.0, 1.0, 0.0)
        token_id = int(max(0, min(int(token_id), self.num_tokens - 1 if self.num_tokens > 0 else 0)))
        signature_cpu = signature.detach().float().cpu()
        seen_distance = self._cosine_distance(signature_cpu, self.seen[token_id]) if self.seen_enabled else 1.0
        active_distance = self._cosine_distance(signature_cpu, self._entries_for_token(token_id))
        novelty_distance = min(active_distance, 1.0)
        priority = novelty_distance + (0.25 if effect_valid else 0.0) + (0.25 if control_valid else 0.0)
        priority -= self._recent_repeat_penalty(token_id)

        if seen_distance < float(seen_threshold):
            if update_state:
                self.rejected_seen_count += 1
                self.repeat_suppressed_count += 1
                self.recent_tokens.append(token_id)
            return self._result(0.0, "already_seen", token_id, seen_distance, active_distance, priority)
        if active_distance < float(active_threshold):
            if update_state:
                self.rejected_active_similar_count += 1
                self.repeat_suppressed_count += 1
                self.recent_tokens.append(token_id)
            return self._result(0.0, "already_active_similar", token_id, seen_distance, active_distance, priority)

        reason = "new_inserted"
        reward = float(milestone_reward)
        inserted = False
        if len(self.active) < self.capacity:
            inserted = True
        elif preemption_enabled:
            priorities = [float(entry["priority"]) for entry in self.active]
            min_index = int(min(range(len(priorities)), key=lambda index: priorities[index]))
            min_priority = priorities[min_index]
            if priority > min_priority + float(preemption_margin):
                if update_state:
                    self.active.pop(min_index)
                    self.eviction_count += 1
                    self.preemption_count += 1
                reason = "preempted_low_priority"
                inserted = True
            else:
                if update_state:
                    self.rejected_low_priority_count += 1
                    self.recent_tokens.append(token_id)
                return self._result(0.0, "rejected_low_priority", token_id, seen_distance, active_distance, priority)
        else:
            if update_state:
                self.rejected_low_priority_count += 1
                self.recent_tokens.append(token_id)
            return self._result(0.0, "rejected_low_priority", token_id, seen_distance, active_distance, priority)

        if inserted and update_state:
            if self.seen_enabled and seen_distance < float(seen_threshold):
                self.reward_repeat_violation_count += 1
            self.active.append({"token_id": token_id, "signature": signature_cpu, "priority": float(priority)})
            if self.seen_enabled:
                self.seen[token_id].append(signature_cpu.clone())
            self.new_count += 1
            self.recent_tokens.append(token_id)
        return self._result(reward, reason, token_id, seen_distance, active_distance, priority)

    def _result(self, reward, reason, token_id, seen_distance, active_distance, priority) -> dict[str, object]:
        return {
            "reward": float(reward),
            "reason": str(reason),
            "token_id": int(token_id),
            "candidate_to_seen_distance": float(seen_distance),
            "candidate_to_active_distance": float(active_distance),
            "candidate_priority": float(priority),
            "active_size": int(len(self.active)),
            "seen_size": int(sum(len(entries) for entries in self.seen)),
        }

    def counts(self) -> dict[str, object]:
        active_per_token = [0 for _ in range(max(self.num_tokens, 1))]
        for entry in self.active:
            active_per_token[int(entry["token_id"])] += 1
        seen_per_token = [len(entries) for entries in self.seen]
        token_coverage = sum(1 for count in seen_per_token if count > 0) / float(max(1, self.num_tokens))
        return {
            "active_count": len(self.active),
            "seen_count": sum(seen_per_token),
            "active_per_token": active_per_token,
            "seen_per_token": seen_per_token,
            "token_coverage": token_coverage,
            "eviction_count": self.eviction_count,
            "preemption_count": self.preemption_count,
            "new_count": self.new_count,
            "rejected_seen_count": self.rejected_seen_count,
            "rejected_active_similar_count": self.rejected_active_similar_count,
            "rejected_low_priority_count": self.rejected_low_priority_count,
            "repeat_suppressed_count": self.repeat_suppressed_count,
            "reward_repeat_violation_count": self.reward_repeat_violation_count,
        }


@dataclass(slots=True)
class DCTEventElement:
    token_id: int
    signature: torch.Tensor
    step: int
    element_key: tuple[int, tuple[float, ...]]


class DCTTemporalSequenceBuffer:
    def __init__(self, capacity: int):
        self.events: deque[DCTEventElement] = deque(maxlen=max(1, int(capacity)))

    def reset(self):
        self.events.clear()

    def append(self, element: DCTEventElement):
        self.events.append(element)

    def __len__(self) -> int:
        return len(self.events)

    def previous_events(self) -> list[DCTEventElement]:
        return list(self.events)


class DCTSequenceMilestoneMemory:
    def __init__(
        self,
        capacity: int,
        seen_enabled: bool = True,
        preemption_enabled: bool = True,
        preemption_margin: float = 0.05,
    ):
        self.capacity = int(capacity)
        self.seen_enabled = bool(seen_enabled)
        self.preemption_enabled = bool(preemption_enabled)
        self.preemption_margin = float(preemption_margin)
        self.active: list[dict[str, object]] = []
        self.seen: list[torch.Tensor] = []
        self.new_count = 0
        self.preemption_count = 0
        self.rejected_seen_count = 0
        self.rejected_active_similar_count = 0
        self.rejected_low_priority_count = 0
        self.repeat_violation_count = 0

    def reset_active(self):
        self.active.clear()

    def reset_all(self):
        self.active.clear()
        self.seen.clear()
        self.new_count = 0
        self.preemption_count = 0
        self.rejected_seen_count = 0
        self.rejected_active_similar_count = 0
        self.rejected_low_priority_count = 0
        self.repeat_violation_count = 0

    @staticmethod
    def _cosine_distance(signature: torch.Tensor, entries: list[torch.Tensor]) -> float:
        if not entries:
            return 1.0
        sig = signature.detach().float().cpu()
        stacked = torch.stack([entry.detach().float().cpu() for entry in entries], dim=0)
        sim = torch.matmul(stacked, sig) / (
            torch.clamp(torch.linalg.vector_norm(stacked, dim=-1), min=1e-8)
            * torch.clamp(torch.linalg.vector_norm(sig), min=1e-8)
        )
        return float((1.0 - sim.max()).clamp(min=0.0, max=2.0).item())

    def evaluate(
        self,
        signature: torch.Tensor,
        tokens: list[int],
        seen_threshold: float,
        active_threshold: float,
        update_state: bool,
    ) -> dict[str, object]:
        signature_cpu = signature.detach().float().cpu()
        seen_distance = self._cosine_distance(signature_cpu, self.seen) if self.seen_enabled else 1.0
        active_entries = [entry["signature"] for entry in self.active]
        active_distance = self._cosine_distance(signature_cpu, active_entries)
        if seen_distance < float(seen_threshold):
            if update_state:
                self.rejected_seen_count += 1
            return self._result(False, "sequence_already_seen", seen_distance, active_distance, 0.0)
        if active_distance < float(active_threshold):
            if update_state:
                self.rejected_active_similar_count += 1
            return self._result(False, "sequence_already_active_similar", seen_distance, active_distance, 0.0)

        novelty_distance = max(0.0, min(active_distance, 1.0))
        unique_tokens = len(set(int(token) for token in tokens))
        diversity_bonus = 0.25 if unique_tokens >= 2 else 0.0
        repeated_token_penalty = 0.25 if unique_tokens == 1 else 0.0
        priority = novelty_distance + diversity_bonus + 0.25 - repeated_token_penalty

        inserted = False
        reason = "sequence_new_inserted"
        if len(self.active) < self.capacity:
            inserted = True
        elif self.preemption_enabled:
            priorities = [float(entry["priority"]) for entry in self.active]
            min_index = int(min(range(len(priorities)), key=lambda index: priorities[index]))
            min_priority = priorities[min_index]
            if priority > min_priority + self.preemption_margin:
                if update_state:
                    self.active.pop(min_index)
                    self.preemption_count += 1
                reason = "sequence_preempted_low_priority"
                inserted = True
            else:
                if update_state:
                    self.rejected_low_priority_count += 1
                return self._result(False, "sequence_rejected_low_priority", seen_distance, active_distance, priority)
        else:
            if update_state:
                self.rejected_low_priority_count += 1
            return self._result(False, "sequence_rejected_low_priority", seen_distance, active_distance, priority)

        if inserted and update_state:
            if self.seen_enabled and seen_distance < float(seen_threshold):
                self.repeat_violation_count += 1
            self.active.append({"signature": signature_cpu, "priority": float(priority), "tokens": list(tokens)})
            if self.seen_enabled:
                self.seen.append(signature_cpu.clone())
            self.new_count += 1
        return self._result(True, reason, seen_distance, active_distance, priority)

    def _result(self, accepted: bool, reason: str, seen_distance: float, active_distance: float, priority: float):
        return {
            "accepted": bool(accepted),
            "reason": str(reason),
            "seen_distance": float(seen_distance),
            "active_distance": float(active_distance),
            "priority": float(priority),
            "active_size": int(len(self.active)),
            "seen_size": int(len(self.seen)),
        }

    def counts(self) -> dict[str, int]:
        return {
            "new_count": self.new_count,
            "seen_count": len(self.seen),
            "active_count": len(self.active),
            "preemption_count": self.preemption_count,
            "rejected_seen_count": self.rejected_seen_count,
            "rejected_active_similar_count": self.rejected_active_similar_count,
            "rejected_low_priority_count": self.rejected_low_priority_count,
            "repeat_violation_count": self.repeat_violation_count,
        }


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
        self.register_buffer("dct_lifetime_step", torch.tensor(0, dtype=torch.long))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.config.signature_seed))
        projection = torch.randn(self.feat_dim, int(self.config.signature_dim), generator=generator, dtype=torch.float32)
        projection = projection / torch.clamp(torch.linalg.vector_norm(projection, dim=0, keepdim=True), min=1e-8)
        self.register_buffer("effect_signature_projection", projection)
        self.dct_memory: DCTMilestoneMemory | None = None
        self.dct_temporal_buffer = DCTTemporalSequenceBuffer(int(self.config.temporal_buffer_size))
        self.bigram_memory: DCTSequenceMilestoneMemory | None = None
        self.trigram_memory: DCTSequenceMilestoneMemory | None = None
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
        sequence_input_dim = int(self.num_tokens + self.config.signature_dim)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.config.sequence_signature_seed))
        bigram_projection = torch.randn(sequence_input_dim * 2, int(self.config.sequence_signature_dim), generator=generator)
        bigram_projection = bigram_projection / torch.clamp(torch.linalg.vector_norm(bigram_projection, dim=0, keepdim=True), min=1e-8)
        trigram_projection = torch.randn(sequence_input_dim * 3, int(self.config.sequence_signature_dim), generator=generator)
        trigram_projection = trigram_projection / torch.clamp(torch.linalg.vector_norm(trigram_projection, dim=0, keepdim=True), min=1e-8)
        self.register_buffer("bigram_sequence_projection", bigram_projection)
        self.register_buffer("trigram_sequence_projection", trigram_projection)
        self.dct_memory = DCTMilestoneMemory(
            num_tokens=self.num_tokens,
            signature_dim=int(self.config.signature_dim),
            capacity=int(self.config.milestone_memory_capacity),
            seen_enabled=bool(self.config.seen_memory_enabled),
            recent_token_window=int(self.config.recent_token_window),
            recent_token_repeat_threshold=float(self.config.recent_token_repeat_threshold),
        )
        self.bigram_memory = DCTSequenceMilestoneMemory(
            capacity=int(self.config.bigram_memory_capacity),
            seen_enabled=bool(self.config.sequence_seen_memory_enabled),
            preemption_enabled=bool(self.config.sequence_preemption_enabled),
            preemption_margin=float(self.config.sequence_preemption_margin),
        )
        self.trigram_memory = DCTSequenceMilestoneMemory(
            capacity=int(self.config.trigram_memory_capacity),
            seen_enabled=bool(self.config.sequence_seen_memory_enabled),
            preemption_enabled=bool(self.config.sequence_preemption_enabled),
            preemption_margin=float(self.config.sequence_preemption_margin),
        )

    def reset_lifetime_state(self, env_reset: bool = True):
        if hasattr(self, "visited_dct_tokens"):
            self.visited_dct_tokens.zero_()
        self.dct_temporal_buffer.reset()
        self.dct_lifetime_step.zero_()
        if self.dct_memory is not None and (
            (env_reset and self.config.reset_memory_on_env_reset)
            or ((not env_reset) and self.config.reset_memory_on_lifetime_reset)
        ):
            self.dct_memory.reset()
        if self.bigram_memory is not None:
            if (env_reset and self.config.reset_sequence_memory_on_env_reset) or (
                (not env_reset) and self.config.reset_sequence_memory_on_lifetime_reset
            ):
                self.bigram_memory.reset_all()
            else:
                self.bigram_memory.reset_active()
        if self.trigram_memory is not None:
            if (env_reset and self.config.reset_sequence_memory_on_env_reset) or (
                (not env_reset) and self.config.reset_sequence_memory_on_lifetime_reset
            ):
                self.trigram_memory.reset_all()
            else:
                self.trigram_memory.reset_active()

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

    def _effect_signature(self, delta_x: torch.Tensor) -> torch.Tensor:
        signature = torch.matmul(delta_x, self.effect_signature_projection.to(device=delta_x.device, dtype=delta_x.dtype))
        if self.config.signature_normalize:
            signature = signature / torch.clamp(torch.linalg.vector_norm(signature, dim=-1, keepdim=True), min=1e-8)
        return torch.nan_to_num(signature, nan=0.0, posinf=0.0, neginf=0.0)

    def _event_element_key(self, token_id: int, signature: torch.Tensor) -> tuple[int, tuple[float, ...]]:
        rounded = torch.round(signature.detach().float().cpu() * 10.0) / 10.0
        return int(token_id), tuple(float(value) for value in rounded.tolist())

    def _sequence_signature(self, elements: list[DCTEventElement], device, dtype) -> torch.Tensor:
        parts = []
        for element in elements:
            token = max(0, min(int(element.token_id), max(self.num_tokens - 1, 0)))
            one_hot = F.one_hot(torch.as_tensor(token, device=device), num_classes=max(self.num_tokens, 1)).to(dtype=dtype)
            parts.append(one_hot)
            parts.append(element.signature.to(device=device, dtype=dtype))
        raw = torch.cat(parts, dim=-1)
        if len(elements) == 2:
            projection = self.bigram_sequence_projection.to(device=device, dtype=dtype)
        elif len(elements) == 3:
            projection = self.trigram_sequence_projection.to(device=device, dtype=dtype)
        else:
            raise ValueError("DCT sequence signatures support only bigram or trigram")
        signature = torch.matmul(raw, projection)
        if self.config.sequence_signature_normalize:
            signature = signature / torch.clamp(torch.linalg.vector_norm(signature), min=1e-8)
        return torch.nan_to_num(signature, nan=0.0, posinf=0.0, neginf=0.0)

    def _empty_sequence_info(self, reward: torch.Tensor) -> dict[str, torch.Tensor | list[str]]:
        zero = torch.zeros((), device=reward.device, dtype=reward.dtype)
        zeros = torch.zeros_like(reward)
        return {
            "dct_v3_sequence_reward": zeros.detach(),
            "dct_v3_sequence_bigram_reward": zeros.detach(),
            "dct_v3_sequence_trigram_reward": zeros.detach(),
            "dct_v3_sequence_reward_mean": zeros.detach().mean(),
            "dct_v3_sequence_reward_nonzero_ratio": zeros.detach().mean(),
            "dct_v3_sequence_reward_step_cap_hit_ratio": zeros.detach().mean(),
            "dct_v3_temporal_buffer_size_mean": zero.detach(),
            "dct_bigram_new_count": zero.detach(),
            "dct_bigram_seen_count": zero.detach(),
            "dct_bigram_active_count": zero.detach(),
            "dct_bigram_preemption_count": zero.detach(),
            "dct_bigram_rejected_seen_count": zero.detach(),
            "dct_bigram_rejected_active_similar_count": zero.detach(),
            "dct_bigram_rejected_low_priority_count": zero.detach(),
            "dct_bigram_repeat_violation_count": zero.detach(),
            "dct_trigram_new_count": zero.detach(),
            "dct_trigram_seen_count": zero.detach(),
            "dct_trigram_active_count": zero.detach(),
            "dct_trigram_preemption_count": zero.detach(),
            "dct_trigram_rejected_seen_count": zero.detach(),
            "dct_trigram_rejected_active_similar_count": zero.detach(),
            "dct_trigram_rejected_low_priority_count": zero.detach(),
            "dct_trigram_repeat_violation_count": zero.detach(),
            "dct_sequence_unique_token_ratio_mean": zero.detach(),
            "dct_sequence_all_same_token_ratio": zero.detach(),
            "dct_sequence_gap_mean": zero.detach(),
            "dct_sequence_span_mean": zero.detach(),
            "dct_sequence_candidate_distance_mean": zero.detach(),
            "dct_sequence_candidate_distance_min": zero.detach(),
            "dct_sequence_priority_mean": zero.detach(),
            "dct_sequence_priority_min": zero.detach(),
            "dct_sequence_priority_max": zero.detach(),
            "dct_sequence_reason_new_inserted": zero.detach(),
            "dct_sequence_reason_preempted_low_priority": zero.detach(),
            "dct_sequence_reason_already_seen": zero.detach(),
            "dct_sequence_reason_already_active_similar": zero.detach(),
            "dct_sequence_reason_rejected_low_priority": zero.detach(),
            "dct_sequence_reason_invalid_gap": zero.detach(),
            "dct_sequence_reason_invalid_span": zero.detach(),
            "dct_sequence_reason": [],
            "dct_sequence_gap": zeros.detach(),
            "dct_sequence_span": zeros.detach(),
            "dct_bigram_signature_norm": zeros.detach(),
            "dct_trigram_signature_norm": zeros.detach(),
        }

    def _dct_temporal_sequence_milestone_v3_reward(
        self,
        token_id: torch.Tensor,
        delta_x: torch.Tensor,
        capacity_mask: torch.Tensor | None,
        effect_gain: torch.Tensor,
        update_state: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | list[str]]]:
        reward = torch.zeros(token_id.shape, device=token_id.device, dtype=torch.float32)
        info = self._empty_dct_info(reward)
        sequence_info = self._empty_sequence_info(reward)
        info.update(sequence_info)
        if token_id.shape != (1, 1):
            if update_state:
                raise RuntimeError(
                    "dct_temporal_sequence_milestone_v3 is online-only and cannot be recomputed on replay batches"
                )
            return reward, info
        if self.bigram_memory is None or self.trigram_memory is None:
            return reward, info

        signatures = self._effect_signature(delta_x).detach()
        signature_norm = torch.linalg.vector_norm(signatures, dim=-1)
        capacity_bool = True
        if capacity_mask is not None:
            capacity_bool = bool((capacity_mask.reshape(token_id.shape) > 0)[0, 0].item())
        delta_x_norm = torch.linalg.vector_norm(delta_x, dim=-1)
        effect_valid = bool((effect_gain > float(self.config.min_effect_gain))[0, 0].item())
        delta_valid = bool((delta_x_norm > float(self.config.min_delta_x_norm))[0, 0].item())
        valid_candidate = bool(capacity_bool and effect_valid and delta_valid)

        info.update(
            {
                "dct_candidate_valid": torch.as_tensor([[float(valid_candidate)]], device=reward.device),
                "dct_effect_signature_norm": signature_norm.detach(),
            }
        )
        if not update_state:
            return reward, info

        step = int(self.dct_lifetime_step.item())
        self.dct_lifetime_step.add_(1)
        if not valid_candidate:
            return reward, info

        token = int(token_id[0, 0].item())
        current = DCTEventElement(
            token_id=token,
            signature=signatures[0, 0].detach().cpu(),
            step=step,
            element_key=self._event_element_key(token, signatures[0, 0]),
        )
        previous = self.dct_temporal_buffer.previous_events()
        bigram_reward = 0.0
        trigram_reward = 0.0
        reasons: list[str] = []
        gaps: list[float] = []
        spans: list[float] = []
        distances: list[float] = []
        priorities: list[float] = []
        unique_ratios: list[float] = []
        all_same_flags: list[float] = []
        bigram_norm = 0.0
        trigram_norm = 0.0
        reason_counts = {
            "sequence_new_inserted": 0,
            "sequence_preempted_low_priority": 0,
            "sequence_already_seen": 0,
            "sequence_already_active_similar": 0,
            "sequence_rejected_low_priority": 0,
            "sequence_invalid_gap": 0,
            "sequence_invalid_span": 0,
        }

        def maybe_eval(elements: list[DCTEventElement], memory: DCTSequenceMilestoneMemory, base_reward: float) -> float:
            nonlocal bigram_norm, trigram_norm
            gaps_local = [elements[index + 1].step - elements[index].step for index in range(len(elements) - 1)]
            span = elements[-1].step - elements[0].step
            if any(gap < int(self.config.min_event_gap_steps) for gap in gaps_local):
                reason_counts["sequence_invalid_gap"] += 1
                reasons.append("sequence_invalid_gap")
                return 0.0
            if span > int(self.config.max_sequence_span_steps):
                reason_counts["sequence_invalid_span"] += 1
                reasons.append("sequence_invalid_span")
                return 0.0
            signature = self._sequence_signature(elements, device=reward.device, dtype=reward.dtype)
            norm = float(torch.linalg.vector_norm(signature).detach().cpu())
            if len(elements) == 2:
                bigram_norm = max(bigram_norm, norm)
            else:
                trigram_norm = max(trigram_norm, norm)
            tokens = [element.token_id for element in elements]
            result = memory.evaluate(
                signature=signature,
                tokens=tokens,
                seen_threshold=float(self.config.sequence_seen_similarity_threshold),
                active_threshold=float(self.config.sequence_active_similarity_threshold),
                update_state=True,
            )
            reason = str(result["reason"])
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            reasons.append(reason)
            gaps.extend(float(gap) for gap in gaps_local)
            spans.append(float(span))
            distances.append(float(result["seen_distance"]))
            priorities.append(float(result["priority"]))
            unique_ratios.append(len(set(tokens)) / float(len(tokens)))
            all_same_flags.append(1.0 if len(set(tokens)) == 1 else 0.0)
            return float(base_reward) if bool(result["accepted"]) else 0.0

        if bool(self.config.use_bigram_milestone):
            for prev in previous:
                bigram_reward += maybe_eval([prev, current], self.bigram_memory, float(self.config.bigram_milestone_reward))
        if bool(self.config.use_trigram_milestone):
            for left_index in range(len(previous)):
                for right_index in range(left_index + 1, len(previous)):
                    trigram_reward += maybe_eval(
                        [previous[left_index], previous[right_index], current],
                        self.trigram_memory,
                        float(self.config.trigram_milestone_reward),
                    )
        self.dct_temporal_buffer.append(current)

        total_before_cap = bigram_reward + trigram_reward
        capped = min(float(self.config.sequence_reward_step_cap), total_before_cap)
        reward[0, 0] = capped
        trigger = (reward > 0).float()
        bigram_counts = self.bigram_memory.counts()
        trigram_counts = self.trigram_memory.counts()
        scalar = lambda value: torch.as_tensor(float(value), device=reward.device, dtype=reward.dtype)
        info.update(
            {
                "milestone_reward": reward.detach(),
                "milestone_trigger": trigger.detach(),
                "topk_entered": trigger.detach(),
                "dct_v3_sequence_reward": reward.detach(),
                "dct_v3_sequence_bigram_reward": torch.as_tensor([[bigram_reward]], device=reward.device, dtype=reward.dtype),
                "dct_v3_sequence_trigram_reward": torch.as_tensor([[trigram_reward]], device=reward.device, dtype=reward.dtype),
                "dct_v3_sequence_reward_mean": reward.detach().mean(),
                "dct_v3_sequence_reward_nonzero_ratio": trigger.detach().mean(),
                "dct_v3_sequence_reward_step_cap_hit_ratio": scalar(float(total_before_cap > float(self.config.sequence_reward_step_cap))),
                "dct_v3_temporal_buffer_size_mean": scalar(len(self.dct_temporal_buffer)),
                "dct_bigram_new_count": scalar(bigram_counts["new_count"]),
                "dct_bigram_seen_count": scalar(bigram_counts["seen_count"]),
                "dct_bigram_active_count": scalar(bigram_counts["active_count"]),
                "dct_bigram_preemption_count": scalar(bigram_counts["preemption_count"]),
                "dct_bigram_rejected_seen_count": scalar(bigram_counts["rejected_seen_count"]),
                "dct_bigram_rejected_active_similar_count": scalar(bigram_counts["rejected_active_similar_count"]),
                "dct_bigram_rejected_low_priority_count": scalar(bigram_counts["rejected_low_priority_count"]),
                "dct_bigram_repeat_violation_count": scalar(bigram_counts["repeat_violation_count"]),
                "dct_trigram_new_count": scalar(trigram_counts["new_count"]),
                "dct_trigram_seen_count": scalar(trigram_counts["seen_count"]),
                "dct_trigram_active_count": scalar(trigram_counts["active_count"]),
                "dct_trigram_preemption_count": scalar(trigram_counts["preemption_count"]),
                "dct_trigram_rejected_seen_count": scalar(trigram_counts["rejected_seen_count"]),
                "dct_trigram_rejected_active_similar_count": scalar(trigram_counts["rejected_active_similar_count"]),
                "dct_trigram_rejected_low_priority_count": scalar(trigram_counts["rejected_low_priority_count"]),
                "dct_trigram_repeat_violation_count": scalar(trigram_counts["repeat_violation_count"]),
                "dct_sequence_unique_token_ratio_mean": scalar(sum(unique_ratios) / max(1, len(unique_ratios))),
                "dct_sequence_all_same_token_ratio": scalar(sum(all_same_flags) / max(1, len(all_same_flags))),
                "dct_sequence_gap_mean": scalar(sum(gaps) / max(1, len(gaps))),
                "dct_sequence_span_mean": scalar(sum(spans) / max(1, len(spans))),
                "dct_sequence_candidate_distance_mean": scalar(sum(distances) / max(1, len(distances))),
                "dct_sequence_candidate_distance_min": scalar(min(distances) if distances else 0.0),
                "dct_sequence_priority_mean": scalar(sum(priorities) / max(1, len(priorities))),
                "dct_sequence_priority_min": scalar(min(priorities) if priorities else 0.0),
                "dct_sequence_priority_max": scalar(max(priorities) if priorities else 0.0),
                "dct_sequence_reason_new_inserted": scalar(reason_counts["sequence_new_inserted"]),
                "dct_sequence_reason_preempted_low_priority": scalar(reason_counts["sequence_preempted_low_priority"]),
                "dct_sequence_reason_already_seen": scalar(reason_counts["sequence_already_seen"]),
                "dct_sequence_reason_already_active_similar": scalar(reason_counts["sequence_already_active_similar"]),
                "dct_sequence_reason_rejected_low_priority": scalar(reason_counts["sequence_rejected_low_priority"]),
                "dct_sequence_reason_invalid_gap": scalar(reason_counts["sequence_invalid_gap"]),
                "dct_sequence_reason_invalid_span": scalar(reason_counts["sequence_invalid_span"]),
                "dct_sequence_reason": reasons,
                "dct_sequence_gap": scalar(gaps[-1] if gaps else 0.0).view(1, 1),
                "dct_sequence_span": scalar(spans[-1] if spans else 0.0).view(1, 1),
                "dct_bigram_signature_norm": scalar(bigram_norm).view(1, 1),
                "dct_trigram_signature_norm": scalar(trigram_norm).view(1, 1),
            }
        )
        return reward, info

    def _empty_dct_info(self, reference: torch.Tensor) -> dict[str, torch.Tensor]:
        zeros = torch.zeros(reference.shape, device=reference.device, dtype=reference.dtype)
        scalar_zero = torch.zeros((), device=reference.device, dtype=reference.dtype)
        return {
            "milestone_reward": zeros.detach(),
            "milestone_trigger": zeros.detach(),
            "habituation_factor": torch.ones_like(zeros).detach(),
            "token_repeat_decay": torch.ones_like(zeros).detach(),
            "operator_rarity": zeros.detach(),
            "running_best_intensity": scalar_zero.detach(),
            "event_intensity": zeros.detach(),
            "event_intensity_top_mean": scalar_zero.detach(),
            "event_intensity_normal_mean": scalar_zero.detach(),
            "topk_rank": zeros.detach(),
            "topk_env_bonus": zeros.detach(),
            "topk_entered": zeros.detach(),
            "topk_rank_mean": scalar_zero.detach(),
            "topk_env_bonus_mean": scalar_zero.detach(),
            "topk_entered_ratio": scalar_zero.detach(),
            "topk_leaderboard_size": scalar_zero.detach(),
            "topk_update_count": scalar_zero.detach(),
            "dct_candidate_valid": zeros.detach(),
            "dct_candidate_to_seen_distance": torch.ones_like(zeros).detach(),
            "dct_candidate_to_active_distance": torch.ones_like(zeros).detach(),
            "dct_candidate_priority": zeros.detach(),
            "dct_effect_signature_norm": zeros.detach(),
            "dct_milestone_reason_code": zeros.detach(),
        }

    @staticmethod
    def _reason_code(reason: str) -> int:
        mapping = {
            "invalid_candidate": 0,
            "new_inserted": 1,
            "preempted_low_priority": 2,
            "already_seen": 3,
            "already_active_similar": 4,
            "rejected_low_priority": 5,
        }
        return mapping.get(str(reason), -1)

    def _dct_effect_milestone_v2_reward(
        self,
        token_id: torch.Tensor,
        delta_x: torch.Tensor,
        capacity_mask: torch.Tensor | None,
        effect_gain: torch.Tensor,
        controllability_gain: torch.Tensor,
        update_state: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | list[str]]]:
        reward = torch.zeros(token_id.shape, device=token_id.device, dtype=torch.float32)
        info = self._empty_dct_info(reward)
        if self.dct_memory is None:
            return reward, info
        signatures = self._effect_signature(delta_x).detach()
        signature_norm = torch.linalg.vector_norm(signatures, dim=-1)
        capacity_bool = torch.ones(token_id.shape, device=token_id.device, dtype=torch.bool)
        if capacity_mask is not None:
            capacity_bool = capacity_mask.to(device=token_id.device).reshape(token_id.shape) > 0
        delta_x_norm = torch.linalg.vector_norm(delta_x, dim=-1)
        effect_valid = effect_gain > float(self.config.min_effect_gain)
        control_valid = controllability_gain > float(self.config.min_controllability_gain)
        valid_candidate = capacity_bool & effect_valid & (delta_x_norm > float(self.config.min_delta_x_norm))

        seen_dist = torch.ones_like(reward)
        active_dist = torch.ones_like(reward)
        priority = torch.zeros_like(reward)
        reason_codes = torch.zeros_like(reward)
        reasons: list[str] = []
        for batch_index in range(token_id.shape[0]):
            for time_index in range(token_id.shape[1]):
                result = self.dct_memory.evaluate(
                    token_id=int(token_id[batch_index, time_index].item()),
                    signature=signatures[batch_index, time_index],
                    valid_candidate=bool(valid_candidate[batch_index, time_index].item()),
                    effect_valid=bool(effect_valid[batch_index, time_index].item()),
                    control_valid=bool(control_valid[batch_index, time_index].item()),
                    seen_threshold=float(self.config.seen_similarity_threshold),
                    active_threshold=float(self.config.active_similarity_threshold),
                    preemption_enabled=bool(self.config.preemption_enabled),
                    preemption_margin=float(self.config.preemption_margin),
                    milestone_reward=float(self.config.milestone_reward),
                    update_state=update_state,
                )
                reward[batch_index, time_index] = float(result["reward"])
                seen_dist[batch_index, time_index] = float(result["candidate_to_seen_distance"])
                active_dist[batch_index, time_index] = float(result["candidate_to_active_distance"])
                priority[batch_index, time_index] = float(result["candidate_priority"])
                reason = str(result["reason"])
                reason_codes[batch_index, time_index] = float(self._reason_code(reason))
                reasons.append(reason)

        counts = self.dct_memory.counts()
        trigger = (reward > 0).float()
        info.update(
            {
                "milestone_reward": reward.detach(),
                "milestone_trigger": trigger.detach(),
                "topk_entered": trigger.detach(),
                "topk_update_count": torch.as_tensor(float(counts["new_count"]), device=reward.device, dtype=reward.dtype),
                "dct_candidate_valid": valid_candidate.float().detach(),
                "dct_candidate_to_seen_distance": seen_dist.detach(),
                "dct_candidate_to_active_distance": active_dist.detach(),
                "dct_candidate_priority": priority.detach(),
                "dct_effect_signature_norm": signature_norm.detach(),
                "dct_milestone_reason_code": reason_codes.detach(),
                "dct_milestone_reason": reasons,
                "dct_milestone_new_count": torch.as_tensor(float(counts["new_count"]), device=reward.device, dtype=reward.dtype),
                "dct_milestone_seen_count": torch.as_tensor(float(counts["seen_count"]), device=reward.device, dtype=reward.dtype),
                "dct_milestone_active_count": torch.as_tensor(float(counts["active_count"]), device=reward.device, dtype=reward.dtype),
                "dct_milestone_eviction_count": torch.as_tensor(float(counts["eviction_count"]), device=reward.device, dtype=reward.dtype),
                "dct_milestone_preemption_count": torch.as_tensor(float(counts["preemption_count"]), device=reward.device, dtype=reward.dtype),
                "dct_milestone_rejected_seen_count": torch.as_tensor(float(counts["rejected_seen_count"]), device=reward.device, dtype=reward.dtype),
                "dct_milestone_rejected_active_similar_count": torch.as_tensor(float(counts["rejected_active_similar_count"]), device=reward.device, dtype=reward.dtype),
                "dct_milestone_rejected_low_priority_count": torch.as_tensor(float(counts["rejected_low_priority_count"]), device=reward.device, dtype=reward.dtype),
                "dct_milestone_repeat_suppressed_count": torch.as_tensor(float(counts["repeat_suppressed_count"]), device=reward.device, dtype=reward.dtype),
                "dct_milestone_reward_repeat_violation_count": torch.as_tensor(float(counts["reward_repeat_violation_count"]), device=reward.device, dtype=reward.dtype),
                "dct_milestone_token_coverage": torch.as_tensor(float(counts["token_coverage"]), device=reward.device, dtype=reward.dtype),
                "dct_milestone_seen_per_token": counts["seen_per_token"],
                "dct_milestone_active_per_token": counts["active_per_token"],
                "dct_milestone_candidate_distance_mean": seen_dist.detach().mean(),
                "dct_milestone_candidate_distance_min": seen_dist.detach().min(),
                "dct_milestone_priority_mean": priority.detach().mean(),
                "dct_milestone_priority_min": priority.detach().min(),
                "dct_milestone_priority_max": priority.detach().max(),
            }
        )
        for reason_name in (
            "new_inserted",
            "preempted_low_priority",
            "already_seen",
            "already_active_similar",
            "rejected_low_priority",
            "invalid_candidate",
        ):
            info[f"milestone_reason_{reason_name}"] = torch.as_tensor(
                float(sum(1 for reason in reasons if reason == reason_name)),
                device=reward.device,
                dtype=reward.dtype,
            )
        return reward, info

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
            elif self.config.reward_mode == "dct_effect_milestone_v2":
                reward, milestone_info = self._dct_effect_milestone_v2_reward(
                    token_id=outputs["token_id"],
                    delta_x=delta_x,
                    capacity_mask=capacity_mask,
                    effect_gain=effect_gain,
                    controllability_gain=controllability_gain,
                    update_state=update_state,
                )
            elif self.config.reward_mode == "dct_temporal_sequence_milestone_v3":
                reward, milestone_info = self._dct_temporal_sequence_milestone_v3_reward(
                    token_id=outputs["token_id"],
                    delta_x=delta_x,
                    capacity_mask=capacity_mask,
                    effect_gain=effect_gain,
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
            "dct_token_id": outputs["token_id"].detach(),
            "dct_effect_signature_norm": milestone_info.get("dct_effect_signature_norm", torch.zeros_like(reward)).detach(),
            "dct_candidate_valid": milestone_info.get("dct_candidate_valid", torch.zeros_like(reward)).detach(),
            "dct_milestone_reward": milestone_info["milestone_reward"].detach(),
            "dct_milestone_reason_code": milestone_info.get("dct_milestone_reason_code", torch.zeros_like(reward)).detach(),
            "dct_milestone_reason": milestone_info.get("dct_milestone_reason", []),
            "dct_candidate_to_seen_distance": milestone_info.get(
                "dct_candidate_to_seen_distance", torch.ones_like(reward)
            ).detach(),
            "dct_candidate_to_active_distance": milestone_info.get(
                "dct_candidate_to_active_distance", torch.ones_like(reward)
            ).detach(),
            "dct_candidate_priority": milestone_info.get("dct_candidate_priority", torch.zeros_like(reward)).detach(),
            "dct_memory_active_size": milestone_info.get("dct_milestone_active_count", torch.zeros((), device=x_t.device)).detach(),
            "dct_memory_seen_size": milestone_info.get("dct_milestone_seen_count", torch.zeros((), device=x_t.device)).detach(),
            "dct_milestone_reward_mean": milestone_info["milestone_reward"].detach().mean(),
            "dct_milestone_reward_nonzero_ratio": milestone_info["milestone_trigger"].detach().float().mean(),
            "dct_milestone_new_count": milestone_info.get("dct_milestone_new_count", torch.zeros((), device=x_t.device)).detach(),
            "dct_milestone_seen_count": milestone_info.get("dct_milestone_seen_count", torch.zeros((), device=x_t.device)).detach(),
            "dct_milestone_active_count": milestone_info.get("dct_milestone_active_count", torch.zeros((), device=x_t.device)).detach(),
            "dct_milestone_eviction_count": milestone_info.get("dct_milestone_eviction_count", torch.zeros((), device=x_t.device)).detach(),
            "dct_milestone_preemption_count": milestone_info.get("dct_milestone_preemption_count", torch.zeros((), device=x_t.device)).detach(),
            "dct_milestone_rejected_seen_count": milestone_info.get("dct_milestone_rejected_seen_count", torch.zeros((), device=x_t.device)).detach(),
            "dct_milestone_rejected_active_similar_count": milestone_info.get(
                "dct_milestone_rejected_active_similar_count", torch.zeros((), device=x_t.device)
            ).detach(),
            "dct_milestone_rejected_low_priority_count": milestone_info.get(
                "dct_milestone_rejected_low_priority_count", torch.zeros((), device=x_t.device)
            ).detach(),
            "dct_milestone_repeat_suppressed_count": milestone_info.get(
                "dct_milestone_repeat_suppressed_count", torch.zeros((), device=x_t.device)
            ).detach(),
            "dct_milestone_reward_repeat_violation_count": milestone_info.get(
                "dct_milestone_reward_repeat_violation_count", torch.zeros((), device=x_t.device)
            ).detach(),
            "dct_milestone_token_coverage": milestone_info.get("dct_milestone_token_coverage", torch.zeros((), device=x_t.device)).detach(),
            "dct_milestone_seen_per_token": milestone_info.get("dct_milestone_seen_per_token", []),
            "dct_milestone_active_per_token": milestone_info.get("dct_milestone_active_per_token", []),
            "dct_milestone_candidate_distance_mean": milestone_info.get(
                "dct_milestone_candidate_distance_mean", torch.zeros((), device=x_t.device)
            ).detach(),
            "dct_milestone_candidate_distance_min": milestone_info.get(
                "dct_milestone_candidate_distance_min", torch.zeros((), device=x_t.device)
            ).detach(),
            "dct_milestone_priority_mean": milestone_info.get("dct_milestone_priority_mean", torch.zeros((), device=x_t.device)).detach(),
            "dct_milestone_priority_min": milestone_info.get("dct_milestone_priority_min", torch.zeros((), device=x_t.device)).detach(),
            "dct_milestone_priority_max": milestone_info.get("dct_milestone_priority_max", torch.zeros((), device=x_t.device)).detach(),
            "milestone_reason_new_inserted": milestone_info.get("milestone_reason_new_inserted", torch.zeros((), device=x_t.device)).detach(),
            "milestone_reason_preempted_low_priority": milestone_info.get(
                "milestone_reason_preempted_low_priority", torch.zeros((), device=x_t.device)
            ).detach(),
            "milestone_reason_already_seen": milestone_info.get("milestone_reason_already_seen", torch.zeros((), device=x_t.device)).detach(),
            "milestone_reason_already_active_similar": milestone_info.get(
                "milestone_reason_already_active_similar", torch.zeros((), device=x_t.device)
            ).detach(),
            "milestone_reason_rejected_low_priority": milestone_info.get(
                "milestone_reason_rejected_low_priority", torch.zeros((), device=x_t.device)
            ).detach(),
            "milestone_reason_invalid_candidate": milestone_info.get(
                "milestone_reason_invalid_candidate", torch.zeros((), device=x_t.device)
            ).detach(),
            "dct_v3_sequence_reward": milestone_info.get("dct_v3_sequence_reward", torch.zeros_like(reward)).detach(),
            "dct_v3_sequence_bigram_reward": milestone_info.get(
                "dct_v3_sequence_bigram_reward", torch.zeros_like(reward)
            ).detach(),
            "dct_v3_sequence_trigram_reward": milestone_info.get(
                "dct_v3_sequence_trigram_reward", torch.zeros_like(reward)
            ).detach(),
            "dct_v3_sequence_reward_mean": milestone_info.get(
                "dct_v3_sequence_reward_mean", torch.zeros((), device=x_t.device)
            ).detach(),
            "dct_v3_sequence_reward_nonzero_ratio": milestone_info.get(
                "dct_v3_sequence_reward_nonzero_ratio", torch.zeros((), device=x_t.device)
            ).detach(),
            "dct_v3_sequence_reward_step_cap_hit_ratio": milestone_info.get(
                "dct_v3_sequence_reward_step_cap_hit_ratio", torch.zeros((), device=x_t.device)
            ).detach(),
            "dct_v3_temporal_buffer_size_mean": milestone_info.get(
                "dct_v3_temporal_buffer_size_mean", torch.zeros((), device=x_t.device)
            ).detach(),
            "dct_bigram_new_count": milestone_info.get("dct_bigram_new_count", torch.zeros((), device=x_t.device)).detach(),
            "dct_bigram_seen_count": milestone_info.get("dct_bigram_seen_count", torch.zeros((), device=x_t.device)).detach(),
            "dct_bigram_active_count": milestone_info.get("dct_bigram_active_count", torch.zeros((), device=x_t.device)).detach(),
            "dct_bigram_preemption_count": milestone_info.get(
                "dct_bigram_preemption_count", torch.zeros((), device=x_t.device)
            ).detach(),
            "dct_bigram_rejected_seen_count": milestone_info.get(
                "dct_bigram_rejected_seen_count", torch.zeros((), device=x_t.device)
            ).detach(),
            "dct_bigram_rejected_active_similar_count": milestone_info.get(
                "dct_bigram_rejected_active_similar_count", torch.zeros((), device=x_t.device)
            ).detach(),
            "dct_bigram_rejected_low_priority_count": milestone_info.get(
                "dct_bigram_rejected_low_priority_count", torch.zeros((), device=x_t.device)
            ).detach(),
            "dct_bigram_repeat_violation_count": milestone_info.get(
                "dct_bigram_repeat_violation_count", torch.zeros((), device=x_t.device)
            ).detach(),
            "dct_trigram_new_count": milestone_info.get("dct_trigram_new_count", torch.zeros((), device=x_t.device)).detach(),
            "dct_trigram_seen_count": milestone_info.get("dct_trigram_seen_count", torch.zeros((), device=x_t.device)).detach(),
            "dct_trigram_active_count": milestone_info.get(
                "dct_trigram_active_count", torch.zeros((), device=x_t.device)
            ).detach(),
            "dct_trigram_preemption_count": milestone_info.get(
                "dct_trigram_preemption_count", torch.zeros((), device=x_t.device)
            ).detach(),
            "dct_trigram_rejected_seen_count": milestone_info.get(
                "dct_trigram_rejected_seen_count", torch.zeros((), device=x_t.device)
            ).detach(),
            "dct_trigram_rejected_active_similar_count": milestone_info.get(
                "dct_trigram_rejected_active_similar_count", torch.zeros((), device=x_t.device)
            ).detach(),
            "dct_trigram_rejected_low_priority_count": milestone_info.get(
                "dct_trigram_rejected_low_priority_count", torch.zeros((), device=x_t.device)
            ).detach(),
            "dct_trigram_repeat_violation_count": milestone_info.get(
                "dct_trigram_repeat_violation_count", torch.zeros((), device=x_t.device)
            ).detach(),
            "dct_sequence_unique_token_ratio_mean": milestone_info.get(
                "dct_sequence_unique_token_ratio_mean", torch.zeros((), device=x_t.device)
            ).detach(),
            "dct_sequence_all_same_token_ratio": milestone_info.get(
                "dct_sequence_all_same_token_ratio", torch.zeros((), device=x_t.device)
            ).detach(),
            "dct_sequence_gap_mean": milestone_info.get(
                "dct_sequence_gap_mean", torch.zeros((), device=x_t.device)
            ).detach(),
            "dct_sequence_span_mean": milestone_info.get(
                "dct_sequence_span_mean", torch.zeros((), device=x_t.device)
            ).detach(),
            "dct_sequence_candidate_distance_mean": milestone_info.get(
                "dct_sequence_candidate_distance_mean", torch.zeros((), device=x_t.device)
            ).detach(),
            "dct_sequence_candidate_distance_min": milestone_info.get(
                "dct_sequence_candidate_distance_min", torch.zeros((), device=x_t.device)
            ).detach(),
            "dct_sequence_priority_mean": milestone_info.get(
                "dct_sequence_priority_mean", torch.zeros((), device=x_t.device)
            ).detach(),
            "dct_sequence_priority_min": milestone_info.get(
                "dct_sequence_priority_min", torch.zeros((), device=x_t.device)
            ).detach(),
            "dct_sequence_priority_max": milestone_info.get(
                "dct_sequence_priority_max", torch.zeros((), device=x_t.device)
            ).detach(),
            "dct_sequence_reason": milestone_info.get("dct_sequence_reason", []),
            "dct_sequence_gap": milestone_info.get("dct_sequence_gap", torch.zeros_like(reward)).detach(),
            "dct_sequence_span": milestone_info.get("dct_sequence_span", torch.zeros_like(reward)).detach(),
            "dct_bigram_signature_norm": milestone_info.get("dct_bigram_signature_norm", torch.zeros_like(reward)).detach(),
            "dct_trigram_signature_norm": milestone_info.get("dct_trigram_signature_norm", torch.zeros_like(reward)).detach(),
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
