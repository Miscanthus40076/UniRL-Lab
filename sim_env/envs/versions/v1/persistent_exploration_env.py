from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .base_env import BaseEnv


@dataclass(slots=True)
class PersistentExplorationConfig:
    enabled: bool = False
    max_lifetime_steps: int = 2000
    reset_on_env_done: bool = False
    reset_on_success: bool = False
    reset_on_failure: bool = False
    stale_reset_enabled: bool = True
    stale_window: int = 200
    stale_min_obs_change: float = 0.001
    operator_event_reset_delay_enabled: bool = True
    operator_event_reset_delay_steps: int = 300
    operator_event_threshold: float = 0.1
    force_reset_if_nan: bool = True
    force_reset_if_action_nan: bool = True

    def validate(self):
        if self.max_lifetime_steps <= 0:
            raise ValueError("persistent_exploration.max_lifetime_steps must be > 0")
        if self.stale_window <= 0:
            raise ValueError("persistent_exploration.stale_window must be > 0")
        if self.stale_min_obs_change < 0.0:
            raise ValueError("persistent_exploration.stale_min_obs_change must be >= 0")
        if self.operator_event_reset_delay_steps < 0:
            raise ValueError("persistent_exploration.operator_event_reset_delay_steps must be >= 0")
        if self.operator_event_threshold < 0.0:
            raise ValueError("persistent_exploration.operator_event_threshold must be >= 0")


class PersistentExplorationEnv(BaseEnv):
    """Lifecycle wrapper that keeps longer lifetimes and records reset diagnostics."""

    def __init__(self, env: BaseEnv, config: PersistentExplorationConfig):
        self._env = env
        self.config = config
        self.config.validate()
        self._configure_underlying_horizon()
        self._reset_history_state()
        self._reset_pending_reason: str | None = None
        self._started = False
        self._reset_runtime_state()

    def _default_step_metrics(self) -> dict[str, float | str]:
        return {
            "persistent_enabled": 1.0,
            "lifetime_step": 0.0,
            "operator_event_delay_active": 0.0,
            "post_event_continuation_steps": 0.0,
            "unique_token_count_so_far": 0.0,
            "reset_reason": "NA",
        }

    def _reset_runtime_state(self):
        self._lifetime_step = 0
        self._obs_change_history: deque[float] = deque(maxlen=max(1, int(self.config.stale_window)))
        self._last_obs_flat: np.ndarray | None = None
        self._last_operator_event_step: int | None = None
        self._post_event_anchor_step: int | None = None
        self._episode_token_counter: Counter[int] = Counter()
        self._last_step_metrics = self._default_step_metrics()

    def _reset_history_state(self):
        self._total_resets = 0
        self._reason_counts = {
            "max_lifetime": 0,
            "stale": 0,
            "env_done": 0,
            "nan": 0,
            "manual": 0,
        }
        self._lifetime_history: list[int] = []
        self._post_event_continuation_history: list[int] = []
        self._episode_unique_token_history: list[float] = []
        self._episode_token_perplexity_history: list[float] = []
        self._delay_active_steps = 0
        self._total_finalize_steps = 0
        self._last_step_metrics = self._default_step_metrics()

    def clear_persistent_history(self):
        self._reset_history_state()
        self._reset_pending_reason = None

    def _configure_underlying_horizon(self):
        max_path_length = int(self.config.max_lifetime_steps)
        candidates = [
            self._env,
            getattr(self._env, "_env", None),
            getattr(getattr(self._env, "_env", None), "unwrapped", None),
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            if hasattr(candidate, "max_path_length"):
                try:
                    candidate.max_path_length = max(int(getattr(candidate, "max_path_length")), max_path_length)
                except Exception:
                    candidate.max_path_length = max_path_length

    def _flatten_obs(self, obs) -> np.ndarray:
        if isinstance(obs, (list, tuple)):
            pieces = [np.asarray(item, dtype=np.float32).reshape(-1) for item in obs]
            if not pieces:
                return np.zeros((0,), dtype=np.float32)
            return np.concatenate(pieces, axis=0)
        return np.asarray(obs, dtype=np.float32).reshape(-1)

    def _obs_change(self, obs) -> float:
        flat = self._flatten_obs(obs)
        if self._last_obs_flat is None or self._last_obs_flat.shape != flat.shape:
            self._last_obs_flat = flat
            return float("inf")
        delta = float(np.mean(np.abs(flat - self._last_obs_flat)))
        self._last_obs_flat = flat
        return delta

    def _has_bad_numeric(self, value) -> bool:
        try:
            arr = np.asarray(value, dtype=np.float32)
        except Exception:
            return True
        return not np.isfinite(arr).all()

    def _event_active(self, diagnostics: dict[str, Any] | None) -> bool:
        if not diagnostics:
            return False
        operator_reward = diagnostics.get("operator_reward")
        capacity_gate = diagnostics.get("event_capacity_gate")
        try:
            operator_reward_value = 0.0 if operator_reward is None else float(operator_reward)
        except (TypeError, ValueError):
            operator_reward_value = 0.0
        try:
            capacity_gate_value = 0.0 if capacity_gate is None else float(capacity_gate)
        except (TypeError, ValueError):
            capacity_gate_value = 0.0
        return (
            operator_reward_value > float(self.config.operator_event_threshold)
            or capacity_gate_value > float(self.config.operator_event_threshold)
        )

    def _event_delay_active(self) -> bool:
        if not self.config.operator_event_reset_delay_enabled:
            return False
        if self._last_operator_event_step is None:
            return False
        return (self._lifetime_step - self._last_operator_event_step) <= int(self.config.operator_event_reset_delay_steps)

    def _finalize_episode_stats(self):
        if self._lifetime_step <= 0:
            return
        self._lifetime_history.append(int(self._lifetime_step))
        if self._post_event_anchor_step is not None:
            self._post_event_continuation_history.append(int(self._lifetime_step - self._post_event_anchor_step))
        total_tokens = sum(self._episode_token_counter.values())
        self._episode_unique_token_history.append(float(len(self._episode_token_counter)))
        if total_tokens <= 0:
            self._episode_token_perplexity_history.append(0.0)
        else:
            probs = np.asarray(list(self._episode_token_counter.values()), dtype=np.float64) / float(total_tokens)
            entropy = float(-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum())
            self._episode_token_perplexity_history.append(float(math.exp(entropy)))

    def _mark_reset(self, reason: str):
        reason_key = str(reason)
        if reason_key not in self._reason_counts:
            reason_key = "manual"
        self._reason_counts[reason_key] += 1
        self._total_resets += 1
        self._reset_pending_reason = reason_key
        self._last_step_metrics["reset_reason"] = reason_key
        self._finalize_episode_stats()

    def reset(self):
        if self._started and self._reset_pending_reason is None:
            self._mark_reset("manual")
        obs = self._env.reset()
        self._started = True
        self._reset_pending_reason = None
        self._reset_runtime_state()
        self._obs_change(self._flatten_obs(obs))
        return obs

    def step(self, action):
        return self._env.step(action)

    def finalize_step(self, action, next_obs, raw_done: bool, info: dict | None, diagnostics: dict | None):
        self._lifetime_step += 1
        self._total_finalize_steps += 1
        delay_active = False
        obs_change = self._obs_change(next_obs)
        if math.isfinite(obs_change):
            self._obs_change_history.append(obs_change)
        if diagnostics:
            token_id = diagnostics.get("operator_token_id")
            try:
                if token_id is not None and math.isfinite(float(token_id)):
                    self._episode_token_counter[int(float(token_id))] += 1
            except (TypeError, ValueError):
                pass
            if self._event_active(diagnostics):
                if self._post_event_anchor_step is None:
                    self._post_event_anchor_step = self._lifetime_step
                self._last_operator_event_step = self._lifetime_step
        delay_active = self._event_delay_active()
        if delay_active:
            self._delay_active_steps += 1
        post_event_continuation_steps = (
            0.0 if self._post_event_anchor_step is None else float(self._lifetime_step - self._post_event_anchor_step)
        )
        self._last_step_metrics = {
            "persistent_enabled": 1.0,
            "lifetime_step": float(self._lifetime_step),
            "operator_event_delay_active": float(delay_active),
            "post_event_continuation_steps": post_event_continuation_steps,
            "unique_token_count_so_far": float(len(self._episode_token_counter)),
            "reset_reason": "NA",
        }

        reason: str | None = None
        if self.config.force_reset_if_action_nan and self._has_bad_numeric(action):
            reason = "nan"
        elif self.config.force_reset_if_nan and self._has_bad_numeric(next_obs):
            reason = "nan"
        elif self._lifetime_step >= int(self.config.max_lifetime_steps):
            reason = "max_lifetime"
        elif raw_done and self.config.reset_on_env_done:
            reason = "env_done"
        elif self.config.stale_reset_enabled and len(self._obs_change_history) >= int(self.config.stale_window):
            stale_mean = float(np.mean(self._obs_change_history))
            if stale_mean <= float(self.config.stale_min_obs_change) and not delay_active:
                reason = "stale"

        if reason is not None:
            self._mark_reset(reason)
            return {"done": True, "reset_reason": reason}
        return {"done": False, "reset_reason": "NA"}

    def get_persistent_step_metrics(self) -> dict[str, float | str]:
        return dict(self._last_step_metrics)

    def get_persistent_stats(self) -> dict[str, float]:
        lifetime_array = np.asarray(self._lifetime_history, dtype=np.float32)
        post_event_array = np.asarray(self._post_event_continuation_history, dtype=np.float32)
        unique_token_array = np.asarray(self._episode_unique_token_history, dtype=np.float32)
        token_ppl_array = np.asarray(self._episode_token_perplexity_history, dtype=np.float32)
        total_steps = max(1, int(self._total_finalize_steps))
        return {
            "persistent_enabled": 1.0,
            "lifetime_step_mean": float(lifetime_array.mean()) if lifetime_array.size else 0.0,
            "lifetime_step_max": float(lifetime_array.max()) if lifetime_array.size else 0.0,
            "reset_count": float(self._total_resets),
            "reset_reason_max_lifetime": float(self._reason_counts["max_lifetime"]),
            "reset_reason_stale": float(self._reason_counts["stale"]),
            "reset_reason_env_done": float(self._reason_counts["env_done"]),
            "reset_reason_nan": float(self._reason_counts["nan"]),
            "reset_reason_manual": float(self._reason_counts["manual"]),
            "operator_event_delay_active_ratio": float(self._delay_active_steps / total_steps),
            "post_event_continuation_steps_mean": float(post_event_array.mean()) if post_event_array.size else 0.0,
            "post_event_continuation_steps_max": float(post_event_array.max()) if post_event_array.size else 0.0,
            "episode_unique_token_count_mean": float(unique_token_array.mean()) if unique_token_array.size else 0.0,
            "episode_token_perplexity_mean": float(token_ppl_array.mean()) if token_ppl_array.size else 0.0,
        }

    @property
    def obs_dim(self):
        return self._env.obs_dim

    @property
    def action_dim(self):
        return self._env.action_dim

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()
