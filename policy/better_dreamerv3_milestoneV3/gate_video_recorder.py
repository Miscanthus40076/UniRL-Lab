from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch

from .event_dynamics import DreamerEventDynamicsConfig, EventCapacitySelector, build_context_change_mask


@dataclass(slots=True)
class GateVideoConfig:
    enabled: bool = True
    interval: int = 1000
    episodes: int = 3
    max_episode_steps: int = 200
    deterministic: bool = True
    save_gif: bool = True
    gif_fps: int = 20
    overlay_metrics: bool = False
    output_subdir: str = "gate_videos"
    peak_threshold: float = 0.5

    @classmethod
    def from_root_config(cls, config: dict | None) -> "GateVideoConfig":
        raw = dict((config or {}).get("gate_video", {}))
        return cls(**raw)


def parse_args():
    parser = argparse.ArgumentParser(description="Record gate evaluation rollout videos for Dreamer diagnostics.")
    parser.add_argument("--config", default=None, help="Optional path to a config file. Trainer integration usually calls this module directly.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory.")
    parser.add_argument("--global-step", type=int, default=0, help="Global step for output naming.")
    return parser.parse_args()


def _policy_gate_enabled(config: dict) -> bool:
    dreamer_cfg = dict(config.get("policy", {}).get("dreamerv3", {}))
    thick_enabled = bool(dict(dreamer_cfg.get("thick_context", {})).get("enabled", False))
    return thick_enabled


def _event_dynamics_config(config: dict) -> dict:
    return dict(config.get("policy", {}).get("dreamerv3", {}).get("event_dynamics", {}))


def maybe_record_gate_eval_videos(
    agent,
    env,
    output_dir,
    global_step: int,
    config: dict,
    device: str | None = None,
):
    gate_cfg = GateVideoConfig.from_root_config(config)
    if not gate_cfg.enabled:
        return None
    if gate_cfg.interval <= 0 or int(global_step) % int(gate_cfg.interval) != 0:
        return None
    if not _policy_gate_enabled(config):
        return None
    return record_gate_eval_videos(
        agent=agent,
        env=env,
        output_dir=output_dir,
        global_step=global_step,
        config=config,
        device=device,
    )


def _call_reset(agent):
    reset = getattr(agent, "reset", None)
    if callable(reset):
        reset()


def _call_act(agent, obs, deterministic: bool):
    act = getattr(agent, "act")
    try:
        return act(obs, deterministic=deterministic)
    except TypeError:
        return act(obs)


def _call_observe_transition(agent, next_obs, done: bool, reward: float | None = None):
    observe = getattr(agent, "observe_transition", None)
    if callable(observe):
        return observe(next_obs, done=done, reward=reward)
    return None


def _call_preview_transition(agent, next_obs, done: bool, reward: float | None = None):
    preview = getattr(agent, "preview_transition", None)
    if callable(preview):
        return preview(next_obs, done=done, reward=reward)
    return None


def _numeric_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return value
    return value


def _vector_from_value(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size == 0:
        return None
    return array


def _extract_contact_signal(info: dict[str, Any]) -> float | None:
    for key in ("contact", "is_contact", "touch", "collision"):
        value = _numeric_or_none(info.get(key))
        if value is not None:
            return 1.0 if value > 0.0 else 0.0
    return None


def _extract_success_signal(info: dict[str, Any]) -> float | None:
    for key in ("success", "is_success"):
        value = _numeric_or_none(info.get(key))
        if value is not None:
            return 1.0 if value > 0.0 else 0.0
    return None


def _extract_object_position(info: dict[str, Any]) -> np.ndarray | None:
    for key in ("object_pos", "obj_pos", "rod_pos", "peg_pos", "peg_head_pos", "target_object_pos"):
        value = _vector_from_value(info.get(key))
        if value is not None and value.size >= 2:
            return value
    return None


def _format_overlay_value(value: Any) -> str:
    value = _numeric_or_none(value)
    if value is None or not math.isfinite(value):
        return "NA"
    return f"{value:.4f}"


def _looks_like_image_observation(obs) -> bool:
    if isinstance(obs, (list, tuple)):
        return bool(obs)
    arr = np.asarray(obs)
    return arr.ndim >= 3


def _to_hwc(frame) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"Unsupported frame shape: {arr.shape}")
    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.dtype != np.uint8:
        max_value = float(arr.max()) if arr.size else 0.0
        scale = 255.0 if max_value <= 1.0 else 1.0
        arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
    return arr


def _observation_to_frame(env, obs) -> np.ndarray:
    if _looks_like_image_observation(obs):
        if isinstance(obs, (list, tuple)):
            frames = [_to_hwc(item) for item in obs]
        else:
            obs_array = np.asarray(obs)
            if obs_array.ndim == 4:
                frames = [_to_hwc(item) for item in obs_array]
            else:
                frames = [_to_hwc(obs_array)]
        target_height = max(frame.shape[0] for frame in frames)
        padded = []
        for frame in frames:
            if frame.shape[0] == target_height:
                padded.append(frame)
                continue
            pad = target_height - frame.shape[0]
            padded.append(np.pad(frame, ((0, pad), (0, 0), (0, 0)), mode="constant"))
        return np.concatenate(padded, axis=1)
    return _to_hwc(env.render())


def _overlay_frame(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    overlay_height = 10 + 16 * len(lines)
    draw.rectangle((0, 0, image.width, overlay_height), fill=(0, 0, 0))
    y = 6
    for line in lines:
        draw.text((8, y), line, fill=(255, 255, 255))
        y += 16
    return np.asarray(image)


def _collect_gate_metrics(agent) -> dict[str, Any]:
    getter = getattr(agent, "get_gate_diagnostics", None)
    if callable(getter):
        return dict(getter())
    getter = getattr(agent, "get_diagnostics", None)
    if callable(getter):
        payload = getter()
        if isinstance(payload, dict):
            if isinstance(payload.get("gate"), dict):
                return dict(payload["gate"])
            metrics = {}
            if isinstance(payload.get("context"), dict):
                metrics.update(payload["context"])
            if isinstance(payload.get("event"), dict):
                metrics.update(payload["event"])
            return metrics
    return {}


def _finalize_persistent_step(env, action, next_obs, raw_done: bool, info: dict, diagnostics: dict | None):
    finalize = getattr(env, "finalize_step", None)
    if callable(finalize):
        payload = finalize(
            action=action,
            next_obs=next_obs,
            raw_done=bool(raw_done),
            info=info or {},
            diagnostics=diagnostics or {},
        )
        if isinstance(payload, dict):
            return payload
    return {"done": bool(raw_done), "reset_reason": ("env_done" if raw_done else "NA")}


def _persistent_step_metrics(env) -> dict[str, Any]:
    getter = getattr(env, "get_persistent_step_metrics", None)
    if callable(getter):
        payload = getter()
        if isinstance(payload, dict):
            return dict(payload)
    return {}


def _jsonable_value(value: Any):
    numeric = _numeric_or_none(value)
    if numeric is None:
        return "NA"
    if not math.isfinite(numeric):
        return "NA"
    return float(numeric)


def _string_or_na(value: Any):
    if value is None:
        return "NA"
    text = str(value).strip()
    return text or "NA"


def _sigmoid_float(value: float) -> float:
    value = max(min(float(value), 60.0), -60.0)
    return 1.0 / (1.0 + math.exp(-value))


def _finite_values(records: list[dict], key: str) -> tuple[list[float], int]:
    values: list[float] = []
    bad_numeric_count = 0
    for record in records:
        value = _numeric_or_none(record.get(key))
        if value is None:
            continue
        if not math.isfinite(value):
            bad_numeric_count += 1
            continue
        values.append(float(value))
    return values, bad_numeric_count


def _mean_or_na(values: list[float]):
    if not values:
        return "NA"
    return float(sum(values) / len(values))


def _max_or_na(values: list[float]):
    if not values:
        return "NA"
    return float(max(values))


def _min_or_na(values: list[float]):
    if not values:
        return "NA"
    return float(min(values))


def _std_or_na(values: list[float]):
    if not values:
        return "NA"
    return float(np.asarray(values, dtype=np.float64).std())


def _dynamic_peak_stats(values: list[float]):
    if not values:
        return "NA", "NA", "NA"
    mean = float(np.asarray(values, dtype=np.float64).mean())
    std = float(np.asarray(values, dtype=np.float64).std())
    threshold = mean + std
    count = int(sum(value > threshold for value in values))
    ratio = float(count / max(len(values), 1))
    return threshold, count, ratio


def _top10_peak_stats(records: list[dict]):
    grouped: dict[int, list[float]] = {}
    for record in records:
        episode_index = int(record.get("episode_index", 0))
        gate = _numeric_or_none(record.get("event_gate"))
        if gate is None or not math.isfinite(gate):
            continue
        grouped.setdefault(episode_index, []).append(float(gate))
    if not grouped:
        return "NA", "NA"
    total_count = 0
    total_steps = 0
    for values in grouped.values():
        if not values:
            continue
        total_steps += len(values)
        total_count += max(1, int(math.ceil(len(values) * 0.1)))
    if total_steps <= 0:
        return "NA", "NA"
    return int(total_count), float(total_count / total_steps)


def _relative_improvement(ordinary_mean, mixed_mean):
    if not isinstance(ordinary_mean, (float, int)) or not isinstance(mixed_mean, (float, int)):
        return "NA"
    denom = max(float(ordinary_mean), 1e-8)
    return float((float(ordinary_mean) - float(mixed_mean)) / denom)


def _top_error_gate_means(records: list[dict]):
    indexed = []
    for index, record in enumerate(records):
        err = _numeric_or_none(record.get("event_prediction_error"))
        if err is None or not math.isfinite(err):
            continue
        indexed.append((index, float(err)))
    if not indexed:
        return "NA", "NA"
    indexed.sort(key=lambda item: item[1], reverse=True)
    top_count = max(1, int(math.ceil(len(indexed) * 0.1)))
    top_indices = {index for index, _ in indexed[:top_count]}
    top_values = []
    normal_values = []
    for index, record in enumerate(records):
        gate = _numeric_or_none(record.get("event_gate"))
        if gate is None or not math.isfinite(gate):
            continue
        if index in top_indices:
            top_values.append(float(gate))
        else:
            normal_values.append(float(gate))
    return _mean_or_na(top_values), _mean_or_na(normal_values)


def _top_fraction_means(records: list[dict], value_key: str, fraction: float = 0.1):
    indexed = []
    for index, record in enumerate(records):
        value = _numeric_or_none(record.get(value_key))
        if value is None or not math.isfinite(value):
            continue
        indexed.append((index, float(value)))
    if not indexed:
        return "NA", "NA"
    indexed.sort(key=lambda item: item[1], reverse=True)
    top_count = max(1, int(math.ceil(len(indexed) * max(float(fraction), 0.0))))
    top_indices = {index for index, _ in indexed[:top_count]}
    top_values = []
    normal_values = []
    for index, record in enumerate(records):
        value = _numeric_or_none(record.get(value_key))
        if value is None or not math.isfinite(value):
            continue
        if index in top_indices:
            top_values.append(float(value))
        else:
            normal_values.append(float(value))
    return _mean_or_na(top_values), _mean_or_na(normal_values)


def _peak_stats(values: list[float], total_steps: int, threshold: float):
    if not values or total_steps <= 0:
        return "NA", "NA"
    count = int(sum(value > threshold for value in values))
    return count, float(count / total_steps)


def _weighted_mean_from_records(records: list[dict], value_key: str, mask_key: str, invert: bool = False):
    weighted_sum = 0.0
    total_weight = 0.0
    for record in records:
        value = _numeric_or_none(record.get(value_key))
        mask = _numeric_or_none(record.get(mask_key))
        if value is None or mask is None or not math.isfinite(value) or not math.isfinite(mask):
            continue
        weight = max(0.0, min(1.0, float(mask)))
        if invert:
            weight = 1.0 - weight
        if weight <= 0.0:
            continue
        weighted_sum += float(value) * weight
        total_weight += weight
    if total_weight <= 0.0:
        return "NA"
    return float(weighted_sum / total_weight)


def _bool_check(condition: bool) -> bool:
    return bool(condition)


def _build_summary(records: list[dict], global_step: int, num_episodes: int, peak_threshold: float) -> dict[str, Any]:
    total_steps = len(records)
    context_values, context_bad = _finite_values(records, "context_gate")
    context_change_mask_values, context_change_mask_bad = _finite_values(records, "context_change_mask")
    event_capacity_gate_values, event_capacity_gate_bad = _finite_values(records, "event_capacity_gate")
    event_capacity_k_values, event_capacity_k_bad = _finite_values(records, "event_capacity_k")
    context_l0_values, context_l0_bad = _finite_values(records, "context_l0_open_prob")
    context_gate_hard_values, context_gate_hard_bad = _finite_values(records, "context_gate_hard")
    context_gate_soft_values, context_gate_soft_bad = _finite_values(records, "context_gate_soft")
    context_gate_logit_values, context_gate_logit_bad = _finite_values(records, "context_gate_logit")
    context_update_loss_values, context_update_loss_bad = _finite_values(records, "context_update_loss_raw")
    context_update_loss_scaled_values, context_update_loss_scaled_bad = _finite_values(records, "context_update_loss_scaled")
    operator_reward_values, operator_reward_bad = _finite_values(records, "operator_reward")
    operator_token_id_values, operator_token_id_bad = _finite_values(records, "operator_token_id")
    operator_token_novelty_values, operator_token_novelty_bad = _finite_values(records, "operator_token_novelty")
    operator_rarity_values, operator_rarity_bad = _finite_values(records, "operator_rarity")
    operator_effect_gain_values, operator_effect_gain_bad = _finite_values(records, "operator_effect_gain")
    operator_controllability_gain_values, operator_controllability_gain_bad = _finite_values(
        records, "operator_controllability_gain"
    )
    operator_valid_effect_values, operator_valid_effect_bad = _finite_values(records, "operator_valid_effect")
    operator_valid_control_values, operator_valid_control_bad = _finite_values(records, "operator_valid_control")
    event_intensity_values, event_intensity_bad = _finite_values(records, "event_intensity")
    milestone_reward_values, milestone_reward_bad = _finite_values(records, "milestone_reward")
    milestone_trigger_values, milestone_trigger_bad = _finite_values(records, "milestone_trigger")
    running_best_intensity_values, running_best_intensity_bad = _finite_values(records, "running_best_intensity")
    habituation_factor_values, habituation_factor_bad = _finite_values(records, "habituation_factor")
    token_repeat_decay_values, token_repeat_decay_bad = _finite_values(records, "token_repeat_decay")
    topk_rank_values, topk_rank_bad = _finite_values(records, "topk_rank")
    topk_env_bonus_values, topk_env_bonus_bad = _finite_values(records, "topk_env_bonus")
    topk_entered_values, topk_entered_bad = _finite_values(records, "topk_entered")
    topk_leaderboard_size_values, topk_leaderboard_size_bad = _finite_values(records, "topk_leaderboard_size")
    topk_update_count_values, topk_update_count_bad = _finite_values(records, "topk_update_count")
    operator_token_perplexity_values, operator_token_perplexity_bad = _finite_values(
        records, "operator_token_perplexity_online"
    )
    operator_num_active_tokens_values, operator_num_active_tokens_bad = _finite_values(
        records, "operator_num_active_tokens_online"
    )
    operator_token_count_min_values, operator_token_count_min_bad = _finite_values(records, "operator_token_count_min")
    operator_token_count_max_values, operator_token_count_max_bad = _finite_values(records, "operator_token_count_max")
    operator_raw_reward_values, operator_raw_reward_bad = _finite_values(records, "operator_raw_reward")
    operator_clipped_reward_values, operator_clipped_reward_bad = _finite_values(records, "operator_clipped_reward")
    operator_reward_saturation_values, operator_reward_saturation_bad = _finite_values(records, "operator_reward_saturation")
    operator_nonzero_outside_values, operator_nonzero_outside_bad = _finite_values(
        records, "operator_reward_nonzero_outside_capacity"
    )
    operator_mask_violation_values, operator_mask_violation_bad = _finite_values(
        records, "operator_reward_mask_violation_count"
    )
    operator_control_suppressed_values, operator_control_suppressed_bad = _finite_values(
        records, "operator_control_only_suppressed"
    )
    operator_valid_transition_values, operator_valid_transition_bad = _finite_values(records, "operator_valid_transition")
    operator_stale_suppressed_values, operator_stale_suppressed_bad = _finite_values(
        records, "operator_stale_suppressed"
    )
    operator_latent_delta_values, operator_latent_delta_bad = _finite_values(records, "operator_latent_delta_norm")
    operator_eval_window_ready_values, operator_eval_window_ready_bad = _finite_values(
        records, "operator_eval_capacity_window_ready"
    )
    operator_eval_window_size_values, operator_eval_window_size_bad = _finite_values(
        records, "operator_eval_capacity_window_size"
    )
    event_values, event_bad = _finite_values(records, "event_gate")
    event_logit_values, event_logit_bad = _finite_values(records, "event_logit")
    event_logit_sigmoid_values, event_logit_sigmoid_bad = _finite_values(records, "event_logit_sigmoid")
    ordinary_values, ordinary_bad = _finite_values(records, "pred_next_feat_error_ordinary_only")
    mixed_values, mixed_bad = _finite_values(records, "pred_next_feat_error_event_mixed")
    event_err_values, event_err_bad = _finite_values(records, "event_prediction_error")
    event_residual_norm_values, event_residual_norm_bad = _finite_values(records, "event_residual_norm")
    context_delta_values, context_delta_bad = _finite_values(records, "context_delta_norm")
    lifetime_step_values, lifetime_step_bad = _finite_values(records, "lifetime_step")
    operator_event_delay_values, operator_event_delay_bad = _finite_values(records, "operator_event_delay_active")
    post_event_values, post_event_bad = _finite_values(records, "post_event_continuation_steps")
    unique_token_so_far_values, unique_token_so_far_bad = _finite_values(records, "unique_token_count_so_far")
    contact_rate_values, contact_rate_bad = _finite_values(records, "contact_rate")
    object_motion_values, object_motion_bad = _finite_values(records, "object_motion")
    success_rate_values, success_rate_bad = _finite_values(records, "success_rate")
    bad_numeric_count = (
        context_bad
        + context_change_mask_bad
        + event_capacity_gate_bad
        + event_capacity_k_bad
        + context_l0_bad
        + context_gate_hard_bad
        + context_gate_soft_bad
        + context_gate_logit_bad
        + context_update_loss_bad
        + context_update_loss_scaled_bad
        + operator_reward_bad
        + operator_token_id_bad
        + operator_token_novelty_bad
        + operator_rarity_bad
        + operator_effect_gain_bad
        + operator_controllability_gain_bad
        + operator_valid_effect_bad
        + operator_valid_control_bad
        + event_intensity_bad
        + milestone_reward_bad
        + milestone_trigger_bad
        + running_best_intensity_bad
        + habituation_factor_bad
        + token_repeat_decay_bad
        + topk_rank_bad
        + topk_env_bonus_bad
        + topk_entered_bad
        + topk_leaderboard_size_bad
        + topk_update_count_bad
        + operator_token_perplexity_bad
        + operator_num_active_tokens_bad
        + operator_token_count_min_bad
        + operator_token_count_max_bad
        + operator_raw_reward_bad
        + operator_clipped_reward_bad
        + operator_reward_saturation_bad
        + operator_nonzero_outside_bad
        + operator_mask_violation_bad
        + operator_control_suppressed_bad
        + operator_valid_transition_bad
        + operator_stale_suppressed_bad
        + operator_latent_delta_bad
        + operator_eval_window_ready_bad
        + operator_eval_window_size_bad
        + event_bad
        + event_logit_bad
        + event_logit_sigmoid_bad
        + ordinary_bad
        + mixed_bad
        + event_err_bad
        + event_residual_norm_bad
        + context_delta_bad
        + lifetime_step_bad
        + operator_event_delay_bad
        + post_event_bad
        + unique_token_so_far_bad
        + contact_rate_bad
        + object_motion_bad
        + success_rate_bad
    )

    context_gate_type = "NA"
    context_change_mask_source = "NA"
    context_change_mask_mode = "NA"
    context_change_mask_top_percent = "NA"
    event_capacity_enabled = "NA"
    event_capacity_mode = "NA"
    event_capacity_ratio = "NA"
    capacity_v2_detach_event_input = "NA"
    capacity_v2_detach_event_target = "NA"
    capacity_v2_use_sigmoid_gate_multiplier = "NA"
    event_loss_updates_backbone = "NA"
    for record in records:
        if record.get("context_gate_type") not in (None, "", "NA"):
            context_gate_type = str(record.get("context_gate_type"))
        if record.get("context_change_mask_source") not in (None, "", "NA"):
            context_change_mask_source = str(record.get("context_change_mask_source"))
        if record.get("context_change_mask_mode") not in (None, "", "NA"):
            context_change_mask_mode = str(record.get("context_change_mask_mode"))
        top_percent = _numeric_or_none(record.get("context_change_mask_top_percent"))
        if top_percent is not None and math.isfinite(top_percent):
            context_change_mask_top_percent = float(top_percent)
        capacity_enabled_value = _numeric_or_none(record.get("event_capacity_enabled"))
        if capacity_enabled_value is not None and math.isfinite(capacity_enabled_value):
            event_capacity_enabled = float(capacity_enabled_value)
        if record.get("event_capacity_mode") not in (None, "", "NA"):
            event_capacity_mode = str(record.get("event_capacity_mode"))
        capacity_ratio_value = _numeric_or_none(record.get("event_capacity_ratio"))
        if capacity_ratio_value is not None and math.isfinite(capacity_ratio_value):
            event_capacity_ratio = float(capacity_ratio_value)
        detach_input_value = _numeric_or_none(record.get("capacity_v2_detach_event_input"))
        if detach_input_value is not None and math.isfinite(detach_input_value):
            capacity_v2_detach_event_input = bool(detach_input_value)
        detach_target_value = _numeric_or_none(record.get("capacity_v2_detach_event_target"))
        if detach_target_value is not None and math.isfinite(detach_target_value):
            capacity_v2_detach_event_target = bool(detach_target_value)
        gate_multiplier_value = _numeric_or_none(record.get("capacity_v2_use_sigmoid_gate_multiplier"))
        if gate_multiplier_value is not None and math.isfinite(gate_multiplier_value):
            capacity_v2_use_sigmoid_gate_multiplier = bool(gate_multiplier_value)
        event_loss_updates_value = _numeric_or_none(record.get("event_loss_updates_backbone"))
        if event_loss_updates_value is not None and math.isfinite(event_loss_updates_value):
            event_loss_updates_backbone = bool(event_loss_updates_value)
        if (
            context_gate_type != "NA"
            and context_change_mask_source != "NA"
            and context_change_mask_mode != "NA"
            and context_change_mask_top_percent != "NA"
        ):
            break
    if not event_logit_sigmoid_values and event_logit_values:
        event_logit_sigmoid_values = [_sigmoid_float(value) for value in event_logit_values]
    mean_context_gate = _mean_or_na(context_values)
    context_change_mask_mean = _mean_or_na(context_change_mask_values)
    if context_change_mask_values and total_steps > 0:
        context_change_mask_nonzero_ratio = float(sum(value > 0.0 for value in context_change_mask_values) / total_steps)
    else:
        context_change_mask_nonzero_ratio = "NA"
    context_l0_open_prob = _mean_or_na(context_l0_values)
    context_gate_hard_mean = _mean_or_na(context_gate_hard_values)
    context_gate_soft_mean = _mean_or_na(context_gate_soft_values)
    context_gate_logit_mean = _mean_or_na(context_gate_logit_values)
    context_gate_logit_std = _std_or_na(context_gate_logit_values)
    context_update_loss_raw = _mean_or_na(context_update_loss_values)
    context_update_loss_scaled = _mean_or_na(context_update_loss_scaled_values)
    operator_intrinsic_reward_mean = _mean_or_na(operator_reward_values)
    operator_intrinsic_reward_std = _std_or_na(operator_reward_values)
    operator_intrinsic_reward_max = _max_or_na(operator_reward_values)
    if operator_reward_values and total_steps > 0:
        operator_intrinsic_reward_nonzero_ratio = float(sum(value > 0.0 for value in operator_reward_values) / total_steps)
    else:
        operator_intrinsic_reward_nonzero_ratio = "NA"
    operator_token_novelty_mean = _mean_or_na(operator_token_novelty_values)
    operator_rarity_mean = _mean_or_na(operator_rarity_values)
    operator_effect_gain_mean = _mean_or_na(operator_effect_gain_values)
    operator_controllability_gain_mean = _mean_or_na(operator_controllability_gain_values)
    operator_valid_effect_ratio = _mean_or_na(operator_valid_effect_values)
    operator_valid_control_ratio = _mean_or_na(operator_valid_control_values)
    event_intensity_mean = _mean_or_na(event_intensity_values)
    event_intensity_std = _std_or_na(event_intensity_values)
    event_intensity_max = _max_or_na(event_intensity_values)
    milestone_reward_mean = _mean_or_na(milestone_reward_values)
    if milestone_trigger_values:
        milestone_trigger_count = float(sum(milestone_trigger_values))
        milestone_nonzero_ratio = float(sum(value > 0.0 for value in milestone_trigger_values) / total_steps)
    else:
        milestone_trigger_count = "NA"
        milestone_nonzero_ratio = "NA"
    running_best_intensity = _mean_or_na(running_best_intensity_values)
    habituation_factor_mean = _mean_or_na(habituation_factor_values)
    token_repeat_decay_mean = _mean_or_na(token_repeat_decay_values)
    topk_rank_mean = _mean_or_na(topk_rank_values)
    topk_env_bonus_mean = _mean_or_na(topk_env_bonus_values)
    topk_entered_ratio = _mean_or_na(topk_entered_values)
    topk_leaderboard_size = _mean_or_na(topk_leaderboard_size_values)
    topk_update_count = _max_or_na(topk_update_count_values)
    operator_token_perplexity_online = _mean_or_na(operator_token_perplexity_values)
    operator_num_active_tokens_online = _mean_or_na(operator_num_active_tokens_values)
    operator_token_count_min = _min_or_na(operator_token_count_min_values)
    operator_token_count_max = _max_or_na(operator_token_count_max_values)
    operator_raw_reward_mean = _mean_or_na(operator_raw_reward_values)
    operator_clipped_reward_mean = _mean_or_na(operator_clipped_reward_values)
    operator_reward_saturation_ratio = _mean_or_na(operator_reward_saturation_values)
    operator_reward_nonzero_outside_capacity = _mean_or_na(operator_nonzero_outside_values)
    operator_reward_mask_violation_count = _max_or_na(operator_mask_violation_values)
    operator_control_only_suppressed_ratio = _mean_or_na(operator_control_suppressed_values)
    operator_valid_transition_ratio = _mean_or_na(operator_valid_transition_values)
    operator_stale_suppressed_ratio = _mean_or_na(operator_stale_suppressed_values)
    operator_latent_delta_norm_mean = _mean_or_na(operator_latent_delta_values)
    operator_eval_capacity_window_ready_ratio = _mean_or_na(operator_eval_window_ready_values)
    operator_eval_capacity_window_size_max = _max_or_na(operator_eval_window_size_values)
    mean_event_gate = _mean_or_na(event_values)
    event_capacity_k_mean = _mean_or_na(event_capacity_k_values)
    event_capacity_gate_mean = _mean_or_na(event_capacity_gate_values)
    event_capacity_gate_std = _std_or_na(event_capacity_gate_values)
    event_capacity_gate_min = _min_or_na(event_capacity_gate_values)
    event_capacity_gate_max = _max_or_na(event_capacity_gate_values)
    if event_capacity_gate_values and total_steps > 0:
        event_capacity_nonzero_ratio = float(sum(value > 0.0 for value in event_capacity_gate_values) / total_steps)
    else:
        event_capacity_nonzero_ratio = "NA"
    context_gate_std = _std_or_na(context_values)
    event_gate_std = _std_or_na(event_values)
    context_gate_min = _min_or_na(context_values)
    event_gate_min = _min_or_na(event_values)
    max_context_gate = _max_or_na(context_values)
    max_event_gate = _max_or_na(event_values)
    event_logit_mean = _mean_or_na(event_logit_values)
    event_logit_std = _std_or_na(event_logit_values)
    event_logit_sigmoid_mean = _mean_or_na(event_logit_sigmoid_values)
    ordinary_only_error_mean = _mean_or_na(ordinary_values)
    event_mixed_error_mean = _mean_or_na(mixed_values)
    relative_prediction_improvement = _relative_improvement(ordinary_only_error_mean, event_mixed_error_mean)
    event_gate_at_top_error_mean, event_gate_at_normal_steps_mean = _top_error_gate_means(records)
    context_delta_norm_mean = _mean_or_na(context_delta_values)
    event_prediction_error_mean = _mean_or_na(event_err_values)
    lifetime_step_mean = _mean_or_na(lifetime_step_values)
    lifetime_step_max = _max_or_na(lifetime_step_values)
    operator_event_delay_active_ratio = _mean_or_na(operator_event_delay_values)
    post_event_continuation_steps_mean = _mean_or_na(post_event_values)
    post_event_continuation_steps_max = _max_or_na(post_event_values)
    reset_count = 0
    reset_reason_counts = {
        "max_lifetime": 0,
        "stale": 0,
        "env_done": 0,
        "nan": 0,
        "manual": 0,
    }
    for record in records:
        reason = str(record.get("reset_reason", "NA"))
        if reason in reset_reason_counts:
            reset_reason_counts[reason] += 1
            reset_count += 1
    episode_unique_token_terminal_values = []
    episode_token_perplexity_values = []
    per_episode_tokens: dict[int, list[int]] = {}
    per_episode_unique_counts: dict[int, float] = {}
    for record in records:
        token_id = _numeric_or_none(record.get("operator_token_id"))
        if token_id is None or not math.isfinite(token_id):
            pass
        else:
            per_episode_tokens.setdefault(int(record.get("episode_index", 0)), []).append(int(round(token_id)))
        unique_value = _numeric_or_none(record.get("unique_token_count_so_far"))
        if unique_value is not None and math.isfinite(unique_value):
            episode_index = int(record.get("episode_index", 0))
            per_episode_unique_counts[episode_index] = max(
                float(unique_value),
                per_episode_unique_counts.get(episode_index, 0.0),
            )
    if per_episode_unique_counts:
        episode_unique_token_terminal_values.extend(per_episode_unique_counts.values())
    for token_ids in per_episode_tokens.values():
        if not token_ids:
            continue
        counts = np.asarray(list(Counter(token_ids).values()), dtype=np.float64)
        probs = counts / counts.sum()
        entropy = float(-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum())
        episode_token_perplexity_values.append(float(math.exp(entropy)))
    episode_token_perplexity_mean = _mean_or_na(episode_token_perplexity_values)
    episode_unique_token_count_mean = _mean_or_na(episode_unique_token_terminal_values)
    contact_rate_mean = _mean_or_na(contact_rate_values)
    object_motion_mean = _mean_or_na(object_motion_values)
    success_rate_mean = _mean_or_na(success_rate_values)
    event_logit_top_mean = _weighted_mean_from_records(records, "event_logit", "event_capacity_gate", invert=False)
    event_logit_normal_mean = _weighted_mean_from_records(records, "event_logit", "event_capacity_gate", invert=True)
    event_logit_sigmoid_capacity_mean = _weighted_mean_from_records(
        records,
        "event_logit_sigmoid",
        "event_capacity_gate",
        invert=False,
    )
    event_logit_sigmoid_outside_capacity_mean = _weighted_mean_from_records(
        records,
        "event_logit_sigmoid",
        "event_capacity_gate",
        invert=True,
    )
    event_gate_capacity_mean = _weighted_mean_from_records(records, "event_gate", "event_capacity_gate", invert=False)
    event_gate_outside_capacity_mean = _weighted_mean_from_records(records, "event_gate", "event_capacity_gate", invert=True)
    event_intensity_top_mean, event_intensity_normal_mean = _top_fraction_means(records, "event_intensity", fraction=0.1)
    ordinary_error_capacity_mean = _weighted_mean_from_records(
        records,
        "pred_next_feat_error_ordinary_only",
        "event_capacity_gate",
        invert=False,
    )
    mixed_error_capacity_mean = _weighted_mean_from_records(
        records,
        "pred_next_feat_error_event_mixed",
        "event_capacity_gate",
        invert=False,
    )
    relative_prediction_improvement_capacity = _relative_improvement(
        ordinary_error_capacity_mean,
        mixed_error_capacity_mean,
    )
    event_residual_norm_capacity_mean = _weighted_mean_from_records(
        records,
        "event_residual_norm",
        "event_capacity_gate",
        invert=False,
    )
    event_residual_norm_outside_capacity_mean = _weighted_mean_from_records(
        records,
        "event_residual_norm",
        "event_capacity_gate",
        invert=True,
    )
    event_gate_masked_mean = _weighted_mean_from_records(records, "event_gate", "context_change_mask", invert=False)
    event_gate_unmasked_mean = _weighted_mean_from_records(records, "event_gate", "context_change_mask", invert=True)
    ordinary_error_all_mean = ordinary_only_error_mean
    mixed_error_all_mean = event_mixed_error_mean
    ordinary_error_masked_mean = _weighted_mean_from_records(
        records,
        "pred_next_feat_error_ordinary_only",
        "context_change_mask",
        invert=False,
    )
    mixed_error_masked_mean = _weighted_mean_from_records(
        records,
        "pred_next_feat_error_event_mixed",
        "context_change_mask",
        invert=False,
    )
    relative_prediction_improvement_all = relative_prediction_improvement
    relative_prediction_improvement_masked = _relative_improvement(ordinary_error_masked_mean, mixed_error_masked_mean)
    event_residual_norm_masked_mean = _weighted_mean_from_records(
        records,
        "event_residual_norm",
        "context_change_mask",
        invert=False,
    )
    event_residual_norm_unmasked_mean = _weighted_mean_from_records(
        records,
        "event_residual_norm",
        "context_change_mask",
        invert=True,
    )
    event_gate_peak_count, event_gate_peak_ratio = _peak_stats(event_values, total_steps, peak_threshold)
    event_gate_threshold_dynamic, event_gate_dynamic_peak_count, event_gate_dynamic_peak_ratio = _dynamic_peak_stats(event_values)
    event_gate_top10_peak_count, event_gate_top10_peak_ratio = _top10_peak_stats(records)
    context_gate_peak_count, _ = _peak_stats(context_values, total_steps, peak_threshold)

    checks = {
        "gate_finite": _bool_check(bad_numeric_count == 0),
        "event_gate_not_all_zero": _bool_check(isinstance(mean_event_gate, (float, int)) and mean_event_gate > 0.001),
        "event_gate_not_all_one": _bool_check(isinstance(mean_event_gate, (float, int)) and mean_event_gate < 0.8),
        "context_gate_not_all_zero": _bool_check(isinstance(mean_context_gate, (float, int)) and mean_context_gate > 0.001),
        "context_gate_not_all_one": _bool_check(isinstance(mean_context_gate, (float, int)) and mean_context_gate < 0.8),
        "event_mixed_not_worse_than_ordinary": _bool_check(
            isinstance(event_mixed_error_mean, (float, int))
            and isinstance(ordinary_only_error_mean, (float, int))
            and event_mixed_error_mean <= ordinary_only_error_mean * 1.05
        ),
        "event_gate_correlates_with_error": _bool_check(
            isinstance(event_gate_at_top_error_mean, (float, int))
            and isinstance(event_gate_at_normal_steps_mean, (float, int))
            and event_gate_at_top_error_mean > event_gate_at_normal_steps_mean
        ),
        "sparse_peak_pattern": _bool_check(
            isinstance(event_gate_peak_ratio, (float, int)) and 0.001 < event_gate_peak_ratio < 0.5
        ),
        "dynamic_peak_pattern": _bool_check(
            isinstance(event_gate_dynamic_peak_ratio, (float, int)) and 0.0 < event_gate_dynamic_peak_ratio < 0.5
        ),
        "top10_peak_pattern": _bool_check(
            isinstance(event_gate_top10_peak_ratio, (float, int)) and 0.05 <= event_gate_top10_peak_ratio <= 0.15
        ),
    }
    return {
        "global_step": int(global_step),
        "num_episodes": int(num_episodes),
        "total_steps": int(total_steps),
        "peak_threshold": float(peak_threshold),
        "context_gate_type": context_gate_type,
        "mean_context_gate": mean_context_gate,
        "context_change_mask_mean": context_change_mask_mean,
        "context_change_mask_nonzero_ratio": context_change_mask_nonzero_ratio,
        "context_change_mask_source": context_change_mask_source,
        "context_change_mask_mode": context_change_mask_mode,
        "context_change_mask_top_percent": context_change_mask_top_percent,
        "event_capacity_enabled": event_capacity_enabled,
        "event_capacity_mode": event_capacity_mode,
        "event_capacity_ratio": event_capacity_ratio,
        "capacity_v2_detach_event_input": capacity_v2_detach_event_input,
        "capacity_v2_detach_event_target": capacity_v2_detach_event_target,
        "capacity_v2_use_sigmoid_gate_multiplier": capacity_v2_use_sigmoid_gate_multiplier,
        "event_loss_updates_backbone": event_loss_updates_backbone,
        "event_capacity_k_mean": event_capacity_k_mean,
        "event_capacity_gate_mean": event_capacity_gate_mean,
        "event_capacity_gate_std": event_capacity_gate_std,
        "event_capacity_gate_min": event_capacity_gate_min,
        "event_capacity_gate_max": event_capacity_gate_max,
        "event_capacity_nonzero_ratio": event_capacity_nonzero_ratio,
        "context_l0_open_prob": context_l0_open_prob,
        "context_gate_hard_mean": context_gate_hard_mean,
        "context_gate_soft_mean": context_gate_soft_mean,
        "context_gate_logit_mean": context_gate_logit_mean,
        "context_gate_logit_std": context_gate_logit_std,
        "context_update_loss_raw": context_update_loss_raw,
        "context_update_loss_scaled": context_update_loss_scaled,
        "operator_intrinsic_reward_mean": operator_intrinsic_reward_mean,
        "operator_intrinsic_reward_std": operator_intrinsic_reward_std,
        "operator_intrinsic_reward_max": operator_intrinsic_reward_max,
        "operator_intrinsic_reward_nonzero_ratio": operator_intrinsic_reward_nonzero_ratio,
        "operator_token_novelty_mean": operator_token_novelty_mean,
        "operator_rarity_mean": operator_rarity_mean,
        "operator_effect_gain_mean": operator_effect_gain_mean,
        "operator_controllability_gain_mean": operator_controllability_gain_mean,
        "operator_valid_effect_ratio": operator_valid_effect_ratio,
        "operator_valid_control_ratio": operator_valid_control_ratio,
        "event_intensity_mean": event_intensity_mean,
        "event_intensity_std": event_intensity_std,
        "event_intensity_max": event_intensity_max,
        "milestone_reward_mean": milestone_reward_mean,
        "milestone_trigger_count": milestone_trigger_count,
        "running_best_intensity": running_best_intensity,
        "habituation_factor_mean": habituation_factor_mean,
        "token_repeat_decay_mean": token_repeat_decay_mean,
        "milestone_nonzero_ratio": milestone_nonzero_ratio,
        "topk_rank_mean": topk_rank_mean,
        "topk_env_bonus_mean": topk_env_bonus_mean,
        "topk_entered_ratio": topk_entered_ratio,
        "topk_leaderboard_size": topk_leaderboard_size,
        "topk_update_count": topk_update_count,
        "operator_token_perplexity_online": operator_token_perplexity_online,
        "operator_num_active_tokens_online": operator_num_active_tokens_online,
        "operator_token_count_min": operator_token_count_min,
        "operator_token_count_max": operator_token_count_max,
        "operator_raw_reward_mean": operator_raw_reward_mean,
        "operator_clipped_reward_mean": operator_clipped_reward_mean,
        "operator_reward_saturation_ratio": operator_reward_saturation_ratio,
        "operator_reward_nonzero_outside_capacity": operator_reward_nonzero_outside_capacity,
        "operator_reward_mask_violation_count": operator_reward_mask_violation_count,
        "operator_control_only_suppressed_ratio": operator_control_only_suppressed_ratio,
        "operator_valid_transition_ratio": operator_valid_transition_ratio,
        "operator_stale_suppressed_ratio": operator_stale_suppressed_ratio,
        "operator_latent_delta_norm_mean": operator_latent_delta_norm_mean,
        "operator_eval_capacity_window_ready_ratio": operator_eval_capacity_window_ready_ratio,
        "operator_eval_capacity_window_size_max": operator_eval_capacity_window_size_max,
        "mean_event_gate": mean_event_gate,
        "event_gate_mean": mean_event_gate,
        "context_gate_std": context_gate_std,
        "event_gate_std": event_gate_std,
        "context_gate_min": context_gate_min,
        "event_gate_min": event_gate_min,
        "max_context_gate": max_context_gate,
        "max_event_gate": max_event_gate,
        "event_gate_max": max_event_gate,
        "event_logit_mean": event_logit_mean,
        "event_logit_std": event_logit_std,
        "event_logit_sigmoid_mean": event_logit_sigmoid_mean,
        "event_logit_top_mean": event_logit_top_mean,
        "event_logit_normal_mean": event_logit_normal_mean,
        "event_logit_sigmoid_capacity_mean": event_logit_sigmoid_capacity_mean,
        "event_logit_sigmoid_outside_capacity_mean": event_logit_sigmoid_outside_capacity_mean,
        "event_gate_threshold_dynamic": event_gate_threshold_dynamic,
        "context_gate_peak_count": context_gate_peak_count,
        "event_gate_peak_count": event_gate_peak_count,
        "event_gate_peak_ratio": event_gate_peak_ratio,
        "event_gate_dynamic_peak_count": event_gate_dynamic_peak_count,
        "event_gate_dynamic_peak_ratio": event_gate_dynamic_peak_ratio,
        "event_gate_top10_peak_count": event_gate_top10_peak_count,
        "event_gate_top10_peak_ratio": event_gate_top10_peak_ratio,
        "ordinary_only_error_mean": ordinary_only_error_mean,
        "event_mixed_error_mean": event_mixed_error_mean,
        "relative_prediction_improvement": relative_prediction_improvement,
        "ordinary_error_all_mean": ordinary_error_all_mean,
        "mixed_error_all_mean": mixed_error_all_mean,
        "ordinary_error_capacity_mean": ordinary_error_capacity_mean,
        "mixed_error_capacity_mean": mixed_error_capacity_mean,
        "relative_prediction_improvement_capacity": relative_prediction_improvement_capacity,
        "ordinary_error_masked_mean": ordinary_error_masked_mean,
        "mixed_error_masked_mean": mixed_error_masked_mean,
        "relative_prediction_improvement_all": relative_prediction_improvement_all,
        "relative_prediction_improvement_masked": relative_prediction_improvement_masked,
        "event_gate_capacity_mean": event_gate_capacity_mean,
        "event_gate_outside_capacity_mean": event_gate_outside_capacity_mean,
        "event_intensity_top_mean": event_intensity_top_mean,
        "event_intensity_normal_mean": event_intensity_normal_mean,
        "event_gate_masked_mean": event_gate_masked_mean,
        "event_gate_unmasked_mean": event_gate_unmasked_mean,
        "event_residual_norm_capacity_mean": event_residual_norm_capacity_mean,
        "event_residual_norm_outside_capacity_mean": event_residual_norm_outside_capacity_mean,
        "event_residual_norm_masked_mean": event_residual_norm_masked_mean,
        "event_residual_norm_unmasked_mean": event_residual_norm_unmasked_mean,
        "event_gate_at_top_error_mean": event_gate_at_top_error_mean,
        "event_gate_at_normal_steps_mean": event_gate_at_normal_steps_mean,
        "context_delta_norm_mean": context_delta_norm_mean,
        "event_prediction_error_mean": event_prediction_error_mean,
        "lifetime_step_mean": lifetime_step_mean,
        "lifetime_step_max": lifetime_step_max,
        "reset_count": int(reset_count),
        "reset_reason_max_lifetime": int(reset_reason_counts["max_lifetime"]),
        "reset_reason_stale": int(reset_reason_counts["stale"]),
        "reset_reason_env_done": int(reset_reason_counts["env_done"]),
        "reset_reason_nan": int(reset_reason_counts["nan"]),
        "reset_reason_manual": int(reset_reason_counts["manual"]),
        "operator_event_delay_active_ratio": operator_event_delay_active_ratio,
        "post_event_continuation_steps_mean": post_event_continuation_steps_mean,
        "post_event_continuation_steps_max": post_event_continuation_steps_max,
        "episode_unique_token_count_mean": episode_unique_token_count_mean,
        "episode_token_perplexity_mean": episode_token_perplexity_mean,
        "env/contact_rate": contact_rate_mean,
        "env/object_motion": object_motion_mean,
        "env/success_rate": success_rate_mean,
        "dct_v3_sequence_reward_mean": _mean_or_na(_finite_values(records, "dct_sequence_reward")[0]),
        "dct_v3_sequence_reward_nonzero_ratio": (
            float(sum(value > 0.0 for value in _finite_values(records, "dct_sequence_reward")[0]) / total_steps)
            if total_steps > 0 and _finite_values(records, "dct_sequence_reward")[0]
            else "NA"
        ),
        "dct_sequence_bigram_reward_mean": _mean_or_na(_finite_values(records, "dct_sequence_bigram_reward")[0]),
        "dct_sequence_trigram_reward_mean": _mean_or_na(_finite_values(records, "dct_sequence_trigram_reward")[0]),
        "dct_bigram_seen_count": _max_or_na(_finite_values(records, "dct_bigram_seen_size")[0]),
        "dct_trigram_seen_count": _max_or_na(_finite_values(records, "dct_trigram_seen_size")[0]),
        "dct_bigram_active_count": _max_or_na(_finite_values(records, "dct_bigram_active_size")[0]),
        "dct_trigram_active_count": _max_or_na(_finite_values(records, "dct_trigram_active_size")[0]),
        "dct_sequence_gap_mean": _mean_or_na(_finite_values(records, "dct_sequence_gap")[0]),
        "dct_sequence_span_mean": _mean_or_na(_finite_values(records, "dct_sequence_span")[0]),
        "bad_numeric_count": int(bad_numeric_count),
        "checks": checks,
    }


def _save_gate_curve(path: Path, episode_records: list[dict]):
    x = np.arange(len(episode_records))
    curves = {
        "context_gate": [np.nan if record.get("context_gate") == "NA" else record.get("context_gate") for record in episode_records],
        "context_change_mask": [np.nan if record.get("context_change_mask") == "NA" else record.get("context_change_mask") for record in episode_records],
        "event_gate": [np.nan if record.get("event_gate") == "NA" else record.get("event_gate") for record in episode_records],
        "event_logit": [np.nan if record.get("event_logit") == "NA" else record.get("event_logit") for record in episode_records],
        "event_capacity_gate": [np.nan if record.get("event_capacity_gate") == "NA" else record.get("event_capacity_gate") for record in episode_records],
        "context_delta_norm": [np.nan if record.get("context_delta_norm") == "NA" else record.get("context_delta_norm") for record in episode_records],
        "event_prediction_error": [np.nan if record.get("event_prediction_error") == "NA" else record.get("event_prediction_error") for record in episode_records],
        "ordinary_only_error": [np.nan if record.get("pred_next_feat_error_ordinary_only") == "NA" else record.get("pred_next_feat_error_ordinary_only") for record in episode_records],
        "event_mixed_error": [np.nan if record.get("pred_next_feat_error_event_mixed") == "NA" else record.get("pred_next_feat_error_event_mixed") for record in episode_records],
        "event_residual_norm": [np.nan if record.get("event_residual_norm") == "NA" else record.get("event_residual_norm") for record in episode_records],
        "operator_reward": [np.nan if record.get("operator_reward") == "NA" else record.get("operator_reward") for record in episode_records],
        "event_intensity": [np.nan if record.get("event_intensity") == "NA" else record.get("event_intensity") for record in episode_records],
        "milestone_reward": [np.nan if record.get("milestone_reward") == "NA" else record.get("milestone_reward") for record in episode_records],
    }
    plt.figure(figsize=(10, 5))
    for label, values in curves.items():
        plt.plot(x, values, label=label, linewidth=1.5)
    plt.xlabel("step")
    plt.ylabel("value")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _apply_episode_context_mask(config: dict, episode_records: list[dict]):
    event_cfg = _event_dynamics_config(config)
    if not bool(event_cfg.get("context_mask_enabled", False)) or not episode_records:
        return
    gate_values = []
    delta_values = []
    for record in episode_records:
        gate_value = _numeric_or_none(record.get("context_gate"))
        delta_value = _numeric_or_none(record.get("context_delta_norm"))
        gate_values.append(0.0 if gate_value is None or not math.isfinite(gate_value) else float(gate_value))
        delta_values.append(0.0 if delta_value is None or not math.isfinite(delta_value) else float(delta_value))
    gate_seq = torch.as_tensor(gate_values, dtype=torch.float32).view(1, len(episode_records), 1)
    delta_seq = torch.as_tensor(delta_values, dtype=torch.float32).view(1, len(episode_records), 1)
    mask = build_context_change_mask(
        context_gate_seq=gate_seq,
        context_delta_norm_seq=delta_seq,
        mode=str(event_cfg.get("context_mask_mode", "topk")),
        top_percent=float(event_cfg.get("context_mask_top_percent", 0.10)),
        window=int(event_cfg.get("context_mask_window", 1)),
        detach=bool(event_cfg.get("context_mask_detach", True)),
        source=str(event_cfg.get("context_mask_source", "context_gate")),
    )
    mask_values = mask.squeeze(0).squeeze(-1).detach().cpu().tolist()
    for record, mask_value in zip(episode_records, mask_values):
        record["context_change_mask"] = float(mask_value)
        record["context_change_mask_source"] = str(event_cfg.get("context_mask_source", "context_gate"))
        record["context_change_mask_mode"] = str(event_cfg.get("context_mask_mode", "topk"))
        record["context_change_mask_top_percent"] = float(event_cfg.get("context_mask_top_percent", 0.10))


def _apply_episode_event_capacity(config: dict, episode_records: list[dict]):
    event_cfg_raw = _event_dynamics_config(config)
    if not bool(event_cfg_raw.get("capacity_enabled", False)) or not episode_records:
        return
    event_cfg_kwargs = dict(event_cfg_raw)
    event_cfg_kwargs.setdefault("enabled", True)
    event_cfg = DreamerEventDynamicsConfig(**event_cfg_kwargs)
    selector = EventCapacitySelector(event_cfg)
    selector.eval()
    event_logits = []
    for record in episode_records:
        logit_value = _numeric_or_none(record.get("event_logit"))
        event_logits.append(0.0 if logit_value is None or not math.isfinite(logit_value) else float(logit_value))
    logit_seq = torch.as_tensor(event_logits, dtype=torch.float32).view(1, len(episode_records), 1)
    with torch.no_grad():
        capacity_gate, capacity_stats = selector(logit_seq, valid_mask=torch.ones_like(logit_seq))
    gate_values = capacity_gate.squeeze(0).squeeze(-1).cpu().tolist()
    k_value = float(capacity_stats["event_capacity_k"].squeeze().cpu().item())
    for record, gate_value in zip(episode_records, gate_values):
        record["event_capacity_gate"] = float(gate_value)
        record["event_capacity_enabled"] = 1.0
        record["event_capacity_mode"] = str(event_cfg.capacity_mode)
        record["event_capacity_ratio"] = float(event_cfg.capacity_ratio)
        record["event_capacity_k"] = k_value


def record_gate_eval_videos(
    agent,
    env,
    output_dir,
    global_step,
    config,
    device,
):
    del device
    gate_cfg = GateVideoConfig.from_root_config(config)
    output_root = Path(output_dir)
    step_dir = output_root / f"step_{int(global_step):06d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    clear_history = getattr(env, "clear_persistent_history", None)
    if callable(clear_history):
        clear_history()

    all_records: list[dict[str, Any]] = []
    for episode_index in range(int(gate_cfg.episodes)):
        _call_reset(agent)
        obs = env.reset()
        done = False
        step = 0
        episode_frames = []
        episode_records = []
        prev_object_pos = None

        while not done and step < int(gate_cfg.max_episode_steps):
            action = _call_act(agent, obs, deterministic=bool(gate_cfg.deterministic))
            next_obs, reward, raw_done, info = env.step(action)
            info = dict(info or {})
            preview_metrics = _call_preview_transition(agent, next_obs, done=bool(raw_done), reward=float(reward))
            persistent_result = _finalize_persistent_step(
                env,
                action=action,
                next_obs=next_obs,
                raw_done=bool(raw_done),
                info=info,
                diagnostics=preview_metrics or {},
            )
            done = bool(persistent_result.get("done", False))
            _call_observe_transition(agent, next_obs, done=bool(done), reward=float(reward))
            metrics = _collect_gate_metrics(agent)
            persistent_metrics = _persistent_step_metrics(env)
            contact_rate = _extract_contact_signal(info)
            success_rate = _extract_success_signal(info)
            object_pos = _extract_object_position(info)
            object_motion = None
            if object_pos is not None and prev_object_pos is not None and object_pos.shape == prev_object_pos.shape:
                object_motion = float(np.linalg.norm(object_pos - prev_object_pos))
            prev_object_pos = None if object_pos is None else object_pos.copy()
            step += 1
            record = {
                "global_step": int(global_step),
                "episode_index": int(episode_index),
                "step": int(step),
                "reward": float(reward),
                "done": bool(done),
                "lifetime_step": _jsonable_value(persistent_metrics.get("lifetime_step")),
                "operator_event_delay_active": _jsonable_value(persistent_metrics.get("operator_event_delay_active")),
                "post_event_continuation_steps": _jsonable_value(
                    persistent_metrics.get("post_event_continuation_steps")
                ),
                "unique_token_count_so_far": _jsonable_value(persistent_metrics.get("unique_token_count_so_far")),
                "reset_reason": _string_or_na(persistent_metrics.get("reset_reason")),
                "context_gate_type": _string_or_na(metrics.get("context_gate_type")),
                "context_gate": _jsonable_value(metrics.get("context_gate")),
                "context_change_mask": _jsonable_value(metrics.get("context_change_mask")),
                "context_change_mask_source": _string_or_na(metrics.get("context_change_mask_source")),
                "context_change_mask_mode": _string_or_na(metrics.get("context_change_mask_mode")),
                "context_change_mask_top_percent": _jsonable_value(metrics.get("context_change_mask_top_percent")),
                "context_l0_open_prob": _jsonable_value(metrics.get("context_l0_open_prob")),
                "context_gate_hard": _jsonable_value(metrics.get("context_gate_hard")),
                "context_gate_soft": _jsonable_value(metrics.get("context_gate_soft")),
                "context_gate_logit": _jsonable_value(metrics.get("context_gate_logit")),
                "context_update_loss_raw": _jsonable_value(metrics.get("context_update_loss_raw")),
                "context_update_loss_scaled": _jsonable_value(metrics.get("context_update_loss_scaled")),
                "context_delta_norm": _jsonable_value(metrics.get("context_delta_norm")),
                "event_gate": _jsonable_value(metrics.get("event_gate")),
                "event_logit": _jsonable_value(metrics.get("event_logit")),
                "event_logit_sigmoid": _jsonable_value(metrics.get("event_logit_sigmoid")),
                "event_capacity_gate": _jsonable_value(metrics.get("event_capacity_gate")),
                "event_capacity_enabled": _jsonable_value(metrics.get("event_capacity_enabled")),
                "event_capacity_mode": _string_or_na(metrics.get("event_capacity_mode")),
                "event_capacity_ratio": _jsonable_value(metrics.get("event_capacity_ratio")),
                "event_capacity_k": _jsonable_value(metrics.get("event_capacity_k")),
                "capacity_v2_detach_event_input": _jsonable_value(metrics.get("capacity_v2_detach_event_input")),
                "capacity_v2_detach_event_target": _jsonable_value(metrics.get("capacity_v2_detach_event_target")),
                "capacity_v2_use_sigmoid_gate_multiplier": _jsonable_value(
                    metrics.get("capacity_v2_use_sigmoid_gate_multiplier")
                ),
                "event_loss_updates_backbone": _jsonable_value(metrics.get("event_loss_updates_backbone")),
                "event_residual_norm": _jsonable_value(metrics.get("event_residual_norm")),
                "event_prediction_error": _jsonable_value(metrics.get("event_prediction_error")),
                "pred_next_feat_error_ordinary_only": _jsonable_value(metrics.get("pred_next_feat_error_ordinary_only")),
                "pred_next_feat_error_event_mixed": _jsonable_value(metrics.get("pred_next_feat_error_event_mixed")),
                "operator_reward": _jsonable_value(metrics.get("operator_reward")),
                "operator_token_id": _jsonable_value(metrics.get("operator_token_id")),
                "operator_token_novelty": _jsonable_value(metrics.get("operator_token_novelty")),
                "operator_rarity": _jsonable_value(metrics.get("operator_rarity")),
                "operator_effect_gain": _jsonable_value(metrics.get("operator_effect_gain")),
                "operator_controllability_gain": _jsonable_value(metrics.get("operator_controllability_gain")),
                "operator_valid_effect": _jsonable_value(metrics.get("operator_valid_effect")),
                "operator_valid_control": _jsonable_value(metrics.get("operator_valid_control")),
                "event_intensity": _jsonable_value(metrics.get("event_intensity")),
                "milestone_reward": _jsonable_value(metrics.get("milestone_reward")),
                "milestone_trigger": _jsonable_value(metrics.get("milestone_trigger")),
                "running_best_intensity": _jsonable_value(metrics.get("running_best_intensity")),
                "habituation_factor": _jsonable_value(metrics.get("habituation_factor")),
                "token_repeat_decay": _jsonable_value(metrics.get("token_repeat_decay")),
                "topk_rank": _jsonable_value(metrics.get("topk_rank")),
                "topk_env_bonus": _jsonable_value(metrics.get("topk_env_bonus")),
                "topk_entered": _jsonable_value(metrics.get("topk_entered")),
                "topk_leaderboard_size": _jsonable_value(metrics.get("topk_leaderboard_size")),
                "topk_update_count": _jsonable_value(metrics.get("topk_update_count")),
                "operator_token_perplexity_online": _jsonable_value(metrics.get("operator_token_perplexity_online")),
                "operator_num_active_tokens_online": _jsonable_value(metrics.get("operator_num_active_tokens_online")),
                "operator_token_count_min": _jsonable_value(metrics.get("operator_token_count_min")),
                "operator_token_count_max": _jsonable_value(metrics.get("operator_token_count_max")),
                "operator_raw_reward": _jsonable_value(metrics.get("operator_raw_reward")),
                "operator_clipped_reward": _jsonable_value(metrics.get("operator_clipped_reward")),
                "operator_reward_saturation": _jsonable_value(metrics.get("operator_reward_saturation")),
                "operator_reward_nonzero_outside_capacity": _jsonable_value(
                    metrics.get("operator_reward_nonzero_outside_capacity")
                ),
                "operator_reward_mask_violation_count": _jsonable_value(
                    metrics.get("operator_reward_mask_violation_count")
                ),
                "operator_control_only_suppressed": _jsonable_value(metrics.get("operator_control_only_suppressed")),
                "operator_valid_transition": _jsonable_value(metrics.get("operator_valid_transition")),
                "operator_stale_suppressed": _jsonable_value(metrics.get("operator_stale_suppressed")),
                "operator_latent_delta_norm": _jsonable_value(metrics.get("operator_latent_delta_norm")),
                "operator_eval_capacity_window_ready": _jsonable_value(
                    metrics.get("operator_eval_capacity_window_ready")
                ),
                "operator_eval_capacity_window_size": _jsonable_value(metrics.get("operator_eval_capacity_window_size")),
                "dct_token_id": _jsonable_value(metrics.get("dct_token_id")),
                "dct_event_valid": _jsonable_value(metrics.get("dct_candidate_valid")),
                "dct_event_step": _jsonable_value(metrics.get("lifetime_step")),
                "dct_sequence_bigram_reward": _jsonable_value(metrics.get("dct_v3_sequence_bigram_reward")),
                "dct_sequence_trigram_reward": _jsonable_value(metrics.get("dct_v3_sequence_trigram_reward")),
                "dct_sequence_reward": _jsonable_value(metrics.get("dct_v3_sequence_reward")),
                "dct_sequence_reason": _string_or_na(metrics.get("dct_sequence_reason")),
                "dct_bigram_signature_norm": _jsonable_value(metrics.get("dct_bigram_signature_norm")),
                "dct_trigram_signature_norm": _jsonable_value(metrics.get("dct_trigram_signature_norm")),
                "dct_sequence_gap": _jsonable_value(metrics.get("dct_sequence_gap")),
                "dct_sequence_span": _jsonable_value(metrics.get("dct_sequence_span")),
                "dct_bigram_seen_size": _jsonable_value(metrics.get("dct_bigram_seen_count")),
                "dct_trigram_seen_size": _jsonable_value(metrics.get("dct_trigram_seen_count")),
                "dct_bigram_active_size": _jsonable_value(metrics.get("dct_bigram_active_count")),
                "dct_trigram_active_size": _jsonable_value(metrics.get("dct_trigram_active_count")),
                "contact_rate": _jsonable_value(contact_rate),
                "object_motion": _jsonable_value(object_motion),
                "success_rate": _jsonable_value(success_rate),
            }
            episode_records.append(record)
            all_records.append(record)

            frame = _observation_to_frame(env, next_obs)
            if gate_cfg.overlay_metrics:
                lines = [
                    f"step={step}",
                    f"reward={float(reward):.4f}",
                    f"done={int(bool(done))}",
                    f"lifetime={_format_overlay_value(record['lifetime_step'])}",
                    f"reset={record['reset_reason']}",
                    f"context_gate={_format_overlay_value(record['context_gate'])}",
                    f"context_delta={_format_overlay_value(record['context_delta_norm'])}",
                    f"event_gate={_format_overlay_value(record['event_gate'])}",
                    f"capacity_gate={_format_overlay_value(record['event_capacity_gate'])}",
                    f"event_logit={_format_overlay_value(record['event_logit'])}",
                    f"event_res_norm={_format_overlay_value(record['event_residual_norm'])}",
                    f"event_err={_format_overlay_value(record['event_prediction_error'])}",
                    f"ordinary_err={_format_overlay_value(record['pred_next_feat_error_ordinary_only'])}",
                    f"mixed_err={_format_overlay_value(record['pred_next_feat_error_event_mixed'])}",
                    f"operator_reward={_format_overlay_value(record['operator_reward'])}",
                    f"operator_token_id={_format_overlay_value(record['operator_token_id'])}",
                    f"post_event={_format_overlay_value(record['post_event_continuation_steps'])}",
                    f"unique_tokens={_format_overlay_value(record['unique_token_count_so_far'])}",
                ]
                frame = _overlay_frame(frame, lines)
            episode_frames.append(frame)
            obs = next_obs

        if episode_records:
            _apply_episode_context_mask(config, episode_records)
            _apply_episode_event_capacity(config, episode_records)
            all_records[-len(episode_records) :] = episode_records

        if gate_cfg.save_gif and episode_frames:
            gif_path = step_dir / f"episode_{episode_index:03d}.gif"
            frame_duration_ms = max(1.0, 1000.0 / max(float(gate_cfg.gif_fps), 1.0))
            imageio.mimsave(gif_path, episode_frames, format="GIF", duration=frame_duration_ms)
        _save_gate_curve(step_dir / f"episode_{episode_index:03d}_gate_curves.png", episode_records)

    frames_path = step_dir / "gate_eval_frames.jsonl"
    with frames_path.open("w", encoding="utf-8") as handle:
        for record in all_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = _build_summary(
        records=all_records,
        global_step=int(global_step),
        num_episodes=int(gate_cfg.episodes),
        peak_threshold=float(gate_cfg.peak_threshold),
    )
    summary_path = step_dir / "gate_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "step_dir": step_dir,
        "summary_path": summary_path,
        "frames_path": frames_path,
        "summary": summary,
    }


def main():
    parse_args()
    print("Gate video recorder is intended to be called from the training loop.")


if __name__ == "__main__":
    main()
