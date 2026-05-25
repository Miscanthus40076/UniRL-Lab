from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .transforms import symlog, twohot_logprob

FAST_PRED_LOSS_WEIGHT = 0.1
SLOW_TO_FAST_DISTILL_WEIGHT = 0.1
SLOW_GAIN_REWARD_LOSS_WEIGHT = 1.0
SELF_MOTION_LOSS_WEIGHT = 1.0
EXTERNAL_EFFECT_MARGIN = 0.2


@dataclass(slots=True)
class WorldModelLossConfig:
    free_nats: float = 1.0
    recon_scale: float = 1.0
    reward_scale: float = 1.0
    continue_scale: float = 1.0
    dynamics_scale: float = 1.0
    representation_scale: float = 0.1
    grasp_scale: float = 0.2
    contact_scale: float = 0.2
    kl_balance: float = 0.8
    use_symlog_obs: bool = True
    use_symlog_reward: bool = True
    use_twohot_reward: bool = True
    twohot_bins: int = 255
    twohot_low: float = -20.0
    twohot_high: float = 20.0
    context_update_penalty: float = 0.0
    event_penalty: float = 0.0
    event_prediction_scale: float = 1.0
    event_ordinary_prediction_scale: float = 1.0
    event_sparsity_inside_mask_only: bool = False
    event_sparsity_inside_capacity_only: bool = False
    proprio_loss_weight: float = 1.0


def categorical_kl(post: dict[str, torch.Tensor], prior: dict[str, torch.Tensor]) -> torch.Tensor:
    post_log_probs = F.log_softmax(post["logits"], dim=-1)
    prior_log_probs = F.log_softmax(prior["logits"], dim=-1)
    post_probs = post_log_probs.exp()
    kl = (post_probs * (post_log_probs - prior_log_probs)).sum(dim=-1)
    return kl.sum(dim=-1)


def _reduce_except_batch(x: torch.Tensor) -> torch.Tensor:
    if x.ndim <= 1:
        return x
    dims = tuple(range(1, x.ndim))
    return x.mean(dim=dims)


def _reduce_except_batch_time(x: torch.Tensor) -> torch.Tensor:
    if x.ndim <= 2:
        return x.float()
    dims = tuple(range(2, x.ndim))
    return x.float().mean(dim=dims)


def _mask_to_match(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    while mask.ndim > value.ndim and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    return mask


def _masked_mean_per_seq(value: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mask = _mask_to_match(value, mask)
    weighted = (value.float() * mask).sum(dim=tuple(range(1, value.ndim)))
    denom = mask.sum(dim=tuple(range(1, mask.ndim))).clamp_min(eps)
    return weighted / denom


def _masked_scalar_mean(value: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return _masked_mean_per_seq(value, mask, eps=eps).mean()


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    value = value.float()
    weight = _mask_to_match(value, weight).float()
    return (value * weight).sum() / weight.sum().clamp_min(eps)


def _gate_sequence_2d(gate: torch.Tensor | None) -> torch.Tensor | None:
    if gate is None:
        return None
    gate = gate.detach().float()
    if gate.ndim == 0:
        return None
    if gate.ndim == 1:
        return gate.unsqueeze(0)
    if gate.ndim == 2:
        return gate
    if gate.ndim >= 3:
        if gate.shape[-1] == 1:
            gate = gate.squeeze(-1)
        else:
            gate = gate.mean(dim=-1)
    if gate.ndim != 2:
        return None
    return gate


def _slow_gate_oscillation_metrics(gate: torch.Tensor | None) -> dict[str, torch.Tensor]:
    seq = _gate_sequence_2d(gate)
    if seq is None or seq.shape[1] <= 1:
        zero = torch.zeros((), dtype=torch.float32, device=gate.device if gate is not None else None)
        return {
            "wm/slow_switch_count": zero,
            "wm/slow_switch_rate": zero,
            "wm/slow_residence_mean": zero,
            "wm/slow_residence_median": zero,
            "wm/slow_on_segments": zero,
        }

    hard = seq > 0.5
    switches = hard[:, 1:] != hard[:, :-1]
    switch_count = switches.sum(dim=1).float().mean()
    switch_rate = switches.float().mean()

    segment_lengths: list[int] = []
    on_segment_counts: list[int] = []
    for row in hard.cpu().tolist():
        count = 0
        run = 0
        for value in row:
            if bool(value):
                run += 1
            elif run > 0:
                segment_lengths.append(run)
                count += 1
                run = 0
        if run > 0:
            segment_lengths.append(run)
            count += 1
        on_segment_counts.append(count)

    device = seq.device
    if segment_lengths:
        lengths_t = torch.as_tensor(segment_lengths, dtype=torch.float32, device=device)
        residence_mean = lengths_t.mean()
        residence_median = lengths_t.median()
    else:
        residence_mean = torch.zeros((), dtype=torch.float32, device=device)
        residence_median = torch.zeros((), dtype=torch.float32, device=device)
    on_segments = torch.as_tensor(on_segment_counts, dtype=torch.float32, device=device).mean()

    return {
        "wm/slow_switch_count": switch_count,
        "wm/slow_switch_rate": switch_rate,
        "wm/slow_residence_mean": residence_mean,
        "wm/slow_residence_median": residence_median,
        "wm/slow_on_segments": on_segments,
    }


def compute_per_step_prediction_errors(
    outputs: dict,
    batch: dict,
    config: WorldModelLossConfig,
) -> dict[str, torch.Tensor]:
    obs = batch["obs"].float()
    reward = batch["reward"].float()
    done = batch["done"].float()
    continue_target = 1.0 - done
    obs_target = obs / 255.0 if obs.ndim >= 4 and obs.max() > 2.0 else obs
    if obs.ndim < 4 and config.use_symlog_obs:
        obs_target = symlog(obs_target)

    if config.use_symlog_reward:
        reward_target = symlog(reward)
    else:
        reward_target = reward

    recon_error = (outputs["obs_pred"] - obs_target).pow(2)
    recon_per_step = _reduce_except_batch_time(recon_error)

    if config.use_twohot_reward:
        reward_logprob = twohot_logprob(
            outputs["reward_logits"],
            reward_target,
            num_bins=config.twohot_bins,
            low=config.twohot_low,
            high=config.twohot_high,
        )
        reward_per_step = -_reduce_except_batch_time(reward_logprob)
    else:
        reward_error = (outputs["reward_pred"] - reward_target).pow(2)
        reward_per_step = _reduce_except_batch_time(reward_error)

    continue_loss_raw = F.binary_cross_entropy_with_logits(
        outputs["continue_logit"],
        continue_target,
        reduction="none",
    )
    continue_per_step = _reduce_except_batch_time(continue_loss_raw)

    teacher_per_step = (
        config.recon_scale * recon_per_step
        + config.reward_scale * reward_per_step
        + config.continue_scale * continue_per_step
    )

    result = {"with_slow": teacher_per_step}

    if outputs.get("fast_only_obs_pred") is None:
        return result

    fast_recon_error = (outputs["fast_only_obs_pred"] - obs_target).pow(2)
    fast_recon_per_step = _reduce_except_batch_time(fast_recon_error)
    if config.use_twohot_reward and outputs.get("fast_only_reward_logits") is not None:
        fast_reward_logprob = twohot_logprob(
            outputs["fast_only_reward_logits"],
            reward_target,
            num_bins=config.twohot_bins,
            low=config.twohot_low,
            high=config.twohot_high,
        )
        fast_reward_per_step = -_reduce_except_batch_time(fast_reward_logprob)
    elif outputs.get("fast_only_reward_pred") is not None:
        fast_reward_error = (outputs["fast_only_reward_pred"] - reward_target).pow(2)
        fast_reward_per_step = _reduce_except_batch_time(fast_reward_error)
    else:
        fast_reward_per_step = None

    if outputs.get("fast_only_continue_logit") is not None:
        fast_continue_loss_raw = F.binary_cross_entropy_with_logits(
            outputs["fast_only_continue_logit"],
            continue_target,
            reduction="none",
        )
        fast_continue_per_step = _reduce_except_batch_time(fast_continue_loss_raw)
    else:
        fast_continue_per_step = None

    if fast_reward_per_step is None or fast_continue_per_step is None:
        return result

    fast_per_step = (
        config.recon_scale * fast_recon_per_step
        + config.reward_scale * fast_reward_per_step
        + config.continue_scale * fast_continue_per_step
    )
    result["fast"] = fast_per_step
    result["slow_gain"] = fast_per_step - teacher_per_step
    return result


def summarize_slow_gain_per_step(
    per_step_slow_gain: torch.Tensor | None,
    per_step_fast_error: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if per_step_slow_gain is None:
        return {}
    slow_gain = per_step_slow_gain.detach().float()
    flat = slow_gain.reshape(-1)
    positive = slow_gain > 0
    top_k = max(1, int(np.ceil(float(flat.numel()) * 0.2)))
    top_values = torch.topk(flat, k=top_k).values
    top20_threshold = top_values.min()
    top20_mask = slow_gain >= top20_threshold
    metrics = {
        "wm/local_slow_gain_mean": slow_gain.mean(),
        "wm/local_slow_gain_std": slow_gain.std(unbiased=False),
        "wm/local_slow_gain_positive_rate": positive.float().mean(),
        "wm/local_slow_gain_top20_mean": top_values.mean(),
        "wm/high_slow_gain_top20_rate": top20_mask.float().mean(),
        "wm/positive_slow_gain_rate": positive.float().mean(),
    }
    if per_step_fast_error is not None:
        gain_score = torch.clamp(
            slow_gain / (per_step_fast_error.detach().float().abs() + 1e-6),
            min=0.0,
            max=1.0,
        )
        metrics["wm/gain_score_mean"] = gain_score.mean()
        metrics["wm/gain_score_std"] = gain_score.std(unbiased=False)
        metrics["wm/gain_score_max"] = gain_score.max()
    return metrics


def _safe_corrcoef_tensor(left: torch.Tensor | None, right: torch.Tensor | None) -> torch.Tensor:
    if left is None or right is None:
        return torch.zeros(())
    x = left.detach().float().reshape(-1)
    y = right.detach().float().reshape(-1)
    if x.numel() == 0 or y.numel() == 0 or x.numel() != y.numel():
        return torch.zeros((), device=x.device if x.numel() > 0 else None)
    x_std = x.std(unbiased=False)
    y_std = y.std(unbiased=False)
    if float(x_std.detach().cpu()) <= 1e-8 or float(y_std.detach().cpu()) <= 1e-8:
        return torch.zeros((), device=x.device)
    cov = ((x - x.mean()) * (y - y.mean())).mean()
    return cov / (x_std * y_std)


def compute_slow_gain_reward_target(
    outputs: dict,
    batch: dict,
    config: WorldModelLossConfig,
) -> dict[str, torch.Tensor] | None:
    if outputs.get("fast_only_obs_pred") is None or outputs.get("obs_pred") is None:
        return None
    obs = batch["obs"].float()
    obs_target = obs / 255.0 if obs.ndim >= 4 and obs.max() > 2.0 else obs
    if obs.ndim < 4 and config.use_symlog_obs:
        obs_target = symlog(obs_target)
    fast_obs_error = _reduce_except_batch_time((outputs["fast_only_obs_pred"] - obs_target).pow(2))
    teacher_obs_error = _reduce_except_batch_time((outputs["obs_pred"] - obs_target).pow(2))
    local_slow_gain = fast_obs_error - teacher_obs_error
    slow_gain_reward = torch.clamp(
        torch.relu(local_slow_gain) / (fast_obs_error.detach().abs() + 1e-6),
        min=0.0,
        max=1.0,
    ).detach()
    return {
        "fast_obs_error": fast_obs_error.detach(),
        "teacher_obs_error": teacher_obs_error.detach(),
        "local_slow_gain": local_slow_gain.detach(),
        "slow_gain_reward": slow_gain_reward,
    }


def compute_external_bridge_targets(outputs: dict) -> dict[str, torch.Tensor] | None:
    actual_delta = outputs.get("actual_delta")
    self_motion_pred = outputs.get("self_motion_pred")
    if actual_delta is None or self_motion_pred is None:
        return None
    actual_delta_detached = actual_delta.detach().float()
    self_motion_pred_detached = self_motion_pred.detach().float()
    external_residual = actual_delta_detached - self_motion_pred_detached
    actual_delta_norm = torch.linalg.vector_norm(actual_delta_detached, dim=-1)
    self_delta_pred_norm = torch.linalg.vector_norm(self_motion_pred_detached, dim=-1)
    external_residual_norm = torch.linalg.vector_norm(external_residual, dim=-1)
    denom = actual_delta_norm.mean().detach() + 1e-6
    external_effect_raw = (external_residual_norm / denom).detach()
    external_effect_score = torch.clamp(
        (external_effect_raw - EXTERNAL_EFFECT_MARGIN) / (1.0 - EXTERNAL_EFFECT_MARGIN + 1e-6),
        min=0.0,
        max=1.0,
    ).detach()
    return {
        "actual_delta": actual_delta_detached,
        "self_motion_pred": self_motion_pred_detached,
        "external_residual": external_residual.detach(),
        "actual_delta_norm": actual_delta_norm.detach(),
        "self_delta_pred_norm": self_delta_pred_norm.detach(),
        "external_residual_norm": external_residual_norm.detach(),
        "external_effect_raw": external_effect_raw,
        "external_effect_score": external_effect_score,
    }


def world_model_loss(outputs: dict, batch: dict, config: WorldModelLossConfig) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
    if config.use_symlog_reward:
        reward_target = symlog(reward)
    else:
        reward_target = reward
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
    dynamics_kl_per_seq = torch.clamp(categorical_kl(post_sg, outputs["prior"]), min=float(config.free_nats)).mean(dim=1)
    representation_kl_per_seq = torch.clamp(categorical_kl(outputs["post"], prior_sg), min=float(config.free_nats)).mean(dim=1)
    dynamics_kl = dynamics_kl_per_seq.mean()
    representation_kl = representation_kl_per_seq.mean()
    dynamics_loss_per_seq = config.dynamics_scale * dynamics_kl_per_seq
    representation_loss_per_seq = config.representation_scale * representation_kl_per_seq
    dynamics_loss = dynamics_loss_per_seq.mean()
    representation_loss = representation_loss_per_seq.mean()
    kl_loss = dynamics_loss + representation_loss

    prediction_loss_per_seq = (
        config.recon_scale * recon_loss_per_seq
        + config.reward_scale * reward_loss_per_seq
        + config.continue_scale * continue_loss_per_seq
    )
    prediction_loss = prediction_loss_per_seq.mean()
    fast_prediction_loss = None
    fast_prediction_loss_per_seq = None
    distill_loss = None
    distill_loss_per_seq = None
    if outputs.get("fast_only_obs_pred") is not None:
        fast_recon_error = (outputs["fast_only_obs_pred"] - obs_target).pow(2)
        fast_recon_loss_per_seq = _reduce_except_batch(fast_recon_error)
        if config.use_twohot_reward and outputs.get("fast_only_reward_logits") is not None:
            fast_reward_logprob = twohot_logprob(
                outputs["fast_only_reward_logits"],
                reward_target,
                num_bins=config.twohot_bins,
                low=config.twohot_low,
                high=config.twohot_high,
            )
            fast_reward_loss_per_seq = -_reduce_except_batch(fast_reward_logprob)
        elif outputs.get("fast_only_reward_pred") is not None:
            fast_reward_error = (outputs["fast_only_reward_pred"] - reward_target).pow(2)
            fast_reward_loss_per_seq = _reduce_except_batch(fast_reward_error)
        else:
            fast_reward_loss_per_seq = None
        if outputs.get("fast_only_continue_logit") is not None:
            fast_continue_loss_raw = F.binary_cross_entropy_with_logits(
                outputs["fast_only_continue_logit"],
                continue_target,
                reduction="none",
            )
            fast_continue_loss_per_seq = _reduce_except_batch(fast_continue_loss_raw)
        else:
            fast_continue_loss_per_seq = None
        if fast_reward_loss_per_seq is not None and fast_continue_loss_per_seq is not None:
            fast_prediction_loss_per_seq = (
                config.recon_scale * fast_recon_loss_per_seq
                + config.reward_scale * fast_reward_loss_per_seq
                + config.continue_scale * fast_continue_loss_per_seq
            )
            fast_prediction_loss = fast_prediction_loss_per_seq.mean()
            teacher_obs_target = outputs["obs_pred"].detach()
            distill_recon_error = (outputs["fast_only_obs_pred"] - teacher_obs_target).pow(2)
            distill_recon_loss_per_seq = _reduce_except_batch(distill_recon_error)
            teacher_reward_target = outputs["reward_pred"].detach()
            distill_reward_error = (outputs["fast_only_reward_pred"] - teacher_reward_target).pow(2)
            distill_reward_loss_per_seq = _reduce_except_batch(distill_reward_error)
            teacher_continue_target = torch.sigmoid(outputs["continue_logit"].detach())
            distill_continue_loss_raw = F.binary_cross_entropy_with_logits(
                outputs["fast_only_continue_logit"],
                teacher_continue_target,
                reduction="none",
            )
            distill_continue_loss_per_seq = _reduce_except_batch(distill_continue_loss_raw)
            distill_loss_per_seq = (
                config.recon_scale * distill_recon_loss_per_seq
                + config.reward_scale * distill_reward_loss_per_seq
                + config.continue_scale * distill_continue_loss_per_seq
            )
            distill_loss = distill_loss_per_seq.mean()
    per_step_errors = compute_per_step_prediction_errors(outputs, batch, config)
    per_step_fast_error = per_step_errors.get("fast")
    per_step_with_slow_error = per_step_errors.get("with_slow")
    per_step_slow_gain = per_step_errors.get("slow_gain")
    outputs["per_step_fast_error"] = per_step_fast_error.detach() if per_step_fast_error is not None else None
    outputs["per_step_with_slow_error"] = (
        per_step_with_slow_error.detach() if per_step_with_slow_error is not None else None
    )
    outputs["per_step_slow_gain"] = per_step_slow_gain.detach() if per_step_slow_gain is not None else None
    proprio = batch.get("proprio")
    proprio_pred = outputs.get("proprio_pred")
    proprio_loss = None
    proprio_loss_per_seq = None
    proprio_hand_pos_mse = None
    proprio_hand_z_mse = None
    proprio_gripper_mse = None
    proprio_input_dim = 0
    proprio_embed_dim = 0
    if isinstance(proprio, torch.Tensor) and isinstance(proprio_pred, torch.Tensor):
        proprio_target = proprio.detach().float()
        proprio_pred = proprio_pred.float()
        proprio_error = (proprio_pred - proprio_target).pow(2)
        proprio_loss_per_seq = _reduce_except_batch(proprio_error)
        proprio_loss = proprio_loss_per_seq.mean()
        proprio_input_dim = int(proprio_target.shape[-1])
        proprio_embed = outputs.get("proprio_embed")
        if isinstance(proprio_embed, torch.Tensor):
            proprio_embed_dim = int(proprio_embed.shape[-1])
        proprio_hand_pos_mse = proprio_error[..., : min(3, proprio_error.shape[-1])].mean()
        if proprio_error.shape[-1] >= 3:
            proprio_hand_z_mse = proprio_error[..., 2].mean()
        if proprio_error.shape[-1] >= 4:
            proprio_gripper_mse = proprio_error[..., 3].mean()
    slow_gain_reward_components = compute_slow_gain_reward_target(outputs, batch, config)
    slow_gain_reward_loss = None
    slow_gain_reward_loss_per_seq = None
    if slow_gain_reward_components is not None and outputs.get("slow_gain_reward_pred") is not None:
        slow_gain_reward_target = slow_gain_reward_components["slow_gain_reward"]
        slow_gain_reward_pred = outputs["slow_gain_reward_pred"]
        slow_gain_reward_loss_raw = (slow_gain_reward_pred - slow_gain_reward_target).pow(2)
        slow_gain_reward_loss_per_seq = _reduce_except_batch(slow_gain_reward_loss_raw)
        slow_gain_reward_loss = slow_gain_reward_loss_per_seq.mean()
    self_motion_loss = None
    self_motion_loss_per_seq = None
    external_bridge = compute_external_bridge_targets(outputs)
    if outputs.get("self_motion_pred") is not None and outputs.get("actual_delta") is not None:
        self_motion_loss_raw = (outputs["self_motion_pred"] - outputs["actual_delta"]).pow(2)
        self_motion_loss_per_seq = _reduce_except_batch(self_motion_loss_raw)
        self_motion_loss = self_motion_loss_per_seq.mean()

    total_per_seq = prediction_loss_per_seq + dynamics_loss_per_seq + representation_loss_per_seq
    if fast_prediction_loss_per_seq is not None:
        total_per_seq = total_per_seq + FAST_PRED_LOSS_WEIGHT * fast_prediction_loss_per_seq
    if distill_loss_per_seq is not None:
        total_per_seq = total_per_seq + SLOW_TO_FAST_DISTILL_WEIGHT * distill_loss_per_seq
    if proprio_loss_per_seq is not None:
        total_per_seq = total_per_seq + config.proprio_loss_weight * proprio_loss_per_seq
    if slow_gain_reward_loss_per_seq is not None:
        total_per_seq = total_per_seq + SLOW_GAIN_REWARD_LOSS_WEIGHT * slow_gain_reward_loss_per_seq
    if self_motion_loss_per_seq is not None:
        total_per_seq = total_per_seq + SELF_MOTION_LOSS_WEIGHT * self_motion_loss_per_seq
    model_loss_without_context_penalty = total_per_seq.mean()
    context_gate = outputs.get("context_gate")
    context_gate_soft = outputs.get("context_gate_soft")
    context_gate_hard = outputs.get("context_gate_hard")
    context_gate_logit = outputs.get("context_gate_logit")
    context_l0_open_prob = outputs.get("context_l0_open_prob")
    context_update_loss = None
    context_update_loss_scaled = None
    if context_gate is not None:
        context_update_source = context_l0_open_prob if context_l0_open_prob is not None else context_gate
        context_update_per_seq = _reduce_except_batch(context_update_source.float())
        context_update_loss = context_update_per_seq.mean()
        context_update_loss_scaled = config.context_update_penalty * context_update_per_seq.mean()
        total_per_seq = total_per_seq + config.context_update_penalty * context_update_per_seq
    event_gate = outputs.get("event_gate")
    event_logit = outputs.get("event_logit")
    ordinary_error = outputs.get("pred_next_feat_error_ordinary_only")
    mixed_error = outputs.get("pred_next_feat_error_event_mixed")
    event_prediction_error = outputs.get("event_prediction_error")
    context_change_mask = outputs.get("context_change_mask")
    event_capacity_gate = outputs.get("event_capacity_gate")
    event_capacity_k = outputs.get("event_capacity_k")
    event_residual_norm = outputs.get("event_residual_norm")
    event_logit_sigmoid = outputs.get("event_logit_sigmoid")
    event_prediction_loss = None
    ordinary_prediction_loss = None
    event_prediction_loss_scaled = None
    event_gate_penalty_loss = None
    event_gate_penalty_scaled = None
    if event_gate is not None and ordinary_error is not None and mixed_error is not None:
        ordinary_prediction_per_seq = _reduce_except_batch(ordinary_error.float())
        capacity_constraint_applied = bool(outputs.get("event_capacity_constraint_applied", True))
        if event_capacity_gate is not None and capacity_constraint_applied:
            event_prediction_per_seq = _masked_mean_per_seq(mixed_error.float(), event_capacity_gate)
        elif context_change_mask is not None:
            event_prediction_per_seq = _masked_mean_per_seq(mixed_error.float(), context_change_mask)
        else:
            event_prediction_per_seq = _reduce_except_batch(mixed_error.float())
        ordinary_prediction_loss = ordinary_prediction_per_seq.mean()
        event_prediction_loss = event_prediction_per_seq.mean()
        ordinary_prediction_loss_scaled = config.event_ordinary_prediction_scale * ordinary_prediction_loss
        event_prediction_loss_scaled = config.event_prediction_scale * event_prediction_loss
        event_gate_penalty_loss = torch.zeros((), dtype=event_gate.dtype, device=event_gate.device)
        event_gate_penalty_scaled = torch.zeros((), dtype=event_gate.dtype, device=event_gate.device)
        total_per_seq = (
            total_per_seq
            + config.event_ordinary_prediction_scale * ordinary_prediction_per_seq
            + config.event_prediction_scale * event_prediction_per_seq
        )
    total = total_per_seq.mean()
    metrics = {
        "model_loss": total.detach(),
        "model_loss_without_context_penalty": model_loss_without_context_penalty.detach(),
        "model_loss_with_context_penalty": total.detach(),
        "prediction_loss": prediction_loss.detach(),
        "recon_loss": recon_loss.detach(),
        "reward_loss": reward_loss.detach(),
        "continue_loss": continue_loss.detach(),
        "wm/teacher_pred_loss": prediction_loss.detach(),
        "wm/teacher_error": prediction_loss.detach(),
        "kl_loss": kl_loss.detach(),
        "dynamics_loss": dynamics_loss.detach(),
        "representation_loss": representation_loss.detach(),
        "dynamics_kl_loss": dynamics_kl.detach(),
        "representation_kl_loss": representation_kl.detach(),
        "priority": total_per_seq.detach(),
    }
    zero = prediction_loss.detach().new_zeros(())
    if fast_prediction_loss is not None:
        metrics["wm/fast_error"] = fast_prediction_loss.detach()
        metrics["wm/with_slow_error"] = prediction_loss.detach()
        metrics["wm/slow_gain"] = (fast_prediction_loss - prediction_loss).detach()
        metrics["wm/fast_pred_loss"] = fast_prediction_loss.detach()
        metrics["wm/fast_pred_loss_weighted"] = (FAST_PRED_LOSS_WEIGHT * fast_prediction_loss).detach()
    metrics.update(summarize_slow_gain_per_step(per_step_slow_gain, per_step_fast_error))
    if slow_gain_reward_components is not None and outputs.get("slow_gain_reward_pred") is not None:
        raw_slow_gain_reward_target = slow_gain_reward_components["slow_gain_reward"]
        slow_gain_reward_pred = outputs["slow_gain_reward_pred"].detach()
        external_effect_score = None if external_bridge is None else external_bridge["external_effect_score"]
        if external_effect_score is not None:
            external_slow_gain_reward_target = (
                raw_slow_gain_reward_target[:, :-1] * external_effect_score
            ).detach()
        else:
            external_slow_gain_reward_target = raw_slow_gain_reward_target[:, :-1].detach()
        metrics["wm/local_slow_gain_mean"] = slow_gain_reward_components["local_slow_gain"].mean()
        metrics["wm/local_slow_gain_std"] = slow_gain_reward_components["local_slow_gain"].std(unbiased=False)
        metrics["wm/local_slow_gain_positive_rate"] = (slow_gain_reward_components["local_slow_gain"] > 0).float().mean()
        flat_local = slow_gain_reward_components["local_slow_gain"].reshape(-1)
        top_k = max(1, int(np.ceil(float(flat_local.numel()) * 0.2)))
        metrics["wm/local_slow_gain_top20_mean"] = torch.topk(flat_local, k=top_k).values.mean()
        metrics["wm/local_slow_gain_max"] = slow_gain_reward_components["local_slow_gain"].max()
        metrics["wm/raw_slow_gain_reward_mean"] = raw_slow_gain_reward_target.mean()
        metrics["wm/raw_slow_gain_reward_nonzero_rate"] = (raw_slow_gain_reward_target > 0).float().mean()
        metrics["wm/raw_slow_gain_reward_max"] = raw_slow_gain_reward_target.max()
        metrics["wm/slow_gain_reward_mean"] = external_slow_gain_reward_target.mean()
        metrics["wm/slow_gain_reward_std"] = external_slow_gain_reward_target.std(unbiased=False)
        metrics["wm/slow_gain_reward_nonzero_rate"] = (external_slow_gain_reward_target > 0).float().mean()
        metrics["wm/slow_gain_reward_max"] = external_slow_gain_reward_target.max()
        metrics["wm/external_slow_gain_reward_mean"] = external_slow_gain_reward_target.mean()
        metrics["wm/external_slow_gain_reward_nonzero_rate"] = (external_slow_gain_reward_target > 0).float().mean()
        metrics["wm/external_slow_gain_reward_max"] = external_slow_gain_reward_target.max()
        metrics["wm/event_replay_target_mean"] = external_slow_gain_reward_target.mean()
        metrics["wm/event_replay_target_std"] = external_slow_gain_reward_target.std(unbiased=False)
        metrics["wm/event_replay_target_nonzero_rate"] = (external_slow_gain_reward_target > 0).float().mean()
        metrics["wm/event_replay_target_max"] = external_slow_gain_reward_target.max()
        metrics["wm/event_replay_pred_mean"] = slow_gain_reward_pred.mean()
        metrics["wm/event_replay_pred_std"] = slow_gain_reward_pred.std(unbiased=False)
        metrics["wm/event_replay_pred_nonzero_rate"] = (slow_gain_reward_pred > 1e-6).float().mean()
        metrics["wm/event_replay_target_pred_corr"] = _safe_corrcoef_tensor(external_slow_gain_reward_target, slow_gain_reward_pred[:, :-1])
        metrics["wm/event_replay_head_used"] = torch.zeros((), dtype=slow_gain_reward_pred.dtype, device=slow_gain_reward_pred.device)
        metrics["wm/high_gain_label_used"] = torch.zeros((), dtype=slow_gain_reward_pred.dtype, device=slow_gain_reward_pred.device)
        metrics["wm/raw_slow_gain_reward_used_for_actor"] = torch.zeros((), dtype=slow_gain_reward_pred.dtype, device=slow_gain_reward_pred.device)
        metrics["wm/external_slow_gain_reward_used_directly_for_actor"] = torch.zeros((), dtype=slow_gain_reward_pred.dtype, device=slow_gain_reward_pred.device)
        metrics["wm/v8_confirm_reward_used_for_actor"] = torch.ones((), dtype=slow_gain_reward_pred.dtype, device=slow_gain_reward_pred.device)
        flat_gain = slow_gain_reward_components["local_slow_gain"].reshape(-1)
        top_k = max(1, int(np.ceil(float(flat_gain.numel()) * 0.2)))
        top20_threshold = torch.topk(flat_gain, k=top_k).values.min()
        top20_mask = (slow_gain_reward_components["local_slow_gain"] >= top20_threshold).float()
        not_top20_mask = 1.0 - top20_mask
        raw_reward_aligned = raw_slow_gain_reward_target[:, :-1]
        external_reward_aligned = external_slow_gain_reward_target
        external_pred_aligned = slow_gain_reward_pred[:, :-1]
        metrics["intrinsic/raw_slow_gain_reward_mean"] = raw_reward_aligned.mean()
        metrics["intrinsic/external_slow_gain_reward_mean"] = external_reward_aligned.mean()
        metrics["intrinsic/total_intrinsic_reward_mean"] = external_reward_aligned.mean()
        metrics["intrinsic/slow_gain_reward_mean"] = external_reward_aligned.mean()
        metrics["intrinsic/slow_gain_reward_nonzero_rate"] = (external_reward_aligned > 1e-6).float().mean()
        metrics["intrinsic/event_replay_reward_mean"] = external_reward_aligned.mean()
        metrics["intrinsic/event_replay_reward_nonzero_rate"] = (external_reward_aligned > 1e-6).float().mean()
        top20_mask_aligned = top20_mask[:, :-1]
        not_top20_mask_aligned = not_top20_mask[:, :-1]
        metrics["intrinsic/slow_gain_reward_when_slow_gain_top20"] = _masked_scalar_mean(
            external_reward_aligned, top20_mask_aligned
        )
        metrics["intrinsic/slow_gain_reward_when_slow_gain_not_top20"] = _masked_scalar_mean(
            external_reward_aligned, not_top20_mask_aligned
        )
        metrics["intrinsic/event_replay_reward_when_slow_gain_top20"] = _masked_scalar_mean(
            external_reward_aligned, top20_mask_aligned
        )
        metrics["intrinsic/event_replay_reward_when_slow_gain_not_top20"] = _masked_scalar_mean(
            external_reward_aligned, not_top20_mask_aligned
        )
        metrics["intrinsic/raw_slow_gain_reward_when_no_contact"] = zero
        metrics["intrinsic/external_slow_gain_reward_when_no_contact"] = zero
        metrics["intrinsic/raw_slow_gain_reward_when_object_static"] = zero
        metrics["intrinsic/external_slow_gain_reward_when_object_static"] = zero
        metrics["intrinsic/raw_slow_gain_reward_when_hand_high_object_static"] = zero
        metrics["intrinsic/external_slow_gain_reward_when_hand_high_object_static"] = zero
        metrics["intrinsic/raw_slow_gain_reward_when_object_moving"] = zero
        metrics["intrinsic/external_slow_gain_reward_when_object_moving"] = zero
        metrics["intrinsic/raw_slow_gain_reward_when_object_coupled"] = zero
        metrics["intrinsic/external_slow_gain_reward_when_object_coupled"] = zero
        metrics["intrinsic/v8_confirm_reward_when_no_contact"] = zero
        metrics["intrinsic/v8_confirm_reward_when_object_static"] = zero
        metrics["intrinsic/v8_confirm_reward_when_hand_high_object_static"] = zero
        metrics["intrinsic/v8_confirm_reward_when_object_moving"] = zero
        metrics["intrinsic/v8_confirm_reward_when_object_coupled"] = zero
        metrics["intrinsic/no_contact_v8_to_raw_ratio"] = zero
        metrics["intrinsic/no_contact_v8_to_external_ratio"] = zero
        metrics["intrinsic/object_static_v8_to_raw_ratio"] = zero
        metrics["intrinsic/object_static_v8_to_external_ratio"] = zero
        metrics["intrinsic/hand_high_object_static_v8_to_raw_ratio"] = zero
        metrics["intrinsic/hand_high_object_static_v8_to_external_ratio"] = zero
        metrics["intrinsic/object_moving_v8_to_raw_ratio"] = zero
        metrics["intrinsic/object_moving_v8_to_external_ratio"] = zero
        metrics["intrinsic/object_coupled_v8_to_raw_ratio"] = zero
        metrics["intrinsic/object_coupled_v8_to_external_ratio"] = zero
        metrics["wm/slow_gain_event_replay_reward_corr"] = _safe_corrcoef_tensor(
            slow_gain_reward_components["local_slow_gain"][:, :-1], external_reward_aligned
        )
        metrics["wm/slow_gain_reward_corr"] = _safe_corrcoef_tensor(
            slow_gain_reward_components["local_slow_gain"][:, :-1], external_reward_aligned
        )
        if slow_gain_reward_loss is not None:
            metrics["wm/slow_gain_reward_loss"] = slow_gain_reward_loss.detach()
    if external_bridge is not None:
        metrics["wm/actual_delta_norm_mean"] = external_bridge["actual_delta_norm"].mean()
        metrics["wm/self_delta_pred_norm_mean"] = external_bridge["self_delta_pred_norm"].mean()
        metrics["wm/external_residual_norm_mean"] = external_bridge["external_residual_norm"].mean()
        metrics["wm/external_residual_norm_std"] = external_bridge["external_residual_norm"].std(unbiased=False)
        metrics["wm/external_residual_norm_max"] = external_bridge["external_residual_norm"].max()
        metrics["wm/external_effect_margin"] = external_bridge["external_effect_score"].new_tensor(EXTERNAL_EFFECT_MARGIN)
        metrics["wm/external_effect_raw_mean"] = external_bridge["external_effect_raw"].mean()
        metrics["wm/external_effect_raw_std"] = external_bridge["external_effect_raw"].std(unbiased=False)
        metrics["wm/external_effect_raw_max"] = external_bridge["external_effect_raw"].max()
        metrics["wm/external_effect_score_pre_margin_mean"] = external_bridge["external_effect_raw"].mean()
        metrics["wm/external_effect_score_mean"] = external_bridge["external_effect_score"].mean()
        metrics["wm/external_effect_score_post_margin_mean"] = external_bridge["external_effect_score"].mean()
        metrics["wm/external_effect_score_std"] = external_bridge["external_effect_score"].std(unbiased=False)
        metrics["wm/external_effect_score_nonzero_rate"] = (external_bridge["external_effect_score"] > 1e-6).float().mean()
        metrics["wm/external_effect_score_max"] = external_bridge["external_effect_score"].max()
        metrics["wm/external_effect_margin_suppressed_rate"] = (
            external_bridge["external_effect_raw"] <= EXTERNAL_EFFECT_MARGIN
        ).float().mean()
        metrics["wm/external_effect_gate_active_rate"] = (external_bridge["external_effect_score"] > 1e-6).float().mean()
        metrics["wm/external_effect_gate_strong_rate"] = (external_bridge["external_effect_score"] > 0.5).float().mean()
        metrics["wm/self_motion_explained_ratio"] = (
            external_bridge["self_delta_pred_norm"].mean()
            / (external_bridge["actual_delta_norm"].mean() + 1e-8)
        )
    zero = prediction_loss.detach().new_zeros(())
    if self_motion_loss is not None:
        metrics["wm/self_motion_loss"] = self_motion_loss.detach()
        metrics["wm/self_motion_loss_weighted"] = (SELF_MOTION_LOSS_WEIGHT * self_motion_loss).detach()
    if proprio_loss is not None:
        metrics["wm/proprio_loss"] = proprio_loss.detach()
        metrics["wm/proprio_loss_weighted"] = (config.proprio_loss_weight * proprio_loss).detach()
        metrics["wm/proprio_input_dim"] = zero.new_tensor(float(proprio_input_dim))
        metrics["wm/proprio_embed_dim"] = zero.new_tensor(float(proprio_embed_dim))
        if proprio_hand_pos_mse is not None:
            metrics["wm/proprio_hand_pos_mse"] = proprio_hand_pos_mse.detach()
        if proprio_hand_z_mse is not None:
            metrics["wm/proprio_hand_z_mse"] = proprio_hand_z_mse.detach()
        if proprio_gripper_mse is not None:
            metrics["wm/proprio_gripper_mse"] = proprio_gripper_mse.detach()
    if distill_loss is not None:
        metrics["wm/distill_loss"] = distill_loss.detach()
        metrics["wm/distill_loss_weighted"] = (SLOW_TO_FAST_DISTILL_WEIGHT * distill_loss).detach()
    metrics["wm/slow_rate_limit_loss_raw"] = zero
    metrics["wm/slow_rate_limit_loss_weighted"] = zero
    if context_update_loss is not None:
        gate = context_gate.detach().float()
        metrics["context_update_loss"] = context_update_loss.detach()
        metrics["context_update_loss_raw"] = context_update_loss.detach()
        metrics["context_update_loss_scaled"] = context_update_loss_scaled.detach()
        metrics["wm/slow_penalty_raw"] = context_update_loss.detach()
        metrics["wm/slow_penalty"] = context_update_loss_scaled.detach()
        metrics["wm/slow_update_penalty"] = context_update_loss_scaled.detach()
        metrics["context_gate_mean"] = gate.mean()
        metrics["context_gate_std"] = gate.std(unbiased=False)
        metrics["context_gate_min"] = gate.min()
        metrics["context_gate_max"] = gate.max()
        metrics["wm/slow_gate_rate"] = gate.mean()
        metrics["wm/slow_gate_std"] = gate.std(unbiased=False)
        if context_l0_open_prob is not None:
            metrics["context_l0_open_prob"] = context_l0_open_prob.detach().float().mean()
        if context_gate_hard is not None:
            metrics["context_gate_hard_mean"] = context_gate_hard.detach().float().mean()
        if context_gate_soft is not None:
            metrics["context_gate_soft_mean"] = context_gate_soft.detach().float().mean()
        if context_gate_logit is not None:
            logit = context_gate_logit.detach().float()
            metrics["context_gate_logit_mean"] = logit.mean()
            metrics["context_gate_logit_std"] = logit.std(unbiased=False)
        context_delta_norm = outputs.get("context_delta_norm")
        if context_delta_norm is not None:
            context_delta_detached = context_delta_norm.detach().float()
            metrics["context_delta_norm_mean"] = context_delta_detached.mean()
            metrics["wm/slow_update_rate"] = (context_delta_detached > 1e-6).float().mean()
        context = outputs.get("context")
        if context is not None:
            metrics["context_norm_mean"] = torch.linalg.vector_norm(context.detach().float(), dim=-1).mean()
        metrics.update(_slow_gate_oscillation_metrics(context_gate_hard if context_gate_hard is not None else gate))
    if event_prediction_loss is not None:
        gate = event_gate.detach().float()
        ordinary_detached = ordinary_error.detach().float()
        mixed_detached = mixed_error.detach().float()
        capacity_v2_detach_event_input = bool(outputs.get("capacity_v2_detach_event_input", False))
        capacity_v2_detach_event_target = bool(outputs.get("capacity_v2_detach_event_target", False))
        capacity_v2_use_sigmoid_gate_multiplier = bool(outputs.get("capacity_v2_use_sigmoid_gate_multiplier", False))
        event_loss_updates_backbone = bool(outputs.get("event_loss_updates_backbone", True))
        metrics["event_prediction_loss"] = event_prediction_loss.detach()
        metrics["event_prediction_loss_scaled"] = event_prediction_loss_scaled.detach()
        metrics["ordinary_prediction_loss"] = ordinary_prediction_loss.detach()
        metrics["event_gate_penalty_loss"] = event_gate_penalty_loss.detach()
        metrics["event_gate_penalty_scaled"] = event_gate_penalty_scaled.detach()
        metrics["capacity_v2_detach_event_input"] = torch.as_tensor(
            float(capacity_v2_detach_event_input),
            dtype=torch.float32,
            device=gate.device,
        )
        metrics["capacity_v2_detach_event_target"] = torch.as_tensor(
            float(capacity_v2_detach_event_target),
            dtype=torch.float32,
            device=gate.device,
        )
        metrics["capacity_v2_use_sigmoid_gate_multiplier"] = torch.as_tensor(
            float(capacity_v2_use_sigmoid_gate_multiplier),
            dtype=torch.float32,
            device=gate.device,
        )
        metrics["event_loss_updates_backbone"] = torch.as_tensor(
            float(event_loss_updates_backbone),
            dtype=torch.float32,
            device=gate.device,
        )
        metrics["event_gate_mean"] = gate.mean()
        metrics["event/event_gate_mean"] = gate.mean()
        metrics["event/old_event_gate_mean"] = gate.mean()
        metrics["event/old_event_gate_saturation_rate"] = (gate > 0.95).float().mean()
        metrics["event/old_event_gate_used_for_milestone"] = torch.zeros((), dtype=gate.dtype, device=gate.device)
        metrics["event_gate_std"] = gate.std(unbiased=False)
        metrics["event_gate_min"] = gate.min()
        metrics["event_gate_max"] = gate.max()
        if event_logit_sigmoid is not None:
            event_logit_sigmoid_detached = event_logit_sigmoid.detach().float()
            metrics["event_logit_sigmoid_mean"] = event_logit_sigmoid_detached.mean()
        metrics["ordinary_only_error_mean"] = ordinary_detached.mean()
        metrics["event_mixed_error_mean"] = mixed_detached.mean()
        metrics["ordinary_error_all_mean"] = ordinary_detached.mean()
        metrics["mixed_error_all_mean"] = mixed_detached.mean()
        improvement = (ordinary_detached.mean() - mixed_detached.mean()) / torch.clamp(ordinary_detached.mean(), min=1e-8)
        metrics["relative_prediction_improvement"] = improvement
        metrics["relative_prediction_improvement_all"] = improvement
        diag_error = event_prediction_error if event_prediction_error is not None else ordinary_error
        metrics["event_prediction_error_mean"] = diag_error.detach().float().mean()
        if event_capacity_gate is not None:
            capacity_gate = event_capacity_gate.detach().float()
            capacity_binary = (capacity_gate > 0).float()
            metrics["event_capacity_enabled"] = torch.as_tensor(1.0, device=gate.device)
            capacity_ratio_value = outputs.get("event_capacity_ratio", 0.0)
            if isinstance(capacity_ratio_value, torch.Tensor):
                metrics["event_capacity_ratio"] = capacity_ratio_value.detach().float().mean().to(device=gate.device)
            else:
                metrics["event_capacity_ratio"] = torch.as_tensor(
                    float(capacity_ratio_value),
                    dtype=torch.float32,
                    device=gate.device,
                )
            if event_capacity_k is not None:
                metrics["event_capacity_k_mean"] = event_capacity_k.detach().float().mean()
            metrics["event_capacity_gate_mean"] = capacity_gate.mean()
            metrics["event_capacity_gate_std"] = capacity_gate.std(unbiased=False)
            metrics["event_capacity_gate_min"] = capacity_gate.min()
            metrics["event_capacity_gate_max"] = capacity_gate.max()
            metrics["event_capacity_nonzero_ratio"] = capacity_binary.mean()
            if event_logit is not None:
                event_logit_detached = event_logit.detach().float()
                metrics["event_logit_mean"] = event_logit_detached.mean()
                metrics["event_logit_std"] = event_logit_detached.std(unbiased=False)
                metrics["event_logit_top_mean"] = _weighted_mean(event_logit_detached, capacity_gate)
                metrics["event_logit_normal_mean"] = _weighted_mean(event_logit_detached, 1.0 - capacity_gate)
            if event_logit_sigmoid is not None:
                event_logit_sigmoid_detached = event_logit_sigmoid.detach().float()
                metrics["event_logit_sigmoid_capacity_mean"] = _weighted_mean(event_logit_sigmoid_detached, capacity_gate)
                metrics["event_logit_sigmoid_outside_capacity_mean"] = _weighted_mean(
                    event_logit_sigmoid_detached,
                    1.0 - capacity_gate,
                )
            metrics["event_gate_capacity_mean"] = _weighted_mean(gate, capacity_gate)
            metrics["event_gate_outside_capacity_mean"] = _weighted_mean(gate, 1.0 - capacity_gate)
            ordinary_capacity_mean = _weighted_mean(ordinary_detached, capacity_gate)
            mixed_capacity_mean = _weighted_mean(mixed_detached, capacity_gate)
            metrics["ordinary_error_capacity_mean"] = ordinary_capacity_mean
            metrics["mixed_error_capacity_mean"] = mixed_capacity_mean
            metrics["relative_prediction_improvement_capacity"] = (
                ordinary_capacity_mean - mixed_capacity_mean
            ) / torch.clamp(ordinary_capacity_mean, min=1e-8)
            if event_residual_norm is not None:
                residual_norm = event_residual_norm.detach().float()
                metrics["event_residual_norm_capacity_mean"] = _weighted_mean(residual_norm, capacity_gate)
                metrics["event_residual_norm_outside_capacity_mean"] = _weighted_mean(
                    residual_norm,
                    1.0 - capacity_gate,
                )
        elif context_change_mask is not None:
            mask = context_change_mask.detach().float()
            mask_binary = (mask > 0).float()
            metrics["context_change_mask_mean"] = mask.mean()
            metrics["context_change_mask_nonzero_ratio"] = mask_binary.mean()
            metrics["context_change_mask_top_percent"] = torch.as_tensor(
                float(outputs.get("context_change_mask_top_percent", 0.0)),
                dtype=torch.float32,
                device=gate.device,
            )
            metrics["event_gate_masked_mean"] = _weighted_mean(gate, mask)
            metrics["event_gate_unmasked_mean"] = _weighted_mean(gate, 1.0 - mask)
            ordinary_masked_mean = _weighted_mean(ordinary_detached, mask)
            mixed_masked_mean = _weighted_mean(mixed_detached, mask)
            metrics["ordinary_error_masked_mean"] = ordinary_masked_mean
            metrics["mixed_error_masked_mean"] = mixed_masked_mean
            metrics["relative_prediction_improvement_masked"] = (
                ordinary_masked_mean - mixed_masked_mean
            ) / torch.clamp(ordinary_masked_mean, min=1e-8)
            if event_residual_norm is not None:
                residual_norm = event_residual_norm.detach().float()
                metrics["event_residual_norm_masked_mean"] = _weighted_mean(residual_norm, mask)
                metrics["event_residual_norm_unmasked_mean"] = _weighted_mean(residual_norm, 1.0 - mask)

    if outputs.get("grasp_logit") is not None and "is_grasping" in batch:
        grasp_loss = F.binary_cross_entropy_with_logits(outputs["grasp_logit"], batch["is_grasping"].float())
        total = total + config.grasp_scale * grasp_loss
        metrics["model_loss"] = total.detach()
        metrics["grasp_loss"] = grasp_loss.detach()
    if outputs.get("contact_logits") is not None and "contact_mode" in batch:
        contact_loss = F.cross_entropy(
            outputs["contact_logits"].reshape(-1, outputs["contact_logits"].shape[-1]),
            batch["contact_mode"].long().reshape(-1),
        )
        total = total + config.contact_scale * contact_loss
        metrics["model_loss"] = total.detach()
        metrics["contact_loss"] = contact_loss.detach()

    return total, metrics
