from __future__ import annotations

import math

import torch

from .actor import DreamerActor
from .heads import TwoHotSymlogHead
from .world_model import DreamerV3WorldModel

EXTERNAL_EFFECT_MARGIN = 0.2
V8_CONFIRM_REWARD_SCALE = 1.0


def _stack_states(states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([state[key] for state in states], dim=0) for key in states[0]}


def _time_encoding(age: torch.Tensor) -> torch.Tensor:
    return torch.stack([age, torch.ones_like(age)], dim=-1)


def imagine_rollout(
    world_model: DreamerV3WorldModel,
    actor: DreamerActor,
    start_state: dict[str, torch.Tensor],
    horizon: int,
    memory_modules: dict[str, torch.nn.Module],
    memory_bank: dict[str, torch.Tensor],
    start_context: torch.Tensor | None = None,
    v8_confirm_reward_scale: float = V8_CONFIRM_REWARD_SCALE,
    pose_bias_weight: float = 1.0,
    pose_sigma: float = 0.05,
    confirm_pose_sigma: float = 0.03,
    pose_conf_threshold: float = 0.8,
    visual_change_threshold: float = 1.0,
    initial_visual_change_ema: float = 1.0,
    credit_window: int = 50,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    state = {key: value for key, value in start_state.items()}
    context = start_context
    feats = []
    policy_feats = []
    actions = []
    rewards = []
    env_rewards = []
    raw_slow_gain_intrinsics = []
    external_slow_gain_intrinsics = []
    v8_confirm_intrinsics = []
    external_effect_scores = []
    memory_contexts = []
    continues = []
    log_probs = []
    entropies = []
    states = []
    attn_entropies = []
    attn_top1_weights = []
    attn_effective_counts = []
    selected_pose_dists = []
    pose_confidences = []
    visual_change_norms = []
    visual_confirm_pass_rates = []
    pose_confirm_pass_rates = []
    confirmed_rates = []

    external_recent: list[torch.Tensor] = []
    visual_change_ema = float(initial_visual_change_ema)

    mem_proprio = memory_bank.get("proprio")
    mem_visual = memory_bank.get("stable_visual")
    mem_summary = memory_bank.get("recent_summary")
    mem_time = memory_bank.get("time_encoding")

    for _ in range(int(horizon)):
        base_feat = world_model.get_base_feat(state)
        feat = world_model.concat_context(base_feat, context)
        recent_stats = []
        if external_recent:
            stacked = torch.stack(external_recent[-max(1, int(credit_window)) :], dim=0)
            recent_stats = [
                stacked.mean(dim=0),
                stacked.max(dim=0).values,
            ]
        else:
            zeros = torch.zeros(feat.shape[0], dtype=feat.dtype, device=feat.device)
            recent_stats = [zeros, zeros]
        raw_stats = recent_stats
        summary = torch.stack(
            [
                recent_stats[0],
                recent_stats[1],
                raw_stats[0],
                raw_stats[1],
                recent_stats[0],
                recent_stats[1],
            ],
            dim=-1,
        )
        proprio = world_model.proprio_head(base_feat).detach()
        decoded_obs = world_model.decoder(feat).detach()
        stable_visual = world_model.encode_stable_visual(decoded_obs).detach()

        if mem_proprio is not None and mem_proprio.numel() > 0:
            query_in = torch.cat([feat, summary], dim=-1)
            query = memory_modules["query_net"](query_in)
            key_in = torch.cat([mem_proprio, mem_summary, mem_time], dim=-1)
            keys = memory_modules["key_net"](key_in)
            value_in = torch.cat([mem_proprio, mem_visual, mem_summary, mem_time], dim=-1)
            values = memory_modules["value_net"](value_in)
            learned = torch.matmul(query, keys.transpose(0, 1)) / math.sqrt(max(1, keys.shape[-1]))
            pose_dist = torch.cdist(proprio.float(), mem_proprio.float(), p=2)
            pose_bias = float(pose_bias_weight) * torch.exp(-pose_dist / max(float(pose_sigma), 1e-6))
            scores = learned + pose_bias
            attention = torch.softmax(scores, dim=-1)
            memory_context = torch.matmul(attention, values)
            selected_proprio = torch.matmul(attention, mem_proprio)
            selected_visual = torch.matmul(attention, mem_visual)
            selected_pose_dist = torch.linalg.vector_norm(proprio - selected_proprio, dim=-1)
            pose_confidence = torch.exp(-selected_pose_dist / max(float(confirm_pose_sigma), 1e-6))
            visual_change = torch.linalg.vector_norm(stable_visual - selected_visual, dim=-1)
            visual_change_norm = visual_change / (visual_change_ema + 1e-6)
            pose_pass = pose_confidence > float(pose_conf_threshold)
            visual_pass = visual_change_norm > float(visual_change_threshold)
            confirmed = (pose_pass & visual_pass).float().detach()
            attn_entropy = -(attention * torch.log(attention.clamp_min(1e-8))).sum(dim=-1)
            attn_top1 = attention.max(dim=-1).values
            attn_eff = torch.exp(attn_entropy)
        else:
            memory_context = torch.zeros(
                feat.shape[0],
                memory_modules["value_net"][-1].out_features,
                dtype=feat.dtype,
                device=feat.device,
            )
            selected_pose_dist = torch.zeros(feat.shape[0], dtype=feat.dtype, device=feat.device)
            pose_confidence = torch.zeros(feat.shape[0], dtype=feat.dtype, device=feat.device)
            visual_change_norm = torch.zeros(feat.shape[0], dtype=feat.dtype, device=feat.device)
            pose_pass = torch.zeros(feat.shape[0], dtype=torch.bool, device=feat.device)
            visual_pass = torch.zeros(feat.shape[0], dtype=torch.bool, device=feat.device)
            confirmed = torch.zeros(feat.shape[0], dtype=feat.dtype, device=feat.device)
            attn_entropy = torch.zeros(feat.shape[0], dtype=feat.dtype, device=feat.device)
            attn_top1 = torch.zeros(feat.shape[0], dtype=feat.dtype, device=feat.device)
            attn_eff = torch.zeros(feat.shape[0], dtype=feat.dtype, device=feat.device)

        policy_feat = memory_modules["context_adapter"](torch.cat([feat, memory_context], dim=-1))
        action, log_prob, entropy = actor.sample(policy_feat)
        if isinstance(world_model.reward_head, TwoHotSymlogHead):
            env_reward = world_model.reward_head.mean(feat)
        else:
            env_reward = world_model.reward_head(feat)
        raw_slow_gain_intrinsic = world_model.predict_slow_gain_reward(feat)["slow_gain_reward_pred"].detach()
        next_state = world_model.rssm.imagine_step(state, action)
        next_base_feat = world_model.get_base_feat(next_state)
        if world_model.slow_context is not None:
            next_details = world_model.get_augmented_feat(next_state, prev_context=context, return_details=True)
            next_feat = next_details["augmented_feat"]
        else:
            next_feat = world_model.concat_context(next_base_feat, context)
        self_motion_pred = world_model.predict_self_motion_delta(feat, action, detach_input=True)["self_motion_pred"].detach()
        actual_delta = (next_feat.detach() - feat.detach()).float()
        actual_delta_norm = torch.linalg.vector_norm(actual_delta, dim=-1)
        external_residual = actual_delta - self_motion_pred.float()
        external_residual_norm = torch.linalg.vector_norm(external_residual, dim=-1)
        external_effect_raw = (external_residual_norm / (actual_delta_norm.mean().detach() + 1e-6)).detach()
        external_effect_score = torch.clamp(
            (external_effect_raw - EXTERNAL_EFFECT_MARGIN) / (1.0 - EXTERNAL_EFFECT_MARGIN + 1e-6),
            min=0.0,
            max=1.0,
        ).detach()
        external_slow_gain_intrinsic = torch.clamp(
            raw_slow_gain_intrinsic * external_effect_score,
            min=0.0,
            max=1.0,
        ).detach()
        external_recent.append(external_slow_gain_intrinsic)
        sequence_score = torch.stack(external_recent[-max(1, int(credit_window)) :], dim=0).max(dim=0).values.detach()
        v8_confirm_intrinsic = (confirmed * sequence_score).detach()
        reward = env_reward + float(v8_confirm_reward_scale) * v8_confirm_intrinsic
        continue_prob = torch.sigmoid(world_model.continue_head(feat))
        state = next_state
        if world_model.slow_context is not None:
            context = next_details["context"]
        visual_change_ema = 0.99 * visual_change_ema + 0.01 * float(visual_change_norm.mean().detach().cpu())

        feats.append(feat)
        policy_feats.append(policy_feat)
        actions.append(action)
        rewards.append(reward)
        env_rewards.append(env_reward)
        raw_slow_gain_intrinsics.append(raw_slow_gain_intrinsic)
        external_slow_gain_intrinsics.append(external_slow_gain_intrinsic)
        v8_confirm_intrinsics.append(v8_confirm_intrinsic)
        external_effect_scores.append(external_effect_score)
        memory_contexts.append(memory_context)
        continues.append(continue_prob)
        log_probs.append(log_prob)
        entropies.append(entropy)
        states.append(state)
        attn_entropies.append(attn_entropy)
        attn_top1_weights.append(attn_top1)
        attn_effective_counts.append(attn_eff)
        selected_pose_dists.append(selected_pose_dist)
        pose_confidences.append(pose_confidence)
        visual_change_norms.append(visual_change_norm)
        visual_confirm_pass_rates.append(visual_pass.float())
        pose_confirm_pass_rates.append(pose_pass.float())
        confirmed_rates.append(confirmed)

    return {
        "feats": torch.stack(feats, dim=0),
        "policy_feats": torch.stack(policy_feats, dim=0),
        "actions": torch.stack(actions, dim=0),
        "rewards": torch.stack(rewards, dim=0),
        "env_rewards": torch.stack(env_rewards, dim=0),
        "raw_slow_gain_intrinsics": torch.stack(raw_slow_gain_intrinsics, dim=0),
        "external_slow_gain_intrinsics": torch.stack(external_slow_gain_intrinsics, dim=0),
        "v8_confirm_intrinsics": torch.stack(v8_confirm_intrinsics, dim=0),
        "external_effect_scores": torch.stack(external_effect_scores, dim=0),
        "memory_contexts": torch.stack(memory_contexts, dim=0),
        "continues": torch.stack(continues, dim=0),
        "log_probs": torch.stack(log_probs, dim=0),
        "entropies": torch.stack(entropies, dim=0),
        "states": _stack_states(states),
        "last_context": context,
        "attention_entropy": torch.stack(attn_entropies, dim=0),
        "attention_top1_weight": torch.stack(attn_top1_weights, dim=0),
        "attention_effective_memory_count": torch.stack(attn_effective_counts, dim=0),
        "selected_pose_dist": torch.stack(selected_pose_dists, dim=0),
        "pose_confidence": torch.stack(pose_confidences, dim=0),
        "visual_change_norm": torch.stack(visual_change_norms, dim=0),
        "visual_confirm_pass_rate": torch.stack(visual_confirm_pass_rates, dim=0),
        "pose_confirm_pass_rate": torch.stack(pose_confirm_pass_rates, dim=0),
        "confirmed_rate": torch.stack(confirmed_rates, dim=0),
    }
