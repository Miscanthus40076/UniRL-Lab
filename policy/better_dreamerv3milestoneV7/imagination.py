from __future__ import annotations

import torch

from .actor import DreamerActor
from .heads import TwoHotSymlogHead
from .world_model import DreamerV3WorldModel

EXTERNAL_SLOW_GAIN_REWARD_SCALE = 1.0
EXTERNAL_EFFECT_MARGIN = 0.2


def _stack_states(states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([state[key] for state in states], dim=0) for key in states[0]}


def imagine_rollout(
    world_model: DreamerV3WorldModel,
    actor: DreamerActor,
    start_state: dict[str, torch.Tensor],
    horizon: int,
    start_context: torch.Tensor | None = None,
    external_slow_gain_reward_scale: float = EXTERNAL_SLOW_GAIN_REWARD_SCALE,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    state = {key: value for key, value in start_state.items()}
    context = start_context
    feats = []
    actions = []
    rewards = []
    env_rewards = []
    raw_slow_gain_intrinsics = []
    external_slow_gain_intrinsics = []
    external_effect_scores = []
    continues = []
    log_probs = []
    entropies = []
    states = []

    for _ in range(int(horizon)):
        base_feat = world_model.get_base_feat(state)
        feat = world_model.concat_context(base_feat, context)
        action, log_prob, entropy = actor.sample(feat)
        if isinstance(world_model.reward_head, TwoHotSymlogHead):
            env_reward = world_model.reward_head.mean(feat)
        else:
            env_reward = world_model.reward_head(feat)
        raw_slow_gain_outputs = world_model.predict_slow_gain_reward(feat)
        raw_slow_gain_intrinsic = raw_slow_gain_outputs["slow_gain_reward_pred"]
        next_state = world_model.rssm.imagine_step(state, action)
        next_base_feat = world_model.get_base_feat(next_state)
        if world_model.slow_context is not None:
            next_details = world_model.get_augmented_feat(next_state, prev_context=context, return_details=True)
            next_feat = next_details["augmented_feat"]
        else:
            next_feat = world_model.concat_context(next_base_feat, context)
        self_motion_outputs = world_model.predict_self_motion_delta(feat, action, detach_input=True)
        self_motion_pred = self_motion_outputs["self_motion_pred"].detach()
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
            raw_slow_gain_intrinsic.detach() * external_effect_score,
            min=0.0,
            max=1.0,
        )
        reward = env_reward + float(external_slow_gain_reward_scale) * external_slow_gain_intrinsic
        continue_prob = torch.sigmoid(world_model.continue_head(feat))
        state = next_state
        if world_model.slow_context is not None:
            context = next_details["context"]

        feats.append(feat)
        actions.append(action)
        rewards.append(reward)
        env_rewards.append(env_reward)
        raw_slow_gain_intrinsics.append(raw_slow_gain_intrinsic.detach())
        external_slow_gain_intrinsics.append(external_slow_gain_intrinsic)
        external_effect_scores.append(external_effect_score)
        continues.append(continue_prob)
        log_probs.append(log_prob)
        entropies.append(entropy)
        states.append(state)

    return {
        "feats": torch.stack(feats, dim=0),
        "actions": torch.stack(actions, dim=0),
        "rewards": torch.stack(rewards, dim=0),
        "env_rewards": torch.stack(env_rewards, dim=0),
        "slow_gain_intrinsics": torch.stack(external_slow_gain_intrinsics, dim=0),
        "raw_slow_gain_intrinsics": torch.stack(raw_slow_gain_intrinsics, dim=0),
        "external_slow_gain_intrinsics": torch.stack(external_slow_gain_intrinsics, dim=0),
        "external_effect_scores": torch.stack(external_effect_scores, dim=0),
        "continues": torch.stack(continues, dim=0),
        "log_probs": torch.stack(log_probs, dim=0),
        "entropies": torch.stack(entropies, dim=0),
        "states": _stack_states(states),
        "last_context": context,
    }
