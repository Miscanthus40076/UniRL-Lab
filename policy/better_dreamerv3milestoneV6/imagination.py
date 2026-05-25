from __future__ import annotations

import torch

from .actor import DreamerActor
from .heads import TwoHotSymlogHead
from .world_model import DreamerV3WorldModel

SLOW_GAIN_REWARD_SCALE = 1.0


def _stack_states(states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([state[key] for state in states], dim=0) for key in states[0]}


def imagine_rollout(
    world_model: DreamerV3WorldModel,
    actor: DreamerActor,
    start_state: dict[str, torch.Tensor],
    horizon: int,
    start_context: torch.Tensor | None = None,
    slow_gain_reward_scale: float = SLOW_GAIN_REWARD_SCALE,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    state = {key: value for key, value in start_state.items()}
    context = start_context
    feats = []
    actions = []
    rewards = []
    env_rewards = []
    slow_gain_intrinsics = []
    raw_slow_gain_intrinsics = []
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
        continue_prob = torch.sigmoid(world_model.continue_head(feat))
        state = world_model.rssm.imagine_step(state, action)
        if world_model.slow_context is not None:
            next_details = world_model.get_augmented_feat(state, prev_context=context, return_details=True)
            context = next_details["context"]
            next_feat = next_details["augmented_feat"]
        else:
            next_feat = world_model.get_base_feat(state)
        if world_model.event_dynamics is not None:
            transition_outputs = world_model.predict_event_transition(
                feat,
                action=action,
                target_next_feat=next_feat.detach(),
                apply_capacity_constraint=False,
            )
            reward_components = world_model.compute_interest_reward_from_transition(transition_outputs)
            if reward_components is None:
                slow_gain_intrinsic = torch.zeros_like(env_reward)
                raw_slow_gain_intrinsic = torch.zeros_like(env_reward)
            else:
                slow_gain_intrinsic = reward_components["interest_slow_gain_reward"]
                raw_slow_gain_intrinsic = reward_components["raw_slow_gain_reward"]
        else:
            slow_gain_intrinsic = torch.zeros_like(env_reward)
            raw_slow_gain_intrinsic = torch.zeros_like(env_reward)
        reward = env_reward + float(slow_gain_reward_scale) * slow_gain_intrinsic

        feats.append(feat)
        actions.append(action)
        rewards.append(reward)
        env_rewards.append(env_reward)
        slow_gain_intrinsics.append(slow_gain_intrinsic)
        raw_slow_gain_intrinsics.append(raw_slow_gain_intrinsic)
        continues.append(continue_prob)
        log_probs.append(log_prob)
        entropies.append(entropy)
        states.append(state)

    return {
        "feats": torch.stack(feats, dim=0),
        "actions": torch.stack(actions, dim=0),
        "rewards": torch.stack(rewards, dim=0),
        "env_rewards": torch.stack(env_rewards, dim=0),
        "slow_gain_intrinsics": torch.stack(slow_gain_intrinsics, dim=0),
        "raw_slow_gain_intrinsics": torch.stack(raw_slow_gain_intrinsics, dim=0),
        "continues": torch.stack(continues, dim=0),
        "log_probs": torch.stack(log_probs, dim=0),
        "entropies": torch.stack(entropies, dim=0),
        "states": _stack_states(states),
        "last_context": context,
    }
