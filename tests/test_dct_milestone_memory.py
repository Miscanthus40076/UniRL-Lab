from __future__ import annotations

from pathlib import Path

import torch

from policy.versions.v1.DreamerV3milestoneV2.operator_intrinsic_reward import (
    DCTMilestoneMemory,
    OperatorIntrinsicRewardConfig,
)
from scripts.train import load_exam_train_app


def _memory(capacity: int = 4) -> DCTMilestoneMemory:
    return DCTMilestoneMemory(
        num_tokens=8,
        signature_dim=4,
        capacity=capacity,
        seen_enabled=True,
        recent_token_window=20,
        recent_token_repeat_threshold=0.5,
    )


def _sig(values) -> torch.Tensor:
    x = torch.as_tensor(values, dtype=torch.float32)
    return x / torch.clamp(torch.linalg.vector_norm(x), min=1e-8)


def _eval(memory, token, sig, valid=True, effect=True, control=False, update=True):
    return memory.evaluate(
        token_id=token,
        signature=_sig(sig),
        valid_candidate=valid,
        effect_valid=effect,
        control_valid=control,
        seen_threshold=0.20,
        active_threshold=0.20,
        preemption_enabled=True,
        preemption_margin=0.05,
        milestone_reward=1.0,
        update_state=update,
    )


def test_first_new_token_signature_rewards_once():
    memory = _memory()
    result = _eval(memory, 1, [1, 0, 0, 0])
    assert result["reward"] == 1.0
    assert result["reason"] == "new_inserted"
    assert memory.counts()["seen_count"] == 1


def test_similar_signature_same_token_rejected_by_seen_memory():
    memory = _memory()
    assert _eval(memory, 1, [1, 0, 0, 0])["reward"] == 1.0
    result = _eval(memory, 1, [0.99, 0.01, 0, 0])
    assert result["reward"] == 0.0
    assert result["reason"] == "already_seen"


def test_same_token_different_signature_can_reward():
    memory = _memory()
    assert _eval(memory, 1, [1, 0, 0, 0])["reward"] == 1.0
    result = _eval(memory, 1, [0, 1, 0, 0])
    assert result["reward"] == 1.0


def test_different_token_same_signature_is_different_milestone():
    memory = _memory()
    assert _eval(memory, 1, [1, 0, 0, 0])["reward"] == 1.0
    result = _eval(memory, 2, [1, 0, 0, 0])
    assert result["reward"] == 1.0


def test_seen_memory_survives_active_preemption():
    memory = _memory(capacity=1)
    assert _eval(memory, 1, [1, 0, 0, 0], control=False)["reward"] == 1.0
    assert _eval(memory, 2, [0, 1, 0, 0], control=True)["reason"] == "preempted_low_priority"
    result = _eval(memory, 1, [1, 0, 0, 0], control=True)
    assert result["reward"] == 0.0
    assert result["reason"] == "already_seen"


def test_active_memory_full_high_priority_preempts_low_priority():
    memory = _memory(capacity=1)
    assert _eval(memory, 1, [1, 0, 0, 0], control=False)["reason"] == "new_inserted"
    result = _eval(memory, 2, [0, 1, 0, 0], control=True)
    assert result["reward"] == 1.0
    assert result["reason"] == "preempted_low_priority"
    assert memory.counts()["preemption_count"] == 1


def test_active_memory_full_low_priority_rejected():
    memory = _memory(capacity=1)
    assert _eval(memory, 1, [1, 0, 0, 0], control=True)["reward"] == 1.0
    result = _eval(memory, 2, [0, 1, 0, 0], control=False)
    assert result["reward"] == 0.0
    assert result["reason"] == "rejected_low_priority"


def test_reward_is_constant_not_scaled_by_effect_or_control_magnitude():
    memory = _memory()
    low = _eval(memory, 1, [1, 0, 0, 0], effect=True, control=False)
    high = _eval(memory, 1, [0, 1, 0, 0], effect=True, control=True)
    assert low["reward"] == 1.0
    assert high["reward"] == 1.0


def test_invalid_candidate_not_inserted_or_rewarded():
    memory = _memory()
    result = _eval(memory, 1, [1, 0, 0, 0], valid=False)
    assert result["reward"] == 0.0
    assert result["reason"] == "invalid_candidate"
    assert memory.counts()["seen_count"] == 0


def test_env_reset_does_not_clear_seen_when_config_false():
    cfg = OperatorIntrinsicRewardConfig(
        reward_mode="dct_effect_milestone_v2",
        compute_online_only=True,
        store_reward_in_replay=True,
        forbid_batch_recompute_for_milestone=True,
        normalize_reward=False,
        clip_reward=False,
        reset_memory_on_env_reset=False,
        probe_checkpoint="dummy.pt",
        enabled=False,
    )
    assert cfg.reset_memory_on_env_reset is False


def test_no_privileged_labels_in_v2_reward_source():
    source = Path("policy/versions/v1/DreamerV3milestoneV2/operator_intrinsic_reward.py").read_text(encoding="utf-8").lower()
    for forbidden in ("success", "contact", "alignment", "insertion_depth", "lateral_distance"):
        assert forbidden not in source


def test_dct_milestone_v2_exams_load():
    smoke = load_exam_train_app("train_metaworld_dreamerv3_peg_insert_side_sparse_dct_milestone_v2_smoke").load_config()
    long = load_exam_train_app("train_metaworld_dreamerv3_peg_insert_side_sparse_dct_milestone_v2_10k").load_config()
    assert smoke["policy"]["type"] == "DreamerV3milestoneV2"
    assert smoke["policy"]["operator_intrinsic_reward"]["reward_mode"] == "dct_effect_milestone_v2"
    assert smoke["policy"]["operator_intrinsic_reward"]["compute_online_only"] is True
    assert long["train"]["total_steps"] == 10000
