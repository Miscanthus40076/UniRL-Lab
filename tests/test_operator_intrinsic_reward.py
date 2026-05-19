from __future__ import annotations

from pathlib import Path

import torch

from policy.dreamerv3.operator_intrinsic_reward import (
    OperatorIntrinsicReward,
    OperatorIntrinsicRewardConfig,
    combine_train_reward,
)
from policy.operator_token_probe.config import OperatorTokenProbePolicyConfig
from policy.operator_token_probe.operator_token_probe_agent import OperatorTokenProbeAgent
from scripts.train import load_exam_train_app


def _mock_probe_checkpoint(tmp_path: Path, feat_dim: int = 6, action_dim: int = 3) -> Path:
    cfg = OperatorTokenProbePolicyConfig(
        device="cpu",
        allow_cpu_fallback=True,
        source_checkpoint="dummy.ckpt",
        collect_steps=8,
        train_steps=1,
        batch_size=2,
        seq_len=4,
        learning_rate=3e-4,
    )
    agent = OperatorTokenProbeAgent(feat_dim=feat_dim, action_dim=action_dim, config=cfg)
    path = tmp_path / "operator_token_probe.pt"
    torch.save(
        {
            "policy_type": "operator_token_probe",
            "action_dim": action_dim,
            "feat_dim": feat_dim,
            "agent": agent.state_dict(),
        },
        path,
    )
    return path


def test_operator_intrinsic_reward_loads_probe_checkpoint(tmp_path: Path):
    checkpoint = _mock_probe_checkpoint(tmp_path)
    module = OperatorIntrinsicReward(
        feat_dim=6,
        action_dim=3,
        config=OperatorIntrinsicRewardConfig(enabled=True, probe_checkpoint=str(checkpoint)),
    )
    assert module.probe_model is not None
    assert module.num_tokens == 8


def test_operator_intrinsic_reward_output_shape(tmp_path: Path):
    checkpoint = _mock_probe_checkpoint(tmp_path)
    module = OperatorIntrinsicReward(
        feat_dim=6,
        action_dim=3,
        config=OperatorIntrinsicRewardConfig(enabled=True, probe_checkpoint=str(checkpoint)),
    )
    reward, _ = module(
        x_t=torch.randn(2, 5, 6),
        x_tp1=torch.randn(2, 5, 6),
        action_t=torch.randn(2, 5, 3),
        capacity_gate_t=torch.ones(2, 5, 1),
        update_state=False,
    )
    assert reward.shape == (2, 5)


def test_operator_intrinsic_reward_no_privileged_labels():
    source = Path("policy/dreamerv3/operator_intrinsic_reward.py").read_text(encoding="utf-8").lower()
    for forbidden in ("success", "contact", "inserted", "ball_in_cup", "alignment", "insertion_depth"):
        assert forbidden not in source


def test_capacity_only_zeroes_reward_when_mask_empty(tmp_path: Path):
    checkpoint = _mock_probe_checkpoint(tmp_path)
    module = OperatorIntrinsicReward(
        feat_dim=6,
        action_dim=3,
        config=OperatorIntrinsicRewardConfig(enabled=True, probe_checkpoint=str(checkpoint), capacity_only=True),
    )
    reward, _ = module(
        x_t=torch.randn(2, 5, 6),
        x_tp1=torch.randn(2, 5, 6),
        action_t=torch.randn(2, 5, 3),
        capacity_gate_t=torch.zeros(2, 5, 1),
        update_state=False,
    )
    assert torch.count_nonzero(reward) == 0


def test_token_count_updates_and_novelty_drops(tmp_path: Path):
    checkpoint = _mock_probe_checkpoint(tmp_path)
    module = OperatorIntrinsicReward(
        feat_dim=6,
        action_dim=3,
        config=OperatorIntrinsicRewardConfig(enabled=True, probe_checkpoint=str(checkpoint), normalize_reward=False, clip_reward=False),
    )
    x_t = torch.randn(1, 4, 6)
    x_tp1 = torch.randn(1, 4, 6)
    action_t = torch.randn(1, 4, 3)
    mask = torch.ones(1, 4, 1)
    _, info_a = module(x_t, x_tp1, action_t, capacity_gate_t=mask, update_state=True)
    _, info_b = module(x_t, x_tp1, action_t, capacity_gate_t=mask, update_state=True)
    assert float(module.token_counts.max().item()) > 1.0
    assert float(info_b["operator_token_novelty"].mean().item()) <= float(info_a["operator_token_novelty"].mean().item())


def test_effect_gain_positive_increases_reward(tmp_path: Path):
    checkpoint = _mock_probe_checkpoint(tmp_path)
    module = OperatorIntrinsicReward(
        feat_dim=6,
        action_dim=3,
        config=OperatorIntrinsicRewardConfig(enabled=True, probe_checkpoint=str(checkpoint), normalize_reward=False, clip_reward=False),
    )
    x_t = torch.randn(1, 3, 6)
    x_tp1 = x_t + 0.5
    action_t = torch.randn(1, 3, 3)
    reward, info = module(x_t, x_tp1, action_t, capacity_gate_t=torch.ones(1, 3, 1), update_state=False)
    assert float(info["operator_effect_gain"].mean().item()) >= 0.0
    assert float(reward.mean().item()) >= 0.0


def test_controllability_gain_positive_increases_reward(tmp_path: Path):
    checkpoint = _mock_probe_checkpoint(tmp_path)
    module = OperatorIntrinsicReward(
        feat_dim=6,
        action_dim=3,
        config=OperatorIntrinsicRewardConfig(enabled=True, probe_checkpoint=str(checkpoint), normalize_reward=False, clip_reward=False),
    )
    reward, info = module(
        x_t=torch.randn(1, 3, 6),
        x_tp1=torch.randn(1, 3, 6),
        action_t=torch.randn(1, 3, 3),
        capacity_gate_t=torch.ones(1, 3, 1),
        update_state=False,
    )
    assert float(info["operator_controllability_gain"].mean().item()) >= 0.0
    assert float(reward.mean().item()) >= 0.0


def test_normalize_and_clip_no_nan(tmp_path: Path):
    checkpoint = _mock_probe_checkpoint(tmp_path)
    module = OperatorIntrinsicReward(
        feat_dim=6,
        action_dim=3,
        config=OperatorIntrinsicRewardConfig(enabled=True, probe_checkpoint=str(checkpoint), normalize_reward=True, clip_reward=True),
    )
    reward, _ = module(
        x_t=torch.randn(2, 5, 6),
        x_tp1=torch.randn(2, 5, 6),
        action_t=torch.randn(2, 5, 3),
        capacity_gate_t=torch.ones(2, 5, 1),
        update_state=True,
    )
    assert torch.isfinite(reward).all()
    assert float(reward.min().item()) >= 0.0
    assert float(reward.max().item()) <= 1.0


def test_freeze_probe_disables_probe_gradients(tmp_path: Path):
    checkpoint = _mock_probe_checkpoint(tmp_path)
    module = OperatorIntrinsicReward(
        feat_dim=6,
        action_dim=3,
        config=OperatorIntrinsicRewardConfig(enabled=True, probe_checkpoint=str(checkpoint), freeze_probe=True),
    )
    assert all(not param.requires_grad for param in module.probe_model.parameters())


def test_detach_features_blocks_input_gradients(tmp_path: Path):
    checkpoint = _mock_probe_checkpoint(tmp_path)
    module = OperatorIntrinsicReward(
        feat_dim=6,
        action_dim=3,
        config=OperatorIntrinsicRewardConfig(enabled=True, probe_checkpoint=str(checkpoint), freeze_probe=True, detach_features=True),
    )
    x_t = torch.randn(2, 5, 6, requires_grad=True)
    x_tp1 = torch.randn(2, 5, 6, requires_grad=True)
    action_t = torch.randn(2, 5, 3, requires_grad=True)
    reward, _ = module(x_t, x_tp1, action_t, capacity_gate_t=torch.ones(2, 5, 1), update_state=False)
    assert reward.requires_grad is False
    assert x_t.grad is None and x_tp1.grad is None and action_t.grad is None


def test_enabled_false_keeps_train_reward_equal_env_reward():
    env_reward = torch.randn(2, 5)
    train_reward, env_component, operator_component = combine_train_reward(
        env_reward=env_reward,
        operator_reward=None,
        config=OperatorIntrinsicRewardConfig(enabled=False),
    )
    assert torch.allclose(train_reward, env_reward)
    assert torch.allclose(env_component, env_reward)
    assert torch.count_nonzero(operator_component) == 0


def test_new_exam_loads():
    app = load_exam_train_app("test_metaworld_dreamerv3_peg_insert_side_operator_reward_persistent_smoke")
    config = app.load_config()
    assert config["policy"]["operator_intrinsic_reward"]["enabled"] is True


def test_operator_reward_v1_smoke_exam_loads():
    app = load_exam_train_app("train_metaworld_dreamerv3_peg_insert_side_sparse_operator_reward_v1_smoke")
    config = app.load_config()
    reward_cfg = config["policy"]["operator_intrinsic_reward"]
    assert config["env"]["task_name"] == "peg-insert-side-sparse-v3"
    assert reward_cfg["post_mask_after_normalize"] is True
    assert reward_cfg["eval_use_rolling_capacity"] is True
    assert reward_cfg["require_effect_for_control"] is True
    assert reward_cfg["normalize_reward"] is False
    assert reward_cfg["reward_clip_max"] == 0.5


def test_operator_reward_v1_5k_exam_loads():
    app = load_exam_train_app("train_metaworld_dreamerv3_peg_insert_side_sparse_operator_reward_v1_5k")
    config = app.load_config()
    assert config["train"]["total_steps"] == 5000
    assert config["policy"]["dreamerv3"]["device"] == "cuda"
    assert config["policy"]["operator_intrinsic_reward"]["capacity_only"] is True


def test_topk_exponential_fill_stage_is_linear(tmp_path: Path):
    checkpoint = _mock_probe_checkpoint(tmp_path)
    module = OperatorIntrinsicReward(
        feat_dim=6,
        action_dim=3,
        config=OperatorIntrinsicRewardConfig(
            enabled=True,
            probe_checkpoint=str(checkpoint),
            reward_mode="topk_exponential",
            normalize_reward=False,
            clip_reward=False,
            topk_leaderboard_capacity=4,
            topk_decay=1.0,
        ),
    )
    reward, info = module._topk_exponential_reward(
        intensity=torch.tensor([[0.3, 0.2, 0.1]], dtype=torch.float32),
        operator_rarity=torch.ones(1, 3, dtype=torch.float32),
        token_id=torch.tensor([[0, 1, 2]], dtype=torch.long),
        capacity_mask=torch.ones(1, 3, dtype=torch.float32),
        env_reward_t=torch.zeros(1, 3, dtype=torch.float32),
        update_state=True,
    )
    expected = torch.tensor([[4.0, 2.0, 4.0 / 3.0]], dtype=torch.float32)
    assert torch.allclose(reward, expected, atol=1e-5)
    assert float(info["topk_leaderboard_size"].item()) == 3.0
    assert float(info["topk_update_count"].item()) == 0.0


def test_topk_exponential_replacement_increments_counter(tmp_path: Path):
    checkpoint = _mock_probe_checkpoint(tmp_path)
    module = OperatorIntrinsicReward(
        feat_dim=6,
        action_dim=3,
        config=OperatorIntrinsicRewardConfig(
            enabled=True,
            probe_checkpoint=str(checkpoint),
            reward_mode="topk_exponential",
            normalize_reward=False,
            clip_reward=False,
            topk_leaderboard_capacity=2,
            topk_decay=1.0,
            frequency_habituation_power=1.0,
        ),
    )
    module._topk_exponential_reward(
        intensity=torch.tensor([[0.4, 0.3]], dtype=torch.float32),
        operator_rarity=torch.ones(1, 2, dtype=torch.float32),
        token_id=torch.tensor([[0, 1]], dtype=torch.long),
        capacity_mask=torch.ones(1, 2, dtype=torch.float32),
        env_reward_t=torch.zeros(1, 2, dtype=torch.float32),
        update_state=True,
    )
    reward, info = module._topk_exponential_reward(
        intensity=torch.tensor([[0.5]], dtype=torch.float32),
        operator_rarity=torch.ones(1, 1, dtype=torch.float32),
        token_id=torch.tensor([[2]], dtype=torch.long),
        capacity_mask=torch.ones(1, 1, dtype=torch.float32),
        env_reward_t=torch.zeros(1, 1, dtype=torch.float32),
        update_state=True,
    )
    assert torch.allclose(reward, torch.tensor([[4.0]], dtype=torch.float32), atol=1e-5)
    assert float(info["topk_update_count"].item()) == 1.0
    assert float(info["topk_rank_mean"].item()) == 1.0


def test_topk_exponential_env_bonus_uses_env_reward_signal(tmp_path: Path):
    checkpoint = _mock_probe_checkpoint(tmp_path)
    module = OperatorIntrinsicReward(
        feat_dim=6,
        action_dim=3,
        config=OperatorIntrinsicRewardConfig(
            enabled=True,
            probe_checkpoint=str(checkpoint),
            reward_mode="topk_exponential",
            normalize_reward=False,
            clip_reward=False,
            topk_leaderboard_capacity=2,
            topk_decay=1.0,
            topk_env_bonus_scale=1.0,
        ),
    )
    reward, info = module._topk_exponential_reward(
        intensity=torch.tensor([[0.2]], dtype=torch.float32),
        operator_rarity=torch.ones(1, 1, dtype=torch.float32),
        token_id=torch.tensor([[0]], dtype=torch.long),
        capacity_mask=torch.ones(1, 1, dtype=torch.float32),
        env_reward_t=torch.ones(1, 1, dtype=torch.float32),
        update_state=True,
    )
    assert float(info["topk_env_bonus_mean"].item()) == 2.0
    assert float(reward.item()) == 4.0


def test_topk_exponential_env_bonus_decays_on_repeat_token(tmp_path: Path):
    checkpoint = _mock_probe_checkpoint(tmp_path)
    module = OperatorIntrinsicReward(
        feat_dim=6,
        action_dim=3,
        config=OperatorIntrinsicRewardConfig(
            enabled=True,
            probe_checkpoint=str(checkpoint),
            reward_mode="topk_exponential",
            normalize_reward=False,
            clip_reward=False,
            topk_leaderboard_capacity=4,
            topk_decay=1.0,
            topk_env_bonus_scale=1.0,
            topk_env_bonus_first_only=True,
            topk_env_bonus_repeat_decay=0.1,
            topk_env_bonus_count_decay=1.0,
        ),
    )
    reward_a, info_a = module._topk_exponential_reward(
        intensity=torch.tensor([[0.3]], dtype=torch.float32),
        operator_rarity=torch.ones(1, 1, dtype=torch.float32),
        token_id=torch.tensor([[0]], dtype=torch.long),
        capacity_mask=torch.ones(1, 1, dtype=torch.float32),
        env_reward_t=torch.ones(1, 1, dtype=torch.float32),
        update_state=True,
    )
    reward_b, info_b = module._topk_exponential_reward(
        intensity=torch.tensor([[0.4]], dtype=torch.float32),
        operator_rarity=torch.ones(1, 1, dtype=torch.float32),
        token_id=torch.tensor([[0]], dtype=torch.long),
        capacity_mask=torch.ones(1, 1, dtype=torch.float32),
        env_reward_t=torch.ones(1, 1, dtype=torch.float32),
        update_state=True,
    )
    assert float(info_b["topk_env_bonus_mean"].item()) < float(info_a["topk_env_bonus_mean"].item())
    assert float(reward_b.item()) < float(reward_a.item())


def test_dct_unigram_milestone_rewards_token_once_per_lifetime(tmp_path: Path):
    checkpoint = _mock_probe_checkpoint(tmp_path)
    module = OperatorIntrinsicReward(
        feat_dim=6,
        action_dim=3,
        config=OperatorIntrinsicRewardConfig(
            enabled=True,
            probe_checkpoint=str(checkpoint),
            reward_mode="dct_unigram_milestone",
            r_new_dct=1.0,
            normalize_reward=False,
            clip_reward=False,
        ),
    )
    reward, info = module._dct_unigram_milestone_reward(
        token_id=torch.tensor([[1, 1, 2, 1]], dtype=torch.long),
        capacity_mask=torch.ones(1, 4, dtype=torch.float32),
        lifetime_reset_t=torch.zeros(1, 4, dtype=torch.float32),
        update_state=False,
    )
    assert torch.allclose(reward, torch.tensor([[1.0, 0.0, 1.0, 0.0]]))
    assert torch.allclose(info["milestone_trigger"], torch.tensor([[1.0, 0.0, 1.0, 0.0]]))


def test_dct_unigram_milestone_resets_on_is_first(tmp_path: Path):
    checkpoint = _mock_probe_checkpoint(tmp_path)
    module = OperatorIntrinsicReward(
        feat_dim=6,
        action_dim=3,
        config=OperatorIntrinsicRewardConfig(
            enabled=True,
            probe_checkpoint=str(checkpoint),
            reward_mode="dct_unigram_milestone",
            r_new_dct=1.0,
            normalize_reward=False,
            clip_reward=False,
        ),
    )
    reward, _ = module._dct_unigram_milestone_reward(
        token_id=torch.tensor([[1, 1, 1]], dtype=torch.long),
        capacity_mask=torch.ones(1, 3, dtype=torch.float32),
        lifetime_reset_t=torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32),
        update_state=False,
    )
    assert torch.allclose(reward, torch.tensor([[1.0, 1.0, 0.0]]))


def test_dct_unigram_milestone_requires_capacity(tmp_path: Path):
    checkpoint = _mock_probe_checkpoint(tmp_path)
    module = OperatorIntrinsicReward(
        feat_dim=6,
        action_dim=3,
        config=OperatorIntrinsicRewardConfig(
            enabled=True,
            probe_checkpoint=str(checkpoint),
            reward_mode="dct_unigram_milestone",
            r_new_dct=1.0,
            normalize_reward=False,
            clip_reward=False,
        ),
    )
    reward, _ = module._dct_unigram_milestone_reward(
        token_id=torch.tensor([[1, 2, 2]], dtype=torch.long),
        capacity_mask=torch.tensor([[0.0, 1.0, 1.0]], dtype=torch.float32),
        lifetime_reset_t=torch.zeros(1, 3, dtype=torch.float32),
        update_state=False,
    )
    assert torch.allclose(reward, torch.tensor([[0.0, 1.0, 0.0]]))


def test_sparse_milestone_operator_reward_exam_loads():
    app = load_exam_train_app("train_metaworld_dreamerv3_peg_insert_side_sparse_milestone_operator_reward_v1_10k")
    config = app.load_config()
    reward_cfg = config["policy"]["operator_intrinsic_reward"]
    assert config["env"]["task_name"] == "peg-insert-side-sparse-v3"
    assert reward_cfg["reward_mode"] == "dct_unigram_milestone"
    assert reward_cfg["r_new_dct"] == 1.0
    assert reward_cfg["beta"] == 0.1


def test_dct_sequence_milestone_v3_exam_loads():
    app = load_exam_train_app("train_metaworld_dreamerv3_peg_insert_side_sparse_dct_sequence_milestone_v3_smoke")
    config = app.load_config()
    reward_cfg = config["policy"]["operator_intrinsic_reward"]
    assert config["env"]["task_name"] == "peg-insert-side-sparse-v3"
    assert config["policy"]["type"] == "dreamerv3_milestoneV3"
    assert reward_cfg["reward_mode"] == "dct_temporal_sequence_milestone_v3"
    assert reward_cfg["compute_online_only"] is True
    assert reward_cfg["store_reward_in_replay"] is True
    assert reward_cfg["forbid_batch_recompute_for_sequence_milestone"] is True
    assert reward_cfg["enable_v2_single_event_reward"] is False
    assert reward_cfg["normalize_reward"] is False
    assert reward_cfg["clip_reward"] is False


def test_new_topk_exp_exam_loads():
    app = load_exam_train_app("test_metaworld_dreamerv3_peg_insert_side_sparse_operator_topk_milestone_smoke")
    config = app.load_config()
    assert config["policy"]["operator_intrinsic_reward"]["reward_mode"] == "topk_exponential"


def test_new_topk_exp_env_once_exam_loads():
    app = load_exam_train_app("test_metaworld_dreamerv3_peg_insert_side_sparse_operator_topk_milestone_smoke")
    config = app.load_config()
    assert config["policy"]["operator_intrinsic_reward"]["topk_env_bonus_first_only"] is True


def test_new_topk_exp_env_once_tuned_exam_loads():
    app = load_exam_train_app("test_metaworld_dreamerv3_peg_insert_side_sparse_operator_topk_milestone_smoke")
    config = app.load_config()
    assert config["policy"]["operator_intrinsic_reward"]["topk_env_bonus_repeat_decay"] == 0.0
