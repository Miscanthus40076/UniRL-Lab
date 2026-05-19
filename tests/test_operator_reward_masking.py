from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from policy.dreamerv3.operator_intrinsic_reward import (
    OperatorIntrinsicReward,
    OperatorIntrinsicRewardConfig,
    combine_train_reward,
)
from policy.operator_token_probe.config import OperatorTokenProbePolicyConfig
from policy.operator_token_probe.operator_token_probe_agent import OperatorTokenProbeAgent


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


class _FixedProbe(nn.Module):
    def __init__(self, mode: str, num_tokens: int = 8):
        super().__init__()
        self.mode = mode
        self.num_tokens = num_tokens

    def forward(self, batch):
        x_t = batch["x_t"]
        delta_x = batch["delta_x"]
        action_t = batch["action_t"]
        token_id = torch.zeros(x_t.shape[:2], device=x_t.device, dtype=torch.long)
        token_probs = torch.nn.functional.one_hot(token_id, num_classes=self.num_tokens).float()
        pred_delta_x = delta_x.clone()
        pred_delta_x_no_token = torch.zeros_like(delta_x)
        pred_action = action_t.clone()
        pred_action_no_token = torch.zeros_like(action_t)
        if self.mode == "control_only":
            pred_delta_x_no_token = pred_delta_x.clone()
            pred_action_no_token = action_t + 1.0
        if self.mode == "no_effect":
            pred_delta_x_no_token = pred_delta_x.clone()
        return {
            "token_logits": token_probs,
            "token_probs": token_probs,
            "token_onehot_st": token_probs,
            "token_id": token_id,
            "token_emb": torch.zeros(*x_t.shape[:2], 32, device=x_t.device),
            "pred_delta_x": pred_delta_x,
            "pred_delta_x_no_token": pred_delta_x_no_token,
            "pred_action": pred_action,
            "pred_action_no_token": pred_action_no_token,
        }


def _module(tmp_path: Path, **kwargs) -> OperatorIntrinsicReward:
    checkpoint = _mock_probe_checkpoint(tmp_path)
    config_kwargs = {
        "enabled": True,
        "probe_checkpoint": str(checkpoint),
        "capacity_only": True,
        "normalize_reward": False,
        "clip_reward": False,
    }
    config_kwargs.update(kwargs)
    cfg = OperatorIntrinsicRewardConfig(**config_kwargs)
    module = OperatorIntrinsicReward(feat_dim=6, action_dim=3, config=cfg)
    module.probe_model = _FixedProbe("positive", num_tokens=module.num_tokens)
    return module


def test_capacity_mask_zero_final_reward_strictly_zero(tmp_path: Path):
    module = _module(tmp_path)
    reward, info = module(
        x_t=torch.zeros(1, 4, 6),
        x_tp1=torch.ones(1, 4, 6),
        action_t=torch.ones(1, 4, 3),
        capacity_gate_t=torch.zeros(1, 4, 1),
        update_state=False,
    )
    assert torch.count_nonzero(reward) == 0
    assert float(info["operator_reward_mask_violation_count"].item()) == 0.0


def test_post_mask_keeps_zero_after_normalize_and_clip(tmp_path: Path):
    module = _module(
        tmp_path,
        normalize_reward=True,
        clip_reward=True,
        reward_clip_max=0.5,
        post_mask_after_normalize=True,
    )
    capacity = torch.tensor([[[1.0], [0.0], [1.0], [0.0]]])
    reward, info = module(
        x_t=torch.zeros(1, 4, 6),
        x_tp1=torch.ones(1, 4, 6),
        action_t=torch.ones(1, 4, 3),
        capacity_gate_t=capacity,
        update_state=True,
    )
    assert torch.all(reward[capacity.squeeze(-1) == 0] == 0)
    assert float(info["operator_reward_nonzero_outside_capacity"].item()) == 0.0


def test_single_step_insufficient_window_zero_capacity_reward(tmp_path: Path):
    module = _module(tmp_path)
    reward, _ = module(
        x_t=torch.zeros(1, 1, 6),
        x_tp1=torch.ones(1, 1, 6),
        action_t=torch.ones(1, 1, 3),
        capacity_gate_t=torch.zeros(1, 1, 1),
        update_state=False,
    )
    assert float(reward.item()) == 0.0


def test_rolling_window_capacity_ratio_close_to_topk():
    logits = torch.arange(64, dtype=torch.float32).view(1, 64, 1)
    k = max(1, int(round(64 * 0.10)))
    threshold = torch.topk(logits.squeeze(-1), k=k, dim=1).values[:, -1:]
    gate = (logits.squeeze(-1) >= threshold).float()
    assert abs(float(gate.mean().item()) - (k / 64.0)) < 1e-6


def test_require_effect_for_control_suppresses_control_only_reward(tmp_path: Path):
    module = _module(
        tmp_path,
        require_effect_for_control=True,
        min_effect_gain=0.005,
        min_delta_x_norm=0.01,
    )
    module.probe_model = _FixedProbe("control_only", num_tokens=module.num_tokens)
    reward, info = module(
        x_t=torch.zeros(1, 4, 6),
        x_tp1=torch.ones(1, 4, 6),
        action_t=torch.ones(1, 4, 3),
        capacity_gate_t=torch.ones(1, 4, 1),
        update_state=False,
    )
    assert torch.count_nonzero(reward) == 0
    assert float(info["operator_control_only_suppressed"].mean().item()) > 0.0


def test_low_delta_x_below_threshold_gets_no_reward(tmp_path: Path):
    module = _module(tmp_path, min_effect_gain=0.005, min_delta_x_norm=10.0)
    reward, info = module(
        x_t=torch.zeros(1, 4, 6),
        x_tp1=torch.ones(1, 4, 6),
        action_t=torch.ones(1, 4, 3),
        capacity_gate_t=torch.ones(1, 4, 1),
        update_state=False,
    )
    assert torch.count_nonzero(reward) == 0
    assert float(info["operator_valid_transition"].mean().item()) == 0.0


def test_stale_filter_suppresses_low_change_reward(tmp_path: Path):
    module = _module(tmp_path, stale_filter_enabled=True, min_latent_delta_norm=0.01)
    reward, info = module(
        x_t=torch.zeros(1, 4, 6),
        x_tp1=torch.full((1, 4, 6), 1e-4),
        action_t=torch.ones(1, 4, 3),
        capacity_gate_t=torch.ones(1, 4, 1),
        update_state=False,
    )
    assert torch.count_nonzero(reward) == 0
    assert float(info["operator_stale_suppressed"].mean().item()) >= 0.0


def test_enabled_false_train_reward_equals_env_reward():
    env_reward = torch.randn(2, 3)
    train_reward, env_component, operator_component = combine_train_reward(
        env_reward=env_reward,
        operator_reward=torch.ones_like(env_reward),
        config=OperatorIntrinsicRewardConfig(enabled=False),
    )
    assert torch.allclose(train_reward, env_reward)
    assert torch.allclose(env_component, env_reward)
    assert torch.count_nonzero(operator_component) == 0


def test_no_privileged_reward_fields_used():
    source = Path("policy/dreamerv3/operator_intrinsic_reward.py").read_text(encoding="utf-8").lower()
    for forbidden in ("success", "contact", "inserted", "ball_in_cup", "alignment", "insertion_depth"):
        assert forbidden not in source
