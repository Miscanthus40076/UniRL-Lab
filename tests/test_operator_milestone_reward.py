from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from policy.dreamerv3.operator_intrinsic_reward import OperatorIntrinsicReward, OperatorIntrinsicRewardConfig
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


class FixedProbe(nn.Module):
    def __init__(self, token_ids: torch.Tensor, num_tokens: int):
        super().__init__()
        self.register_buffer("token_ids", token_ids.long())
        self.num_tokens = int(num_tokens)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x_t = batch["x_t"]
        delta_x = batch["delta_x"]
        action_t = batch["action_t"]
        token_id = self.token_ids.to(device=x_t.device)
        token_probs = F.one_hot(token_id, num_classes=self.num_tokens).to(dtype=x_t.dtype)
        return {
            "token_logits": torch.log(token_probs + 1e-6),
            "token_probs": token_probs,
            "token_onehot_st": token_probs,
            "token_id": token_id,
            "pred_delta_x": delta_x,
            "pred_delta_x_no_token": torch.zeros_like(delta_x),
            "pred_action": action_t,
            "pred_action_no_token": torch.zeros_like(action_t),
        }


def _make_module(tmp_path: Path, token_ids: torch.Tensor, **overrides) -> OperatorIntrinsicReward:
    checkpoint = _mock_probe_checkpoint(tmp_path)
    raw = {
        "enabled": True,
        "probe_checkpoint": str(checkpoint),
        "reward_mode": "milestone",
        "normalize_reward": False,
        "clip_reward": False,
        "running_best_rate": 1.0,
        "milestone_margin": 0.1,
        "frequency_habituation_power": 1.0,
        "repeat_decay_tau": 1.0,
        "repeat_decay_min": 0.05,
    }
    raw.update(overrides)
    cfg = OperatorIntrinsicRewardConfig(**raw)
    module = OperatorIntrinsicReward(feat_dim=6, action_dim=3, config=cfg)
    module.probe_model = FixedProbe(token_ids=token_ids, num_tokens=module.num_tokens)
    module.event_logit_norm.mean = 0.0
    module.event_logit_norm.var = 1.0
    module.event_logit_norm.initialized = True
    module.context_delta_norm.mean = 0.0
    module.context_delta_norm.var = 1.0
    module.context_delta_norm.initialized = True
    module.effect_gain_norm.mean = 0.0
    module.effect_gain_norm.var = 1.0
    module.effect_gain_norm.initialized = True
    module.rarity_norm.mean = 0.0
    module.rarity_norm.var = 1.0
    module.rarity_norm.initialized = True
    return module


def test_milestone_reward_output_shape(tmp_path: Path):
    token_ids = torch.tensor([[0, 1, 2, 3], [1, 1, 2, 2]])
    module = _make_module(tmp_path, token_ids)
    reward, _ = module(
        x_t=torch.randn(2, 4, 6),
        x_tp1=torch.randn(2, 4, 6),
        action_t=torch.randn(2, 4, 3),
        capacity_gate_t=torch.ones(2, 4, 1),
        event_logit_t=torch.randn(2, 4, 1),
        context_delta_norm_t=torch.randn(2, 4, 1),
        ordinary_error_t=torch.ones(2, 4, 1),
        mixed_error_t=torch.zeros(2, 4, 1),
        update_state=False,
    )
    assert reward.shape == (2, 4)


def test_milestone_reward_no_privileged_labels():
    source = Path("policy/dreamerv3/operator_intrinsic_reward.py").read_text(encoding="utf-8").lower()
    for forbidden in ("success", "contact", "inserted", "ball_in_cup", "alignment", "insertion_depth"):
        assert forbidden not in source


def test_capacity_only_zeroes_milestone_reward_when_mask_empty(tmp_path: Path):
    token_ids = torch.tensor([[0, 1, 2, 3]])
    module = _make_module(tmp_path, token_ids, capacity_only=True)
    reward, info = module(
        x_t=torch.randn(1, 4, 6),
        x_tp1=torch.randn(1, 4, 6),
        action_t=torch.randn(1, 4, 3),
        capacity_gate_t=torch.zeros(1, 4, 1),
        event_logit_t=torch.ones(1, 4, 1),
        context_delta_norm_t=torch.ones(1, 4, 1),
        ordinary_error_t=torch.ones(1, 4, 1),
        mixed_error_t=torch.zeros(1, 4, 1),
        update_state=False,
    )
    assert torch.count_nonzero(reward) == 0
    assert float(info["milestone_trigger_count"].item()) == 0.0


def test_milestone_reward_updates_running_best_and_reduces_repeat_reward(tmp_path: Path):
    token_ids = torch.tensor([[0, 1, 2, 3]])
    module = _make_module(
        tmp_path,
        token_ids,
        intensity_context_delta_weight=0.0,
        intensity_effect_gain_weight=0.0,
        intensity_operator_rarity_weight=0.0,
        frequency_habituation_power=0.0,
        repeat_decay_tau=1000.0,
        repeat_decay_min=1.0,
    )
    event_logit = torch.tensor([[[0.0], [0.25], [0.6], [0.8]]], dtype=torch.float32)
    x_t = torch.randn(1, 4, 6)
    x_tp1 = torch.randn(1, 4, 6)
    action_t = torch.randn(1, 4, 3)
    mask = torch.ones(1, 4, 1)
    ordinary = torch.ones(1, 4, 1)
    mixed = torch.zeros(1, 4, 1)

    reward_a, info_a = module(
        x_t=x_t,
        x_tp1=x_tp1,
        action_t=action_t,
        capacity_gate_t=mask,
        event_logit_t=event_logit,
        context_delta_norm_t=torch.zeros(1, 4, 1),
        ordinary_error_t=ordinary,
        mixed_error_t=mixed,
        update_state=True,
    )
    reward_b, info_b = module(
        x_t=x_t,
        x_tp1=x_tp1,
        action_t=action_t,
        capacity_gate_t=mask,
        event_logit_t=event_logit,
        context_delta_norm_t=torch.zeros(1, 4, 1),
        ordinary_error_t=ordinary,
        mixed_error_t=mixed,
        update_state=False,
    )
    assert float(info_a["milestone_trigger_count"].item()) > 0.0
    assert float(info_a["running_best_intensity"].item()) >= 0.79
    assert float(reward_b.sum().item()) < float(reward_a.sum().item())


def test_repeat_decay_and_habituation_reduce_repeated_token_reward(tmp_path: Path):
    token_ids = torch.tensor([[2, 2, 2, 2]])
    module = _make_module(tmp_path, token_ids)
    module.token_counts.fill_(16.0)
    reward, info = module(
        x_t=torch.randn(1, 4, 6),
        x_tp1=torch.randn(1, 4, 6),
        action_t=torch.randn(1, 4, 3),
        capacity_gate_t=torch.ones(1, 4, 1),
        event_logit_t=torch.ones(1, 4, 1),
        context_delta_norm_t=torch.ones(1, 4, 1),
        ordinary_error_t=torch.ones(1, 4, 1),
        mixed_error_t=torch.zeros(1, 4, 1),
        update_state=False,
    )
    assert float(info["habituation_factor_mean"].item()) < 1.0
    assert float(info["token_repeat_decay_mean"].item()) < 1.0
    assert float(reward[0, -1].item()) < float(reward[0, 0].item())
    assert torch.isfinite(reward).all()


def test_detach_features_blocks_gradients_in_milestone_mode(tmp_path: Path):
    token_ids = torch.tensor([[0, 1, 2, 3]])
    module = _make_module(tmp_path, token_ids, detach_features=True)
    x_t = torch.randn(1, 4, 6, requires_grad=True)
    x_tp1 = torch.randn(1, 4, 6, requires_grad=True)
    action_t = torch.randn(1, 4, 3, requires_grad=True)
    reward, _ = module(
        x_t=x_t,
        x_tp1=x_tp1,
        action_t=action_t,
        capacity_gate_t=torch.ones(1, 4, 1),
        event_logit_t=torch.ones(1, 4, 1),
        context_delta_norm_t=torch.ones(1, 4, 1),
        ordinary_error_t=torch.ones(1, 4, 1),
        mixed_error_t=torch.zeros(1, 4, 1),
        update_state=False,
    )
    assert reward.requires_grad is False
    assert x_t.grad is None and x_tp1.grad is None and action_t.grad is None


def test_milestone_exam_loads():
    app = load_exam_train_app("test_metaworld_dreamerv3_peg_insert_side_operator_milestone_smoke")
    config = app.load_config()
    reward_cfg = config["operator_intrinsic_reward"]
    assert reward_cfg["enabled"] is True
    assert reward_cfg["reward_mode"] == "milestone"
