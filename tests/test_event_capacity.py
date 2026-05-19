from __future__ import annotations

import torch
from torch import nn

from policy.dreamerv3.event_dynamics import (
    DreamerEventDynamicsConfig,
    EventCapacitySelector,
    EventDynamicsModule,
)
from policy.dreamerv3.thick_context import ThickContextConfig
from policy.dreamerv3.world_model import DreamerV3WorldModel, DreamerV3WorldModelConfig


def _make_config(capacity_enabled: bool = True, **overrides) -> DreamerEventDynamicsConfig:
    base = {
        "enabled": True,
        "hidden_dim": 16,
        "gate_bias_init": -3.0,
        "event_penalty": 0.0005,
        "prediction_loss_scale": 1.0,
        "action_input": True,
        "detach_prediction_target": True,
        "detach_base_input": False,
        "capacity_enabled": capacity_enabled,
        "capacity_mode": "soft_topk_st",
        "capacity_ratio": 0.10,
        "capacity_temperature": 1.0,
        "capacity_min_k": 1,
        "capacity_eval_hard": True,
        "capacity_use_valid_mask": True,
        "event_sparsity_inside_capacity_only": True,
        "capacity_v2_detach_event_input": False,
        "capacity_v2_detach_event_target": False,
        "capacity_v2_use_sigmoid_gate_multiplier": False,
        "capacity_v2_freeze_backbone_for_event_loss": False,
    }
    base.update(overrides)
    return DreamerEventDynamicsConfig(**base)


def _make_v2_module(**overrides) -> EventDynamicsModule:
    config = _make_config(
        capacity_enabled=True,
        event_penalty=0.0,
        capacity_v2_detach_event_input=True,
        capacity_v2_detach_event_target=True,
        capacity_v2_use_sigmoid_gate_multiplier=False,
        capacity_v2_freeze_backbone_for_event_loss=True,
        **overrides,
    )
    return EventDynamicsModule(feat_dim=8, action_dim=3, config=config)


class FixedOutput(nn.Module):
    def __init__(self, output: torch.Tensor):
        super().__init__()
        self.register_buffer("output", output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != self.output.ndim:
            raise AssertionError(f"Expected ndim {self.output.ndim}, got {x.ndim}")
        if x.shape[:-1] != self.output.shape[:-1]:
            raise AssertionError(f"Expected leading shape {self.output.shape[:-1]}, got {x.shape[:-1]}")
        return self.output.clone()


def _event_only_capacity_loss(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    capacity_gate = outputs["event_capacity_gate"]
    ordinary_error = outputs["pred_next_feat_error_ordinary_only"]
    mixed_error = outputs["pred_next_feat_error_event_mixed"]
    return ordinary_error.mean() + (
        (capacity_gate.squeeze(-1) * mixed_error).sum() / capacity_gate.sum().clamp_min(1.0)
    )


def _all_zero_or_none(parameters) -> bool:
    for param in parameters:
        if param.grad is None:
            continue
        if not torch.allclose(param.grad, torch.zeros_like(param.grad)):
            return False
    return True


def _make_world_model() -> DreamerV3WorldModel:
    return DreamerV3WorldModel(
        DreamerV3WorldModelConfig(
            obs_dim=8,
            action_dim=3,
            encoder_type="mlp",
            embed_dim=16,
            deter_dim=16,
            stoch_dim=4,
            stoch_classes=4,
            hidden_dim=32,
            num_layers=2,
            use_symlog_obs=False,
            use_symlog_reward=False,
            use_twohot_reward=False,
            thick_context=ThickContextConfig(
                enabled=True,
                context_dim=6,
                hidden_dim=12,
                update_penalty=0.001,
                gate_bias_init=-2.0,
            ),
            event_dynamics=_make_config(
                capacity_enabled=True,
                event_penalty=0.0,
                capacity_v2_detach_event_input=True,
                capacity_v2_detach_event_target=True,
                capacity_v2_use_sigmoid_gate_multiplier=False,
                capacity_v2_freeze_backbone_for_event_loss=True,
            ),
        )
    )


def _make_batch(batch_size: int = 2, seq_len: int = 5) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "obs": torch.randn(batch_size, seq_len, 8),
        "action": torch.randn(batch_size, seq_len, 3),
        "reward": torch.randn(batch_size, seq_len),
        "done": torch.zeros(batch_size, seq_len),
        "is_first": torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0, 0.0]])[:batch_size, :seq_len],
    }


def test_event_capacity_selector_shape_and_min_keep():
    selector = EventCapacitySelector(_make_config())
    selector.train()
    logits = torch.randn(3, 8, 1)
    gate, stats = selector(logits)
    assert gate.shape == (3, 8, 1)
    assert stats["event_capacity_k"].shape == (3, 1, 1)
    assert torch.all(stats["event_capacity_k"] >= 1.0)


def test_event_capacity_gate_range_is_valid():
    selector = EventCapacitySelector(_make_config())
    selector.train()
    logits = torch.randn(2, 6, 1)
    gate, _ = selector(logits)
    assert torch.all(gate >= 0.0)
    assert torch.all(gate <= 1.0)


def test_soft_topk_st_backpropagates_to_event_logits():
    selector = EventCapacitySelector(_make_config())
    selector.train()
    logits = torch.randn(2, 6, 1, requires_grad=True)
    gate, _ = selector(logits)
    weights = torch.arange(1, 7, dtype=logits.dtype).view(1, 6, 1)
    loss = (gate * weights).sum()
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.detach().abs().sum() > 0


def test_eval_hard_mode_outputs_binary_gate():
    selector = EventCapacitySelector(_make_config())
    selector.eval()
    logits = torch.tensor([[[0.1], [0.9], [0.2], [0.3]]], dtype=torch.float32)
    gate, _ = selector(logits)
    unique = set(gate.reshape(-1).tolist())
    assert unique.issubset({0.0, 1.0})


def test_valid_mask_blocks_invalid_transitions():
    selector = EventCapacitySelector(_make_config())
    selector.eval()
    logits = torch.tensor([[[0.1], [10.0], [0.2], [0.3]]], dtype=torch.float32)
    valid_mask = torch.tensor([[[1.0], [0.0], [1.0], [1.0]]], dtype=torch.float32)
    gate, _ = selector(logits, valid_mask=valid_mask)
    assert float(gate[0, 1, 0]) == 0.0


def test_v2_prediction_does_not_multiply_sigmoid_gate():
    module = _make_v2_module(capacity_eval_hard=True)
    module.eval()
    feat = torch.zeros(1, 4, 8)
    action = torch.zeros(1, 4, 3)
    module.gate_net = FixedOutput(torch.tensor([[[-10.0], [-20.0], [-20.0], [-20.0]]], dtype=torch.float32))
    module.ordinary_head = FixedOutput(torch.full((1, 4, 8), 0.5, dtype=torch.float32))
    module.event_head = FixedOutput(torch.full((1, 4, 8), 2.0, dtype=torch.float32))
    outputs = module.predict(feat, action=action, target_next_feat=torch.zeros_like(feat))
    assert outputs["event_capacity_gate"][0, 0, 0].item() == 1.0
    assert outputs["event_logit_sigmoid"][0, 0, 0].item() < 1e-3
    assert torch.allclose(outputs["pred_next_feat_event_mixed"][0, 0], torch.full((8,), 2.5))
    assert torch.allclose(outputs["pred_next_feat_event_mixed"][0, 1:], torch.full((3, 8), 0.5))


def test_event_logit_only_changes_capacity_ranking():
    module = _make_v2_module(capacity_eval_hard=True)
    module.eval()
    feat = torch.zeros(1, 4, 8)
    action = torch.zeros(1, 4, 3)
    module.ordinary_head = FixedOutput(torch.full((1, 4, 8), 0.5, dtype=torch.float32))
    module.event_head = FixedOutput(torch.full((1, 4, 8), 2.0, dtype=torch.float32))
    module.gate_net = FixedOutput(torch.tensor([[[-10.0], [-20.0], [-20.0], [-20.0]]], dtype=torch.float32))
    outputs_a = module.predict(feat, action=action, target_next_feat=torch.zeros_like(feat))
    module.gate_net = FixedOutput(torch.tensor([[[10.0], [0.0], [0.0], [0.0]]], dtype=torch.float32))
    outputs_b = module.predict(feat, action=action, target_next_feat=torch.zeros_like(feat))
    assert torch.allclose(outputs_a["event_capacity_gate"], outputs_b["event_capacity_gate"])
    assert torch.allclose(outputs_a["pred_next_feat_event_mixed"], outputs_b["pred_next_feat_event_mixed"])
    assert not torch.allclose(outputs_a["event_logit_sigmoid"], outputs_b["event_logit_sigmoid"])


def test_capacity_gate_controls_event_residual_entry():
    module = _make_v2_module(capacity_eval_hard=True)
    module.eval()
    feat = torch.zeros(1, 4, 8)
    action = torch.zeros(1, 4, 3)
    module.gate_net = FixedOutput(torch.tensor([[[5.0], [4.0], [3.0], [2.0]]], dtype=torch.float32))
    module.ordinary_head = FixedOutput(torch.zeros(1, 4, 8))
    module.event_head = FixedOutput(torch.ones(1, 4, 8))
    outputs = module.predict(feat, action=action, target_next_feat=torch.zeros_like(feat))
    expected = outputs["event_capacity_gate"] * torch.ones_like(outputs["pred_next_feat_event_mixed"])
    assert torch.allclose(outputs["pred_next_feat_event_mixed"], expected)


def test_v2_event_input_detach_blocks_grad_to_input_and_target():
    torch.manual_seed(0)
    module = _make_v2_module()
    module.train()
    feat = torch.randn(2, 5, 8, requires_grad=True)
    action = torch.randn(2, 5, 3)
    target = torch.randn(2, 5, 8, requires_grad=True)
    outputs = module.predict(feat, action=action, target_next_feat=target)
    loss = _event_only_capacity_loss(outputs)
    loss.backward()
    assert feat.grad is None or torch.allclose(feat.grad, torch.zeros_like(feat.grad))
    assert target.grad is None or torch.allclose(target.grad, torch.zeros_like(target.grad))
    assert module.gate_net[-1].weight.grad is not None
    assert module.ordinary_head[-1].weight.grad is not None
    assert module.event_head[-1].weight.grad is not None
    assert module.gate_net[-1].weight.grad.detach().abs().sum() > 0
    assert module.ordinary_head[-1].weight.grad.detach().abs().sum() > 0
    assert module.event_head[-1].weight.grad.detach().abs().sum() > 0


def test_isolated_event_loss_does_not_update_backbone():
    torch.manual_seed(0)
    model = _make_world_model()
    model.train()
    outputs = model(_make_batch(), None)
    loss = _event_only_capacity_loss(outputs)
    model.zero_grad(set_to_none=True)
    loss.backward()
    assert outputs["event_loss_updates_backbone"] is False
    assert _all_zero_or_none(model.encoder.parameters())
    assert _all_zero_or_none(model.rssm.parameters())
    assert _all_zero_or_none(model.slow_context.parameters())
    assert _all_zero_or_none(model.decoder.parameters())
    assert _all_zero_or_none(model.reward_head.parameters())
    assert _all_zero_or_none(model.continue_head.parameters())
    assert not _all_zero_or_none(model.event_dynamics.gate_net.parameters())
    assert not _all_zero_or_none(model.event_dynamics.ordinary_head.parameters())
    assert not _all_zero_or_none(model.event_dynamics.event_head.parameters())


def test_capacity_disabled_keeps_old_event_mixing_behavior():
    config = _make_config(capacity_enabled=False)
    module = EventDynamicsModule(feat_dim=8, action_dim=3, config=config)
    feat = torch.randn(1, 4, 8)
    action = torch.randn(1, 4, 3)
    outputs = module.predict(feat, action=action, target_next_feat=torch.randn(1, 4, 8))
    assert outputs["event_capacity_gate"] is None
    assert outputs["event_capacity_enabled"] is False
    assert not torch.allclose(outputs["pred_next_feat_event_mixed"], outputs["pred_next_feat_ordinary_only"])


def test_no_bce_cooldown_or_privileged_label_fields():
    fields = DreamerEventDynamicsConfig.__dataclass_fields__
    assert "bce_event_loss" not in fields
    assert "cooldown_penalty" not in fields
    assert "refractory_penalty" not in fields
    assert "success_label" not in fields
    assert "contact_label" not in fields
    assert "inserted_label" not in fields
