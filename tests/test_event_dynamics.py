from __future__ import annotations

import torch

from policy.dreamerv3.event_dynamics import (
    DreamerEventDynamicsConfig,
    EventDynamicsModule,
    build_context_change_mask,
)


def _make_module(context_mask_enabled: bool = True) -> EventDynamicsModule:
    config = DreamerEventDynamicsConfig(
        enabled=True,
        hidden_dim=16,
        gate_bias_init=-3.0,
        event_penalty=0.0005,
        prediction_loss_scale=1.0,
        action_input=True,
        detach_prediction_target=True,
        detach_base_input=False,
        context_mask_enabled=context_mask_enabled,
        context_mask_source="context_gate",
        context_mask_mode="topk",
        context_mask_top_percent=0.25,
        context_mask_window=1,
        context_mask_detach=True,
        event_sparsity_inside_mask_only=True,
    )
    return EventDynamicsModule(feat_dim=8, action_dim=3, config=config)


def test_build_context_change_mask_topk_shape_and_keep_one():
    gate = torch.full((2, 5, 1), 0.1)
    mask = build_context_change_mask(gate, mode="topk", top_percent=0.05, window=0, detach=True)
    assert mask.shape == (2, 5, 1)
    assert torch.all(mask.sum(dim=1) >= 1.0)


def test_build_context_change_mask_window_expands_neighbors():
    gate = torch.tensor([[[0.1], [0.2], [0.9], [0.3], [0.1]]], dtype=torch.float32)
    mask = build_context_change_mask(gate, mode="topk", top_percent=0.2, window=1, detach=True)
    expected = torch.tensor([[[0.0], [1.0], [1.0], [1.0], [0.0]]], dtype=torch.float32)
    assert torch.allclose(mask, expected)


def test_build_context_change_mask_detach_blocks_grad():
    gate = torch.tensor([[[0.2], [0.9], [0.1]]], dtype=torch.float32, requires_grad=True)
    other = torch.ones_like(gate, requires_grad=True)
    mask = build_context_change_mask(gate, mode="soft", detach=True)
    loss = (mask * other).sum()
    loss.backward()
    assert gate.grad is None
    assert other.grad is not None


def test_masked_event_loss_backpropagates_to_gate_and_event_heads():
    torch.manual_seed(0)
    module = _make_module(context_mask_enabled=True)
    feat = torch.randn(2, 4, 8)
    action = torch.randn(2, 4, 3)
    target = torch.randn(2, 4, 8)
    mask = torch.tensor(
        [
            [[0.0], [1.0], [1.0], [0.0]],
            [[1.0], [0.0], [1.0], [0.0]],
        ],
        dtype=torch.float32,
    )
    outputs = module.predict(feat, action=action, target_next_feat=target, context_change_mask=mask)
    masked_error = (mask.squeeze(-1) * outputs["pred_next_feat_error_event_mixed"]).sum() / mask.sum().clamp_min(1.0)
    loss = outputs["pred_next_feat_error_ordinary_only"].mean() + masked_error
    loss = loss + ((mask * outputs["event_gate"]).sum() / mask.sum().clamp_min(1.0))
    loss.backward()
    assert module.gate_net[-1].weight.grad is not None
    assert module.event_head[-1].weight.grad is not None
    assert module.gate_net[-1].weight.grad.detach().abs().sum() > 0
    assert module.event_head[-1].weight.grad.detach().abs().sum() > 0


def test_zero_mask_blocks_event_head_gradients():
    torch.manual_seed(0)
    module = _make_module(context_mask_enabled=True)
    feat = torch.randn(2, 4, 8)
    action = torch.randn(2, 4, 3)
    target = torch.randn(2, 4, 8)
    mask = torch.zeros(2, 4, 1, dtype=torch.float32)
    outputs = module.predict(feat, action=action, target_next_feat=target, context_change_mask=mask)
    masked_error = (mask.squeeze(-1) * outputs["pred_next_feat_error_event_mixed"]).sum() / mask.sum().clamp_min(1.0)
    loss = outputs["pred_next_feat_error_ordinary_only"].mean() + masked_error
    loss = loss + ((mask * outputs["event_gate"]).sum() / mask.sum().clamp_min(1.0))
    loss.backward()
    event_grad = module.event_head[-1].weight.grad
    gate_grad = module.gate_net[-1].weight.grad
    assert event_grad is None or torch.allclose(event_grad, torch.zeros_like(event_grad))
    assert gate_grad is None or torch.allclose(gate_grad, torch.zeros_like(gate_grad))


def test_context_mask_disabled_keeps_old_mixing_behavior():
    torch.manual_seed(0)
    module = _make_module(context_mask_enabled=False)
    feat = torch.randn(1, 3, 8)
    action = torch.randn(1, 3, 3)
    with torch.no_grad():
        module.gate_net[-1].bias.fill_(8.0)
    outputs = module.predict(
        feat,
        action=action,
        target_next_feat=torch.randn(1, 3, 8),
        context_change_mask=torch.zeros(1, 3, 1),
    )
    assert outputs["context_change_mask"] is None
    assert not torch.allclose(outputs["pred_next_feat_event_mixed"], outputs["pred_next_feat_ordinary_only"])


def test_no_pseudo_label_bce_or_cooldown_fields():
    fields = DreamerEventDynamicsConfig.__dataclass_fields__
    assert "pseudo_event_label" not in fields
    assert "bce_event_loss" not in fields
    assert "cooldown_penalty" not in fields
    assert "refractory_penalty" not in fields
