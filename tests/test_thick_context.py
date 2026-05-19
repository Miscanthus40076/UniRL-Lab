from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import torch

from policy.dreamerv3.dreamerv3_agent import DreamerV3Agent
from policy.dreamerv3.dreamerv3_model import DreamerObservationSpec, DreamerV3ModelConfig
from policy.dreamerv3.losses import WorldModelLossConfig
from policy.dreamerv3.world_model import DreamerV3WorldModel, DreamerV3WorldModelConfig
from policy.dreamerv3.thick_context import SlowContextModule, ThickContextConfig


ROOT = Path(__file__).resolve().parents[1]


def _make_obs_spec():
    return DreamerObservationSpec(mode="vector", obs_dim=8)


def _make_thick_context(enabled: bool, **kwargs) -> ThickContextConfig:
    base = {
        "enabled": enabled,
        "context_dim": 6,
        "hidden_dim": 12,
        "update_penalty": 0.001,
        "gate_bias_init": -2.0,
    }
    base.update(kwargs)
    return ThickContextConfig(**base)


def _make_model_config(enabled: bool, *, thick_context: ThickContextConfig | None = None) -> DreamerV3ModelConfig:
    return DreamerV3ModelConfig(
        action_dim=3,
        observation=_make_obs_spec(),
        device="cpu",
        embed_dim=16,
        deter_dim=16,
        stoch_dim=4,
        stoch_classes=4,
        hidden_dim=32,
        num_layers=2,
        actor_hidden_dim=32,
        actor_num_layers=2,
        value_hidden_dim=32,
        value_num_layers=2,
        imagination_horizon=4,
        twohot_bins=31,
        thick_context=thick_context or _make_thick_context(enabled),
    )


def _make_world_model(enabled: bool, *, thick_context: ThickContextConfig | None = None) -> DreamerV3WorldModel:
    cfg = _make_model_config(enabled, thick_context=thick_context)
    return DreamerV3WorldModel(
        DreamerV3WorldModelConfig(
            obs_dim=cfg.observation.obs_dim,
            obs_shape=cfg.observation.obs_shape,
            action_dim=cfg.action_dim,
            encoder_type=cfg.encoder_type,
            embed_dim=cfg.embed_dim,
            deter_dim=cfg.deter_dim,
            stoch_dim=cfg.stoch_dim,
            stoch_classes=cfg.stoch_classes,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            use_symlog_obs=cfg.use_symlog_obs,
            use_symlog_reward=cfg.use_symlog_reward,
            use_twohot_reward=cfg.use_twohot_reward,
            reward_bins=cfg.twohot_bins,
            reward_low=cfg.twohot_low,
            reward_high=cfg.twohot_high,
            rssm_unimix=cfg.rssm_unimix,
            thick_context=cfg.thick_context,
        )
    )


def _make_batch(batch_size: int = 2, seq_len: int = 4):
    torch.manual_seed(0)
    return {
        "obs": torch.randn(batch_size, seq_len, 8),
        "action": torch.randn(batch_size, seq_len, 3),
        "reward": torch.randn(batch_size, seq_len),
        "done": torch.zeros(batch_size, seq_len),
        "is_first": torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]])[:batch_size, :seq_len],
    }


def test_slow_context_module_vector_shape():
    module = SlowContextModule(base_feat_dim=10, config=ThickContextConfig(enabled=True, context_dim=6, hidden_dim=12))
    base_feat = torch.randn(4, 10)
    context, gate, candidate, augmented = module(base_feat)
    assert context.shape == (4, 6)
    assert gate.shape == (4, 1)
    assert candidate.shape == (4, 6)
    assert augmented.shape == (4, 16)


def test_slow_context_module_sequence_shape():
    module = SlowContextModule(base_feat_dim=10, config=ThickContextConfig(enabled=True, context_dim=6, hidden_dim=12))
    base_feat = torch.randn(3, 5, 10)
    context, gate, candidate, augmented = module(base_feat)
    assert context.shape == (3, 5, 6)
    assert gate.shape == (3, 5, 1)
    assert candidate.shape == (3, 5, 6)
    assert augmented.shape == (3, 5, 16)


def test_gate_range_is_valid():
    module = SlowContextModule(base_feat_dim=7, config=ThickContextConfig(enabled=True, context_dim=5, hidden_dim=9))
    _, gate, _, _ = module(torch.randn(8, 7))
    assert torch.all(gate >= 0.0)
    assert torch.all(gate <= 1.0)


def test_sigmoid_gate_type_matches_previous_soft_behavior():
    module = SlowContextModule(
        base_feat_dim=7,
        config=_make_thick_context(True, context_dim=5, hidden_dim=9, gate_type="sigmoid"),
    )
    details = module.forward_details(torch.randn(8, 7))
    assert details["gate_type"] == "sigmoid"
    assert torch.allclose(details["gate"], details["gate_soft"])
    assert torch.allclose(details["gate"], details["gate_hard"])
    assert torch.allclose(details["gate"], details["open_prob"])


def test_l0_st_gate_range_is_valid():
    torch.manual_seed(0)
    module = SlowContextModule(
        base_feat_dim=7,
        config=_make_thick_context(
            True,
            context_dim=5,
            hidden_dim=9,
            gate_type="l0_st",
            l0_temperature=0.5,
            l0_noise=True,
            l0_straight_through=True,
        ),
    )
    details = module.forward_details(torch.randn(8, 7))
    assert details["gate_type"] == "l0_st"
    assert torch.all(details["gate"] >= 0.0)
    assert torch.all(details["gate"] <= 1.0)
    assert torch.all(details["gate_soft"] >= 0.0)
    assert torch.all(details["gate_soft"] <= 1.0)
    assert torch.all(details["gate_hard"] >= 0.0)
    assert torch.all(details["gate_hard"] <= 1.0)
    assert torch.isfinite(details["open_prob"]).all()


def test_is_first_resets_context():
    module = SlowContextModule(base_feat_dim=7, config=ThickContextConfig(enabled=True, context_dim=5, hidden_dim=9))
    base_feat = torch.randn(2, 3, 7)
    prev_context = torch.randn(2, 5)
    is_first = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    details = module.forward_details(base_feat, prev_context=prev_context, is_first=is_first)
    assert torch.allclose(details["prev_context"][0, 1], torch.zeros(5), atol=1e-6)
    assert torch.allclose(details["prev_context"][1, 1], details["context"][1, 0], atol=1e-6)


def test_l0_st_sequence_shape_and_reset():
    torch.manual_seed(1)
    module = SlowContextModule(
        base_feat_dim=7,
        config=_make_thick_context(
            True,
            context_dim=5,
            hidden_dim=9,
            gate_type="l0_st",
            l0_temperature=0.5,
            l0_noise=False,
            l0_straight_through=True,
        ),
    )
    base_feat = torch.randn(2, 4, 7)
    prev_context = torch.randn(2, 5)
    is_first = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    details = module.forward_details(base_feat, prev_context=prev_context, is_first=is_first)
    assert details["context"].shape == (2, 4, 5)
    assert details["gate"].shape == (2, 4, 1)
    assert details["gate_soft"].shape == (2, 4, 1)
    assert details["gate_hard"].shape == (2, 4, 1)
    assert details["gate_logit"].shape == (2, 4, 1)
    assert details["open_prob"].shape == (2, 4, 1)
    assert torch.allclose(details["prev_context"][0, 1], torch.zeros(5), atol=1e-6)
    assert torch.allclose(details["prev_context"][1, 2], torch.zeros(5), atol=1e-6)


def test_update_penalty_backward_and_metrics():
    model = _make_world_model(enabled=True)
    batch = _make_batch()
    loss_cfg = WorldModelLossConfig(
        use_symlog_obs=True,
        use_symlog_reward=True,
        use_twohot_reward=True,
        twohot_bins=31,
        twohot_low=-20.0,
        twohot_high=20.0,
        context_update_penalty=0.001,
    )
    outputs = model(batch, loss_cfg)
    outputs["loss"].backward()
    assert model.slow_context is not None
    assert model.slow_context.gate_net[0].weight.grad is not None
    assert "context_update_loss" in outputs["metrics"]
    assert "context_update_loss_raw" in outputs["metrics"]
    assert "context_update_loss_scaled" in outputs["metrics"]
    assert "model_loss_without_context_penalty" in outputs["metrics"]
    assert "model_loss_with_context_penalty" in outputs["metrics"]
    assert "context_gate_mean" in outputs["metrics"]
    assert "context_delta_norm_mean" in outputs["metrics"]
    assert torch.isfinite(outputs["metrics"]["context_update_loss"])
    raw = outputs["metrics"]["context_update_loss_raw"]
    scaled = outputs["metrics"]["context_update_loss_scaled"]
    without_penalty = outputs["metrics"]["model_loss_without_context_penalty"]
    with_penalty = outputs["metrics"]["model_loss_with_context_penalty"]
    assert torch.allclose(raw, outputs["metrics"]["context_update_loss"])
    assert torch.allclose(with_penalty, without_penalty + scaled, atol=1e-6, rtol=1e-6)


def test_l0_st_update_penalty_backward_and_metrics():
    torch.manual_seed(0)
    model = _make_world_model(
        enabled=True,
        thick_context=_make_thick_context(
            True,
            gate_type="l0_st",
            l0_temperature=0.5,
            l0_noise=True,
            l0_straight_through=True,
        ),
    )
    batch = _make_batch()
    loss_cfg = WorldModelLossConfig(
        use_symlog_obs=True,
        use_symlog_reward=True,
        use_twohot_reward=True,
        twohot_bins=31,
        twohot_low=-20.0,
        twohot_high=20.0,
        context_update_penalty=0.001,
    )
    outputs = model(batch, loss_cfg)
    outputs["loss"].backward()
    metrics = outputs["metrics"]
    assert model.slow_context is not None
    assert model.slow_context.gate_net[0].weight.grad is not None
    assert "context_l0_open_prob" in metrics
    assert "context_gate_hard_mean" in metrics
    assert "context_gate_soft_mean" in metrics
    assert "context_gate_logit_mean" in metrics
    assert "context_gate_logit_std" in metrics
    assert torch.isfinite(metrics["context_l0_open_prob"])
    assert torch.isfinite(metrics["context_gate_hard_mean"])
    assert torch.isfinite(metrics["context_gate_soft_mean"])
    assert torch.isfinite(metrics["context_gate_logit_mean"])
    assert torch.isfinite(metrics["context_gate_logit_std"])
    assert torch.allclose(metrics["context_update_loss_raw"], metrics["context_l0_open_prob"], atol=1e-6, rtol=1e-6)


def test_disabled_context_keeps_original_feat_dim():
    cfg = _make_model_config(enabled=False)
    agent = DreamerV3Agent(cfg)
    state = agent.world_model.rssm.init_state(2, device="cpu")
    feat = agent.world_model.get_augmented_feat(state)
    assert cfg.base_feat_dim == cfg.augmented_feat_dim
    assert feat.shape == (2, cfg.base_feat_dim)
    assert agent.actor.config.feat_dim == cfg.base_feat_dim


def test_enabled_context_expands_feat_dim():
    cfg = _make_model_config(enabled=True)
    agent = DreamerV3Agent(cfg)
    state = agent.world_model.rssm.init_state(2, device="cpu")
    details = agent.world_model.get_augmented_feat(state, return_details=True)
    assert cfg.augmented_feat_dim == cfg.base_feat_dim + cfg.thick_context.context_dim
    assert details["augmented_feat"].shape == (2, cfg.augmented_feat_dim)
    assert agent.actor.config.feat_dim == cfg.augmented_feat_dim


def test_no_cooldown_or_refractory_fields():
    assert "cooldown_penalty" not in ThickContextConfig.__dataclass_fields__
    assert "refractory_penalty" not in ThickContextConfig.__dataclass_fields__
    assert "cooldown_penalty" not in WorldModelLossConfig.__dataclass_fields__
    outputs = _make_world_model(enabled=True)(
        _make_batch(),
        WorldModelLossConfig(
            context_update_penalty=0.001,
            twohot_bins=31,
            twohot_low=-20.0,
            twohot_high=20.0,
        ),
    )
    assert not any("cooldown" in key or "refractory" in key for key in outputs["metrics"])


def test_visualize_script_help_runs():
    script = ROOT / "scripts" / "62_visualize_thick_context.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()
