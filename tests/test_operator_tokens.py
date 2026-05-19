from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import torch

from policy.dreamerv3.operator_tokens import (
    InverseActionProbe,
    NoTokenEffectModel,
    NoTokenInverseActionProbe,
    OperatorEffectModel,
    OperatorTokenConfig,
    OperatorTokenPrior,
    OperatorTokenizerPosterior,
    compute_token_usage_metrics,
    masked_mean,
)
from policy.operator_token_probe.operator_token_probe_model import OperatorTokenProbeModel


def _cfg(**overrides):
    raw = {
        "enabled": True,
        "num_tokens": 4,
        "token_dim": 8,
        "hidden_dim": 16,
        "gumbel_temperature": 1.0,
        "straight_through": True,
        "train_on_capacity_only": True,
        "effect_loss_scale": 1.0,
        "inverse_loss_scale": 0.5,
        "prior_loss_scale": 0.1,
        "usage_entropy_scale": 0.01,
        "detach_features": True,
    }
    raw.update(overrides)
    return OperatorTokenConfig(**raw)


def test_operator_tokenizer_shapes():
    cfg = _cfg()
    module = OperatorTokenizerPosterior(feat_dim=6, action_dim=3, config=cfg)
    x_t = torch.randn(2, 5, 6)
    x_tp1 = torch.randn(2, 5, 6)
    delta_x = x_tp1 - x_t
    action = torch.randn(2, 5, 3)
    out = module(x_t, x_tp1, delta_x, action)
    assert out["token_logits"].shape == (2, 5, 4)
    assert out["token_probs"].shape == (2, 5, 4)
    assert out["token_onehot_st"].shape == (2, 5, 4)
    assert out["token_id"].shape == (2, 5)


def test_gumbel_st_onehot_and_backward():
    cfg = _cfg(straight_through=True)
    module = OperatorTokenizerPosterior(feat_dim=4, action_dim=2, config=cfg)
    module.train()
    x_t = torch.randn(2, 3, 4)
    x_tp1 = torch.randn(2, 3, 4)
    delta_x = x_tp1 - x_t
    action = torch.randn(2, 3, 2)
    out = module(x_t, x_tp1, delta_x, action)
    assert torch.allclose(out["token_onehot_st"].sum(dim=-1), torch.ones(2, 3), atol=1e-5)
    loss = out["token_onehot_st"][..., 0].sum()
    loss.backward()
    grads = [param.grad for param in module.parameters() if param.requires_grad]
    assert any(grad is not None and torch.count_nonzero(grad) > 0 for grad in grads)


def test_operator_token_prior_shape():
    cfg = _cfg()
    module = OperatorTokenPrior(feat_dim=6, action_dim=3, config=cfg)
    out = module(torch.randn(2, 5, 6), torch.randn(2, 5, 3))
    assert out.shape == (2, 5, 4)


def test_effect_models_shape():
    token = OperatorEffectModel(feat_dim=6, action_dim=3, token_dim=8, hidden_dim=16)
    no_token = NoTokenEffectModel(feat_dim=6, action_dim=3, hidden_dim=16)
    x_t = torch.randn(2, 5, 6)
    action = torch.randn(2, 5, 3)
    token_emb = torch.randn(2, 5, 8)
    assert token(x_t, action, token_emb).shape == (2, 5, 6)
    assert no_token(x_t, action).shape == (2, 5, 6)


def test_inverse_probes_shape():
    inverse = InverseActionProbe(feat_dim=6, action_dim=3, token_dim=8, hidden_dim=16)
    baseline = NoTokenInverseActionProbe(feat_dim=6, action_dim=3, hidden_dim=16)
    x_t = torch.randn(2, 5, 6)
    delta_x = torch.randn(2, 5, 6)
    token_emb = torch.randn(2, 5, 8)
    assert inverse(x_t, delta_x, token_emb).shape == (2, 5, 3)
    assert baseline(x_t, delta_x).shape == (2, 5, 3)


def test_masked_loss_uses_only_active_transitions():
    values = torch.tensor([[1.0, 5.0, 9.0]])
    mask = torch.tensor([[0.0, 1.0, 0.0]])
    result = masked_mean(values, mask)
    assert torch.isclose(result, torch.tensor(5.0))


def test_token_perplexity_no_nan():
    probs = torch.tensor(
        [
            [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]],
            [[0.2, 0.3, 0.5], [0.3, 0.3, 0.4]],
        ],
        dtype=torch.float32,
    )
    mask = torch.ones(2, 2)
    metrics = compute_token_usage_metrics(probs, mask)
    assert torch.isfinite(metrics["token_perplexity"])
    assert metrics["token_perplexity"].item() > 0.0


def test_usage_entropy_backward():
    probs = torch.tensor(
        [[[0.25, 0.25, 0.25, 0.25], [0.1, 0.2, 0.3, 0.4]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    mask = torch.ones(1, 2)
    metrics = compute_token_usage_metrics(probs, mask)
    (-0.01 * metrics["token_entropy"]).backward()
    assert probs.grad is not None


def test_detach_features_blocks_input_gradients():
    model = OperatorTokenProbeModel(feat_dim=6, action_dim=3, config=_cfg(detach_features=True))
    batch = {
        "x_t": torch.randn(2, 5, 6, requires_grad=True),
        "x_tp1": torch.randn(2, 5, 6, requires_grad=True),
        "delta_x": torch.randn(2, 5, 6, requires_grad=True),
        "action_t": torch.randn(2, 5, 3, requires_grad=True),
    }
    outputs = model(batch)
    loss = outputs["pred_delta_x"].sum() + outputs["pred_action"].sum()
    loss.backward()
    assert batch["x_t"].grad is None
    assert batch["x_tp1"].grad is None
    assert batch["delta_x"].grad is None
    assert batch["action_t"].grad is None


def test_no_privileged_label_dependencies():
    source = Path("policy/dreamerv3/operator_tokens.py").read_text(encoding="utf-8")
    lowered = source.lower()
    for forbidden in ("success", "contact", "inserted", "ball_in_cup"):
        assert forbidden not in lowered


def test_operator_token_probe_script_help_runs():
    result = subprocess.run(
        [sys.executable, "scripts/64_train_operator_token_probe.py", "--help"],
        cwd="/home/y/Simer",
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--run-dir" in result.stdout
