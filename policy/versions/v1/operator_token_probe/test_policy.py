import numpy as np
import torch

from policy.versions.v1.operator_token_probe.operator_token_probe_policy_impl import OperatorTokenProbePolicyFolder


def _config():
    return {
        "policy": {
            "type": "operator_token_probe",
            "checkpoint": {"save": False},
            "operator_token_probe": {
                "device": "cpu",
                "allow_cpu_fallback": True,
                "source_checkpoint": "dummy.ckpt",
                "train_steps": 1,
                "collect_steps": 8,
                "batch_size": 2,
                "seq_len": 4,
            },
        },
        "operator_token": {
            "enabled": True,
            "num_tokens": 4,
            "token_dim": 8,
            "hidden_dim": 16,
        },
    }


def test_operator_token_probe_policy_act_shape():
    policy = OperatorTokenProbePolicyFolder(action_dim=3, observation_example=np.zeros((3, 8, 8), dtype=np.float32), policy_config=_config())
    action = policy.act(np.zeros((3, 8, 8), dtype=np.float32))
    assert action.shape == (3,)
    assert action.dtype == np.float32


def test_operator_token_probe_policy_update_metrics_payload():
    policy = OperatorTokenProbePolicyFolder(action_dim=2, observation_example=np.zeros((3, 8, 8), dtype=np.float32), policy_config=_config())
    batch = {
        "x_t": torch.randn(2, 3, 6),
        "x_tp1": torch.randn(2, 3, 6),
        "delta_x": torch.randn(2, 3, 6),
        "action_t": torch.randn(2, 3, 2),
        "capacity_gate": torch.ones(2, 3, 1),
    }
    payload = policy.update(batch)
    assert payload["schema"] == "policy_metrics/v1"
    assert "effect_loss_token" in payload["scalars"]
    assert "token_perplexity" in payload["scalars"]
