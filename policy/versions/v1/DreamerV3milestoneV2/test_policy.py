import numpy as np

from policy.versions.v1.dreamerv3.dreamerv3_policy_impl import DreamerV3PolicyFolder


class DummyEnv:
    obs_dim = 8
    action_dim = 3


def _make_config():
    return {
        "type": "dreamerv3",
        "name": "test_dreamerv3",
        "checkpoint": {
            "load": False,
            "path": None,
            "save": False,
        },
        "dreamerv3": {
            "device": "cpu",
            "batch_size": 2,
            "seq_len": 4,
            "warmup_steps": 32,
            "train_every": 1,
            "train_ratio": 1.0,
            "hidden_dim": 32,
            "embed_dim": 16,
            "deter_dim": 16,
            "stoch_dim": 8,
            "stoch_classes": 8,
            "actor_hidden_dim": 32,
            "value_hidden_dim": 32,
        },
    }


def test_dreamerv3_policy_act():
    env = DummyEnv()
    obs = np.zeros(env.obs_dim, dtype=np.float32)
    policy = DreamerV3PolicyFolder(
        action_dim=env.action_dim,
        observation_example=obs,
        policy_config=_make_config(),
    )
    action = policy.act(obs)
    assert isinstance(action, np.ndarray)
    assert action.shape == (env.action_dim,)


def test_dreamerv3_policy_update_metrics_payload():
    env = DummyEnv()
    obs = np.zeros(env.obs_dim, dtype=np.float32)
    policy = DreamerV3PolicyFolder(
        action_dim=env.action_dim,
        observation_example=obs,
        policy_config=_make_config(),
    )
    _ = policy.act(obs)
    payload = policy.update(
        {
            "obs": obs,
            "action": np.zeros(env.action_dim, dtype=np.float32),
            "reward": 0.0,
            "next_obs": obs,
            "done": False,
            "info": {},
            "step": 1,
        }
    )
    assert isinstance(payload, dict)
    assert payload["schema"] == "policy_metrics/v1"
    assert isinstance(payload["scalars"], dict)
    assert isinstance(payload["metadata"], dict)
