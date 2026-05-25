from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.train import load_exam_train_app
from sim_env.envs.base_env import BaseEnv
from sim_env.envs.make_env import _maybe_wrap_persistent
from sim_env.envs.persistent_exploration_env import PersistentExplorationConfig, PersistentExplorationEnv


class _DummyEnv(BaseEnv):
    def __init__(self):
        self._obs = np.zeros(4, dtype=np.float32)
        self._step = 0
        self.max_path_length = 3

    def reset(self):
        self._step = 0
        self._obs = np.zeros(4, dtype=np.float32)
        return self._obs.copy()

    def step(self, action):
        self._step += 1
        self._obs = self._obs + np.asarray(action, dtype=np.float32).reshape(-1)[:4]
        return self._obs.copy(), 0.0, False, {}

    @property
    def obs_dim(self):
        return 4

    @property
    def action_dim(self):
        return 4

    def render(self):
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def close(self):
        return None


def _step_env(env: PersistentExplorationEnv, action=None, diagnostics=None):
    action = np.zeros(4, dtype=np.float32) if action is None else np.asarray(action, dtype=np.float32)
    obs, reward, raw_done, info = env.step(action)
    result = env.finalize_step(
        action=action,
        next_obs=obs,
        raw_done=raw_done,
        info=info,
        diagnostics=diagnostics or {},
    )
    return obs, reward, raw_done, result


def test_persistent_disabled_keeps_original_env():
    env = _DummyEnv()
    wrapped = _maybe_wrap_persistent(env, {"persistent_exploration": {"enabled": False}})
    assert wrapped is env


def test_enabled_true_lifetime_accumulates():
    env = PersistentExplorationEnv(_DummyEnv(), PersistentExplorationConfig(enabled=True, max_lifetime_steps=10))
    env.reset()
    _step_env(env, action=np.ones(4, dtype=np.float32))
    _step_env(env, action=np.ones(4, dtype=np.float32))
    assert env.get_persistent_step_metrics()["lifetime_step"] == 2.0


def test_max_lifetime_forces_reset():
    env = PersistentExplorationEnv(_DummyEnv(), PersistentExplorationConfig(enabled=True, max_lifetime_steps=3))
    env.reset()
    for _ in range(2):
        _, _, _, result = _step_env(env, action=np.ones(4, dtype=np.float32))
        assert result["done"] is False
    _, _, _, result = _step_env(env, action=np.ones(4, dtype=np.float32))
    assert result["done"] is True
    assert result["reset_reason"] == "max_lifetime"


def test_env_done_does_not_reset_when_disabled():
    env = PersistentExplorationEnv(
        _DummyEnv(),
        PersistentExplorationConfig(enabled=True, max_lifetime_steps=10, reset_on_env_done=False),
    )
    env.reset()
    result = env.finalize_step(
        action=np.zeros(4, dtype=np.float32),
        next_obs=np.zeros(4, dtype=np.float32),
        raw_done=True,
        info={},
        diagnostics={},
    )
    assert result["done"] is False
    assert result["reset_reason"] == "NA"


def test_env_done_resets_when_enabled():
    env = PersistentExplorationEnv(
        _DummyEnv(),
        PersistentExplorationConfig(enabled=True, max_lifetime_steps=10, reset_on_env_done=True),
    )
    env.reset()
    result = env.finalize_step(
        action=np.zeros(4, dtype=np.float32),
        next_obs=np.zeros(4, dtype=np.float32),
        raw_done=True,
        info={},
        diagnostics={},
    )
    assert result["done"] is True
    assert result["reset_reason"] == "env_done"


def test_stale_window_resets_when_obs_change_too_small():
    env = PersistentExplorationEnv(
        _DummyEnv(),
        PersistentExplorationConfig(
            enabled=True,
            max_lifetime_steps=20,
            stale_window=2,
            stale_min_obs_change=0.001,
        ),
    )
    env.reset()
    _step_env(env, action=np.zeros(4, dtype=np.float32))
    _, _, _, result = _step_env(env, action=np.zeros(4, dtype=np.float32))
    assert result["done"] is True
    assert result["reset_reason"] == "stale"


def test_recent_operator_event_delays_stale_reset():
    env = PersistentExplorationEnv(
        _DummyEnv(),
        PersistentExplorationConfig(
            enabled=True,
            max_lifetime_steps=20,
            stale_window=2,
            stale_min_obs_change=0.001,
            operator_event_reset_delay_steps=3,
            operator_event_threshold=0.1,
        ),
    )
    env.reset()
    _step_env(env, action=np.zeros(4, dtype=np.float32), diagnostics={"operator_reward": 0.2})
    _, _, _, result = _step_env(env, action=np.zeros(4, dtype=np.float32))
    assert result["done"] is False
    _, _, _, result = _step_env(env, action=np.zeros(4, dtype=np.float32))
    assert result["done"] is False
    _, _, _, result = _step_env(env, action=np.zeros(4, dtype=np.float32))
    assert result["done"] is False
    _, _, _, result = _step_env(env, action=np.zeros(4, dtype=np.float32))
    assert result["done"] is True
    assert result["reset_reason"] == "stale"


def test_nan_obs_or_action_forces_reset():
    env = PersistentExplorationEnv(_DummyEnv(), PersistentExplorationConfig(enabled=True, max_lifetime_steps=20))
    env.reset()
    obs = np.full(4, np.nan, dtype=np.float32)
    result = env.finalize_step(action=np.zeros(4, dtype=np.float32), next_obs=obs, raw_done=False, info={}, diagnostics={})
    assert result["done"] is True
    assert result["reset_reason"] == "nan"

    env.reset()
    result = env.finalize_step(
        action=np.full(4, np.nan, dtype=np.float32),
        next_obs=np.zeros(4, dtype=np.float32),
        raw_done=False,
        info={},
        diagnostics={},
    )
    assert result["done"] is True
    assert result["reset_reason"] == "nan"


def test_reset_reason_and_post_event_stats_recorded():
    env = PersistentExplorationEnv(_DummyEnv(), PersistentExplorationConfig(enabled=True, max_lifetime_steps=4))
    env.reset()
    _step_env(env, action=np.ones(4, dtype=np.float32), diagnostics={"operator_reward": 0.2, "operator_token_id": 3})
    _step_env(env, action=np.ones(4, dtype=np.float32), diagnostics={"operator_token_id": 3})
    _step_env(env, action=np.ones(4, dtype=np.float32), diagnostics={"operator_token_id": 2})
    _, _, _, result = _step_env(env, action=np.ones(4, dtype=np.float32), diagnostics={"operator_token_id": 2})
    assert result["done"] is True
    stats = env.get_persistent_stats()
    assert stats["reset_reason_max_lifetime"] == 1.0
    assert stats["post_event_continuation_steps_mean"] > 0.0
    assert stats["episode_unique_token_count_mean"] >= 2.0
    assert stats["episode_token_perplexity_mean"] > 1.0


def test_persistent_exam_config_loads():
    app = load_exam_train_app("test_metaworld_dreamerv3_peg_insert_side_operator_reward_persistent_smoke")
    config = app.load_config()
    assert config["persistent_exploration"]["enabled"] is True
