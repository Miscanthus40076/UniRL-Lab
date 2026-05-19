from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("metaworld")

from sim_env.envs.metaworld_env import _CUSTOM_METAWORLD_ENVS
from sim_env.envs.sawyer_peg_insertion_side_sparse_v3 import SawyerPegInsertionSideSparseEnvV3


def test_sparse_env_aliases_registered():
    assert "peg-insert-side-sparse-v3" in _CUSTOM_METAWORLD_ENVS
    assert "peg-insert-side-reverse-sparse-v3" in _CUSTOM_METAWORLD_ENVS


def test_sparse_success_threshold():
    assert SawyerPegInsertionSideSparseEnvV3._success_from_obj_to_target(0.05, 0.07) is True
    assert SawyerPegInsertionSideSparseEnvV3._success_from_obj_to_target(0.08, 0.07) is False


def test_sparse_evaluate_state_returns_binary_reward():
    env = SawyerPegInsertionSideSparseEnvV3.__new__(SawyerPegInsertionSideSparseEnvV3)
    env.TARGET_RADIUS = 0.07
    env.obj_init_pos = np.array([0.0, 0.6, 0.02], dtype=np.float64)
    env._dense_terms = lambda action, obs: (
        3.5,
        0.01,
        0.5,
        0.05,
        1.0,
        0.8,
        0.7,
        0.9,
    )
    reward, info = SawyerPegInsertionSideSparseEnvV3.evaluate_state(
        env,
        obs=np.array([0, 0, 0, 0.5, 0.0, 0.6, 0.2], dtype=np.float64),
        action=np.zeros(4, dtype=np.float32),
    )
    assert reward == 1.0
    assert info["success"] == 1.0
    assert info["dense_reward"] == 3.5
    assert info["unscaled_reward"] == 1.0
