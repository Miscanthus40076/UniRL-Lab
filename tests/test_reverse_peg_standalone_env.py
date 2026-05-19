from __future__ import annotations

import inspect

import pytest

pytest.importorskip("metaworld")

from sim_env.envs.make_env import make_env
from sim_env.envs.registry import load_env_components
from sim_env.envs.versions.v1.peg_insert_side_reverse_sparse import (
    PegInsertSideReverseSparseStandaloneEnv,
    SawyerPegInsertionSideReverseSparseStandaloneV1,
)


def _standalone_config():
    return {
        "type": "peg_insert_side_reverse_sparse",
        "version": "v1",
        "observation": {
            "type": "vector",
            "num_cams": 1,
            "camera_indices": None,
        },
        "render": {
            "enabled": False,
            "height": 64,
            "width": 64,
            "camera_id": 1,
            "camera_indices": None,
            "backend": None,
            "backend_priority": ["egl", "osmesa"],
            "allow_software_render_fallback": False,
        },
        "metaworld": {
            "seed": 0,
            "reward_function_version": "v2",
        },
    }


def test_standalone_reverse_sparse_component_is_registered():
    components = load_env_components({"type": "peg_insert_side_reverse_sparse", "version": "v1"})
    assert components["PegInsertSideReverseSparseStandaloneEnv"] is PegInsertSideReverseSparseStandaloneEnv


def test_standalone_reverse_sparse_does_not_import_forward_wrapper():
    source = inspect.getsource(PegInsertSideReverseSparseStandaloneEnv)
    task_source = inspect.getsource(SawyerPegInsertionSideReverseSparseStandaloneV1)
    combined = source + task_source
    assert "metaworld_env" not in combined
    assert "sawyer_peg_insertion_side_sparse_v3" not in combined


def test_standalone_reverse_sparse_factory_smoke():
    env = make_env(_standalone_config())
    try:
        obs = env.reset()
        assert obs is not None
        assert env.action_dim > 0
    finally:
        env.close()
