from pathlib import Path

from policy.make_policy import make_policy
from policy.registry import maybe_record_gate_eval_videos_for_policy, resolve_policy_spec
from sim_env.envs.registry import load_env_components, resolve_env_version


class _DummyEnv:
    action_dim = 2
    obs_dim = 4

    def reset(self):
        return [0.0, 0.0, 0.0, 0.0]


def test_policy_registry_defaults_to_v1():
    spec = resolve_policy_spec({"type": "random"})
    assert spec.version == "v1"
    assert spec.module_path == "policy.versions.v1.random"


def test_policy_registry_supports_inline_version():
    spec = resolve_policy_spec({"type": "dreamerv3@v1"})
    assert spec.version == "v1"
    assert spec.module_path == "policy.versions.v1.dreamerv3"


def test_policy_registry_supports_ego_snapshot():
    spec = resolve_policy_spec({"type": "EGO", "version": "v1"})
    assert spec.version == "v1"
    assert spec.module_path == "policy.versions.v1.EGO"
    assert spec.gate_video_module_path == "policy.versions.v1.EGO.gate_video_recorder"


def test_make_policy_uses_versioned_snapshot():
    policy = make_policy(
        {
            "type": "random",
            "version": "v1",
            "checkpoint": {"save": False},
            "random": {"action_low": -1.0, "action_high": 1.0},
        },
        env=_DummyEnv(),
        exam_dir=Path("/tmp/exam"),
    )
    assert policy.__class__.__module__.startswith("policy.versions.v1.random")


def test_gate_video_dispatch_for_non_dreamer_returns_none():
    result = maybe_record_gate_eval_videos_for_policy(
        {"type": "random", "version": "v1"},
        agent=None,
        env=None,
        output_dir=Path("/tmp"),
        global_step=0,
        config={},
        device=None,
    )
    assert result is None


def test_env_registry_defaults_to_v1():
    assert resolve_env_version({"type": "metaworld"}) == "v1"


def test_env_registry_loads_versioned_components():
    components = load_env_components({"type": "metaworld", "version": "v1"})
    assert components["MetaWorldEnv"].__module__ == "sim_env.envs.versions.v1.metaworld_env"
    assert components["PersistentExplorationEnv"].__module__ == "sim_env.envs.versions.v1.persistent_exploration_env"
