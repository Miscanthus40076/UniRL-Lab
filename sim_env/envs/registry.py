from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True, slots=True)
class EnvComponentSpec:
    version: str
    dm_control_module: str
    dm_control_class: str
    ball_in_cup_module: str
    ball_in_cup_class: str
    metaworld_module: str
    metaworld_class: str
    peg_reverse_sparse_module: str
    peg_reverse_sparse_class: str
    persistent_module: str
    persistent_config_class: str
    persistent_env_class: str


_ENV_COMPONENT_SPECS: dict[str, EnvComponentSpec] = {
    "v1": EnvComponentSpec(
        version="v1",
        dm_control_module="sim_env.envs.versions.v1.dm_control_env",
        dm_control_class="DMControlEnv",
        ball_in_cup_module="sim_env.envs.versions.v1.ball_in_cup_take_out_env",
        ball_in_cup_class="BallInCupTakeOutEnv",
        metaworld_module="sim_env.envs.versions.v1.metaworld_env",
        metaworld_class="MetaWorldEnv",
        peg_reverse_sparse_module="sim_env.envs.versions.v1.peg_insert_side_reverse_sparse",
        peg_reverse_sparse_class="PegInsertSideReverseSparseStandaloneEnv",
        persistent_module="sim_env.envs.versions.v1.persistent_exploration_env",
        persistent_config_class="PersistentExplorationConfig",
        persistent_env_class="PersistentExplorationEnv",
    )
}

_DEFAULT_ENV_VERSION = "v1"


def resolve_env_version(config: dict) -> str:
    return str(config.get("version", _DEFAULT_ENV_VERSION))


def load_env_components(config: dict) -> dict[str, type]:
    version = resolve_env_version(config)
    if version not in _ENV_COMPONENT_SPECS:
        supported = ", ".join(sorted(_ENV_COMPONENT_SPECS))
        raise ValueError(f"Unsupported env version: {version}. Supported: {supported}")
    spec = _ENV_COMPONENT_SPECS[version]

    def _load(module_path: str, class_name: str):
        module = import_module(module_path)
        return getattr(module, class_name)

    components = {
        "DMControlEnv": _load(spec.dm_control_module, spec.dm_control_class),
        "BallInCupTakeOutEnv": _load(spec.ball_in_cup_module, spec.ball_in_cup_class),
        "MetaWorldEnv": _load(spec.metaworld_module, spec.metaworld_class),
        "PersistentExplorationConfig": _load(spec.persistent_module, spec.persistent_config_class),
        "PersistentExplorationEnv": _load(spec.persistent_module, spec.persistent_env_class),
    }
    if str(config.get("type", "")) == "peg_insert_side_reverse_sparse":
        components["PegInsertSideReverseSparseStandaloneEnv"] = _load(
            spec.peg_reverse_sparse_module,
            spec.peg_reverse_sparse_class,
        )
    return components
