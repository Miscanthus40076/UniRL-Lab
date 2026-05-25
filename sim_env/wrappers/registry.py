from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any


@dataclass(frozen=True, slots=True)
class WrapperSpec:
    wrapper_type: str
    version: str
    module_path: str
    class_name: str


_WRAPPER_SPECS: dict[str, dict[str, WrapperSpec]] = {
    "simer_to_embodied": {
        "v1": WrapperSpec(
            wrapper_type="simer_to_embodied",
            version="v1",
            module_path="sim_env.wrappers.simer_to_embodied_v1",
            class_name="SimerToEmbodiedEnv",
        ),
    },
}

_DEFAULT_WRAPPER_VERSION = {
    wrapper_type: "v1"
    for wrapper_type in _WRAPPER_SPECS
}


def _normalise_wrapper_entries(config: dict | list | None) -> list[dict[str, Any]]:
    if config is None:
        return []
    if isinstance(config, list):
        entries = config
    elif isinstance(config, dict):
        if "chain" in config:
            entries = config["chain"]
        elif "wrappers" in config:
            entries = config["wrappers"]
        else:
            entries = [config]
    else:
        raise TypeError("wrapper config must be a mapping or list")
    if not isinstance(entries, list):
        raise TypeError("wrapper chain must be a list")
    normalised = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise TypeError(f"wrapper chain entry {index} must be a mapping")
        normalised.append(dict(entry))
    return normalised


def wrapper_config_from_exam(config: dict) -> list[dict[str, Any]]:
    if "wrappers" in config:
        return _normalise_wrapper_entries(config.get("wrappers"))
    if "wrapper" in config:
        return _normalise_wrapper_entries(config.get("wrapper"))
    return []


def resolve_wrapper_specs(config: dict | list | None) -> list[WrapperSpec]:
    specs = []
    for entry in _normalise_wrapper_entries(config):
        wrapper_type = str(entry.get("type", "")).strip()
        if not wrapper_type:
            raise ValueError("wrapper entry must contain type")
        if wrapper_type not in _WRAPPER_SPECS:
            supported = ", ".join(sorted(_WRAPPER_SPECS))
            raise ValueError(f"Unsupported wrapper type: {wrapper_type}. Supported: {supported}")
        version = str(entry.get("version") or _DEFAULT_WRAPPER_VERSION[wrapper_type])
        versions = _WRAPPER_SPECS[wrapper_type]
        if version not in versions:
            supported_versions = ", ".join(sorted(versions))
            raise ValueError(
                f"Unsupported wrapper version for {wrapper_type}: {version}. "
                f"Supported: {supported_versions}"
            )
        specs.append(versions[version])
    return specs


def load_wrapper_class(entry: dict) -> type:
    spec = resolve_wrapper_specs([entry])[0]
    module = import_module(spec.module_path)
    return getattr(module, spec.class_name)


def apply_wrappers(env, config: dict | list | None):
    wrapped = env
    for entry in _normalise_wrapper_entries(config):
        wrapper_cls = load_wrapper_class(entry)
        kwargs = {key: value for key, value in entry.items() if key not in {"type", "version"}}
        wrapped = wrapper_cls(wrapped, **kwargs)
    return wrapped
