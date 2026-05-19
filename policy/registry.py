from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Callable, Type


@dataclass(frozen=True, slots=True)
class PolicyVersionSpec:
    policy_type: str
    version: str
    module_path: str
    class_name: str
    gate_video_module_path: str | None = None


_POLICY_SPECS: dict[str, dict[str, PolicyVersionSpec]] = {
    "random": {
        "v1": PolicyVersionSpec(
            policy_type="random",
            version="v1",
            module_path="policy.versions.v1.random",
            class_name="RandomPolicyFolder",
        ),
    },
    "mlp": {
        "v1": PolicyVersionSpec(
            policy_type="mlp",
            version="v1",
            module_path="policy.versions.v1.mlp",
            class_name="MLPPolicyFolder",
        ),
    },
    "dreamerv3": {
        "v1": PolicyVersionSpec(
            policy_type="dreamerv3",
            version="v1",
            module_path="policy.versions.v1.dreamerv3",
            class_name="DreamerV3PolicyFolder",
            gate_video_module_path="policy.versions.v1.dreamerv3.gate_video_recorder",
        ),
    },
    "DreamerV3milestoneV2": {
        "v1": PolicyVersionSpec(
            policy_type="DreamerV3milestoneV2",
            version="v1",
            module_path="policy.versions.v1.DreamerV3milestoneV2",
            class_name="DreamerV3PolicyFolder",
            gate_video_module_path="policy.versions.v1.DreamerV3milestoneV2.gate_video_recorder",
        ),
    },
    "dreamerv3_milestoneV2": {
        "v1": PolicyVersionSpec(
            policy_type="dreamerv3_milestoneV2",
            version="v1",
            module_path="policy.dreamerv3_milestoneV2",
            class_name="DreamerV3PolicyFolder",
            gate_video_module_path="policy.dreamerv3_milestoneV2.gate_video_recorder",
        ),
    },
    "dreamerv3milestoneV2": {
        "v1": PolicyVersionSpec(
            policy_type="dreamerv3milestoneV2",
            version="v1",
            module_path="policy.dreamerv3_milestoneV2",
            class_name="DreamerV3PolicyFolder",
            gate_video_module_path="policy.dreamerv3_milestoneV2.gate_video_recorder",
        ),
    },
    "dreamerv3_milestoneV3": {
        "v1": PolicyVersionSpec(
            policy_type="dreamerv3_milestoneV3",
            version="v1",
            module_path="policy.dreamerv3_milestoneV3",
            class_name="DreamerV3PolicyFolder",
            gate_video_module_path="policy.dreamerv3_milestoneV3.gate_video_recorder",
        ),
    },
    "DreamerV3milestoneV3": {
        "v1": PolicyVersionSpec(
            policy_type="DreamerV3milestoneV3",
            version="v1",
            module_path="policy.dreamerv3_milestoneV3",
            class_name="DreamerV3PolicyFolder",
            gate_video_module_path="policy.dreamerv3_milestoneV3.gate_video_recorder",
        ),
    },
    "dreamerv3milestoneV1": {
        "v2": PolicyVersionSpec(
            policy_type="dreamerv3milestoneV1",
            version="v2",
            module_path="policy.dreamerv3milestoneV1",
            class_name="DreamerV3PolicyFolder",
            gate_video_module_path="policy.dreamerv3milestoneV1.gate_video_recorder",
        ),
    },
    "EGO": {
        "v1": PolicyVersionSpec(
            policy_type="EGO",
            version="v1",
            module_path="policy.versions.v1.EGO",
            class_name="DreamerV3PolicyFolder",
            gate_video_module_path="policy.versions.v1.EGO.gate_video_recorder",
        ),
    },
    "operator_token_probe": {
        "v1": PolicyVersionSpec(
            policy_type="operator_token_probe",
            version="v1",
            module_path="policy.versions.v1.operator_token_probe",
            class_name="OperatorTokenProbePolicyFolder",
        ),
    },
}

_DEFAULT_POLICY_VERSION = {
    policy_type: "v1"
    for policy_type in _POLICY_SPECS
}


def _split_type_and_version(raw_policy_type: str) -> tuple[str, str | None]:
    text = str(raw_policy_type)
    if "@" not in text:
        return text, None
    policy_type, version = text.split("@", 1)
    return policy_type, version or None


def resolve_policy_spec(config: dict) -> PolicyVersionSpec:
    raw_policy_type = str(config["type"])
    policy_type, inline_version = _split_type_and_version(raw_policy_type)
    requested_version = config.get("version", inline_version) or _DEFAULT_POLICY_VERSION.get(policy_type)
    if policy_type not in _POLICY_SPECS:
        raise ValueError(f"Unsupported policy type: {raw_policy_type}")
    versions = _POLICY_SPECS[policy_type]
    if requested_version not in versions:
        supported = ", ".join(sorted(versions))
        raise ValueError(f"Unsupported policy version for {policy_type}: {requested_version}. Supported: {supported}")
    return versions[str(requested_version)]


def load_policy_class(config: dict) -> Type:
    spec = resolve_policy_spec(config)
    module = import_module(spec.module_path)
    return getattr(module, spec.class_name)


def maybe_record_gate_eval_videos_for_policy(
    policy_config: dict,
    *,
    agent,
    env,
    output_dir,
    global_step: int,
    config: dict,
    device: str | None = None,
):
    spec = resolve_policy_spec(policy_config)
    if spec.gate_video_module_path is None:
        return None
    module = import_module(spec.gate_video_module_path)
    fn: Callable = getattr(module, "maybe_record_gate_eval_videos")
    return fn(
        agent=agent,
        env=env,
        output_dir=output_dir,
        global_step=global_step,
        config=config,
        device=device,
    )
