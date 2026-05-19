from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from policy.dreamerv3.operator_tokens import OperatorTokenConfig


@dataclass(slots=True)
class OperatorTokenProbeCheckpointConfig:
    load: bool = False
    path: str | None = None
    save: bool = True


@dataclass(slots=True)
class OperatorTokenProbePolicyConfig:
    device: str = "cuda"
    allow_cpu_fallback: bool = False
    source_checkpoint: str | None = None
    source_run_dir: str | None = None
    output_dir: str = "outputs/operator_token_probe/seed_0"
    collect_steps: int = 2000
    train_steps: int = 3000
    batch_size: int = 32
    seq_len: int = 32
    learning_rate: float = 3e-4
    max_episode_steps: int = 200
    deterministic: bool = True
    num_eval_batches: int = 16
    num_example_records: int = 128
    num_examples_per_token: int = 8
    operator_token: OperatorTokenConfig = field(default_factory=OperatorTokenConfig)

    def validate(self):
        if self.device not in ("cpu", "cuda"):
            raise ValueError("operator_token_probe.device must be 'cpu' or 'cuda'")
        if self.collect_steps <= 0:
            raise ValueError("operator_token_probe.collect_steps must be > 0")
        if self.train_steps <= 0:
            raise ValueError("operator_token_probe.train_steps must be > 0")
        if self.batch_size <= 0:
            raise ValueError("operator_token_probe.batch_size must be > 0")
        if self.seq_len <= 1:
            raise ValueError("operator_token_probe.seq_len must be > 1")
        if self.learning_rate <= 0.0:
            raise ValueError("operator_token_probe.learning_rate must be > 0")
        if self.max_episode_steps <= 0:
            raise ValueError("operator_token_probe.max_episode_steps must be > 0")
        if self.num_eval_batches <= 0:
            raise ValueError("operator_token_probe.num_eval_batches must be > 0")
        if self.num_example_records <= 0:
            raise ValueError("operator_token_probe.num_example_records must be > 0")
        if self.num_examples_per_token <= 0:
            raise ValueError("operator_token_probe.num_examples_per_token must be > 0")
        if not str(self.output_dir).strip():
            raise ValueError("operator_token_probe.output_dir must be a non-empty string")
        if self.source_checkpoint is None and self.source_run_dir is None:
            raise ValueError("operator_token_probe requires source_checkpoint or source_run_dir")
        self.operator_token.validate()

    def asdict(self) -> dict:
        return asdict(self)


def _maybe_full_config(config: dict | None) -> tuple[dict, dict]:
    root = dict(config or {})
    policy = dict(root.get("policy", root))
    return root, policy


def build_operator_token_probe_config(
    config: dict | None,
) -> tuple[OperatorTokenProbePolicyConfig, OperatorTokenProbeCheckpointConfig]:
    root, policy = _maybe_full_config(config)
    raw = dict(policy.get("operator_token_probe", {}))
    token_raw = dict(root.get("operator_token", raw.pop("operator_token", {})))
    checkpoint_raw = dict(policy.get("checkpoint", {}))
    cfg = OperatorTokenProbePolicyConfig(
        **raw,
        operator_token=OperatorTokenConfig(**token_raw),
    )
    cfg.validate()
    if cfg.device == "cuda" and not torch.cuda.is_available():
        if cfg.allow_cpu_fallback:
            print("[OperatorTokenProbe] CUDA unavailable, falling back to CPU because allow_cpu_fallback=true")
            cfg.device = "cpu"
        else:
            raise RuntimeError(
                "OperatorTokenProbe is configured with device='cuda' but CUDA is unavailable. "
                "Set policy.operator_token_probe.allow_cpu_fallback=true only for debug fallback."
            )
    checkpoint = OperatorTokenProbeCheckpointConfig(
        load=bool(checkpoint_raw.get("load", False)),
        path=checkpoint_raw.get("path"),
        save=bool(checkpoint_raw.get("save", True)),
    )
    return cfg, checkpoint


def resolve_probe_path(exam_dir: str | Path | None, path: str | Path | None) -> Path | None:
    if path is None:
        return None
    resolved = Path(path)
    if resolved.is_absolute() or exam_dir is None:
        return resolved
    return Path(exam_dir) / resolved
