from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from policy.base_policy import BasePolicy
from policy.dreamerv3.dreamerv3_policy_impl import DreamerV3PolicyFolder
from policy.dreamerv3.replay_buffer import EpisodeReplayBuffer
from scripts.utils import load_config as load_exam_config
from sim_env.envs.make_env import make_env

from .config import build_operator_token_probe_config, resolve_probe_path
from .operator_token_probe_agent import OperatorTokenProbeAgent
from .processor import OperatorTokenProbeProcessor


class OperatorTokenProbePolicyFolder(BasePolicy):
    def __init__(self, action_dim: int, observation_example=None, policy_config: dict | None = None, exam_dir: str | Path | None = None):
        self.action_dim = int(action_dim)
        self.exam_dir = Path(exam_dir) if exam_dir is not None else None
        self.cfg, self.checkpoint_cfg = build_operator_token_probe_config(policy_config)
        self.processor = OperatorTokenProbeProcessor(device=self.cfg.device)
        self.agent: OperatorTokenProbeAgent | None = None
        self.observation_example = observation_example

    def initialize(self, feat_dim: int):
        if self.agent is None:
            self.agent = OperatorTokenProbeAgent(feat_dim=feat_dim, action_dim=self.action_dim, config=self.cfg)

    def act(self, obs, deterministic: bool = False):
        self._prepare_image_observation(obs)
        return np.zeros(self.action_dim, dtype=np.float32)

    def update(self, step_batch: dict) -> dict[str, object]:
        batch = self.processor.prepare_transition_batch(step_batch)
        self.initialize(int(batch["x_t"].shape[-1]))
        assert self.agent is not None
        scalars, metadata = self.agent.update(batch)
        return self._build_metrics_payload(scalars=scalars, metadata=metadata)

    def evaluate(self, step_batch: dict) -> dict[str, float | list[float]]:
        batch = self.processor.prepare_transition_batch(step_batch)
        self.initialize(int(batch["x_t"].shape[-1]))
        assert self.agent is not None
        return self.agent.evaluate(batch)

    def save(self, path: str | Path):
        if self.agent is None:
            raise RuntimeError("OperatorTokenProbePolicyFolder has not been initialized")
        resolved = resolve_probe_path(self.exam_dir, path)
        assert resolved is not None
        resolved.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_type": "operator_token_probe",
                "action_dim": self.action_dim,
                "feat_dim": int(self.agent.model.feat_dim),
                "agent": self.agent.state_dict(),
            },
            resolved,
        )

    def load(self, path: str | Path):
        resolved = resolve_probe_path(self.exam_dir, path)
        assert resolved is not None
        payload = torch.load(resolved, map_location=self.cfg.device)
        if payload.get("policy_type") != "operator_token_probe":
            raise ValueError("Checkpoint is not an operator_token_probe checkpoint")
        if int(payload["action_dim"]) != self.action_dim:
            raise ValueError("OperatorTokenProbe action_dim mismatch")
        self.initialize(int(payload["feat_dim"]))
        assert self.agent is not None
        self.agent.load_state_dict(payload["agent"])


def _deep_update(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_source_exam_config(source_run_dir: Path) -> tuple[dict, Path]:
    if source_run_dir.is_dir() and (source_run_dir / "config.yaml").exists():
        config_path = source_run_dir / "config.yaml"
        exam_dir = source_run_dir
    else:
        exam_dir = source_run_dir.parent if source_run_dir.name == "output" else source_run_dir
        config_path = exam_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Source exam config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return json.loads(json.dumps(__import__("yaml").safe_load(handle))), exam_dir


def _resolve_source_paths(cfg, exam_dir: Path | None) -> tuple[Path, Path | None]:
    source_run_dir = resolve_probe_path(exam_dir, cfg.source_run_dir)
    source_checkpoint = resolve_probe_path(exam_dir, cfg.source_checkpoint)
    if source_checkpoint is None:
        if source_run_dir is None:
            raise ValueError("source_checkpoint or source_run_dir is required")
        if source_run_dir.name == "output":
            source_checkpoint = source_run_dir.parent / "policy.ckpt"
        else:
            source_checkpoint = source_run_dir / "policy.ckpt"
    if not source_checkpoint.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {source_checkpoint}")
    source_checkpoint = source_checkpoint.resolve()
    source_run_dir = source_run_dir.resolve() if source_run_dir is not None else None
    return source_checkpoint, source_run_dir


def _build_source_policy(source_exam_config: dict, source_checkpoint: Path, exam_dir: Path | None):
    env = make_env(source_exam_config["env"])
    observation_example = env.reset()
    policy_config = dict(source_exam_config["policy"])
    checkpoint_cfg = dict(policy_config.get("checkpoint", {}))
    checkpoint_cfg["load"] = True
    checkpoint_cfg["path"] = str(source_checkpoint)
    policy_config["checkpoint"] = checkpoint_cfg
    policy = DreamerV3PolicyFolder(
        action_dim=env.action_dim,
        observation_example=observation_example,
        policy_config=policy_config,
        exam_dir=exam_dir,
    )
    policy.agent.world_model.eval()
    policy.agent.actor.eval()
    policy.agent.value.eval()
    for module in (policy.agent.world_model, policy.agent.actor, policy.agent.value):
        for param in module.parameters():
            param.requires_grad_(False)
    return env, policy


def _collect_replay(source_policy: DreamerV3PolicyFolder, env, cfg) -> EpisodeReplayBuffer:
    replay = EpisodeReplayBuffer(capacity=max(cfg.collect_steps + cfg.max_episode_steps, cfg.collect_steps))
    obs = env.reset()
    source_policy.reset()
    for _ in range(int(cfg.collect_steps)):
        action = source_policy.act(obs, deterministic=cfg.deterministic)
        next_obs, reward, done, _ = env.step(action)
        processed_obs = source_policy.processor.preprocess_obs(obs, source_policy.obs_spec)
        replay.add_step(
            obs=processed_obs,
            action=np.asarray(source_policy._latent_prev_action, dtype=np.float32).reshape(-1),
            reward=float(reward),
            done=bool(done),
        )
        obs = next_obs
        if done:
            source_policy.reset()
            obs = env.reset()
    replay.end_episode()
    return replay


def _build_transition_batch(source_policy: DreamerV3PolicyFolder, replay: EpisodeReplayBuffer, cfg) -> dict:
    batch = replay.sample_batch(
        batch_size=int(cfg.batch_size),
        seq_len=int(cfg.seq_len),
        device=cfg.device,
        include_contact=False,
        include_grasp=False,
    )
    with torch.no_grad():
        outputs = source_policy.agent.world_model.forward(batch, loss_config=None)
    feat = outputs["augmented_feat"].detach()
    x_t = feat[:, :-1]
    x_tp1 = feat[:, 1:]
    delta_x = (x_tp1 - x_t).detach()
    action_t = batch["action"][:, :-1].detach()
    capacity_gate = outputs.get("event_capacity_gate")
    if capacity_gate is None:
        raise ValueError("Frozen world model did not produce event_capacity_gate; Capacity V2 checkpoint is required")
    transition_batch = {
        "x_t": x_t,
        "x_tp1": x_tp1,
        "delta_x": delta_x,
        "action_t": action_t,
        "capacity_gate": capacity_gate.detach(),
        "ordinary_error": outputs.get("pred_next_feat_error_ordinary_only", None),
        "mixed_error": outputs.get("pred_next_feat_error_event_mixed", None),
        "_sample_refs": batch.get("_sample_refs", []),
    }
    return transition_batch


def _aggregate_eval(eval_rows: list[dict[str, Any]], num_tokens: int) -> dict[str, Any]:
    if not eval_rows:
        raise ValueError("No evaluation rows to aggregate")
    scalars = defaultdict(list)
    usage_vectors = []
    bad_numeric = 0.0
    for row in eval_rows:
        for key, value in row.items():
            if key in {"token_id", "effect_loss_per_step", "inverse_loss_per_step", "token_usage"}:
                continue
            if isinstance(value, (int, float)):
                scalars[key].append(float(value))
        usage_vectors.append(np.asarray(row["token_usage"], dtype=np.float64))
        bad_numeric += float(row.get("bad_numeric_count", 0.0))
    summary = {key: float(np.mean(values)) for key, values in scalars.items() if values}
    mean_usage = np.mean(np.stack(usage_vectors, axis=0), axis=0)
    summary["token_usage"] = mean_usage.tolist()
    summary["num_active_tokens"] = float(np.sum(mean_usage > 0.01))
    summary["token_perplexity"] = float(np.exp(-np.sum(np.clip(mean_usage, 1e-8, 1.0) * np.log(np.clip(mean_usage, 1e-8, 1.0)))))
    summary["random_token_accuracy"] = 1.0 / float(max(1, num_tokens))
    summary["bad_numeric_count"] = float(bad_numeric)
    return summary


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path


def _write_jsonl(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _collect_examples(policy: OperatorTokenProbePolicyFolder, source_policy: DreamerV3PolicyFolder, replay: EpisodeReplayBuffer, cfg) -> list[dict[str, Any]]:
    example_rows: list[dict[str, Any]] = []
    for _ in range(max(1, int(cfg.num_eval_batches))):
        batch = _build_transition_batch(source_policy, replay, cfg)
        eval_row = policy.evaluate(batch)
        token_ids = eval_row["token_id"]
        effect_losses = eval_row["effect_loss_per_step"]
        inverse_losses = eval_row["inverse_loss_per_step"]
        ordinary_error = batch.get("ordinary_error")
        mixed_error = batch.get("mixed_error")
        sample_refs = batch.get("_sample_refs", [])
        capacity_gate = batch["capacity_gate"].detach().cpu().squeeze(-1).tolist()
        action_norm = torch.linalg.vector_norm(batch["action_t"].detach(), dim=-1).cpu().tolist()
        delta_norm = torch.linalg.vector_norm(batch["delta_x"].detach(), dim=-1).cpu().tolist()
        for batch_index, ref in enumerate(sample_refs):
            episode_id, start_index = ref
            for time_index, token_id in enumerate(token_ids[batch_index]):
                example_rows.append(
                    {
                        "token_id": int(token_id),
                        "episode_id": int(episode_id),
                        "sample_id": f"{int(episode_id)}:{int(start_index + time_index)}",
                        "time_index": int(start_index + time_index),
                        "capacity_gate": float(capacity_gate[batch_index][time_index]),
                        "effect_loss": float(effect_losses[batch_index][time_index]),
                        "inverse_loss": float(inverse_losses[batch_index][time_index]),
                        "action_norm": float(action_norm[batch_index][time_index]),
                        "delta_x_norm": float(delta_norm[batch_index][time_index]),
                        "ordinary_error": (
                            "NA"
                            if ordinary_error is None
                            else float(ordinary_error.detach().cpu()[batch_index, time_index].item())
                        ),
                        "mixed_error": (
                            "NA"
                            if mixed_error is None
                            else float(mixed_error.detach().cpu()[batch_index, time_index].item())
                        ),
                    }
                )
    example_rows.sort(key=lambda row: (row["token_id"], row["effect_loss"]))
    per_token_counts = defaultdict(int)
    selected = []
    for row in example_rows:
        token_id = int(row["token_id"])
        if per_token_counts[token_id] >= int(cfg.num_examples_per_token):
            continue
        per_token_counts[token_id] += 1
        selected.append(row)
        if len(selected) >= int(cfg.num_example_records):
            break
    return selected


def run_operator_token_probe(config: dict, exam_dir: str | Path | None = None) -> dict[str, Any]:
    exam_dir_path = Path(exam_dir).resolve() if exam_dir is not None else None
    probe_policy = OperatorTokenProbePolicyFolder(action_dim=4, policy_config=config, exam_dir=exam_dir_path)
    cfg = probe_policy.cfg
    source_checkpoint, source_run_dir = _resolve_source_paths(cfg, exam_dir_path)
    if source_run_dir is None:
        source_run_dir = source_checkpoint.parent
    if source_run_dir.name == "output":
        source_exam_dir = source_run_dir.parent
    else:
        source_exam_dir = source_run_dir
    with (source_exam_dir / "config.yaml").open("r", encoding="utf-8") as handle:
        import yaml

        source_exam_config = yaml.safe_load(handle)
    env, source_policy = _build_source_policy(source_exam_config, source_checkpoint, source_exam_dir)
    probe_policy.action_dim = int(env.action_dim)
    replay = _collect_replay(source_policy, env, cfg)
    output_dir = Path(cfg.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    history = []
    for step in range(1, int(cfg.train_steps) + 1):
        batch = _build_transition_batch(source_policy, replay, cfg)
        payload = probe_policy.update(batch)
        row = {"step": step, **payload["scalars"]}
        history.append(row)
    eval_rows = []
    for _ in range(int(cfg.num_eval_batches)):
        batch = _build_transition_batch(source_policy, replay, cfg)
        eval_rows.append(probe_policy.evaluate(batch))
    summary = _aggregate_eval(eval_rows, cfg.operator_token.num_tokens)
    summary["train_steps"] = int(cfg.train_steps)
    summary["capacity_transition_count"] = float(summary.get("capacity_transition_count", 0.0))
    summary["source_checkpoint"] = str(source_checkpoint)
    summary["source_exam_dir"] = str(source_exam_dir)
    summary["training_finished"] = True

    metrics_path = _write_json(output_dir / "operator_token_metrics.json", {"history": history, "eval": eval_rows})
    usage_path = _write_json(output_dir / "operator_token_usage.json", {"token_usage": summary["token_usage"]})
    examples = _collect_examples(probe_policy, source_policy, replay, cfg)
    examples_path = _write_jsonl(output_dir / "operator_token_examples.jsonl", examples)
    summary_path = _write_json(output_dir / "operator_token_summary.json", summary)
    checkpoint_path = output_dir / "operator_token_probe.pt"
    probe_policy.save(checkpoint_path)

    env.close()
    return {
        "output_dir": output_dir,
        "metrics_path": metrics_path,
        "usage_path": usage_path,
        "examples_path": examples_path,
        "summary_path": summary_path,
        "checkpoint_path": checkpoint_path,
        "summary": summary,
    }
