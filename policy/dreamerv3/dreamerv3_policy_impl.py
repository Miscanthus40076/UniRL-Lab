from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..base_policy import BasePolicy
from .config import build_dreamerv3_model_config, build_dreamerv3_policy_config
from .dreamerv3_agent import DreamerV3Agent
from .dreamerv3_model import DreamerObservationSpec
from .processor import DreamerV3Processor
from .replay_buffer import EpisodeReplayBuffer


class DreamerV3PolicyFolder(BasePolicy):
    def __init__(
        self,
        action_dim: int,
        observation_example,
        policy_config: dict | None = None,
        exam_dir: str | Path | None = None,
    ):
        self.action_dim = int(action_dim)
        self.exam_dir = Path(exam_dir) if exam_dir is not None else None

        self.policy_cfg, self.checkpoint_cfg = build_dreamerv3_policy_config(policy_config)
        self.processor = DreamerV3Processor(device=self.policy_cfg.device)
        self.obs_spec = self.processor.infer_observation_spec(observation_example)
        self.model_cfg = build_dreamerv3_model_config(self.policy_cfg, self.action_dim, self.obs_spec)
        self.agent = DreamerV3Agent(self.model_cfg)
        self.contact_label_map: dict[str, int] = {}
        self.last_checkpoint_load_report: dict[str, object] | None = None

        self._make_replay()
        self._reset_runtime_state()

        if self.checkpoint_cfg.load and self.checkpoint_cfg.path:
            self.load(self.checkpoint_cfg.path)
        else:
            self.reset()

    def _make_replay(self):
        self.replay = EpisodeReplayBuffer(
            capacity=self.policy_cfg.replay_capacity,
            priority_enabled=self.policy_cfg.priority_replay,
            priority_exponent=self.policy_cfg.priority_exponent,
            priority_uniform_mix=self.policy_cfg.priority_uniform_mix,
            priority_initial=self.policy_cfg.priority_initial,
        )

    def _reset_runtime_state(self):
        self.batch_size = int(self.policy_cfg.batch_size)
        self.seq_len = int(self.policy_cfg.seq_len)
        self.warmup_steps = int(self.policy_cfg.warmup_steps)
        self.train_every = max(1, int(self.policy_cfg.train_every))
        self.train_ratio = float(self.policy_cfg.train_ratio)
        self.max_updates_per_step = int(self.policy_cfg.max_updates_per_step)
        self.agent.configure_online_update(
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            warmup_steps=self.warmup_steps,
            train_every=self.train_every,
            train_ratio=self.train_ratio,
            max_updates_per_step=self.max_updates_per_step,
        )

    def _resolve_checkpoint_path(self, path: str | Path) -> Path:
        checkpoint_path = Path(path)
        if checkpoint_path.is_absolute() or self.exam_dir is None:
            return checkpoint_path
        return self.exam_dir / checkpoint_path

    def _maybe_contact_label(self, info: dict | None) -> int | None:
        if self.model_cfg.aux.contact_num_classes is None or not info or "contact_mode" not in info:
            return None
        raw = info["contact_mode"]
        if isinstance(raw, (int, np.integer)):
            label = int(raw)
        else:
            key = str(raw)
            if key not in self.contact_label_map:
                next_idx = len(self.contact_label_map)
                limit = int(self.model_cfg.aux.contact_num_classes)
                if next_idx >= limit:
                    return None
                self.contact_label_map[key] = next_idx
            label = self.contact_label_map[key]
        if label < 0 or label >= int(self.model_cfg.aux.contact_num_classes):
            return None
        return label

    def _maybe_grasp_label(self, info: dict | None) -> float | None:
        if not self.model_cfg.aux.predict_grasp or not info or "is_grasping" not in info:
            return None
        return float(bool(info["is_grasping"]))

    def act(self, obs, deterministic: bool = False):
        obs_array = self.processor.preprocess_obs(obs, self.obs_spec)
        prev_action = self._prev_action.copy()
        self.agent.update_latent(obs_array, prev_action, is_first=self._episode_is_first)
        self._latent_prev_action = prev_action
        raw_action = self.agent.act(deterministic=deterministic)
        action = self.processor.postprocess_action(raw_action)
        self._prev_action = action
        self._episode_is_first = False
        return action

    def update(self, step_batch: dict) -> dict[str, object]:
        obs = self.processor.preprocess_obs(step_batch["obs"], self.obs_spec)
        done = bool(step_batch["done"])
        info = step_batch.get("info") or {}

        self.replay.add_step(
            obs=obs,
            action=np.asarray(self._latent_prev_action, dtype=np.float32).reshape(-1),
            reward=float(step_batch["reward"]),
            done=done,
            contact_mode=self._maybe_contact_label(info),
            is_grasping=self._maybe_grasp_label(info),
        )
        if done:
            self.reset()

        include_contact = self.model_cfg.aux.contact_num_classes is not None
        include_grasp = self.model_cfg.aux.predict_grasp
        metrics, metadata = self.agent.update_from_replay(
            replay=self.replay,
            include_contact=include_contact,
            include_grasp=include_grasp,
        )
        return self._build_metrics_payload(scalars=metrics, metadata=metadata)

    def reset(self):
        if self.replay.current["obs"]:
            self.replay.end_episode()
        self.agent.reset_latent(batch_size=1)
        self._prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self._latent_prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self._episode_is_first = True

    def get_diagnostics(self) -> dict[str, object]:
        return {
            "context": self.agent.get_context_diagnostics(),
        }

    def save(self, path: str | Path):
        path = self._resolve_checkpoint_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_type": "dreamerv3",
                "policy_config": self.policy_cfg.asdict(),
                "observation_spec": asdict(self.obs_spec),
                "action_dim": self.action_dim,
                "contact_label_map": self.contact_label_map,
                "agent": self.agent.state_dict(),
            },
            path,
        )

    def load(self, path: str | Path):
        payload = torch.load(self._resolve_checkpoint_path(path), map_location=self.policy_cfg.device)
        if payload.get("policy_type") != "dreamerv3":
            raise ValueError("Checkpoint is not a DreamerV3 policy checkpoint")
        saved_action_dim = int(payload["action_dim"])
        saved_obs_spec = DreamerObservationSpec(**payload["observation_spec"])
        if saved_action_dim != self.action_dim:
            raise ValueError(f"Checkpoint action_dim={saved_action_dim} does not match current action_dim={self.action_dim}")
        if saved_obs_spec != self.obs_spec:
            raise ValueError(
                f"Checkpoint observation spec {saved_obs_spec.asdict()} does not match current spec {self.obs_spec.asdict()}"
            )
        self.agent = DreamerV3Agent.from_state_dict(
            payload["agent"],
            device=self.model_cfg.device,
            model_config=self.model_cfg,
        )
        self.contact_label_map = dict(payload.get("contact_label_map", {}))
        self.last_checkpoint_load_report = getattr(self.agent, "_last_load_report", None)

        self._make_replay()
        self._reset_runtime_state()
        self.reset()
