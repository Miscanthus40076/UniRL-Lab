from __future__ import annotations

from collections import deque
import random

import numpy as np
import torch


class EpisodeReplayBuffer:
    def __init__(
        self,
        capacity: int = 100000,
        priority_enabled: bool = False,
        priority_exponent: float = 0.8,
        priority_uniform_mix: float = 0.1,
        priority_initial: float = 1.0,
    ):
        self.capacity = int(capacity)
        self.priority_enabled = bool(priority_enabled)
        self.priority_exponent = float(priority_exponent)
        self.priority_uniform_mix = float(priority_uniform_mix)
        self.priority_initial = float(priority_initial)
        self._episode_counter = 0
        self.episodes: deque[dict[str, object]] = deque()
        self.current = self._new_episode()
        self.num_steps = 0

    def _new_episode(self) -> dict[str, object]:
        return {
            "episode_id": self._episode_counter,
            "obs": [],
            "proprio": [],
            "pose_probe_target": [],
            "action": [],
            "reward": [],
            "raw_env_reward": [],
            "operator_milestone_reward": [],
            "action_penalty": [],
            "done": [],
            "contact_mode": [],
            "is_grasping": [],
            "has_contact": True,
            "has_grasp": True,
            "priorities": [],
        }

    def add_step(
        self,
        obs,
        action,
        reward,
        done,
        contact_mode: int | None = None,
        is_grasping: float | None = None,
        raw_env_reward: float | None = None,
        operator_milestone_reward: float | None = None,
        action_penalty: float | None = None,
        proprio=None,
        pose_probe_target=None,
    ):
        self.current["obs"].append(np.asarray(obs, dtype=np.float32))
        if proprio is None:
            self.current["proprio"].append(np.full(4, np.nan, dtype=np.float32))
        else:
            self.current["proprio"].append(np.asarray(proprio, dtype=np.float32).reshape(-1))
        if pose_probe_target is None:
            self.current["pose_probe_target"].append(np.full(8, np.nan, dtype=np.float32))
        else:
            self.current["pose_probe_target"].append(np.asarray(pose_probe_target, dtype=np.float32).reshape(-1))
        self.current["action"].append(np.asarray(action, dtype=np.float32))
        self.current["reward"].append(float(reward))
        self.current["raw_env_reward"].append(float(reward if raw_env_reward is None else raw_env_reward))
        self.current["operator_milestone_reward"].append(
            float(0.0 if operator_milestone_reward is None else operator_milestone_reward)
        )
        self.current["action_penalty"].append(float(0.0 if action_penalty is None else action_penalty))
        self.current["done"].append(float(bool(done)))
        self.current["contact_mode"].append(None if contact_mode is None else int(contact_mode))
        self.current["is_grasping"].append(None if is_grasping is None else float(is_grasping))
        self.current["has_contact"] = bool(self.current["has_contact"]) and contact_mode is not None
        self.current["has_grasp"] = bool(self.current["has_grasp"]) and is_grasping is not None
        self.current["priorities"] = [self.priority_initial] * max(
            0,
            len(self.current["obs"]) - 1,
        )
        self.num_steps += 1
        if done:
            self.end_episode()

    def end_episode(self):
        if self.current["obs"]:
            self.episodes.append(self.current)
            self._episode_counter += 1
            self.current = self._new_episode()
            self._trim()

    def _trim(self):
        while self.num_steps > self.capacity and self.episodes:
            old = self.episodes.popleft()
            self.num_steps -= len(old["obs"])

    def __len__(self) -> int:
        return int(self.num_steps)

    def _eligible_episodes(self, seq_len: int, require_contact: bool, require_grasp: bool):
        candidates = list(self.episodes)
        if self.current["obs"]:
            candidates.append(self.current)
        return [
            ep
            for ep in candidates
            if len(ep["obs"]) >= seq_len
            and (not require_contact or ep["has_contact"])
            and (not require_grasp or ep["has_grasp"])
        ]

    def _sequence_candidates(self, seq_len: int, require_contact: bool, require_grasp: bool):
        candidates = []
        for ep in self._eligible_episodes(seq_len, require_contact, require_grasp):
            max_start = len(ep["obs"]) - seq_len + 1
            for start in range(max_start):
                priority = ep["priorities"][start] if start < len(ep["priorities"]) else self.priority_initial
                candidates.append(
                    {
                        "episode": ep,
                        "episode_id": ep["episode_id"],
                        "start": start,
                        "priority": max(float(priority), 1e-6),
                    }
                )
        return candidates

    def can_sample(self, batch_size: int, seq_len: int, require_contact: bool = False, require_grasp: bool = False) -> bool:
        candidates = self._sequence_candidates(seq_len, require_contact, require_grasp)
        return bool(candidates)

    def sample_batch(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device | str = "cpu",
        include_contact: bool = False,
        include_grasp: bool = False,
    ) -> dict[str, torch.Tensor]:
        candidates = self._sequence_candidates(seq_len, include_contact, include_grasp)
        if not candidates:
            raise ValueError(f"Replay buffer has no episode with length >= {seq_len}")
        if self.priority_enabled:
            priorities = np.asarray([c["priority"] for c in candidates], dtype=np.float64)
            weights = priorities ** self.priority_exponent
            weights = weights / weights.sum()
            uniform = np.full_like(weights, 1.0 / len(weights))
            weights = (1.0 - self.priority_uniform_mix) * weights + self.priority_uniform_mix * uniform
            chosen = random.choices(candidates, weights=weights.tolist(), k=int(batch_size))
        else:
            chosen = random.choices(candidates, k=int(batch_size))

        rows = {
            "obs": [],
            "proprio": [],
            "action": [],
            "reward": [],
            "raw_env_reward": [],
            "operator_milestone_reward": [],
            "action_penalty": [],
            "done": [],
            "is_first": [],
            "pose_probe_target": [],
        }
        sample_refs = []
        if include_contact:
            rows["contact_mode"] = []
        if include_grasp:
            rows["is_grasping"] = []

        for candidate in chosen:
            ep = candidate["episode"]
            start = int(candidate["start"])
            end = start + seq_len
            rows["obs"].append(np.asarray(ep["obs"][start:end], dtype=np.float32))
            rows["proprio"].append(np.asarray(ep["proprio"][start:end], dtype=np.float32))
            rows["pose_probe_target"].append(np.asarray(ep["pose_probe_target"][start:end], dtype=np.float32))
            rows["action"].append(np.asarray(ep["action"][start:end], dtype=np.float32))
            rows["reward"].append(np.asarray(ep["reward"][start:end], dtype=np.float32))
            rows["raw_env_reward"].append(np.asarray(ep["raw_env_reward"][start:end], dtype=np.float32))
            rows["operator_milestone_reward"].append(
                np.asarray(ep["operator_milestone_reward"][start:end], dtype=np.float32)
            )
            rows["action_penalty"].append(np.asarray(ep["action_penalty"][start:end], dtype=np.float32))
            rows["done"].append(np.asarray(ep["done"][start:end], dtype=np.float32))
            is_first = np.zeros(seq_len, dtype=np.float32)
            if start == 0:
                is_first[0] = 1.0
            rows["is_first"].append(is_first)
            sample_refs.append((int(candidate["episode_id"]), start))
            if include_contact:
                rows["contact_mode"].append(np.asarray(ep["contact_mode"][start:end], dtype=np.int64))
            if include_grasp:
                rows["is_grasping"].append(np.asarray(ep["is_grasping"][start:end], dtype=np.float32))

        batch = {
            "obs": torch.as_tensor(np.asarray(rows["obs"]), dtype=torch.float32, device=device),
            "proprio": torch.as_tensor(np.asarray(rows["proprio"]), dtype=torch.float32, device=device),
            "pose_probe_target": torch.as_tensor(
                np.asarray(rows["pose_probe_target"]), dtype=torch.float32, device=device
            ),
            "action": torch.as_tensor(np.asarray(rows["action"]), dtype=torch.float32, device=device),
            "reward": torch.as_tensor(np.asarray(rows["reward"]), dtype=torch.float32, device=device),
            "raw_env_reward": torch.as_tensor(np.asarray(rows["raw_env_reward"]), dtype=torch.float32, device=device),
            "operator_milestone_reward": torch.as_tensor(
                np.asarray(rows["operator_milestone_reward"]), dtype=torch.float32, device=device
            ),
            "action_penalty": torch.as_tensor(np.asarray(rows["action_penalty"]), dtype=torch.float32, device=device),
            "done": torch.as_tensor(np.asarray(rows["done"]), dtype=torch.float32, device=device),
            "is_first": torch.as_tensor(np.asarray(rows["is_first"]), dtype=torch.float32, device=device),
        }
        if include_contact:
            batch["contact_mode"] = torch.as_tensor(np.asarray(rows["contact_mode"]), dtype=torch.long, device=device)
        if include_grasp:
            batch["is_grasping"] = torch.as_tensor(np.asarray(rows["is_grasping"]), dtype=torch.float32, device=device)
        batch["_sample_refs"] = sample_refs
        return batch

    def update_priorities(self, sample_refs: list[tuple[int, int]], priorities):
        if not self.priority_enabled or not sample_refs:
            return
        priority_map = {tuple(ref): max(float(prio), 1e-6) for ref, prio in zip(sample_refs, priorities)}
        for ep in list(self.episodes) + ([self.current] if self.current["obs"] else []):
            ep_id = int(ep["episode_id"])
            for start, priority in priority_map.items():
                ref_ep_id, ref_start = start
                if ref_ep_id == ep_id and ref_start < len(ep["priorities"]):
                    ep["priorities"][ref_start] = priority
