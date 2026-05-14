from __future__ import annotations

from collections import deque
from pathlib import Path
import random

import numpy as np
import torch


class EpisodeReplayBuffer:
    def __init__(self, direction: str, capacity: int = 100000):
        if direction not in {"forward", "reverse"}:
            raise ValueError(f"Unsupported replay direction: {direction}")
        self.direction = str(direction)
        self.capacity = int(capacity)
        self._episode_counter = 0
        self.episodes: deque[dict[str, object]] = deque()
        self.current = self._new_episode()
        self.num_steps = 0
        self._action_dim: int | None = None

    def _new_episode(self) -> dict[str, object]:
        return {
            "episode_id": self._episode_counter,
            "direction": self.direction,
            "obs": [],
            "action": [],
            "reward": [],
            "done": [],
            "info": [],
        }

    def _trim(self):
        while self.num_steps > self.capacity and self.episodes:
            old = self.episodes.popleft()
            self.num_steps -= len(old["obs"])

    def __len__(self) -> int:
        return int(self.num_steps)

    @property
    def action_dim(self) -> int | None:
        return self._action_dim

    def add_step(self, obs, action, reward, done, info, direction):
        if str(direction) != self.direction:
            raise ValueError(
                f"Replay direction mismatch: replay={self.direction} got step direction={direction}"
            )
        obs_arr = np.asarray(obs, dtype=np.float32)
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if self._action_dim is None:
            self._action_dim = int(action_arr.shape[0])
        elif int(action_arr.shape[0]) != self._action_dim:
            raise ValueError(
                f"Action dim mismatch: expected {self._action_dim}, got {action_arr.shape[0]}"
            )
        step_info = dict(info or {})
        step_info["direction"] = self.direction
        self.current["obs"].append(obs_arr)
        self.current["action"].append(action_arr)
        self.current["reward"].append(float(reward))
        self.current["done"].append(float(bool(done)))
        self.current["info"].append(step_info)
        self.num_steps += 1
        if done:
            self.end_episode()

    def end_episode(self):
        if self.current["obs"]:
            self.episodes.append(self.current)
            self._episode_counter += 1
            self.current = self._new_episode()
            self._trim()

    def sample_episodes(self, num_episodes: int):
        candidates = list(self.episodes)
        if self.current["obs"]:
            candidates.append(self.current)
        if not candidates:
            raise ValueError("Replay buffer is empty")
        return random.choices(candidates, k=int(num_episodes))

    def iter_episodes(self, include_current: bool = True):
        episodes = list(self.episodes)
        if include_current and self.current["obs"]:
            episodes.append(self.current)
        return episodes

    def _eligible_episodes(self, seq_len: int):
        candidates = list(self.episodes)
        if self.current["obs"]:
            candidates.append(self.current)
        return [ep for ep in candidates if len(ep["obs"]) >= seq_len]

    def _sequence_candidates(self, seq_len: int):
        rows = []
        for episode in self._eligible_episodes(seq_len):
            max_start = len(episode["obs"]) - seq_len + 1
            for start in range(max_start):
                rows.append((episode, int(start)))
        return rows

    def sample_batch(self, batch_size: int, seq_len: int, device: str | torch.device = "cpu") -> dict:
        candidates = self._sequence_candidates(seq_len)
        if not candidates:
            raise ValueError(f"Replay buffer has no episode with length >= {seq_len}")
        chosen = random.choices(candidates, k=int(batch_size))
        rows = {"obs": [], "action": [], "reward": [], "done": [], "is_first": []}
        refs = []
        for episode, start in chosen:
            end = start + seq_len
            obs = np.asarray(episode["obs"][start:end], dtype=np.float32)
            raw_action = np.asarray(episode["action"][start:end], dtype=np.float32).reshape(seq_len, -1)
            raw_reward = np.asarray(episode["reward"][start:end], dtype=np.float32)
            raw_done = np.asarray(episode["done"][start:end], dtype=np.float32)
            action = np.zeros_like(raw_action, dtype=np.float32)
            reward = np.zeros_like(raw_reward, dtype=np.float32)
            done = np.zeros_like(raw_done, dtype=np.float32)
            if seq_len > 1:
                action[1:] = raw_action[:-1]
                reward[1:] = raw_reward[:-1]
                done[1:] = raw_done[:-1]
            is_first = np.zeros(seq_len, dtype=np.float32)
            if start == 0:
                is_first[0] = 1.0
            rows["obs"].append(obs)
            rows["action"].append(action)
            rows["reward"].append(reward)
            rows["done"].append(done)
            rows["is_first"].append(is_first)
            refs.append((int(episode["episode_id"]), start))
        batch = {
            "obs": torch.as_tensor(np.asarray(rows["obs"]), dtype=torch.float32, device=device),
            "action": torch.as_tensor(np.asarray(rows["action"]), dtype=torch.float32, device=device),
            "reward": torch.as_tensor(np.asarray(rows["reward"]), dtype=torch.float32, device=device),
            "done": torch.as_tensor(np.asarray(rows["done"]), dtype=torch.float32, device=device),
            "is_first": torch.as_tensor(np.asarray(rows["is_first"]), dtype=torch.float32, device=device),
            "direction": self.direction,
            "_sample_refs": refs,
        }
        return batch

    def export_npz(self, path: str | Path):
        self.end_episode()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        episodes = list(self.episodes)
        payload = {
            "obs": np.asarray([np.asarray(ep["obs"], dtype=np.float32) for ep in episodes], dtype=object),
            "action": np.asarray([np.asarray(ep["action"], dtype=np.float32) for ep in episodes], dtype=object),
            "reward": np.asarray([np.asarray(ep["reward"], dtype=np.float32) for ep in episodes], dtype=object),
            "done": np.asarray([np.asarray(ep["done"], dtype=np.float32) for ep in episodes], dtype=object),
            "is_first": np.asarray(
                [
                    np.concatenate(
                        [np.asarray([1.0], dtype=np.float32), np.zeros(max(len(ep["obs"]) - 1, 0), dtype=np.float32)]
                    )
                    for ep in episodes
                ],
                dtype=object,
            ),
            "direction": np.asarray([ep["direction"] for ep in episodes], dtype=object),
            "episode_id": np.asarray([int(ep["episode_id"]) for ep in episodes], dtype=np.int64),
            "info": np.asarray([np.asarray(ep["info"], dtype=object) for ep in episodes], dtype=object),
        }
        np.savez(path, **payload)
        return path
