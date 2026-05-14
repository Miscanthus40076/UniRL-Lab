from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import torch


def _is_object_array(value) -> bool:
    return isinstance(value, np.ndarray) and value.dtype == object


def _to_numpy(value):
    return np.asarray(value)


def _infer_is_first(episode_id: np.ndarray) -> np.ndarray:
    out = np.zeros(len(episode_id), dtype=np.float32)
    prev = None
    for idx, item in enumerate(episode_id):
        if idx == 0 or item != prev:
            out[idx] = 1.0
        prev = item
    return out


def _split_flat_by_is_first(data: dict[str, np.ndarray], is_first: np.ndarray) -> list[dict[str, np.ndarray]]:
    starts = np.flatnonzero(is_first.astype(bool))
    if len(starts) == 0 or starts[0] != 0:
        starts = np.concatenate([np.asarray([0]), starts])
    ends = np.concatenate([starts[1:], np.asarray([len(is_first)])])
    episodes = []
    for start, end in zip(starts, ends):
        episode = {}
        for key, value in data.items():
            episode[key] = np.asarray(value[start:end])
        episode["is_first"] = np.zeros(end - start, dtype=np.float32)
        if end > start:
            episode["is_first"][0] = 1.0
        episodes.append(episode)
    return episodes


def normalize_raw_episode(episode: dict[str, np.ndarray], direction: str, action_alignment: str) -> dict[str, np.ndarray]:
    obs = np.asarray(episode["obs"], dtype=np.float32)
    raw_action = np.asarray(episode["action"], dtype=np.float32)
    raw_reward = np.asarray(episode["reward"], dtype=np.float32).reshape(-1)
    raw_done = np.asarray(episode["done"], dtype=np.float32).reshape(-1)
    if len(obs) != len(raw_action) or len(obs) != len(raw_reward) or len(obs) != len(raw_done):
        raise ValueError("Episode obs/action/reward/done lengths must match after normalization")
    is_first = np.asarray(episode.get("is_first"), dtype=np.float32).reshape(-1)
    if len(is_first) != len(obs):
        raise ValueError("Episode is_first length must match obs length")
    action_dim = int(raw_action.reshape(len(raw_action), -1).shape[-1])
    flat_action = raw_action.reshape(len(raw_action), action_dim)

    if action_alignment == "dreamer_prev":
        dreamer_action = flat_action.astype(np.float32)
        dreamer_reward = raw_reward.astype(np.float32)
        dreamer_done = raw_done.astype(np.float32)
        raw_action_index = np.arange(len(obs), dtype=np.int64)
        reward_index = np.arange(len(obs), dtype=np.int64)
        done_index = np.arange(len(obs), dtype=np.int64)
    elif action_alignment == "env_step_current":
        dreamer_action = np.zeros_like(flat_action, dtype=np.float32)
        dreamer_reward = np.zeros_like(raw_reward, dtype=np.float32)
        dreamer_done = np.zeros_like(raw_done, dtype=np.float32)
        if len(obs) > 1:
            dreamer_action[1:] = flat_action[:-1]
            dreamer_reward[1:] = raw_reward[:-1]
            dreamer_done[1:] = raw_done[:-1]
        raw_action_index = np.full(len(obs), -1, dtype=np.int64)
        reward_index = np.full(len(obs), -1, dtype=np.int64)
        done_index = np.full(len(obs), -1, dtype=np.int64)
        if len(obs) > 1:
            raw_action_index[1:] = np.arange(len(obs) - 1, dtype=np.int64)
            reward_index[1:] = np.arange(len(obs) - 1, dtype=np.int64)
            done_index[1:] = np.arange(len(obs) - 1, dtype=np.int64)
    else:
        raise ValueError(f"Unsupported action_alignment: {action_alignment}")

    normalized = {
        "obs": obs,
        "action": dreamer_action.astype(np.float32),
        "reward": dreamer_reward.astype(np.float32),
        "done": dreamer_done.astype(np.float32),
        "is_first": is_first.astype(np.float32),
        "_raw_action_index": raw_action_index,
        "_dreamer_action_index": np.arange(len(obs), dtype=np.int64),
        "_reward_index": reward_index,
        "_done_index": done_index,
        "direction": np.asarray([direction] * len(obs), dtype=object),
    }
    if "contact_mode" in episode:
        normalized["contact_mode"] = np.asarray(episode["contact_mode"], dtype=np.int64).reshape(-1)
    if "grasp_state" in episode:
        normalized["is_grasping"] = np.asarray(episode["grasp_state"], dtype=np.float32).reshape(-1)
    elif "is_grasping" in episode:
        normalized["is_grasping"] = np.asarray(episode["is_grasping"], dtype=np.float32).reshape(-1)
    if "success" in episode:
        normalized["success"] = np.asarray(episode["success"])
    if "info" in episode:
        normalized["info"] = np.asarray(episode["info"], dtype=object)
    return normalized


@dataclass(slots=True)
class DatasetDescription:
    path: str
    direction: str
    num_episodes: int
    mean_episode_length: float
    action_dim: int
    obs_shape: tuple[int, ...]
    obs_dtype: str
    has_is_first: bool
    has_episode_id: bool
    has_contact_mode: bool
    has_grasp_state: bool
    has_success: bool
    action_alignment: str

    def asdict(self) -> dict:
        return {
            "path": self.path,
            "direction": self.direction,
            "num_episodes": self.num_episodes,
            "mean_episode_length": self.mean_episode_length,
            "action_dim": self.action_dim,
            "obs_shape": list(self.obs_shape),
            "obs_dtype": self.obs_dtype,
            "has_is_first": self.has_is_first,
            "has_episode_id": self.has_episode_id,
            "has_contact_mode": self.has_contact_mode,
            "has_grasp_state": self.has_grasp_state,
            "has_success": self.has_success,
            "action_alignment": self.action_alignment,
        }


class BidirectionalEpisodeDataset:
    def __init__(
        self,
        path: str | Path,
        direction: str,
        action_alignment: str = "env_step_current",
        episodes: list[dict[str, np.ndarray]] | None = None,
    ):
        self.path = str(path)
        self.direction = str(direction)
        self.action_alignment = str(action_alignment)
        self.episodes = episodes if episodes is not None else self._load_npz(path)
        if not self.episodes:
            raise ValueError(f"No episodes found in dataset: {path}")
        self.action_dim = int(self.episodes[0]["action"].reshape(len(self.episodes[0]["action"]), -1).shape[-1])
        self.obs_shape = tuple(int(x) for x in self.episodes[0]["obs"].shape[1:])

    def _load_npz(self, path: str | Path) -> list[dict[str, np.ndarray]]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        with np.load(path, allow_pickle=True) as data:
            raw = {key: data[key] for key in data.files}
        if "obs" not in raw or "action" not in raw or "reward" not in raw or "done" not in raw:
            missing = [key for key in ("obs", "action", "reward", "done") if key not in raw]
            raise KeyError(f"Dataset missing required fields: {missing}")

        if _is_object_array(raw["obs"]):
            episodes = []
            num_eps = len(raw["obs"])
            for idx in range(num_eps):
                episode = {}
                for key, value in raw.items():
                    if len(value) != num_eps:
                        continue
                    episode[key] = _to_numpy(value[idx])
                if "is_first" not in episode:
                    episode["is_first"] = np.zeros(len(episode["obs"]), dtype=np.float32)
                    if len(episode["obs"]) > 0:
                        episode["is_first"][0] = 1.0
                episodes.append(normalize_raw_episode(episode, self.direction, self.action_alignment))
            return episodes

        flat = {key: _to_numpy(value) for key, value in raw.items()}
        has_is_first = "is_first" in flat
        has_episode_id = "episode_id" in flat
        if not has_is_first and not has_episode_id:
            raise KeyError("Dataset must contain either 'is_first' or 'episode_id'")
        is_first = np.asarray(flat["is_first"], dtype=np.float32).reshape(-1) if has_is_first else _infer_is_first(np.asarray(flat["episode_id"]).reshape(-1))
        episodes = _split_flat_by_is_first(flat, is_first)
        return [normalize_raw_episode(episode, self.direction, self.action_alignment) for episode in episodes]

    def __len__(self) -> int:
        return len(self.episodes)

    def split(self, val_ratio: float = 0.2, seed: int = 0) -> tuple["BidirectionalEpisodeDataset", "BidirectionalEpisodeDataset"]:
        if not 0.0 <= val_ratio < 1.0:
            raise ValueError("val_ratio must be in [0, 1)")
        rng = np.random.default_rng(seed)
        indices = np.arange(len(self.episodes))
        rng.shuffle(indices)
        val_count = int(round(len(indices) * val_ratio))
        val_indices = set(indices[:val_count].tolist())
        train_eps, val_eps = [], []
        for idx, episode in enumerate(self.episodes):
            (val_eps if idx in val_indices else train_eps).append(episode)
        if not train_eps:
            train_eps = val_eps[:1]
            val_eps = val_eps[1:]
        return (
            BidirectionalEpisodeDataset(self.path, self.direction, self.action_alignment, train_eps),
            BidirectionalEpisodeDataset(self.path, self.direction, self.action_alignment, val_eps or train_eps[:1]),
        )

    def _eligible_sequences(self, seq_len: int) -> list[tuple[int, int]]:
        rows = []
        for ep_idx, episode in enumerate(self.episodes):
            ep_len = len(episode["obs"])
            if ep_len < seq_len:
                continue
            for start in range(ep_len - seq_len + 1):
                rows.append((ep_idx, start))
        return rows

    def sample_batch(self, batch_size: int, seq_len: int, device: str | torch.device | None = None) -> dict[str, torch.Tensor | str]:
        candidates = self._eligible_sequences(seq_len)
        if not candidates:
            raise ValueError(f"No sequences of len={seq_len} available in {self.path}")
        rng = np.random.default_rng()
        picks = rng.choice(len(candidates), size=int(batch_size), replace=True)
        rows: dict[str, list[np.ndarray]] = {
            "obs": [],
            "action": [],
            "reward": [],
            "done": [],
            "is_first": [],
        }
        include_contact = "contact_mode" in self.episodes[0]
        include_grasp = "is_grasping" in self.episodes[0]
        if include_contact:
            rows["contact_mode"] = []
        if include_grasp:
            rows["is_grasping"] = []
        refs = []
        for pick in picks:
            ep_idx, start = candidates[int(pick)]
            episode = self.episodes[ep_idx]
            end = start + seq_len
            rows["obs"].append(np.asarray(episode["obs"][start:end], dtype=np.float32))
            rows["action"].append(np.asarray(episode["action"][start:end], dtype=np.float32))
            rows["reward"].append(np.asarray(episode["reward"][start:end], dtype=np.float32))
            rows["done"].append(np.asarray(episode["done"][start:end], dtype=np.float32))
            rows["is_first"].append(np.asarray(episode["is_first"][start:end], dtype=np.float32))
            if include_contact:
                rows["contact_mode"].append(np.asarray(episode["contact_mode"][start:end], dtype=np.int64))
            if include_grasp:
                rows["is_grasping"].append(np.asarray(episode["is_grasping"][start:end], dtype=np.float32))
            refs.append((ep_idx, start))

        dev = device or "cpu"
        batch: dict[str, torch.Tensor | str] = {
            "obs": torch.as_tensor(np.asarray(rows["obs"]), dtype=torch.float32, device=dev),
            "action": torch.as_tensor(np.asarray(rows["action"]), dtype=torch.float32, device=dev),
            "reward": torch.as_tensor(np.asarray(rows["reward"]), dtype=torch.float32, device=dev),
            "done": torch.as_tensor(np.asarray(rows["done"]), dtype=torch.float32, device=dev),
            "is_first": torch.as_tensor(np.asarray(rows["is_first"]), dtype=torch.float32, device=dev),
            "direction": self.direction,
            "_sample_refs": refs,
        }
        if include_contact:
            batch["contact_mode"] = torch.as_tensor(np.asarray(rows["contact_mode"]), dtype=torch.long, device=dev)
        if include_grasp:
            batch["is_grasping"] = torch.as_tensor(np.asarray(rows["is_grasping"]), dtype=torch.float32, device=dev)
        return batch

    def iter_eval_batches(self, num_batches: int, batch_size: int, seq_len: int, device: str | torch.device | None = None):
        for _ in range(int(num_batches)):
            yield self.sample_batch(batch_size=batch_size, seq_len=seq_len, device=device)

    def describe_schema(self) -> DatasetDescription:
        lengths = [len(ep["obs"]) for ep in self.episodes]
        sample = self.episodes[0]
        return DatasetDescription(
            path=self.path,
            direction=self.direction,
            num_episodes=len(self.episodes),
            mean_episode_length=float(np.mean(lengths)),
            action_dim=self.action_dim,
            obs_shape=tuple(int(x) for x in sample["obs"].shape[1:]),
            obs_dtype=str(sample["obs"].dtype),
            has_is_first=True,
            has_episode_id=False,
            has_contact_mode="contact_mode" in sample,
            has_grasp_state="is_grasping" in sample,
            has_success="success" in sample,
            action_alignment=self.action_alignment,
        )

    def describe_alignment(self, episode_index: int = 0, steps: int = 5) -> list[dict[str, int | float]]:
        episode = self.episodes[int(episode_index)]
        limit = min(int(steps), len(episode["obs"]))
        rows = []
        for idx in range(limit):
            rows.append(
                {
                    "obs_index": idx,
                    "raw_action_index": int(episode["_raw_action_index"][idx]),
                    "dreamer_action_index": int(episode["_dreamer_action_index"][idx]),
                    "reward_index": int(episode["_reward_index"][idx]),
                    "done_index": int(episode["_done_index"][idx]),
                    "is_first": float(episode["is_first"][idx]),
                }
            )
        return rows

    def export_reverse_direction_frames(self, output_dir: str | Path, max_episodes: int = 4):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for ep_idx, episode in enumerate(self.episodes[: int(max_episodes)]):
            obs = np.asarray(episode["obs"])
            if obs.ndim != 4:
                continue
            picks = [0, len(obs) // 2, len(obs) - 1]
            labels = ["first", "middle", "last"]
            for pick, label in zip(picks, labels):
                frame = obs[pick]
                if frame.shape[0] in (1, 3, 4):
                    image = np.transpose(frame, (1, 2, 0))
                else:
                    image = frame
                if image.shape[-1] == 1:
                    image = image[..., 0]
                path = output_dir / f"episode_{ep_idx:03d}_{label}.png"
                import matplotlib.pyplot as plt

                plt.imsave(path, image)
                written.append(path)
        return written

    def save_description(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.describe_schema().asdict(), indent=2) + "\n", encoding="utf-8")


def load_bidirectional_dataset(path: str | Path, direction: str, action_alignment: str = "env_step_current") -> BidirectionalEpisodeDataset:
    return BidirectionalEpisodeDataset(path=path, direction=direction, action_alignment=action_alignment)
