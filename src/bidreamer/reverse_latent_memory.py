from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn.functional as F

from .bidirectional_dataset import normalize_raw_episode


@dataclass(slots=True)
class NearestNeighborResult:
    global_index: int
    episode_id: int
    time_index: int
    distance: float

    def asdict(self) -> dict:
        return {
            "global_index": self.global_index,
            "episode_id": self.episode_id,
            "time_index": self.time_index,
            "distance": self.distance,
        }


class ReverseLatentMemory:
    def __init__(
        self,
        features: torch.Tensor,
        episode_ids: torch.Tensor,
        time_indices: torch.Tensor,
        obs: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        reward: torch.Tensor | None = None,
        done: torch.Tensor | None = None,
        success: torch.Tensor | None = None,
        distance_metric: str = "cosine",
        normalize_features: bool = True,
        feature_whiten: bool = False,
        feature_mean: torch.Tensor | None = None,
        feature_std: torch.Tensor | None = None,
        fallback_used: bool = False,
    ):
        self.features = features.float()
        self.episode_ids = episode_ids.long()
        self.time_indices = time_indices.long()
        self.obs = obs
        self.action = action
        self.reward = reward
        self.done = done
        self.success = success
        self.distance_metric = str(distance_metric)
        self.normalize_features = bool(normalize_features)
        self.feature_whiten = bool(feature_whiten)
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.fallback_used = bool(fallback_used)
        self._lookup = {
            (int(ep.item()), int(t.item())): idx
            for idx, (ep, t) in enumerate(zip(self.episode_ids, self.time_indices))
        }

    @classmethod
    def _build_from_episodes(
        cls,
        model,
        episodes: list[dict[str, np.ndarray]],
        device: str = "cpu",
        direction: str = "reverse",
        use_success_only: bool = True,
        distance_metric: str = "cosine",
        normalize_features: bool = True,
        feature_whiten: bool = False,
        store_obs: bool = True,
    ) -> "ReverseLatentMemory":
        filtered_episodes = []
        fallback_used = False
        for episode in episodes:
            has_success = "success" in episode
            ep_success = bool(np.asarray(episode["success"]).reshape(-1)[-1]) if has_success and len(episode["success"]) else False
            if use_success_only and has_success and not ep_success:
                continue
            filtered_episodes.append(episode)
        if use_success_only and not filtered_episodes:
            filtered_episodes = list(episodes)
            fallback_used = True

        model.eval()
        feats, ep_ids, time_ids = [], [], []
        obs_rows, action_rows, reward_rows, done_rows, success_rows = [], [], [], [], []
        kept_episodes = 0
        effective_success_only = bool(use_success_only and not fallback_used)
        with torch.no_grad():
            for episode in filtered_episodes:
                has_success = "success" in episode
                ep_success = bool(np.asarray(episode["success"]).reshape(-1)[-1]) if has_success and len(episode["success"]) else False
                if effective_success_only and has_success and not ep_success:
                    continue
                if effective_success_only and not has_success:
                    fallback_used = True
                batch = {
                    "obs": torch.as_tensor(episode["obs"][None, ...], dtype=torch.float32, device=device),
                    "action": torch.as_tensor(episode["action"][None, ...], dtype=torch.float32, device=device),
                    "reward": torch.as_tensor(episode["reward"][None, ...], dtype=torch.float32, device=device),
                    "done": torch.as_tensor(episode["done"][None, ...], dtype=torch.float32, device=device),
                    "is_first": torch.as_tensor(episode["is_first"][None, ...], dtype=torch.float32, device=device),
                }
                outputs = model.posterior_outputs(batch, direction)
                feat = model.extract_feat(outputs["post"], deterministic=True).squeeze(0).cpu()
                seq_len = feat.shape[0]
                feats.append(feat)
                ep_ids.append(torch.full((seq_len,), kept_episodes, dtype=torch.long))
                time_ids.append(torch.arange(seq_len, dtype=torch.long))
                if store_obs:
                    obs_rows.append(torch.as_tensor(episode["obs"], dtype=torch.float32))
                action_rows.append(torch.as_tensor(episode["action"], dtype=torch.float32))
                reward_rows.append(torch.as_tensor(episode["reward"], dtype=torch.float32))
                done_rows.append(torch.as_tensor(episode["done"], dtype=torch.float32))
                if has_success:
                    success_rows.append(torch.as_tensor(np.asarray(episode["success"]).reshape(-1), dtype=torch.float32))
                else:
                    success_rows.append(torch.full((seq_len,), float(ep_success), dtype=torch.float32))
                kept_episodes += 1
        if not feats:
            raise ValueError("No reverse episodes available to build latent memory")
        features = torch.cat(feats, dim=0)
        feature_mean = features.mean(dim=0)
        feature_std = features.std(dim=0).clamp_min(1e-6)
        if feature_whiten:
            features = (features - feature_mean) / feature_std
        if normalize_features:
            features = F.normalize(features, dim=-1)
        return cls(
            features=features,
            episode_ids=torch.cat(ep_ids, dim=0),
            time_indices=torch.cat(time_ids, dim=0),
            obs=torch.cat(obs_rows, dim=0) if obs_rows else None,
            action=torch.cat(action_rows, dim=0),
            reward=torch.cat(reward_rows, dim=0),
            done=torch.cat(done_rows, dim=0),
            success=torch.cat(success_rows, dim=0),
            distance_metric=distance_metric,
            normalize_features=normalize_features,
            feature_whiten=feature_whiten,
            feature_mean=feature_mean,
            feature_std=feature_std,
            fallback_used=fallback_used,
        )

    @classmethod
    def build_from_reverse_dataset(
        cls,
        model,
        dataset,
        device: str = "cpu",
        use_success_only: bool = True,
        distance_metric: str = "cosine",
        normalize_features: bool = True,
        feature_whiten: bool = False,
        store_obs: bool = True,
    ) -> "ReverseLatentMemory":
        return cls._build_from_episodes(
            model=model,
            episodes=dataset.episodes,
            device=device,
            direction="reverse",
            use_success_only=use_success_only,
            distance_metric=distance_metric,
            normalize_features=normalize_features,
            feature_whiten=feature_whiten,
            store_obs=store_obs,
        )

    @classmethod
    def build_from_reverse_replay(
        cls,
        model,
        replay,
        device: str = "cpu",
        action_alignment: str = "env_step_current",
        use_success_only: bool = True,
        distance_metric: str = "cosine",
        normalize_features: bool = True,
        feature_whiten: bool = False,
        store_obs: bool = True,
    ) -> "ReverseLatentMemory":
        episodes = []
        for replay_episode in replay.iter_episodes(include_current=True):
            if not replay_episode["obs"]:
                continue
            success = np.asarray(
                [
                    float(bool(info.get("success") or info.get("is_success")))
                    for info in replay_episode["info"]
                ],
                dtype=np.float32,
            )
            raw_episode = {
                "obs": np.asarray(replay_episode["obs"], dtype=np.float32),
                "action": np.asarray(replay_episode["action"], dtype=np.float32),
                "reward": np.asarray(replay_episode["reward"], dtype=np.float32),
                "done": np.asarray(replay_episode["done"], dtype=np.float32),
                "is_first": np.concatenate(
                    [
                        np.asarray([1.0], dtype=np.float32),
                        np.zeros(max(len(replay_episode["obs"]) - 1, 0), dtype=np.float32),
                    ]
                ),
                "success": success,
                "info": np.asarray(replay_episode["info"], dtype=object),
            }
            episodes.append(normalize_raw_episode(raw_episode, "reverse", action_alignment))
        return cls._build_from_episodes(
            model=model,
            episodes=episodes,
            device=device,
            direction="reverse",
            use_success_only=use_success_only,
            distance_metric=distance_metric,
            normalize_features=normalize_features,
            feature_whiten=feature_whiten,
            store_obs=store_obs,
        )

    def _prepare_query(self, feat: torch.Tensor) -> torch.Tensor:
        query = feat.detach().cpu().float().reshape(1, -1)
        if self.feature_whiten and self.feature_mean is not None and self.feature_std is not None:
            query = (query - self.feature_mean) / self.feature_std
        if self.normalize_features:
            query = F.normalize(query, dim=-1)
        return query

    def nearest(self, feat: torch.Tensor, k: int = 1) -> list[NearestNeighborResult]:
        query = self._prepare_query(feat)
        if self.distance_metric == "cosine":
            distances = 1.0 - torch.matmul(self.features, query[0])
        elif self.distance_metric == "l2":
            distances = torch.norm(self.features - query, dim=-1)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
        values, indices = torch.topk(distances, k=int(k), largest=False)
        results = []
        for value, index in zip(values.tolist(), indices.tolist()):
            results.append(
                NearestNeighborResult(
                    global_index=int(index),
                    episode_id=int(self.episode_ids[index].item()),
                    time_index=int(self.time_indices[index].item()),
                    distance=float(value),
                )
            )
        return results

    def get_subgoal(self, nearest_result: NearestNeighborResult | dict, subgoal_step: int = 1) -> dict:
        if isinstance(nearest_result, dict):
            episode_id = int(nearest_result["episode_id"])
            time_index = int(nearest_result["time_index"])
        else:
            episode_id = int(nearest_result.episode_id)
            time_index = int(nearest_result.time_index)
        subgoal_time = max(time_index - int(subgoal_step), 0)
        global_index = self._lookup[(episode_id, subgoal_time)]
        return {
            "global_index": int(global_index),
            "episode_id": episode_id,
            "time_index": subgoal_time,
            "feature": self.features[global_index],
            "obs": None if self.obs is None else self.obs[global_index],
            "action": None if self.action is None else self.action[global_index],
            "reward": None if self.reward is None else self.reward[global_index],
            "done": None if self.done is None else self.done[global_index],
            "success": None if self.success is None else self.success[global_index],
        }

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "features": self.features,
                "episode_ids": self.episode_ids,
                "time_indices": self.time_indices,
                "obs": self.obs,
                "action": self.action,
                "reward": self.reward,
                "done": self.done,
                "success": self.success,
                "distance_metric": self.distance_metric,
                "normalize_features": self.normalize_features,
                "feature_whiten": self.feature_whiten,
                "feature_mean": self.feature_mean,
                "feature_std": self.feature_std,
                "fallback_used": self.fallback_used,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ReverseLatentMemory":
        payload = torch.load(path, map_location="cpu")
        return cls(**payload)

    def summary(self) -> dict:
        norms = torch.norm(self.features, dim=-1)
        nearest_distances = []
        if len(self.features) > 1:
            for idx in range(min(len(self.features), 128)):
                query = self.features[idx]
                results = self.nearest(query, k=2)
                if len(results) > 1:
                    nearest_distances.append(results[1].distance)
        return {
            "num_episodes": int(len(torch.unique(self.episode_ids))),
            "num_latents": int(len(self.features)),
            "feat_dim": int(self.features.shape[-1]),
            "mean_nearest_distance_on_reverse": float(np.mean(nearest_distances)) if nearest_distances else 0.0,
            "example_indices": [int(x) for x in range(min(5, len(self.features)))],
            "feature_norm_mean": float(norms.mean().item()),
            "feature_norm_std": float(norms.std().item()),
            "distance_metric": self.distance_metric,
            "normalize_features": self.normalize_features,
            "feature_whiten": self.feature_whiten,
            "fallback_used": self.fallback_used,
        }

    def save_summary(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary(), indent=2) + "\n", encoding="utf-8")
