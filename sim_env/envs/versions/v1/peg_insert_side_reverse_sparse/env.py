from __future__ import annotations

import os
from typing import Sequence

import numpy as np

from metaworld.envs.sawyer_peg_insertion_side_v3 import SawyerPegInsertionSideEnvV3

from sim_env.envs.versions.v1.base_env import BaseEnv


class SawyerPegInsertionSideReverseSparseStandaloneV1(SawyerPegInsertionSideEnvV3):
    """Standalone reverse task: unplug the inserted peg, place it, and release."""

    TABLE_HEIGHT_Z = 0.0
    TABLE_CONTACT_Z_THRESHOLD = 0.055
    UNPLUG_DISTANCE_X = 0.15
    UNPLUG_SUCCESS_RADIUS = 0.07
    RELEASE_DISTANCE = 0.06

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._has_been_unplugged = False
        self._inserted_obj_init_pos = None
        self._unplug_target_pos = None

    def reset_model(self):
        super().reset_model()
        assert self._target_pos is not None

        inserted_obj_pos = self._target_pos + np.array([0.1, 0.0, 0.0], dtype=np.float64)
        self._set_obj_xyz(inserted_obj_pos)

        self.obj_init_pos = inserted_obj_pos.copy()
        self._inserted_obj_init_pos = inserted_obj_pos.copy()
        self._unplug_target_pos = inserted_obj_pos + np.array(
            [self.UNPLUG_DISTANCE_X, 0.0, 0.0],
            dtype=np.float64,
        )
        self._has_been_unplugged = False

        return self._get_obs()

    def _sparse_success(self, obs):
        assert self._unplug_target_pos is not None
        obj = obs[4:7]
        tcp = self.tcp_center

        tcp_to_obj = float(np.linalg.norm(obj - tcp))
        obj_to_unplug_target = float(np.linalg.norm(obj - self._unplug_target_pos))

        unplugged_now = obj_to_unplug_target <= self.UNPLUG_SUCCESS_RADIUS
        if unplugged_now:
            self._has_been_unplugged = True

        on_table = obj[2] <= (self.TABLE_HEIGHT_Z + self.TABLE_CONTACT_Z_THRESHOLD)
        released = tcp_to_obj >= self.RELEASE_DISTANCE
        success = bool(self._has_been_unplugged and on_table and released)

        return success, tcp_to_obj, obj_to_unplug_target, float(on_table), float(released)

    def evaluate_state(self, obs, action):
        del action
        success, tcp_to_obj, obj_to_unplug_target, on_table, released = self._sparse_success(obs)
        reward = 1.0 if success else 0.0
        info = {
            "success": float(success),
            "near_object": float(tcp_to_obj <= 0.03),
            "grasp_success": float(tcp_to_obj <= 0.04),
            "grasp_reward": 0.0,
            "in_place_reward": 0.0,
            "obj_to_target": obj_to_unplug_target,
            "obj_on_table": on_table,
            "released": released,
            "has_been_unplugged": float(self._has_been_unplugged),
            "unscaled_reward": reward,
        }
        return reward, info

    def compute_reward(self, action, obs):
        del action
        success, tcp_to_obj, obj_to_unplug_target, on_table, released = self._sparse_success(obs)
        reward = 1.0 if success else 0.0
        return (
            reward,
            tcp_to_obj,
            0.0,
            obj_to_unplug_target,
            float(self._has_been_unplugged),
            on_table,
            released,
            0.0,
        )


class PegInsertSideReverseSparseStandaloneEnv(BaseEnv):
    """Project BaseEnv wrapper for the standalone reverse sparse peg task."""

    BENCHMARK_ENV_NAME = "peg-insert-side-v3"

    def __init__(
        self,
        render_height: int = 240,
        render_width: int = 320,
        camera_id: int = 0,
        camera_indices: Sequence[int] | None = None,
        observation_type: str = "vector",
        num_cams: int = 1,
        render_backend: str | None = None,
        render_backend_priority: Sequence[str] | None = None,
        allow_software_render_fallback: bool = False,
        render_enabled: bool = False,
        seed: int = 0,
        reward_function_version: str = "v2",
    ):
        self._observation_type = str(observation_type)
        self._render_enabled = bool(render_enabled)
        self._render_height = int(render_height)
        self._render_width = int(render_width)
        self._seed = int(seed)
        self._rng = np.random.default_rng(int(seed))
        self._selected_render_backend = None
        self._env = None
        self._benchmark = None
        self._tasks = []
        self._reward_function_version = str(reward_function_version)

        if isinstance(camera_indices, (list, tuple)) and camera_indices:
            self._camera_ids = [int(value) for value in camera_indices]
            self._num_cams = len(self._camera_ids)
            self._camera_id = int(self._camera_ids[0])
        else:
            self._num_cams = max(1, int(num_cams))
            self._camera_id = int(camera_id)
            self._camera_ids = [self._camera_id + offset for offset in range(self._num_cams)]

        self._load_env_with_backend_fallback(
            render_backend=render_backend,
            render_backend_priority=render_backend_priority,
            allow_software_render_fallback=allow_software_render_fallback,
        )

        obs = self.reset()
        self._obs_dim = int(np.asarray(obs, dtype=np.float32).size)
        self._action_dim = int(np.prod(self._env.action_space.shape))

    def _requires_render_backend(self) -> bool:
        return self._observation_type == "image" or self._render_enabled

    def _normalize_backend(self, backend):
        if backend is None:
            return None
        value = str(backend).strip().lower()
        if not value or value == "null":
            return None
        return value

    def _resolve_backend_candidates(self, render_backend, render_backend_priority, allow_software_render_fallback):
        if not self._requires_render_backend():
            return [None]

        candidates = []
        explicit_backend = self._normalize_backend(render_backend)
        if explicit_backend is not None:
            candidates.append(explicit_backend)
        else:
            if isinstance(render_backend_priority, (list, tuple)):
                for candidate in render_backend_priority:
                    normalized = self._normalize_backend(candidate)
                    if normalized is not None:
                        candidates.append(normalized)
            if not candidates:
                candidates.append("egl")
        if bool(allow_software_render_fallback) and "osmesa" not in candidates:
            candidates.append("osmesa")

        deduped = []
        for candidate in candidates:
            if candidate not in deduped:
                deduped.append(candidate)
        return deduped

    def _configure_render_backend(self, backend):
        if backend is None:
            return
        os.environ["MUJOCO_GL"] = backend
        if backend in {"egl", "osmesa"}:
            os.environ["PYOPENGL_PLATFORM"] = backend
        else:
            os.environ.pop("PYOPENGL_PLATFORM", None)

    def _build_env(self):
        import metaworld

        benchmark = metaworld.ML1(self.BENCHMARK_ENV_NAME, seed=self._seed)
        env = SawyerPegInsertionSideReverseSparseStandaloneV1(
            render_mode="rgb_array" if self._requires_render_backend() else None,
            camera_id=self._camera_id,
            reward_function_version=self._reward_function_version,
            height=self._render_height,
            width=self._render_width,
        )
        tasks = [task for task in benchmark.train_tasks if task.env_name == self.BENCHMARK_ENV_NAME]
        if not tasks:
            raise ValueError(f"No MetaWorld tasks found for env_name={self.BENCHMARK_ENV_NAME}")
        return benchmark, env, tasks

    def _validate_env(self, env, tasks):
        env.set_task(tasks[0])
        env.reset(seed=int(self._rng.integers(0, 2**31 - 1)))
        if self._requires_render_backend():
            self._render_obs(env)

    def _load_env_with_backend_fallback(self, render_backend, render_backend_priority, allow_software_render_fallback):
        candidates = self._resolve_backend_candidates(
            render_backend=render_backend,
            render_backend_priority=render_backend_priority,
            allow_software_render_fallback=allow_software_render_fallback,
        )

        errors = []
        for backend in candidates:
            env = None
            self._configure_render_backend(backend)
            try:
                benchmark, env, tasks = self._build_env()
                self._validate_env(env, tasks)
                self._benchmark = benchmark
                self._env = env
                self._tasks = tasks
                self._selected_render_backend = backend
                if self._requires_render_backend():
                    print(f"[PegInsertSideReverseSparseStandaloneEnv] Using render backend '{backend}'")
                return
            except Exception as exc:
                errors.append((backend, exc))
                try:
                    env.close()
                except Exception:
                    pass

        details = "; ".join(
            f"{backend or 'none'} -> {type(err).__name__}: {err}"
            for backend, err in errors
        )
        raise ImportError(
            "Failed to initialize standalone reverse sparse peg task. "
            f"Tried: {candidates}. Details: {details}"
        )

    def _sample_task(self):
        index = int(self._rng.integers(0, len(self._tasks)))
        return self._tasks[index]

    def _flatten_obs(self, obs):
        return np.asarray(obs, dtype=np.float32).reshape(-1)

    def _render_single_camera(self, env, camera_id: int):
        renderer = env.mujoco_renderer
        previous_camera_id = renderer.camera_id
        renderer.camera_id = int(camera_id)
        try:
            frame = renderer.render("rgb_array")
        finally:
            renderer.camera_id = previous_camera_id
        return np.asarray(frame, dtype=np.uint8)

    def _render_obs(self, env=None):
        env = env or self._env
        frames = [self._render_single_camera(env, camera_id) for camera_id in self._camera_ids]
        if self._num_cams == 1:
            return frames[0]
        return frames

    def _composite_render(self, env=None):
        env = env or self._env
        frames = [self._render_single_camera(env, camera_id) for camera_id in self._camera_ids]
        if len(frames) == 1:
            return frames[0]
        target_height = max(frame.shape[0] for frame in frames)
        padded = []
        for frame in frames:
            if frame.shape[0] == target_height:
                padded.append(frame)
                continue
            pad = target_height - frame.shape[0]
            padded.append(np.pad(frame, ((0, pad), (0, 0), (0, 0)), mode="constant"))
        return np.concatenate(padded, axis=1)

    def _extract_obs(self, obs):
        if self._observation_type == "image":
            return self._render_obs()
        return self._flatten_obs(obs)

    def reset(self):
        self._env.set_task(self._sample_task())
        obs, _info = self._env.reset(seed=int(self._rng.integers(0, 2**31 - 1)))
        return self._extract_obs(obs)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(self._env.action_space.shape)
        action = np.clip(action, self._env.action_space.low, self._env.action_space.high)

        obs, reward, terminated, truncated, info = self._env.step(action)
        obs = self._extract_obs(obs)
        done = bool(terminated or truncated)
        info = dict(info or {})
        if "success" in info:
            info["is_success"] = bool(info["success"])
        return obs, float(reward), done, info

    @property
    def obs_dim(self):
        return self._obs_dim

    @property
    def action_dim(self):
        return self._action_dim

    def render(self):
        return self._composite_render(self._env)

    def close(self):
        if self._env is not None:
            self._env.close()
