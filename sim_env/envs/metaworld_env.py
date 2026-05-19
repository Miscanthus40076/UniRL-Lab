import os
import importlib
from typing import Sequence

import numpy as np

from .base_env import BaseEnv


_CUSTOM_METAWORLD_ENVS = {
    "peg-insert-side-sparse-v3": {
        "benchmark_env_name": "peg-insert-side-v3",
        "module": "sim_env.envs.sawyer_peg_insertion_side_sparse_v3",
        "class_name": "SawyerPegInsertionSideSparseEnvV3",
    },
    "peg-insert-side-reverse-v3": {
        "benchmark_env_name": "peg-insert-side-v3",
        "module": "sim_env.envs.sawyer_peg_insertion_side_reverse_v3",
        "class_name": "SawyerPegInsertionSideReverseEnvV3",
    },
    "peg-insert-side-reverse-sparse-v3": {
        "benchmark_env_name": "peg-insert-side-v3",
        "module": "sim_env.envs.sawyer_peg_insertion_side_reverse_v3",
        "class_name": "SawyerPegInsertionSideReverseEnvV3",
    },
    "peg-insert-side-reverse-shaped-v3": {
        "benchmark_env_name": "peg-insert-side-v3",
        "module": "sim_env.envs.sawyer_peg_insertion_side_reverse_shaped_v3",
        "class_name": "SawyerPegInsertionSideReverseShapedEnvV3",
    },
}


class MetaWorldEnv(BaseEnv):
    """Minimal MetaWorld ML1 wrapper using the project BaseEnv API."""

    def __init__(
        self,
        env_name: str,
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
            env_name=str(env_name),
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

    def _build_env(self, env_name: str):
        import metaworld

        custom_env_spec = _CUSTOM_METAWORLD_ENVS.get(env_name)
        benchmark_env_name = env_name if custom_env_spec is None else custom_env_spec["benchmark_env_name"]
        benchmark = metaworld.ML1(benchmark_env_name, seed=self._seed)
        if custom_env_spec is None:
            env_cls = benchmark.train_classes[benchmark_env_name]
        else:
            module = importlib.import_module(str(custom_env_spec["module"]))
            env_cls = getattr(module, str(custom_env_spec["class_name"]))
        env = env_cls(
            render_mode="rgb_array" if self._requires_render_backend() else None,
            camera_id=self._camera_id,
            reward_function_version=self._reward_function_version,
            height=self._render_height,
            width=self._render_width,
        )
        tasks = [task for task in benchmark.train_tasks if task.env_name == benchmark_env_name]
        if not tasks:
            raise ValueError(f"No MetaWorld tasks found for env_name={benchmark_env_name}")
        return benchmark, env, tasks

    def _validate_env(self, env, tasks):
        env.set_task(tasks[0])
        env.reset(seed=int(self._rng.integers(0, 2**31 - 1)))
        if self._requires_render_backend():
            self._render_obs(env)

    def _load_env_with_backend_fallback(
        self,
        env_name: str,
        render_backend,
        render_backend_priority,
        allow_software_render_fallback,
    ):
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
                benchmark, env, tasks = self._build_env(env_name)
                self._validate_env(env, tasks)
                self._benchmark = benchmark
                self._env = env
                self._tasks = tasks
                self._selected_render_backend = backend
                if self._requires_render_backend():
                    print(f"[MetaWorldEnv] Using render backend '{backend}'")
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
            "Failed to initialize MetaWorld with configured render backends. "
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
