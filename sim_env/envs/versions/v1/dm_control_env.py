import importlib
import os
import sys

import numpy as np

from .base_env import BaseEnv


class DMControlEnv(BaseEnv):
    """Minimal DeepMind Control Suite wrapper using the project BaseEnv API."""

    def __init__(
        self,
        domain_name,
        task_name,
        render_height=240,
        render_width=320,
        camera_id=0,
        camera_indices=None,
        observation_type="vector",
        num_cams=1,
        render_backend=None,
        render_backend_priority=None,
        allow_software_render_fallback=False,
        render_enabled=False,
    ):
        self._observation_type = str(observation_type)
        self._render_enabled = bool(render_enabled)
        self._selected_render_backend = None
        suite = self._import_suite_with_backend_fallback(
            render_backend=render_backend,
            render_backend_priority=render_backend_priority,
            allow_software_render_fallback=allow_software_render_fallback,
        )
        self.env = suite.load(domain_name=domain_name, task_name=task_name)
        self._action_spec = self.env.action_spec()
        self._render_height = render_height
        self._render_width = render_width
        self._camera_id = camera_id
        if isinstance(camera_indices, (list, tuple)) and camera_indices:
            self._camera_ids = [int(value) for value in camera_indices]
            self._num_cams = len(self._camera_ids)
            self._camera_id = int(self._camera_ids[0])
        else:
            self._num_cams = max(1, int(num_cams))
            self._camera_ids = [int(camera_id) + offset for offset in range(self._num_cams)]
        if self._requires_render_backend():
            print(f"[DMControlEnv] Using render backend '{self._selected_render_backend}'")

        time_step = self.env.reset()
        obs = self._extract_obs(time_step)

        self._obs_dim = int(np.asarray(obs, dtype=np.float32).size)
        self._action_dim = int(np.prod(self._action_spec.shape))

    def _requires_render_backend(self):
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

    def _purge_dm_control_modules(self):
        for module_name in list(sys.modules.keys()):
            if module_name == "dm_control" or module_name.startswith("dm_control."):
                sys.modules.pop(module_name, None)

    def _import_suite_with_backend_fallback(
        self,
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
        for index, backend in enumerate(candidates):
            if index > 0:
                self._purge_dm_control_modules()
            self._configure_render_backend(backend)
            try:
                suite = importlib.import_module("dm_control.suite")
                self._selected_render_backend = backend
                if index > 0:
                    print(f"[DMControlEnv] Switched render backend fallback to '{backend}'")
                return suite
            except Exception as exc:
                errors.append((backend, exc))

        details = "; ".join(
            f"{backend or 'none'} -> {type(err).__name__}: {err}"
            for backend, err in errors
        )
        raise ImportError(
            "Failed to initialize dm_control with configured render backends. "
            f"Tried: {candidates}. Details: {details}"
        )

    def _flatten_obs(self, obs_dict):
        return np.concatenate(
            [value.ravel() for value in obs_dict.values()]
        ).astype(np.float32)

    def _render_obs(self):
        frames = [
            self.env.physics.render(
                height=self._render_height,
                width=self._render_width,
                camera_id=camera_id,
            ).astype(np.uint8)
            for camera_id in self._camera_ids
        ]
        if self._num_cams == 1:
            return frames[0]
        return frames

    def _extract_obs(self, time_step):
        if self._observation_type == "image":
            return self._render_obs()
        return self._flatten_obs(time_step.observation)

    def reset(self):
        time_step = self.env.reset()
        return self._extract_obs(time_step)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(self._action_spec.shape)
        action = np.clip(action, self._action_spec.minimum, self._action_spec.maximum)

        time_step = self.env.step(action)

        obs = self._extract_obs(time_step)
        reward = 0.0 if time_step.reward is None else float(time_step.reward)
        done = bool(time_step.last())
        info = {}

        return obs, reward, done, info

    @property
    def obs_dim(self):
        return self._obs_dim

    @property
    def action_dim(self):
        return self._action_dim

    def render(self):
        return self.env.physics.render(
            height=self._render_height,
            width=self._render_width,
            camera_id=self._camera_id,
        )

    def close(self):
        self.env.close()
