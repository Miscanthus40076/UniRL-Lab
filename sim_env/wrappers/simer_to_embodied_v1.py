from __future__ import annotations

import numpy as np


class SimerToEmbodiedEnv:
    """Adapter from the project BaseEnv API to DreamerV3's embodied.Env API."""

    def __init__(
        self,
        env,
        *,
        obs_key: str = "observation",
        action_key: str = "action",
        action_low: float = -1.0,
        action_high: float = 1.0,
        max_episode_steps: int | None = None,
        terminal_on_done: bool = False,
    ):
        self.env = env
        self.obs_key = str(obs_key)
        self.action_key = str(action_key)
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self.max_episode_steps = None if max_episode_steps is None else int(max_episode_steps)
        self.terminal_on_done = bool(terminal_on_done)
        self._done = True
        self._step = 0
        self._obs_space = None
        self._act_space = None

        initial = self.env.reset()
        self._last_obs = self._format_obs_value(initial)
        self._done = True

    @property
    def obs_space(self):
        if self._obs_space is None:
            elements = self._import_elements()
            obs = self._last_obs
            self._obs_space = {
                "reward": elements.Space(np.float32),
                "is_first": elements.Space(bool),
                "is_last": elements.Space(bool),
                "is_terminal": elements.Space(bool),
                self.obs_key: elements.Space(obs.dtype, obs.shape),
            }
        return self._obs_space

    @property
    def act_space(self):
        if self._act_space is None:
            elements = self._import_elements()
            shape = (int(self.env.action_dim),)
            low = np.full(shape, self.action_low, dtype=np.float32)
            high = np.full(shape, self.action_high, dtype=np.float32)
            self._act_space = {
                "reset": elements.Space(bool),
                self.action_key: elements.Space(np.float32, shape, low, high),
            }
        return self._act_space

    def step(self, action):
        action = dict(action or {})
        reset = bool(np.asarray(action.get("reset", False)).item())
        if reset or self._done:
            obs = self.env.reset()
            self._done = False
            self._step = 0
            self._last_obs = self._format_obs_value(obs)
            return self._obs(
                self._last_obs,
                reward=0.0,
                is_first=True,
                is_last=False,
                is_terminal=False,
            )

        raw_action = np.asarray(action[self.action_key], dtype=np.float32).reshape(-1)
        obs, reward, done, info = self.env.step(raw_action)
        info = dict(info or {})
        self._step += 1
        time_limit_done = self.max_episode_steps is not None and self._step >= self.max_episode_steps
        is_last = bool(done or time_limit_done)
        is_terminal = bool(done and self.terminal_on_done and not time_limit_done)
        self._done = is_last
        self._last_obs = self._format_obs_value(obs)
        return self._obs(
            self._last_obs,
            reward=float(reward),
            is_first=False,
            is_last=is_last,
            is_terminal=is_terminal,
            info=info,
        )

    def close(self):
        close = getattr(self.env, "close", None)
        if callable(close):
            close()

    def _obs(self, obs, *, reward: float, is_first: bool, is_last: bool, is_terminal: bool, info: dict | None = None):
        payload = {
            "reward": np.float32(reward),
            "is_first": bool(is_first),
            "is_last": bool(is_last),
            "is_terminal": bool(is_terminal),
            self.obs_key: obs,
        }
        for key, value in (info or {}).items():
            if key.startswith("log/"):
                payload[key] = np.asarray(value)
        return payload

    def _format_obs_value(self, obs):
        if isinstance(obs, (list, tuple)):
            values = [np.asarray(value) for value in obs]
            if not values:
                raise ValueError("Cannot adapt an empty observation list")
            if all(value.ndim == 3 and value.shape[:2] == values[0].shape[:2] for value in values):
                obs = np.concatenate(values, axis=-1)
            else:
                obs = np.stack(values, axis=0)
        obs = np.asarray(obs)
        if obs.dtype == np.uint8:
            return obs
        return obs.astype(np.float32, copy=False)

    def _import_elements(self):
        try:
            import elements
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "simer_to_embodied requires the DreamerV3 dependency package 'elements'. "
                "Install policy/REF_policy/dreamerv3/requirements.txt in the active environment."
            ) from exc
        return elements
