from __future__ import annotations

from copy import deepcopy

from sim_env.envs.make_env import make_env


def _deep_merge(base, override):
    if not isinstance(base, dict) or not isinstance(override, dict):
        return deepcopy(override)
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _default_env_config() -> dict:
    return {
        "type": "dmcontrol",
        "name": "ball_in_cup_catch",
        "domain_name": "ball_in_cup",
        "task_name": "catch",
        "observation": {
            "type": "image",
            "num_cams": 1,
        },
        "action": {
            "clip": True,
            "normalize": True,
        },
        "render": {
            "enabled": True,
            "save_frames": False,
            "save_video": False,
            "height": 64,
            "width": 64,
            "camera_id": 0,
            "backend": None,
            "backend_priority": ["egl", "osmesa"],
            "allow_software_render_fallback": True,
        },
        "isaac": {
            "task_name": None,
            "num_envs": 1,
            "headless": True,
            "create_cam": None,
        },
    }


class BidirectionalEnvWrapper:
    def __init__(self, env, direction: str):
        self.env = env
        self.direction = str(direction)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        merged = dict(info or {})
        merged["bidreamer_direction"] = self.direction
        return obs, reward, done, merged

    @property
    def obs_dim(self):
        return self.env.obs_dim

    @property
    def action_dim(self):
        return self.env.action_dim

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


def _bidirectional_cfg(config: dict) -> dict:
    train_cfg = config.get("train", {})
    if not isinstance(train_cfg, dict):
        return {}
    bidirectional = train_cfg.get("bidirectional", {})
    if bidirectional is None:
        return {}
    if not isinstance(bidirectional, dict):
        raise TypeError("train.bidirectional must be a mapping")
    return bidirectional


def _direction_env_config(config: dict, direction: str) -> dict:
    base = deepcopy(config.get("env", _default_env_config()))
    bidirectional = _bidirectional_cfg(config)
    forward_override = _deep_merge(config.get("forward_env", {}), bidirectional.get("forward_env", {}))
    reverse_override = _deep_merge(config.get("reverse_env", {}), bidirectional.get("reverse_env", {}))
    if direction == "forward":
        env_cfg = _deep_merge(base, forward_override)
        env_cfg["domain_name"] = forward_override.get("domain_name", env_cfg.get("domain_name", "ball_in_cup"))
        env_cfg["task_name"] = forward_override.get("task_name", env_cfg.get("task_name", "catch"))
        env_cfg["name"] = forward_override.get("name", env_cfg.get("name", "ball_in_cup_catch"))
        return env_cfg
    if direction == "reverse":
        env_cfg = _deep_merge(base, reverse_override)
        env_cfg["domain_name"] = reverse_override.get("domain_name", env_cfg.get("domain_name", "ball_in_cup"))
        env_cfg["task_name"] = reverse_override.get("task_name", "release")
        env_cfg["name"] = reverse_override.get("name", "ball_in_cup_release")
        return env_cfg
    raise ValueError(f"Unsupported direction: {direction}")


def make_bidirectional_env(config: dict, direction: str):
    env_cfg = _direction_env_config(config, direction)
    env = make_env(env_cfg)
    try:
        _ = env.render()
    except Exception as exc:
        if direction == "reverse":
            raise RuntimeError(
                f"Reverse env render/reset sanity failed during creation for task={env_cfg.get('task_name')}"
            ) from exc
        raise
    return BidirectionalEnvWrapper(env, direction=direction)
