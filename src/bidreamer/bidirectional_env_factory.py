from __future__ import annotations

from copy import deepcopy

from sim_env.envs.make_env import make_env


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


def _direction_env_config(config: dict, direction: str) -> dict:
    base = deepcopy(config.get("env", _default_env_config()))
    forward_override = deepcopy(config.get("forward_env", {}))
    reverse_override = deepcopy(config.get("reverse_env", {}))
    if direction == "forward":
        env_cfg = base
        env_cfg.update(forward_override)
        env_cfg["domain_name"] = forward_override.get("domain_name", env_cfg.get("domain_name", "ball_in_cup"))
        env_cfg["task_name"] = forward_override.get("task_name", env_cfg.get("task_name", "catch"))
        env_cfg["name"] = forward_override.get("name", env_cfg.get("name", "ball_in_cup_catch"))
        return env_cfg
    if direction == "reverse":
        env_cfg = base
        env_cfg.update(reverse_override)
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
