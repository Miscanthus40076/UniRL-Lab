from __future__ import annotations

from pathlib import Path
import json

import imageio.v2 as imageio
import numpy as np

from .bidirectional_env_factory import make_bidirectional_env


def _default_config(seed: int = 0) -> dict:
    return {
        "seed": int(seed),
        "output_dir": f"outputs/online_bidreamer_from_scratch/seed_{seed}/env_sanity",
        "env": {
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
        },
        "forward_env": {
            "name": "ball_in_cup_catch",
            "domain_name": "ball_in_cup",
            "task_name": "catch",
        },
        "reverse_env": {
            "name": "ball_in_cup_release",
            "domain_name": "ball_in_cup",
            "task_name": "release",
        },
        "sanity": {
            "num_reset_images": 2,
            "random_rollout_steps": 100,
            "gif_fps": 20,
        },
    }


def _save_image(path: Path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, frame)
    return path


def _save_gif(path: Path, frames, fps: int = 20):
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, format="GIF", fps=fps)
    return path


def _random_action(env, rng: np.random.Generator):
    return rng.uniform(-1.0, 1.0, size=(env.action_dim,)).astype(np.float32)


def _rollout_direction(env, direction: str, output_dir: Path, rollout_steps: int, gif_fps: int, require_reset_render: bool):
    summary = {
        "direction": direction,
        "wrapper_direction": env.direction,
        "reset_images": [],
        "rollout_steps": 0,
        "reward_sum": 0.0,
        "success_count": 0,
        "done_count": 0,
        "all_info_directions_match": True,
        "rollout_gif": None,
    }
    frames = []
    for reset_idx in range(2):
        _ = env.reset()
        frame = env.render()
        if frame is None and require_reset_render:
            raise RuntimeError(f"{direction} reset render returned None")
        image_path = output_dir / f"{direction}_reset_{reset_idx:03d}.png"
        _save_image(image_path, frame)
        summary["reset_images"].append(str(image_path))
        if reset_idx == 0:
            frames.append(frame)

    rng = np.random.default_rng(0)
    obs = env.reset()
    frame = env.render()
    if frame is None and require_reset_render:
        raise RuntimeError(f"{direction} rollout initial render returned None")
    frames = [frame]
    for _ in range(int(rollout_steps)):
        action = _random_action(env, rng)
        obs, reward, done, info = env.step(action)
        info_direction = info.get("bidreamer_direction")
        if info_direction != direction:
            summary["all_info_directions_match"] = False
        summary["reward_sum"] += float(reward)
        summary["success_count"] += int(bool(info.get("success") or info.get("is_success")))
        summary["done_count"] += int(bool(done))
        summary["rollout_steps"] += 1
        frames.append(env.render())
        if done:
            obs = env.reset()
            frames.append(env.render())
    gif_path = output_dir / f"{direction}_random_rollout.gif"
    _save_gif(gif_path, frames, fps=gif_fps)
    summary["rollout_gif"] = str(gif_path)
    return summary


def run_env_sanity(config: dict | None = None, seed: int = 0) -> dict:
    config = dict(_default_config(seed) if config is None else config)
    output_dir = Path(config.get("output_dir", f"outputs/online_bidreamer_from_scratch/seed_{seed}/env_sanity"))
    output_dir.mkdir(parents=True, exist_ok=True)
    sanity_cfg = config.get("sanity", {})
    rollout_steps = int(sanity_cfg.get("random_rollout_steps", 100))
    gif_fps = int(sanity_cfg.get("gif_fps", 20))
    require_reset_render = True

    forward_env = make_bidirectional_env(config, "forward")
    reverse_env = make_bidirectional_env(config, "reverse")
    try:
        forward_summary = _rollout_direction(
            forward_env,
            "forward",
            output_dir,
            rollout_steps=rollout_steps,
            gif_fps=gif_fps,
            require_reset_render=require_reset_render,
        )
        reverse_summary = _rollout_direction(
            reverse_env,
            "reverse",
            output_dir,
            rollout_steps=rollout_steps,
            gif_fps=gif_fps,
            require_reset_render=require_reset_render,
        )
    finally:
        forward_env.close()
        reverse_env.close()

    summary = {
        "seed": int(seed),
        "output_dir": str(output_dir),
        "forward": forward_summary,
        "reverse": reverse_summary,
        "checks": {
            "forward_wrapper_direction_ok": forward_summary["wrapper_direction"] == "forward",
            "reverse_wrapper_direction_ok": reverse_summary["wrapper_direction"] == "reverse",
            "forward_info_direction_ok": bool(forward_summary["all_info_directions_match"]),
            "reverse_info_direction_ok": bool(reverse_summary["all_info_directions_match"]),
        },
    }
    summary_path = output_dir / "env_direction_sanity.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary
