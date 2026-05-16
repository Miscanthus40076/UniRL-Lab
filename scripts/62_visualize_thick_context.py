from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize THICK-lite Dreamer context gate and delta diagnostics.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "train_dreamer_thick_lite.yaml"),
        help="Path to a Dreamer-style YAML config.",
    )
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. Defaults to policy.checkpoint.path in config.")
    parser.add_argument("--episodes", type=int, default=3, help="Number of evaluation episodes.")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "thick_lite_dreamer" / "seed_0" / "context_vis"),
        help="Directory for GIFs and plots.",
    )
    parser.add_argument("--gif-fps", type=int, default=20, help="GIF frame rate.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy actions.")
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="Optional per-episode step cap. Defaults to train.max_episode_steps.",
    )
    return parser.parse_args()


def _load_config(path: Path):
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _eval_env_config(config: dict):
    env_config = deepcopy(config["env"])
    render_config = dict(env_config.get("render", {}))
    render_config["enabled"] = True
    render_config["save_frames"] = False
    render_config["save_video"] = False
    env_config["render"] = render_config
    return env_config


def _policy_config(config: dict, checkpoint: str | None):
    policy_config = deepcopy(config["policy"])
    checkpoint_cfg = dict(policy_config.get("checkpoint", {}))
    checkpoint_cfg["load"] = True
    if checkpoint is not None:
        checkpoint_cfg["path"] = checkpoint
    if not checkpoint_cfg.get("path"):
        raise ValueError("Checkpoint path is required, either in config policy.checkpoint.path or via --checkpoint")
    policy_config["checkpoint"] = checkpoint_cfg
    return policy_config


def _overlay_frame(frame, lines):
    import numpy as np
    from PIL import Image, ImageDraw

    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=-1)
    if frame.ndim == 3 and frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    overlay_height = 10 + 16 * len(lines)
    draw.rectangle((0, 0, image.width, overlay_height), fill=(0, 0, 0))
    y = 6
    for line in lines:
        draw.text((8, y), line, fill=(255, 255, 255))
        y += 16
    return np.asarray(image)


def _save_curve(path: Path, title: str, ylabel: str, values):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 3))
    plt.plot(values, linewidth=1.5)
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    args = parse_args()

    from policy.make_policy import make_policy
    from scripts.utils.media import save_gif
    from sim_env.envs.make_env import make_env

    config_path = Path(args.config).resolve()
    config = _load_config(config_path)
    policy_config = _policy_config(config, args.checkpoint)
    thick_cfg = dict(policy_config.get("dreamerv3", {}).get("thick_context", {}))
    if not bool(thick_cfg.get("enabled", False)):
        raise ValueError("Visualization script requires policy.dreamerv3.thick_context.enabled=true")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    max_episode_steps = args.max_episode_steps
    if max_episode_steps is None:
        max_episode_steps = config.get("train", {}).get("max_episode_steps")

    env = make_env(_eval_env_config(config))
    policy = make_policy(policy_config, env, exam_dir=config_path.parent)
    gate_curve = []
    delta_curve = []
    episode_summaries = []
    gif_paths = []

    try:
        for episode_idx in range(1, int(args.episodes) + 1):
            reset = getattr(policy, "reset", None)
            if callable(reset):
                reset()
            obs = env.reset()
            done = False
            step = 0
            episode_return = 0.0
            frames = []

            while not done:
                action = policy.act(obs, deterministic=bool(args.deterministic))
                diagnostics = getattr(policy, "get_diagnostics", lambda: {})()
                context_diag = dict(diagnostics.get("context", {}))
                obs, reward, done, _ = env.step(action)
                step += 1
                episode_return += float(reward)

                gate = float(context_diag.get("context_gate", 0.0))
                delta = float(context_diag.get("context_delta_norm", 0.0))
                gate_curve.append(gate)
                delta_curve.append(delta)

                frame = env.render()
                lines = [
                    f"step: {step}",
                    f"reward: {float(reward):.4f}",
                    f"done: {int(bool(done))}",
                    f"context_gate: {gate:.4f}",
                    f"context_delta_norm: {delta:.4f}",
                ]
                frames.append(_overlay_frame(frame, lines))

                if max_episode_steps is not None and step >= int(max_episode_steps):
                    done = True

            gif_path = output_dir / f"eval_episode_{episode_idx:02d}.gif"
            save_gif(frames, gif_path, fps=int(args.gif_fps))
            gif_paths.append(str(gif_path))
            episode_summaries.append(
                {
                    "episode_index": episode_idx,
                    "episode_return": float(episode_return),
                    "episode_length": int(step),
                    "gif_path": str(gif_path),
                }
            )
            print(f"Episode {episode_idx} return={episode_return:.4f} length={step} saved={gif_path}")
    finally:
        env.close()

    gate_curve_path = output_dir / "context_gate_curve.png"
    delta_curve_path = output_dir / "context_delta_curve.png"
    _save_curve(gate_curve_path, "Context Gate", "gate", gate_curve)
    _save_curve(delta_curve_path, "Context Delta Norm", "delta_norm", delta_curve)

    summary = {
        "config_path": str(config_path),
        "checkpoint": str(policy_config["checkpoint"]["path"]),
        "episodes": episode_summaries,
        "gif_paths": gif_paths,
        "context_gate_curve_path": str(gate_curve_path),
        "context_delta_curve_path": str(delta_curve_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved summary: {summary_path}")
    print(f"Saved gate curve: {gate_curve_path}")
    print(f"Saved delta curve: {delta_curve_path}")


if __name__ == "__main__":
    main()
