from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
import traceback

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from policy.make_policy import make_policy
from sim_env.envs.make_env import make_env
from scripts.utils import load_config, save_gif, train_cfg


EXAM_ROOT = PROJECT_ROOT / "exam"


def _find_policy_checkpoints(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("*/policy.ckpt") if path.is_file())


def _load_exam_config(exam_dir: Path) -> dict:
    config_path = exam_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _policy_config_for_eval(config: dict, checkpoint_path: Path) -> dict:
    policy_config = deepcopy(config["policy"])
    if "operator_intrinsic_reward" in config:
        policy_config["operator_intrinsic_reward"] = deepcopy(config.get("operator_intrinsic_reward", {}))
    checkpoint_cfg = dict(policy_config.get("checkpoint", {}))
    checkpoint_cfg["load"] = True
    checkpoint_cfg["path"] = str(checkpoint_path.resolve())
    checkpoint_cfg["save"] = False
    policy_config["checkpoint"] = checkpoint_cfg
    return policy_config


def _env_config_for_eval(config: dict) -> dict:
    env_config = deepcopy(config["env"])
    render_config = dict(env_config.get("render", {}))
    render_config["enabled"] = True
    render_config["save_frames"] = False
    render_config["save_video"] = False
    env_config["render"] = render_config
    if "persistent_exploration" in config:
        env_config["persistent_exploration"] = deepcopy(config.get("persistent_exploration", {}))
    return env_config


def _success(success_metric: str, threshold: float, episode_return: float, info: dict) -> bool:
    if success_metric == "info_success":
        return bool(info.get("success", info.get("is_success", False)))
    if success_metric == "episode_return_threshold":
        return float(episode_return) >= float(threshold)
    if success_metric == "episode_return_positive":
        return float(episode_return) > float(threshold)
    return False


def _preview_transition(policy, next_obs, reward: float, raw_done: bool) -> dict:
    preview = getattr(policy, "preview_transition", None)
    if callable(preview):
        return dict(preview(next_obs, done=bool(raw_done), reward=float(reward)) or {})
    return {}


def _finalize_persistent(env, action, next_obs, raw_done: bool, info: dict, diagnostics: dict) -> dict:
    finalize = getattr(env, "finalize_step", None)
    if callable(finalize):
        payload = finalize(action=action, next_obs=next_obs, raw_done=raw_done, info=info or {}, diagnostics=diagnostics)
        if isinstance(payload, dict):
            return payload
    return {"done": bool(raw_done), "reset_reason": ("env_done" if raw_done else "NA")}


def eval_checkpoint(checkpoint_path: Path, output_root: Path, seconds: float, gif_fps: int, max_frames: int) -> dict:
    exam_dir = checkpoint_path.parent
    config = _load_exam_config(exam_dir)
    train = train_cfg(config)
    success_metric = str(train.get("eval_success_metric", "episode_return_positive"))
    success_threshold = float(train.get("eval_success_threshold", 0.0))
    max_episode_steps = train.get("max_episode_steps")

    out_dir = output_root / exam_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(_env_config_for_eval(config))
    policy = make_policy(_policy_config_for_eval(config, checkpoint_path), env, exam_dir=exam_dir)

    frames = []
    episodes = []
    obs = env.reset()
    reset = getattr(policy, "reset", None)
    if callable(reset):
        reset()
    clear_history = getattr(env, "clear_persistent_history", None)
    if callable(clear_history):
        clear_history()

    episode_return = 0.0
    episode_length = 0
    episode_index = 1
    success_count = 0
    total_steps = 0
    final_info = {}
    start = time.monotonic()
    next_frame_at = start
    frame_interval = 1.0 / max(1, int(gif_fps))

    try:
        while time.monotonic() - start < float(seconds):
            now = time.monotonic()
            if len(frames) < max_frames and now >= next_frame_at:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
                next_frame_at = now + frame_interval

            action = policy.act(obs)
            next_obs, reward, raw_done, info = env.step(action)
            info = dict(info or {})
            diagnostics = _preview_transition(policy, next_obs, reward=float(reward), raw_done=bool(raw_done))
            persistent = _finalize_persistent(env, action, next_obs, bool(raw_done), info, diagnostics)

            obs = next_obs
            done = bool(persistent.get("done", False))
            episode_return += float(reward)
            episode_length += 1
            total_steps += 1
            final_info = info

            if max_episode_steps is not None and episode_length >= int(max_episode_steps):
                done = True
            if done:
                ok = _success(success_metric, success_threshold, episode_return, final_info)
                success_count += int(ok)
                episodes.append(
                    {
                        "episode_index": episode_index,
                        "episode_return": float(episode_return),
                        "episode_length": int(episode_length),
                        "success": bool(ok),
                    }
                )
                episode_index += 1
                episode_return = 0.0
                episode_length = 0
                obs = env.reset()
                if callable(reset):
                    reset()

        if episode_length > 0:
            ok = _success(success_metric, success_threshold, episode_return, final_info)
            success_count += int(ok)
            episodes.append(
                {
                    "episode_index": episode_index,
                    "episode_return": float(episode_return),
                    "episode_length": int(episode_length),
                    "success": bool(ok),
                    "partial": True,
                }
            )

        if not frames:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        gif_path = out_dir / "eval_60s.gif"
        if frames:
            save_gif(frames, gif_path, fps=gif_fps)
        else:
            gif_path = None

        summary = {
            "checkpoint": str(checkpoint_path),
            "exam": exam_dir.name,
            "duration_seconds_target": float(seconds),
            "duration_seconds_actual": float(time.monotonic() - start),
            "env_steps": int(total_steps),
            "episodes": episodes,
            "success_count": int(success_count),
            "success_rate": float(success_count / len(episodes)) if episodes else 0.0,
            "success_metric": success_metric,
            "success_threshold": success_threshold,
            "gif_path": str(gif_path) if gif_path is not None else None,
            "frame_count": len(frames),
        }
        getter = getattr(env, "get_persistent_stats", None)
        if callable(getter):
            payload = getter()
            if isinstance(payload, dict):
                summary.update(payload)
        (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return {"status": "ok", "exam": exam_dir.name, "summary_path": str(out_dir / "eval_summary.json"), **summary}
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run wall-clock timed eval for all exam policy checkpoints.")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--output-dir", default="outputs/checkpoint_eval_60s")
    parser.add_argument("--gif-fps", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=1200)
    parser.add_argument("--only", nargs="*", default=None, help="Optional exam names to evaluate.")
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoints = _find_policy_checkpoints(EXAM_ROOT)
    if args.only:
        allowed = set(args.only)
        checkpoints = [path for path in checkpoints if path.parent.name in allowed]

    results = []
    for idx, checkpoint_path in enumerate(checkpoints, 1):
        print(f"[{idx}/{len(checkpoints)}] eval {checkpoint_path}", flush=True)
        try:
            result = eval_checkpoint(
                checkpoint_path=checkpoint_path,
                output_root=output_root,
                seconds=args.seconds,
                gif_fps=args.gif_fps,
                max_frames=args.max_frames,
            )
            print(f"  ok steps={result['env_steps']} summary={result['summary_path']}", flush=True)
        except Exception as exc:  # noqa: BLE001 - batch eval should continue and report failures.
            result = {
                "status": "failed",
                "checkpoint": str(checkpoint_path),
                "exam": checkpoint_path.parent.name,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            print(f"  failed: {exc}", flush=True)
        results.append(result)
        (output_root / "batch_eval_summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Saved batch summary: {output_root / 'batch_eval_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
