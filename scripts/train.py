from pathlib import Path
import argparse
from copy import deepcopy
import json
import os
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from policy.make_policy import make_policy
from sim_env.envs.make_env import make_env
from utils import CsvLog, JsonlLog, load_config, metrics_row_from_payload, multi_task_enabled, normalize_policy_metrics, output_dir_for_task, plot_policy_metrics_jsonl, resolve_env_tasks, run_policy_evaluation, train_cfg, validate_exam_config


EXAM_ROOT = PROJECT_ROOT / "exam"


def maybe_update(policy, batch):
    update = getattr(policy, "update", None)
    if not callable(update):
        return normalize_policy_metrics(None)
    return normalize_policy_metrics(update(batch))


def _policy_config_for_task(config):
    policy_config = deepcopy(config["policy"])
    checkpoint_cfg = dict(policy_config.get("checkpoint", {}))
    if multi_task_enabled(config):
        checkpoint_cfg["load"] = False
        checkpoint_cfg["path"] = None
    policy_config["checkpoint"] = checkpoint_cfg
    return policy_config


def _checkpoint_path_for_task(config, policy_config, output_dir):
    checkpoint_cfg = policy_config.get("checkpoint", {})
    checkpoint_path = checkpoint_cfg.get("path")
    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)
        if multi_task_enabled(config) and not checkpoint_path.is_absolute():
            return output_dir / checkpoint_path
        return checkpoint_path
    return output_dir / "policy.ckpt"


def _save_checkpoint(policy, policy_config, config, output_dir):
    checkpoint_cfg = policy_config.get("checkpoint", {})
    save = getattr(policy, "save", None)
    if not callable(save) or not bool(checkpoint_cfg.get("save", False)):
        return None

    checkpoint_path = _checkpoint_path_for_task(config, policy_config, output_dir)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    save(checkpoint_path)
    return checkpoint_path


def _write_run_manifest(config, policy_config, output_dir, task):
    manifest = {
        "task_name": task["name"],
        "train": dict(config.get("train", {})),
        "env": dict(task.get("env", {})),
        "policy": dict(policy_config),
        "artifacts": {
            "metrics_csv": str(output_dir / "metrics.csv"),
            "policy_metrics_jsonl": str(output_dir / "policy_metrics.jsonl"),
            "plots_dir": str(output_dir / "plots"),
            "eval_dir": str(output_dir / "eval"),
            "checkpoint_path": str(_checkpoint_path_for_task(config, policy_config, output_dir)),
        },
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest_path


def _train_env_config(task_env):
    env_config = deepcopy(task_env)
    render_config = dict(env_config.get("render", {}))
    render_config["enabled"] = False
    render_config["save_frames"] = False
    render_config["save_video"] = False
    env_config["render"] = render_config
    return env_config


def _eval_env_config(task_env):
    env_config = deepcopy(task_env)
    render_config = dict(env_config.get("render", {}))
    render_config["enabled"] = True
    render_config["save_frames"] = False
    render_config["save_video"] = False
    env_config["render"] = render_config
    return env_config


def train_task(config, exam_dir, task):
    validate_exam_config(config)
    cfg = train_cfg(config)

    total_steps = int(cfg.get("total_steps", 1000))
    max_episode_steps = cfg.get("max_episode_steps")
    log_interval = int(cfg.get("log_interval", 100))
    record_interval = int(cfg.get("record_interval", log_interval))
    save_interval = int(cfg.get("save_interval", 0))
    eval_interval = int(cfg.get("eval_interval", 0))
    eval_episodes = int(cfg.get("eval_episodes", 0))

    output_dir = output_dir_for_task(exam_dir, config, task["name"])
    metrics_path = output_dir / "metrics.csv"
    policy_metrics_path = output_dir / "policy_metrics.jsonl"
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(_train_env_config(task["env"]))
    eval_env = None
    policy_config = _policy_config_for_task(config)
    policy = make_policy(policy_config, env, exam_dir=exam_dir)
    log = CsvLog(metrics_path)
    policy_metrics_log = JsonlLog(policy_metrics_path)
    manifest_path = _write_run_manifest(config, policy_config, output_dir, task)

    print(f"Training task: {task['name']}")
    print(f"Saved run manifest: {manifest_path}")
    obs = env.reset()
    episode_return = 0.0
    episode_length = 0
    episode_index = 0
    start_time = time.time()

    try:
        for step in range(1, total_steps + 1):
            action = policy.act(obs)
            next_obs, reward, done, info = env.step(action)
            forced_done = max_episode_steps is not None and episode_length + 1 >= int(max_episode_steps)
            done = bool(done or forced_done)

            episode_return += float(reward)
            episode_length += 1

            policy_metrics = maybe_update(
                policy,
                {
                    "obs": obs,
                    "action": action,
                    "reward": reward,
                    "next_obs": next_obs,
                    "done": done,
                    "truncated": bool(forced_done),
                    "info": info,
                    "step": step,
                },
            )
            metrics = metrics_row_from_payload(policy_metrics)

            if step % record_interval == 0 or done or step == total_steps:
                elapsed = max(time.time() - start_time, 1e-6)
                train_scalars = {
                    "step": step,
                    "episode_index": episode_index,
                    "episode_return": episode_return,
                    "episode_length": episode_length,
                    "fps": step / elapsed,
                }
                row = dict(train_scalars)
                row.update(metrics)
                log.add(row)
                policy_metrics_log.add(
                    {
                        "step": step,
                        "episode_index": episode_index,
                        "episode_return": episode_return,
                        "episode_length": episode_length,
                        "train_scalars": train_scalars,
                        "policy_metrics": policy_metrics,
                    }
                )
                plot_policy_metrics_jsonl(policy_metrics_path, plots_dir, x_key="step")

            if step % log_interval == 0:
                print(f"step={step} episode={episode_index} return={episode_return:.4f} length={episode_length}")

            if eval_episodes > 0 and eval_interval > 0 and step % eval_interval == 0:
                if eval_env is None:
                    eval_env = make_env(_eval_env_config(task["env"]))
                eval_dir = output_dir / "eval" / f"step_{step:07d}"
                print(f"Running evaluation at step={step} ...")
                run_policy_evaluation(
                    env=eval_env,
                    policy=policy,
                    output_dir=eval_dir,
                    eval_episodes=eval_episodes,
                    max_episode_steps=max_episode_steps,
                    gif_fps=20,
                )
                reset = getattr(policy, "reset", None)
                if callable(reset):
                    reset()

            if save_interval > 0 and step % save_interval == 0:
                checkpoint_path = _save_checkpoint(policy, policy_config, config, output_dir)
                if checkpoint_path is not None:
                    print(f"Saved checkpoint: {checkpoint_path}")

            obs = next_obs

            if done:
                reset = getattr(policy, "reset", None)
                if callable(reset):
                    reset()
                episode_index += 1
                obs = env.reset()
                episode_return = 0.0
                episode_length = 0
    finally:
        checkpoint_path = _save_checkpoint(policy, policy_config, config, output_dir)
        if checkpoint_path is not None:
            print(f"Saved final checkpoint: {checkpoint_path}")
        env.close()
        if eval_env is not None:
            eval_env.close()

    plot_paths = plot_policy_metrics_jsonl(policy_metrics_path, plots_dir, x_key="step")
    print(f"Saved metrics CSV: {metrics_path}")
    print(f"Saved policy metrics JSONL: {policy_metrics_path}")
    if plot_paths:
        print("Saved metric plots:")
        for path in plot_paths:
            print(f"  {path}")


def train(exam_name):
    config, exam_dir = load_config(EXAM_ROOT, exam_name)
    tasks = resolve_env_tasks(config)
    for task in tasks:
        train_task(config, exam_dir, task)


def parse_args():
    parser = argparse.ArgumentParser(description="Train an exam from exam/<name>/config.yaml")
    parser.add_argument("exam_name", help="Name of the exam directory under exam/")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.exam_name)
