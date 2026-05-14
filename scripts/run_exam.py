from pathlib import Path
import argparse
from copy import deepcopy
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from policy.make_policy import make_policy
from sim_env.envs.make_env import make_env
from utils import eval_cfg, load_config, multi_task_enabled, output_dir_for_task, render_cfg, resolve_env_tasks, run_policy_evaluation, save_frame, save_line, train_cfg


EXAM_ROOT = PROJECT_ROOT / "exam"


def _policy_config_for_task(config, exam_dir, task_name: str):
    policy_config = deepcopy(config["policy"])
    checkpoint_cfg = dict(policy_config.get("checkpoint", {}))
    if multi_task_enabled(config):
        checkpoint_path = checkpoint_cfg.get("path")
        task_output_dir = output_dir_for_task(exam_dir, config, task_name)
        if checkpoint_path:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.is_absolute():
                checkpoint_cfg["path"] = str(task_output_dir / checkpoint_path)
        elif bool(checkpoint_cfg.get("load", False)):
            checkpoint_cfg["path"] = str(task_output_dir / "policy.ckpt")
    policy_config["checkpoint"] = checkpoint_cfg
    return policy_config


def run_task(config, exam_dir, task):
    env = make_env(task["env"])
    policy = make_policy(_policy_config_for_task(config, exam_dir, task["name"]), env, exam_dir=exam_dir)
    train = train_cfg(config)
    render = render_cfg({"env": task["env"], "train": train})
    eval_run = eval_cfg(config)

    total_steps = int(train.get("total_steps", 1000))
    frame_interval = int(train.get("record_interval", 100))
    save_plot = True
    max_episode_steps = train.get("max_episode_steps")

    output_dir = output_dir_for_task(exam_dir, config, task["name"])
    frames_dir = output_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running task: {task['name']}")
    if render["enabled"] and render["save_frames"]:
        frames_dir.mkdir(exist_ok=True)

    obs = env.reset()
    episode_reward = 0.0
    rewards = []
    saved_frames = []

    try:
        for step in range(1, total_steps + 1):
            action = policy.act(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            rewards.append(reward)

            if render["enabled"] and render["save_frames"] and step % frame_interval == 0:
                saved_frames.append(save_frame(env, frames_dir, step))

            if done:
                print(f"Episode finished at step {step} with return {episode_reward:.4f}")
                reset = getattr(policy, "reset", None)
                if callable(reset):
                    reset()
                obs = env.reset()
                episode_reward = 0.0
    finally:
        env.close()

    if save_plot and rewards:
        plot_path = save_line(
            xs=list(range(1, len(rewards) + 1)),
            ys=rewards,
            path=output_dir / "reward_curve.png",
            xlabel="Step",
            ylabel="Reward",
            title="Per-step reward",
        )
        print(f"Saved reward plot: {plot_path}")

    if saved_frames:
        print("Saved frames:")
        for frame_path in saved_frames:
            print(f"  {frame_path}")

    if eval_run["enabled"] and render["enabled"]:
        env = make_env(task["env"])
        try:
            eval_result = run_policy_evaluation(
                env=env,
                policy=policy,
                output_dir=output_dir / "eval",
                eval_episodes=eval_run["episodes"],
                max_episode_steps=max_episode_steps,
                gif_fps=eval_run["fps"],
                visualize_episodes=eval_run["visualize_episodes"],
                success_metric=eval_run["success_metric"],
                success_threshold=eval_run["success_threshold"],
            )
        finally:
            env.close()

        gif_paths = eval_result["gif_paths"]
        print(f"Saved eval summary: {eval_result['summary_path']}")
        if gif_paths:
            print("Saved eval GIFs:")
            for gif_path in gif_paths:
                print(f"  {gif_path}")


def run_exam(exam_name):
    config, exam_dir = load_config(EXAM_ROOT, exam_name)
    tasks = resolve_env_tasks(config)
    for task in tasks:
        run_task(config, exam_dir, task)


def parse_args():
    parser = argparse.ArgumentParser(description="Run an exam from exam/<name>/config.yaml")
    parser.add_argument("exam_name", help="Name of the exam directory under exam/")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_exam(args.exam_name)
