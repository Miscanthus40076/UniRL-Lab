from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import time

from policy.registry import maybe_record_gate_eval_videos_for_policy
from policy.make_policy import make_policy
from scripts.utils import (
    CsvLog,
    JsonlLog,
    metrics_row_from_payload,
    multi_task_enabled,
    normalize_policy_metrics,
    plot_policy_metrics_jsonl,
    resolve_env_tasks,
    run_policy_evaluation,
    train_cfg,
)
from sim_env.envs.make_env import make_env

from .base import Trainer


def maybe_update(policy, batch):
    update = getattr(policy, "update", None)
    if not callable(update):
        return normalize_policy_metrics(None)
    return normalize_policy_metrics(update(batch))


class OnlinePolicyTrainer(Trainer):
    trainer_name = "online_policy"

    def _task_output_dir(self, task_name: str) -> Path:
        base_output_dir = self.output_dir
        if multi_task_enabled(self.config):
            return base_output_dir / task_name
        return base_output_dir

    def _policy_config_for_task(self):
        policy_config = deepcopy(self.config["policy"])
        if "operator_intrinsic_reward" in self.config:
            policy_config["operator_intrinsic_reward"] = deepcopy(self.config.get("operator_intrinsic_reward", {}))
        checkpoint_cfg = dict(policy_config.get("checkpoint", {}))
        if multi_task_enabled(self.config):
            checkpoint_cfg["load"] = False
            checkpoint_cfg["path"] = None
        policy_config["checkpoint"] = checkpoint_cfg
        return policy_config

    def _checkpoint_path_for_task(self, policy_config, output_dir: Path) -> Path:
        checkpoint_cfg = policy_config.get("checkpoint", {})
        checkpoint_path = checkpoint_cfg.get("path")
        if checkpoint_path:
            checkpoint_path = Path(checkpoint_path)
            if multi_task_enabled(self.config) and not checkpoint_path.is_absolute():
                return output_dir / checkpoint_path
            return checkpoint_path
        return output_dir / "policy.ckpt"

    def _gate_video_output_root(self, output_dir: Path) -> Path:
        output_subdir = str(self.config.get("gate_video", {}).get("output_subdir", "gate_videos"))
        return output_dir / output_subdir

    def _checkpoint_history_path_for_task(self, policy_config, output_dir: Path, step: int | None) -> Path | None:
        if step is None:
            return None
        checkpoint_cfg = policy_config.get("checkpoint", {})
        if not bool(checkpoint_cfg.get("save_history", False)):
            return None
        history_subdir = str(checkpoint_cfg.get("history_subdir", "checkpoints"))
        name_template = str(checkpoint_cfg.get("history_name_template", "policy_step_{step:07d}.ckpt"))
        return output_dir / history_subdir / name_template.format(step=int(step))

    def _save_checkpoint(self, policy, policy_config, output_dir: Path, step: int | None = None):
        checkpoint_cfg = policy_config.get("checkpoint", {})
        save = getattr(policy, "save", None)
        if not callable(save) or not bool(checkpoint_cfg.get("save", False)):
            return None

        checkpoint_path = self._checkpoint_path_for_task(policy_config, output_dir)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        save(checkpoint_path)
        result = {"latest": checkpoint_path}

        history_path = self._checkpoint_history_path_for_task(policy_config, output_dir, step=step)
        if history_path is not None:
            history_path.parent.mkdir(parents=True, exist_ok=True)
            save(history_path)
            result["history"] = history_path

        return result

    def _write_run_manifest(self, policy_config, output_dir: Path, task: dict):
        manifest = {
            "trainer": self.trainer_name,
            "task_name": task["name"],
            "train": dict(self.config.get("train", {})),
            "env": dict(task.get("env", {})),
            "persistent_exploration": dict(self.config.get("persistent_exploration", {})),
            "policy": dict(policy_config),
            "artifacts": {
                "metrics_csv": str(output_dir / "metrics.csv"),
                "policy_metrics_jsonl": str(output_dir / "policy_metrics.jsonl"),
                "plots_dir": str(output_dir / "plots"),
                "eval_dir": str(output_dir / "eval"),
                "gate_video_dir": str(self._gate_video_output_root(output_dir)),
                "checkpoint_path": str(self._checkpoint_path_for_task(policy_config, output_dir)),
                "checkpoint_history_dir": str(output_dir / str(policy_config.get("checkpoint", {}).get("history_subdir", "checkpoints"))),
            },
        }
        return self.write_json(output_dir / "run_manifest.json", manifest)

    def _train_env_config(self, task_env: dict) -> dict:
        env_config = deepcopy(task_env)
        render_config = dict(env_config.get("render", {}))
        render_config["enabled"] = False
        render_config["save_frames"] = False
        render_config["save_video"] = False
        env_config["render"] = render_config
        if "persistent_exploration" in self.config:
            env_config["persistent_exploration"] = deepcopy(self.config.get("persistent_exploration", {}))
        return env_config

    def _eval_env_config(self, task_env: dict) -> dict:
        env_config = deepcopy(task_env)
        render_config = dict(env_config.get("render", {}))
        render_config["enabled"] = True
        render_config["save_frames"] = False
        render_config["save_video"] = False
        env_config["render"] = render_config
        if "persistent_exploration" in self.config:
            env_config["persistent_exploration"] = deepcopy(self.config.get("persistent_exploration", {}))
        return env_config

    def _persistent_enabled(self) -> bool:
        return bool(dict(self.config.get("persistent_exploration", {})).get("enabled", False))

    def _persistent_preview_metrics(self, policy, next_obs, reward: float, raw_done: bool) -> dict:
        preview = getattr(policy, "preview_transition", None)
        if callable(preview):
            payload = preview(next_obs, done=bool(raw_done), reward=float(reward))
            return dict(payload or {})
        return {}

    def _persistent_finalize(self, env, action, next_obs, raw_done: bool, info: dict, diagnostics: dict) -> dict:
        finalize = getattr(env, "finalize_step", None)
        if callable(finalize):
            payload = finalize(
                action=action,
                next_obs=next_obs,
                raw_done=bool(raw_done),
                info=info,
                diagnostics=diagnostics,
            )
            if isinstance(payload, dict):
                return payload
        return {"done": bool(raw_done), "reset_reason": ("env_done" if raw_done else "NA")}

    def _persistent_stats(self, env) -> dict[str, float]:
        getter = getattr(env, "get_persistent_stats", None)
        if callable(getter):
            payload = getter()
            if isinstance(payload, dict):
                return {key: float(value) for key, value in payload.items()}
        return {}

    def _clear_persistent_history(self, env):
        clear = getattr(env, "clear_persistent_history", None)
        if callable(clear):
            clear()

    def train_task(self, task: dict) -> dict:
        cfg = train_cfg(self.config)

        total_steps = int(cfg.get("total_steps", 1000))
        max_episode_steps = cfg.get("max_episode_steps")
        log_interval = int(cfg.get("log_interval", 100))
        record_interval = int(cfg.get("record_interval", log_interval))
        save_interval = int(cfg.get("save_interval", 0))
        eval_interval = int(cfg.get("eval_interval", 0))
        eval_episodes = int(cfg.get("eval_episodes", 0))
        eval_visualize_episodes = int(cfg.get("eval_visualize_episodes", min(eval_episodes, 10)))
        eval_success_metric = str(cfg.get("eval_success_metric", "episode_return_positive"))
        eval_success_threshold = float(cfg.get("eval_success_threshold", 0.0))

        output_dir = self._task_output_dir(task["name"])
        metrics_path = output_dir / "metrics.csv"
        policy_metrics_path = output_dir / "policy_metrics.jsonl"
        plots_dir = output_dir / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)

        env = make_env(self._train_env_config(task["env"]))
        eval_env = None
        policy_config = self._policy_config_for_task()
        policy = make_policy(policy_config, env, exam_dir=self.exam_dir)
        log = CsvLog(metrics_path)
        policy_metrics_log = JsonlLog(policy_metrics_path)
        manifest_path = self._write_run_manifest(policy_config, output_dir, task)

        print(f"Training task: {task['name']}")
        print(f"Saved run manifest: {manifest_path}")
        obs = env.reset()
        if self._persistent_enabled():
            self._clear_persistent_history(env)
        episode_return = 0.0
        episode_length = 0
        episode_index = 0
        start_time = time.time()

        try:
            for step in range(1, total_steps + 1):
                action = policy.act(obs)
                next_obs, reward, raw_done, info = env.step(action)
                info = dict(info or {})
                if self._persistent_enabled():
                    preview_metrics = self._persistent_preview_metrics(policy, next_obs, reward=float(reward), raw_done=bool(raw_done))
                    persistent_result = self._persistent_finalize(
                        env,
                        action=action,
                        next_obs=next_obs,
                        raw_done=bool(raw_done),
                        info=info,
                        diagnostics=preview_metrics,
                    )
                    done = bool(persistent_result.get("done", False))
                    forced_done = False
                    info["persistent_reset_reason"] = str(persistent_result.get("reset_reason", "NA"))
                else:
                    forced_done = max_episode_steps is not None and episode_length + 1 >= int(max_episode_steps)
                    done = bool(raw_done or forced_done)

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
                metrics.update(self._persistent_stats(env))

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
                            "persistent_stats": self._persistent_stats(env),
                            "persistent_step_metrics": (
                                getattr(env, "get_persistent_step_metrics", lambda: {})()
                                if self._persistent_enabled()
                                else {}
                            ),
                        }
                    )
                    plot_policy_metrics_jsonl(policy_metrics_path, plots_dir, x_key="step")

                if step % log_interval == 0:
                    reset_reason = info.get("persistent_reset_reason", "NA") if self._persistent_enabled() else "NA"
                    print(
                        f"step={step} episode={episode_index} return={episode_return:.4f} "
                        f"length={episode_length} reset_reason={reset_reason}"
                    )

                if eval_episodes > 0 and eval_interval > 0 and step % eval_interval == 0:
                    if eval_env is None:
                        eval_env = make_env(self._eval_env_config(task["env"]))
                    eval_dir = output_dir / "eval" / f"step_{step:07d}"
                    print(f"Running evaluation at step={step} ...")
                    eval_result = run_policy_evaluation(
                        env=eval_env,
                        policy=policy,
                        output_dir=eval_dir,
                        eval_episodes=eval_episodes,
                        max_episode_steps=max_episode_steps,
                        gif_fps=20,
                        visualize_episodes=eval_visualize_episodes,
                        success_metric=eval_success_metric,
                        success_threshold=eval_success_threshold,
                    )
                    print(f"Saved eval summary: {eval_result['summary_path']}")
                    gate_video_result = maybe_record_gate_eval_videos_for_policy(
                        policy_config,
                        agent=policy,
                        env=eval_env,
                        output_dir=self._gate_video_output_root(output_dir),
                        global_step=step,
                        config=self.config,
                        device=cfg.get("device"),
                    )
                    if gate_video_result is not None:
                        print(f"Saved gate video summary: {gate_video_result['summary_path']}")
                    reset = getattr(policy, "reset", None)
                    if callable(reset):
                        reset()

                if save_interval > 0 and step % save_interval == 0:
                    checkpoint_paths = self._save_checkpoint(policy, policy_config, output_dir, step=step)
                    if checkpoint_paths is not None:
                        print(f"Saved checkpoint: {checkpoint_paths['latest']}")
                        history_path = checkpoint_paths.get("history")
                        if history_path is not None:
                            print(f"Saved checkpoint snapshot: {history_path}")

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
            checkpoint_paths = self._save_checkpoint(policy, policy_config, output_dir)
            if checkpoint_paths is not None:
                print(f"Saved final checkpoint: {checkpoint_paths['latest']}")
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

        return {
            "task_name": task["name"],
            "output_dir": output_dir,
            "manifest_path": manifest_path,
            "metrics_path": metrics_path,
            "policy_metrics_path": policy_metrics_path,
            "plot_paths": plot_paths,
        }

    def run(self) -> dict:
        self.ensure_output_dir()
        task_results = []
        for task in resolve_env_tasks(self.config):
            task_results.append(self.train_task(task))
        return {"trainer": self.trainer_name, "tasks": task_results}
