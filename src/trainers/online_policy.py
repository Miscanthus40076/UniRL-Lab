from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import time

import numpy as np

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


def _scalar_from_value(value):
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return float(value)
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value.reshape(-1)[0])
        return None
    if isinstance(value, (list, tuple)) and len(value) == 1:
        try:
            return float(value[0])
        except (TypeError, ValueError):
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _vector_from_value(value):
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size == 0:
        return None
    return array


def _extract_contact_signal(info: dict) -> float | None:
    for key in ("contact", "is_contact", "touch", "collision"):
        value = _scalar_from_value(info.get(key))
        if value is not None:
            return 1.0 if value > 0.0 else 0.0
    return None


def _extract_success_signal(info: dict) -> float | None:
    for key in ("success", "is_success"):
        value = _scalar_from_value(info.get(key))
        if value is not None:
            return 1.0 if value > 0.0 else 0.0
    return None


def _extract_object_position(info: dict):
    for key in (
        "object_pos",
        "obj_pos",
        "rod_pos",
        "peg_pos",
        "peg_head_pos",
        "target_object_pos",
    ):
        value = _vector_from_value(info.get(key))
        if value is not None and value.size >= 2:
            return value
    return None


def _extract_hand_position(info: dict):
    for key in ("hand_pos", "tcp_center", "eef_pos", "gripper_pos"):
        value = _vector_from_value(info.get(key))
        if value is not None and value.size >= 3:
            return value[:3]
    return None


def _mean_or_none(values):
    if not values:
        return None
    return float(np.mean(values))


OBJECT_DELTA_THRESHOLD = 1e-3
OBJECT_LIFT_DELTA_THRESHOLD = 5e-4
HAND_OBJECT_DISTANCE_THRESHOLD = 0.04
OBJECT_HIGH_DELTA_THRESHOLD = 0.01
HAND_HIGH_MARGIN_THRESHOLD = 0.05


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
        if hasattr(policy, "set_observation_info"):
            policy.set_observation_info({})
        if self._persistent_enabled():
            self._clear_persistent_history(env)
        episode_return = 0.0
        episode_length = 0
        episode_index = 0
        start_time = time.time()
        contact_values: list[float] = []
        object_motion_values: list[float] = []
        success_values: list[float] = []
        hand_z_values: list[float] = []
        object_z_values: list[float] = []
        object_delta_values: list[float] = []
        object_lift_delta_values: list[float] = []
        hand_object_distance_values: list[float] = []
        contact_proxy_values: list[float] = []
        condition_names = (
            "object_moving",
            "object_static",
            "object_lifting",
            "object_not_lifting",
            "contact_proxy",
            "no_contact",
            "hand_high_object_static",
            "hand_high_object_high",
            "object_coupled",
            "not_object_coupled",
        )
        conditioned_reward_sums = {key: 0.0 for key in condition_names}
        conditioned_reward_counts = {key: 0 for key in condition_names}
        raw_conditioned_reward_sums = {key: 0.0 for key in condition_names}
        external_conditioned_reward_sums = {key: 0.0 for key in condition_names}
        prev_object_pos = None
        prev_object_z: float | None = None
        object_z_baseline: float | None = None
        raw_done_count = 0
        reset_trigger_count = 0
        raw_done_with_reset_count = 0
        raw_done_without_reset_count = 0
        reset_without_raw_done_count = 0
        last_raw_done_step: int | None = None
        last_reset_step: int | None = None
        metrics_window_steps = 0

        try:
            for step in range(1, total_steps + 1):
                metrics_window_steps += 1
                action = policy.act(obs)
                next_obs, reward, raw_done, info = env.step(action)
                info = dict(info or {})
                if hasattr(policy, "set_observation_info"):
                    policy.set_observation_info(info)
                contact_signal = _extract_contact_signal(info)
                if contact_signal is not None:
                    contact_values.append(contact_signal)
                success_signal = _extract_success_signal(info)
                if success_signal is not None:
                    success_values.append(success_signal)
                hand_pos = _extract_hand_position(info)
                object_pos = _extract_object_position(info)
                hand_z = None if hand_pos is None or hand_pos.size < 3 else float(hand_pos[2])
                object_z = None if object_pos is None or object_pos.size < 3 else float(object_pos[2])
                if hand_z is not None:
                    hand_z_values.append(hand_z)
                if object_z is not None:
                    object_z_values.append(object_z)
                    if object_z_baseline is None:
                        object_z_baseline = object_z
                object_delta = None
                if object_pos is not None and prev_object_pos is not None and object_pos.shape == prev_object_pos.shape:
                    object_delta = float(np.linalg.norm(object_pos - prev_object_pos))
                    object_motion_values.append(object_delta)
                    object_delta_values.append(object_delta)
                object_lift_delta = None
                if object_z is not None and prev_object_z is not None:
                    object_lift_delta = float(object_z - prev_object_z)
                    object_lift_delta_values.append(object_lift_delta)
                hand_object_distance = _scalar_from_value(info.get("hand_object_distance"))
                if hand_object_distance is None and hand_pos is not None and object_pos is not None and hand_pos.shape[0] >= 3 and object_pos.shape[0] >= 3:
                    hand_object_distance = float(np.linalg.norm(hand_pos[:3] - object_pos[:3]))
                if hand_object_distance is not None:
                    hand_object_distance_values.append(hand_object_distance)
                object_moving = bool(object_delta is not None and object_delta > OBJECT_DELTA_THRESHOLD)
                object_static = bool(object_delta is not None and object_delta <= OBJECT_DELTA_THRESHOLD)
                object_lifting = bool(object_lift_delta is not None and object_lift_delta > OBJECT_LIFT_DELTA_THRESHOLD)
                object_not_lifting = bool(object_lift_delta is not None and object_lift_delta <= OBJECT_LIFT_DELTA_THRESHOLD)
                contact_proxy = bool(
                    hand_object_distance is not None and hand_object_distance < HAND_OBJECT_DISTANCE_THRESHOLD
                )
                if hand_object_distance is not None:
                    contact_proxy_values.append(1.0 if contact_proxy else 0.0)
                object_high = bool(
                    object_z is not None
                    and object_z_baseline is not None
                    and (object_z - object_z_baseline) > OBJECT_HIGH_DELTA_THRESHOLD
                )
                hand_high = bool(
                    hand_z is not None
                    and object_z is not None
                    and (hand_z - object_z) > HAND_HIGH_MARGIN_THRESHOLD
                )
                object_coupled = bool(contact_proxy and object_moving)
                prev_object_pos = None if object_pos is None else object_pos.copy()
                prev_object_z = object_z
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

                if raw_done:
                    raw_done_count += 1
                    last_raw_done_step = int(step)
                if done:
                    reset_trigger_count += 1
                    last_reset_step = int(step)
                if raw_done and done:
                    raw_done_with_reset_count += 1
                elif raw_done and not done:
                    raw_done_without_reset_count += 1
                elif done and not raw_done:
                    reset_without_raw_done_count += 1

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
                slow_gain_reward_value = _scalar_from_value(metrics.get("intrinsic/slow_gain_reward_mean"))
                raw_slow_gain_reward_value = _scalar_from_value(metrics.get("intrinsic/raw_slow_gain_reward_mean"))
                external_slow_gain_reward_value = _scalar_from_value(metrics.get("intrinsic/external_slow_gain_reward_mean"))
                for name, condition in (
                    ("object_moving", object_moving),
                    ("object_static", object_static),
                    ("object_lifting", object_lifting),
                    ("object_not_lifting", object_not_lifting),
                    ("contact_proxy", contact_proxy),
                    ("no_contact", hand_object_distance is not None and not contact_proxy),
                    ("hand_high_object_static", hand_high and object_static),
                    ("hand_high_object_high", hand_high and object_high),
                    ("object_coupled", object_coupled),
                    ("not_object_coupled", hand_object_distance is not None and not object_coupled),
                ):
                    if not condition:
                        continue
                    conditioned_reward_counts[name] += 1
                    if slow_gain_reward_value is not None:
                        conditioned_reward_sums[name] += float(slow_gain_reward_value)
                    if raw_slow_gain_reward_value is not None:
                        raw_conditioned_reward_sums[name] += float(raw_slow_gain_reward_value)
                    if external_slow_gain_reward_value is not None:
                        external_conditioned_reward_sums[name] += float(external_slow_gain_reward_value)
                if contact_values:
                    metrics["env/contact_rate"] = float(np.mean(contact_values))
                if hand_z_values:
                    metrics["env/hand_z"] = float(np.mean(hand_z_values))
                if object_z_values:
                    metrics["env/object_z"] = float(np.mean(object_z_values))
                if object_motion_values:
                    metrics["env/object_motion"] = float(np.mean(object_motion_values))
                if object_delta_values:
                    metrics["env/object_delta"] = float(np.mean(object_delta_values))
                    metrics["env/object_motion_rate"] = float(
                        np.mean(np.asarray(object_delta_values, dtype=np.float32) > OBJECT_DELTA_THRESHOLD)
                    )
                if object_lift_delta_values:
                    metrics["env/object_lift_delta"] = float(np.mean(object_lift_delta_values))
                    metrics["env/object_lift_rate"] = float(
                        np.mean(np.asarray(object_lift_delta_values, dtype=np.float32) > OBJECT_LIFT_DELTA_THRESHOLD)
                    )
                if hand_object_distance_values:
                    metrics["env/hand_object_distance"] = float(np.mean(hand_object_distance_values))
                if contact_proxy_values:
                    metrics["env/contact_proxy_rate"] = float(np.mean(contact_proxy_values))
                if success_values:
                    metrics["env/success_rate"] = float(np.mean(success_values))
                for name, metric_key in (
                    ("object_moving", "intrinsic/slow_gain_reward_when_object_moving"),
                    ("object_static", "intrinsic/slow_gain_reward_when_object_static"),
                    ("object_lifting", "intrinsic/slow_gain_reward_when_object_lifting"),
                    ("object_not_lifting", "intrinsic/slow_gain_reward_when_object_not_lifting"),
                    ("contact_proxy", "intrinsic/slow_gain_reward_when_contact_proxy"),
                    ("no_contact", "intrinsic/slow_gain_reward_when_no_contact"),
                    ("hand_high_object_static", "intrinsic/slow_gain_reward_when_hand_high_object_static"),
                    ("hand_high_object_high", "intrinsic/slow_gain_reward_when_hand_high_object_high"),
                    ("object_coupled", "intrinsic/slow_gain_reward_when_object_coupled"),
                    ("not_object_coupled", "intrinsic/slow_gain_reward_when_not_object_coupled"),
                ):
                    count = conditioned_reward_counts[name]
                    if count > 0:
                        metrics[metric_key] = float(conditioned_reward_sums[name] / count)
                for name, metric_key in (
                    ("no_contact", "intrinsic/raw_slow_gain_reward_when_no_contact"),
                    ("object_static", "intrinsic/raw_slow_gain_reward_when_object_static"),
                    ("hand_high_object_static", "intrinsic/raw_slow_gain_reward_when_hand_high_object_static"),
                    ("object_moving", "intrinsic/raw_slow_gain_reward_when_object_moving"),
                    ("object_coupled", "intrinsic/raw_slow_gain_reward_when_object_coupled"),
                ):
                    count = conditioned_reward_counts[name]
                    if count > 0:
                        metrics[metric_key] = float(raw_conditioned_reward_sums[name] / count)
                for name, metric_key in (
                    ("no_contact", "intrinsic/external_slow_gain_reward_when_no_contact"),
                    ("object_static", "intrinsic/external_slow_gain_reward_when_object_static"),
                    ("hand_high_object_static", "intrinsic/external_slow_gain_reward_when_hand_high_object_static"),
                    ("object_moving", "intrinsic/external_slow_gain_reward_when_object_moving"),
                    ("object_coupled", "intrinsic/external_slow_gain_reward_when_object_coupled"),
                ):
                    count = conditioned_reward_counts[name]
                    if count > 0:
                        metrics[metric_key] = float(external_conditioned_reward_sums[name] / count)
                for name, metric_key in (
                    ("no_contact", "intrinsic/no_contact_external_to_raw_ratio"),
                    ("object_static", "intrinsic/object_static_external_to_raw_ratio"),
                    ("hand_high_object_static", "intrinsic/hand_high_object_static_external_to_raw_ratio"),
                    ("object_moving", "intrinsic/object_moving_external_to_raw_ratio"),
                    ("object_coupled", "intrinsic/object_coupled_external_to_raw_ratio"),
                ):
                    count = conditioned_reward_counts[name]
                    if count <= 0:
                        continue
                    raw_mean = raw_conditioned_reward_sums[name] / count
                    external_mean = external_conditioned_reward_sums[name] / count
                    metrics[metric_key] = float(external_mean / (raw_mean + 1e-8))
                metrics["env/raw_done_count"] = float(raw_done_count)
                metrics["env/raw_done_rate"] = float(raw_done_count / max(1, metrics_window_steps))
                metrics["env/reset_trigger_count"] = float(reset_trigger_count)
                metrics["env/reset_trigger_rate"] = float(reset_trigger_count / max(1, metrics_window_steps))
                metrics["env/raw_done_with_reset_count"] = float(raw_done_with_reset_count)
                metrics["env/raw_done_without_reset_count"] = float(raw_done_without_reset_count)
                metrics["env/reset_without_raw_done_count"] = float(reset_without_raw_done_count)
                metrics["env/raw_done_suppressed_rate"] = float(
                    raw_done_without_reset_count / max(1, raw_done_count)
                )
                metrics["env/reset_from_raw_done_rate"] = float(
                    raw_done_with_reset_count / max(1, reset_trigger_count)
                )
                metrics["env/last_raw_done_step"] = float(last_raw_done_step) if last_raw_done_step is not None else -1.0
                metrics["env/last_reset_step"] = float(last_reset_step) if last_reset_step is not None else -1.0
                if last_raw_done_step is not None and last_reset_step is not None:
                    metrics["env/raw_done_reset_step_gap"] = float(last_reset_step - last_raw_done_step)
                else:
                    metrics["env/raw_done_reset_step_gap"] = -1.0

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
                            "logged_metrics": metrics,
                            "persistent_stats": self._persistent_stats(env),
                            "persistent_step_metrics": (
                                getattr(env, "get_persistent_step_metrics", lambda: {})()
                                if self._persistent_enabled()
                                else {}
                            ),
                        }
                    )
                    plot_policy_metrics_jsonl(policy_metrics_path, plots_dir, x_key="step")
                    contact_values.clear()
                    hand_z_values.clear()
                    object_z_values.clear()
                    object_motion_values.clear()
                    object_delta_values.clear()
                    object_lift_delta_values.clear()
                    hand_object_distance_values.clear()
                    contact_proxy_values.clear()
                    success_values.clear()
                    for key in conditioned_reward_sums:
                        conditioned_reward_sums[key] = 0.0
                        raw_conditioned_reward_sums[key] = 0.0
                        external_conditioned_reward_sums[key] = 0.0
                        conditioned_reward_counts[key] = 0
                    raw_done_count = 0
                    reset_trigger_count = 0
                    raw_done_with_reset_count = 0
                    raw_done_without_reset_count = 0
                    reset_without_raw_done_count = 0
                    metrics_window_steps = 0

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
                    if hasattr(policy, "set_observation_info"):
                        policy.set_observation_info({})
                    episode_return = 0.0
                    episode_length = 0
                    prev_object_pos = None
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
