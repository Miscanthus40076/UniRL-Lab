from pathlib import Path
import re

import yaml


def load_config(exam_root, exam_name):
    config_path = Path(exam_root) / exam_name / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Exam config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle), config_path.parent


def train_cfg(config):
    if "train" not in config:
        raise KeyError("Config must contain 'train'")
    return config["train"]


def validate_exam_config(config):
    train = train_cfg(config)
    policy = config.get("policy", {})
    checkpoint = policy.get("checkpoint", {})

    required_positive_ints = {
        "total_steps": train.get("total_steps", 0),
        "log_interval": train.get("log_interval", 0),
        "record_interval": train.get("record_interval", 0),
    }
    for field, value in required_positive_ints.items():
        value = int(value)
        if value <= 0:
            raise ValueError(f"train.{field} must be > 0, got {value}")

    save_interval = int(train.get("save_interval", 0))
    if save_interval < 0:
        raise ValueError(f"train.save_interval must be >= 0, got {save_interval}")

    eval_interval = int(train.get("eval_interval", 0))
    if eval_interval < 0:
        raise ValueError(f"train.eval_interval must be >= 0, got {eval_interval}")

    eval_episodes = int(train.get("eval_episodes", 0))
    if eval_episodes < 0:
        raise ValueError(f"train.eval_episodes must be >= 0, got {eval_episodes}")
    if eval_interval > 0 and eval_episodes <= 0:
        raise ValueError("train.eval_interval > 0 requires train.eval_episodes > 0")

    max_episode_steps = train.get("max_episode_steps")
    if max_episode_steps is not None and int(max_episode_steps) <= 0:
        raise ValueError(f"train.max_episode_steps must be > 0 when set, got {max_episode_steps}")

    if save_interval > 0 and not bool(checkpoint.get("save", False)):
        raise ValueError("train.save_interval > 0 requires policy.checkpoint.save = true")


def multi_task_enabled(config) -> bool:
    return bool(train_cfg(config).get("multi_task", False))


def _deep_merge(base, override):
    if not isinstance(base, dict) or not isinstance(override, dict):
        return override
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _default_task_name(env_cfg, index: int) -> str:
    if env_cfg.get("name"):
        return str(env_cfg["name"])
    domain_name = env_cfg.get("domain_name")
    task_name = env_cfg.get("task_name")
    if domain_name and task_name:
        return f"{domain_name}_{task_name}"
    return f"task_{index:02d}"


def _slugify_task_name(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("._")
    return slug or "task"


def resolve_env_tasks(config) -> list[dict]:
    env_cfg = config.get("env", {})
    if not isinstance(env_cfg, dict):
        raise TypeError("Config field 'env' must be a mapping")
    if not multi_task_enabled(config):
        task_name = _slugify_task_name(_default_task_name(env_cfg, 1))
        return [{"name": task_name, "env": dict(env_cfg)}]

    task_entries = env_cfg.get("tasks")
    if not isinstance(task_entries, list) or not task_entries:
        raise ValueError("Multi-task config requires env.tasks to be a non-empty list")

    tasks = []
    seen_names = set()
    shared_env = dict(env_cfg)
    shared_env.pop("tasks", None)
    for index, task_entry in enumerate(task_entries, start=1):
        if not isinstance(task_entry, dict):
            raise TypeError(f"env.tasks[{index - 1}] must be a mapping")
        task_env = _deep_merge(shared_env, task_entry)
        task_env.pop("tasks", None)
        task_name = _slugify_task_name(task_entry.get("name") or _default_task_name(task_env, index))
        if task_name in seen_names:
            raise ValueError(f"Duplicate multi-task env name: {task_name}")
        seen_names.add(task_name)
        task_env["name"] = task_name
        tasks.append({"name": task_name, "env": task_env})
    return tasks


def output_dir_for_task(exam_dir, config, task_name: str) -> Path:
    exam_dir = Path(exam_dir)
    if multi_task_enabled(config):
        return exam_dir / "output" / task_name
    return exam_dir / "output"


def render_cfg(config):
    env_cfg = config.get("env", {})
    render = env_cfg.get("render", {})
    return {
        "enabled": bool(render.get("enabled", False)),
        "save_frames": bool(render.get("save_frames", False)),
        "save_video": bool(render.get("save_video", False)),
        "height": int(render.get("height", 240)),
        "width": int(render.get("width", 320)),
        "camera_id": int(render.get("camera_id", 0)),
    }


def eval_cfg(config):
    train = train_cfg(config)
    episodes = int(train.get("eval_episodes", 0))
    return {
        "enabled": episodes > 0,
        "episodes": episodes,
        "fps": 20,
    }
