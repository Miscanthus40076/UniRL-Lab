from pathlib import Path
import re

import yaml


STANDARD_TRAINERS = {"online_policy", "default", "single_policy"}
BIDIRECTIONAL_TRAINERS = {"bidirectional", "bidirectional_online", "bidreamer"}


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


def bidirectional_train_cfg(config) -> dict:
    train = train_cfg(config)
    bidirectional = train.get("bidirectional", {})
    if bidirectional is None:
        return {}
    if not isinstance(bidirectional, dict):
        raise TypeError("train.bidirectional must be a mapping")
    return bidirectional


def _has_bidirectional_training_fields(config) -> bool:
    train = config.get("train", {})
    if not isinstance(train, dict):
        return any(key in config for key in ("forward_env", "reverse_env"))

    trainer = str(train.get("trainer", "")).strip().lower()
    if trainer in BIDIRECTIONAL_TRAINERS:
        return True

    bidirectional = train.get("bidirectional")
    if bidirectional is None:
        return any(key in config for key in ("forward_env", "reverse_env"))
    if not isinstance(bidirectional, dict):
        raise TypeError("train.bidirectional must be a mapping")
    if "enabled" in bidirectional:
        return bool(bidirectional.get("enabled", False))
    return "forward_env" in bidirectional or "reverse_env" in bidirectional or any(key in config for key in ("forward_env", "reverse_env"))


def trainer_name(config) -> str:
    train = train_cfg(config)
    trainer = str(train.get("trainer", "")).strip().lower()
    if trainer:
        return trainer
    if _has_bidirectional_training_fields(config):
        return "bidirectional"
    return "online_policy"


def _validate_positive_int(field_name: str, value, *, allow_zero: bool = False):
    value = int(value)
    minimum = 0 if allow_zero else 1
    op = ">=" if allow_zero else ">"
    if value < minimum or (not allow_zero and value == 0):
        raise ValueError(f"{field_name} must be {op} 0, got {value}")
    return value


def _validate_online_policy_exam_config(config):
    train = train_cfg(config)
    policy = config.get("policy", {})
    checkpoint = policy.get("checkpoint", {})

    required_positive_ints = {
        "train.total_steps": train.get("total_steps", 0),
        "train.log_interval": train.get("log_interval", 0),
        "train.record_interval": train.get("record_interval", 0),
    }
    for field, value in required_positive_ints.items():
        _validate_positive_int(field, value)

    save_interval = _validate_positive_int("train.save_interval", train.get("save_interval", 0), allow_zero=True)
    eval_interval = _validate_positive_int("train.eval_interval", train.get("eval_interval", 0), allow_zero=True)
    eval_episodes = _validate_positive_int("train.eval_episodes", train.get("eval_episodes", 0), allow_zero=True)
    if eval_interval > 0 and eval_episodes <= 0:
        raise ValueError("train.eval_interval > 0 requires train.eval_episodes > 0")

    eval_visualize_episodes = _validate_positive_int(
        "train.eval_visualize_episodes",
        train.get("eval_visualize_episodes", min(eval_episodes, 10)),
        allow_zero=True,
    )
    if eval_visualize_episodes > eval_episodes:
        raise ValueError("train.eval_visualize_episodes must be <= train.eval_episodes")

    max_episode_steps = train.get("max_episode_steps")
    if max_episode_steps is not None and int(max_episode_steps) <= 0:
        raise ValueError(f"train.max_episode_steps must be > 0 when set, got {max_episode_steps}")

    if save_interval > 0 and not bool(checkpoint.get("save", False)):
        raise ValueError("train.save_interval > 0 requires policy.checkpoint.save = true")


def _validate_bidirectional_exam_config(config):
    train = train_cfg(config)
    if bool(train.get("multi_task", False)):
        raise ValueError("Bidirectional trainer does not support train.multi_task = true")

    env_cfg = config.get("env", {})
    if not isinstance(env_cfg, dict):
        raise TypeError("Config field 'env' must be a mapping")

    training_cfg = config.get("training")
    if not isinstance(training_cfg, dict):
        raise KeyError("Bidirectional config must contain a 'training' mapping")

    replay_cfg = config.get("replay")
    if not isinstance(replay_cfg, dict):
        raise KeyError("Bidirectional config must contain a 'replay' mapping")

    bidirectional = bidirectional_train_cfg(config)
    forward_env = bidirectional.get("forward_env", config.get("forward_env"))
    reverse_env = bidirectional.get("reverse_env", config.get("reverse_env"))
    if not isinstance(forward_env, dict) or not forward_env:
        raise ValueError("Bidirectional training requires train.bidirectional.forward_env")
    if not isinstance(reverse_env, dict) or not reverse_env:
        raise ValueError("Bidirectional training requires train.bidirectional.reverse_env")

    required_positive_ints = {
        "training.total_env_steps": training_cfg.get("total_env_steps", 0),
        "training.batch_size": training_cfg.get("batch_size", 0),
        "training.seq_len": training_cfg.get("seq_len", 0),
        "training.eval_interval": training_cfg.get("eval_interval", 0),
        "replay.capacity": replay_cfg.get("capacity", 0),
    }
    for field, value in required_positive_ints.items():
        _validate_positive_int(field, value)

    _validate_positive_int(
        "training.warmup_env_steps_per_direction",
        training_cfg.get("warmup_env_steps_per_direction", 0),
        allow_zero=True,
    )


def validate_exam_config(config):
    name = trainer_name(config)
    if name in STANDARD_TRAINERS:
        _validate_online_policy_exam_config(config)
        return
    if name in BIDIRECTIONAL_TRAINERS:
        _validate_bidirectional_exam_config(config)
        return
    raise ValueError(f"Unsupported trainer: {name}")


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
        "visualize_episodes": int(train.get("eval_visualize_episodes", min(episodes, 10))),
        "success_metric": str(train.get("eval_success_metric", "episode_return_positive")),
        "success_threshold": float(train.get("eval_success_threshold", 0.0)),
        "fps": 20,
    }
