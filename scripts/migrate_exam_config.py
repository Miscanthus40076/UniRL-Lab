from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import yaml


EXAM_ROOT = PROJECT_ROOT / "exam"


def is_new_schema(config):
    return "train" in config and "env" in config and "policy" in config


def migrate_config(config, exam_name):
    if is_new_schema(config):
        return config, False

    experiment = config.get("experiment", {})
    env = config.get("env", {})
    policy = config.get("policy", {})
    runtime = config.get("runtime", {})
    render = env.get("render", {})

    policy_type = policy.get("type", "random")

    migrated = {
        "train": {
            "seed": int(experiment.get("seed", 0)),
            "total_steps": int(experiment.get("total_steps", 10000)),
            "max_episode_steps": experiment.get("max_episode_steps"),
            "log_interval": int(runtime.get("visualization_interval", 100)),
            "save_interval": 1000,
            "eval_interval": 5000,
            "eval_episodes": int(runtime.get("eval_episodes", 0)),
            "record_interval": int(runtime.get("visualization_interval", 100)),
            "device": "cpu",
        },
        "env": {
            "type": env.get("type", "dmcontrol"),
            "name": f"{env.get('domain_name', 'env')}_{env.get('task_name', 'task')}",
            "domain_name": env.get("domain_name", "cartpole"),
            "task_name": env.get("task_name", "swingup"),
            "observation": {
                "type": "image",
                "num_cams": 1,
            },
            "action": {
                "clip": True,
                "normalize": True,
            },
            "render": {
                "enabled": bool(runtime.get("visualization", False)),
                "save_frames": bool(runtime.get("visualization", False)),
                "save_video": bool(runtime.get("save_video", False)),
                "height": int(render.get("height", 240)),
                "width": int(render.get("width", 320)),
                "camera_id": int(render.get("camera_id", 0)),
            },
            "isaac": {
                "task_name": None,
                "num_envs": 1,
                "headless": True,
                "create_cam": None,
            },
        },
        "policy": {
            "type": policy_type,
            "name": f"{policy_type}_baseline",
            "checkpoint": {
                "load": False,
                "path": None,
                "save": False,
            },
        },
    }

    if policy_type == "random":
        migrated["policy"]["random"] = {
            "action_low": -1.0,
            "action_high": 1.0,
        }

    if runtime.get("eval_enabled", False):
        migrated["train"]["eval_episodes"] = int(runtime.get("eval_episodes", 1))

    return migrated, True


def migrate_exam(exam_dir):
    config_path = exam_dir / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    migrated, changed = migrate_config(config, exam_dir.name)
    if not changed:
        print(f"skip {exam_dir.name}")
        return False

    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(migrated, handle, sort_keys=False)

    print(f"migrated {exam_dir.name}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Migrate exam configs to train/env/policy schema")
    parser.add_argument("exam_name", nargs="?", help="Optional single exam name under exam/")
    args = parser.parse_args()

    if args.exam_name:
        exam_dir = EXAM_ROOT / args.exam_name
        if not exam_dir.exists():
            raise FileNotFoundError(f"Exam not found: {exam_dir}")
        migrate_exam(exam_dir)
        return

    for exam_dir in sorted(EXAM_ROOT.iterdir()):
        if exam_dir.is_dir() and (exam_dir / "config.yaml").exists():
            migrate_exam(exam_dir)


if __name__ == "__main__":
    main()
