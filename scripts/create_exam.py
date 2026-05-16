from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from scripts.train import render_exam_train_template
from scripts.utils import validate_exam_config


EXAM_ROOT = PROJECT_ROOT / "exam"
POLICY_ROOT = PROJECT_ROOT / "policy"


def ask_text(prompt, default=None):
    if default is None:
        value = input(f"{prompt}: ").strip()
    else:
        value = input(f"{prompt} [{default}]: ").strip()
        if value == "":
            value = default
    return value


def ask_int(prompt, default):
    value = input(f"{prompt} [{default}]: ").strip()
    if value == "":
        return default
    return int(value)


def ask_positive_int(prompt, default):
    while True:
        value = ask_int(prompt, default)
        if value > 0:
            return value
        print("Value must be > 0, try again.")


def ask_non_negative_int(prompt, default):
    while True:
        value = ask_int(prompt, default)
        if value >= 0:
            return value
        print("Value must be >= 0, try again.")


def ask_bool(prompt, default=False):
    default_text = "y" if default else "n"
    value = input(f"{prompt} (y/n) [{default_text}]: ").strip().lower()

    if value == "":
        return default

    return value in ["y", "yes", "true", "1"]


def list_policy_types():
    policies = []

    # Look for both directories and Python files that represent actual policies
    for path in POLICY_ROOT.iterdir():
        if path.is_dir() and not path.name.startswith("__"):
            policies.append(path.name)
        elif path.is_file() and path.suffix == ".py" and path.name != "__init__.py" and not path.name.startswith("__"):
            # Skip utility files like base_policy.py and make_policy.py
            if path.stem not in ["base_policy", "make_policy"]:
                policies.append(path.stem)

    return sorted(policies)


def choose_policy():
    policies = list_policy_types()

    if not policies:
        raise RuntimeError(f"No policy folders found in {POLICY_ROOT}")

    print("\nAvailable policies:")

    for index, name in enumerate(policies):
        print(f"  [{index}] {name}")

    while True:
        choice = input("Choose policy index: ").strip()

        try:
            index = int(choice)
            return policies[index]
        except (ValueError, IndexError):
            print("Invalid choice, try again.")


def main():
    EXAM_ROOT.mkdir(exist_ok=True)

    print("Create new experiment")

    exam_name = ask_text("Exam name", "dmcontrol_cartpole_random")
    exam_dir = EXAM_ROOT / exam_name

    if exam_dir.exists():
        raise FileExistsError(f"Exam already exists: {exam_dir}")

    policy_type = choose_policy()

    seed = ask_int("Seed", 0)
    total_steps = ask_positive_int("Total steps", 10000)
    max_episode_steps = ask_positive_int("Max episode steps", 200)
    log_interval = ask_positive_int("Log interval", 100)
    save_interval = ask_non_negative_int("Save interval", 1000)
    eval_interval = ask_positive_int("Eval interval", 5000)
    eval_episodes = ask_positive_int("Eval episodes", 2)
    record_interval = ask_positive_int("Record interval", 100)
    device = ask_text("Device", "cpu")
    is_multi_task = ask_bool("Multi-task training", False)

    env_type = ask_text("Env type", "dmcontrol")

    if env_type != "dmcontrol":
        raise ValueError("Currently only dmcontrol is supported")

    task_entries = []
    if is_multi_task:
        task_count = ask_int("Number of tasks", 2)
        for index in range(task_count):
            print(f"\nConfigure task {index + 1}")
            env_name = ask_text("Env name", f"cartpole_task_{index + 1}")
            domain_name = ask_text("dm_control domain_name", "cartpole")
            task_name = ask_text("dm_control task_name", "swingup")
            task_entries.append(
                {
                    "name": env_name,
                    "domain_name": domain_name,
                    "task_name": task_name,
                }
            )
    else:
        env_name = ask_text("Env name", "cartpole_swingup")
        domain_name = ask_text("dm_control domain_name", "cartpole")
        task_name = ask_text("dm_control task_name", "swingup")
    num_cams = ask_int("Number of cameras", 1)

    enable_render = ask_bool("Enable render", True)
    save_frames = ask_bool("Save frames", True)
    save_video = ask_bool("Save video", False)
    checkpoint_load = ask_bool("Load checkpoint", False)
    checkpoint_path = ask_text("Checkpoint path", "null") if checkpoint_load else None
    checkpoint_save = ask_bool("Save checkpoint", save_interval > 0)

    config = {
        "train": {
            "seed": seed,
            "total_steps": total_steps,
            "max_episode_steps": max_episode_steps,
            "log_interval": log_interval,
            "save_interval": save_interval,
            "eval_interval": eval_interval,
            "eval_episodes": eval_episodes,
            "record_interval": record_interval,
            "device": device,
            "multi_task": is_multi_task,
        },
        "env": {
            "type": env_type,
            "observation": {
                "type": "image",
                "num_cams": num_cams,
            },
            "action": {
                "clip": True,
                "normalize": True,
            },
            "render": {
                "enabled": enable_render,
                "save_frames": save_frames,
                "save_video": save_video,
                "height": 240,
                "width": 320,
                "camera_id": 0,
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
                "load": checkpoint_load,
                "path": checkpoint_path,
                "save": checkpoint_save,
            },
        },
    }

    if is_multi_task:
        config["env"]["tasks"] = task_entries
    else:
        config["env"]["name"] = env_name
        config["env"]["domain_name"] = domain_name
        config["env"]["task_name"] = task_name

    if policy_type == "random":
        config["policy"]["random"] = {
            "action_low": -1.0,
            "action_high": 1.0,
        }

    validate_exam_config(config)

    exam_dir.mkdir(parents=True)

    config_path = exam_dir / "config.yaml"
    train_path = exam_dir / "train.py"

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    train_path.write_text(render_exam_train_template(), encoding="utf-8")

    print(f"\nCreated exam:")
    print(f"  {exam_dir}")
    print(f"Config:")
    print(f"  {config_path}")
    print(f"Train entry:")
    print(f"  {train_path}")


if __name__ == "__main__":
    main()
