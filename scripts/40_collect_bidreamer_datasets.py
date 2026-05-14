from pathlib import Path
import argparse
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bidreamer.bidirectional_dataset import load_bidirectional_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset helper for bidreamer trajectories")
    parser.add_argument("--validate", action="store_true", help="Validate forward/reverse dataset schema")
    parser.add_argument("--forward_path", type=str, default="")
    parser.add_argument("--reverse_path", type=str, default="")
    parser.add_argument("--action_alignment", type=str, default="env_step_current")
    return parser.parse_args()


def _validate(path: str, direction: str, action_alignment: str):
    dataset = load_bidirectional_dataset(path, direction, action_alignment=action_alignment)
    print(json.dumps(dataset.describe_schema().asdict(), indent=2))
    print("alignment_preview:")
    print(json.dumps(dataset.describe_alignment(), indent=2))


def main():
    args = parse_args()
    if not args.validate:
        print("This script currently provides dataset validation only.")
        print("Expected fields: obs, action, reward, done, and either is_first or episode_id.")
        print("Optional fields: contact_mode, grasp_state/is_grasping, success, info.")
        return
    if not args.forward_path or not args.reverse_path:
        raise ValueError("--validate requires --forward_path and --reverse_path")
    _validate(args.forward_path, "forward", args.action_alignment)
    _validate(args.reverse_path, "reverse", args.action_alignment)


if __name__ == "__main__":
    main()
