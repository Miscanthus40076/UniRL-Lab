from pathlib import Path
import argparse
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bidreamer.bidirectional_dataset import load_bidirectional_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Validate bidreamer dataset alignment")
    parser.add_argument("--path", required=True)
    parser.add_argument("--direction", required=True, choices=["forward", "reverse"])
    parser.add_argument("--action_alignment", default="env_step_current")
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--save_reverse_direction_check", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = load_bidirectional_dataset(args.path, args.direction, action_alignment=args.action_alignment)
    print(json.dumps(dataset.describe_schema().asdict(), indent=2))
    print(json.dumps(dataset.describe_alignment(episode_index=args.episode_index, steps=args.steps), indent=2))
    if args.save_reverse_direction_check and args.direction == "reverse":
        output_dir = Path("outputs") / "bidreamer_dataset_validation" / "reverse_direction_check"
        paths = dataset.export_reverse_direction_frames(output_dir)
        print(f"Saved reverse direction check frames: {len(paths)} -> {output_dir}")


if __name__ == "__main__":
    main()
