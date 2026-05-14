from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bidreamer.online_train_loop import train_online_bidreamer_from_scratch


def parse_args():
    parser = argparse.ArgumentParser(description="Train online bidreamer from scratch")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run-steps", type=int, default=0)
    parser.add_argument("--stage", type=str, default="actor_value_with_prior")
    return parser.parse_args()


def main():
    args = parse_args()
    result = train_online_bidreamer_from_scratch(
        config_path=args.config,
        seed=args.seed,
        dry_run_steps=(args.dry_run_steps or None),
        stage=args.stage,
    )
    print(f"Saved outputs to: {result['output_dir']}")


if __name__ == "__main__":
    main()
