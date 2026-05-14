from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bidreamer.train_shared_world_model import train_shared_world_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train shared bidreamer world model")
    parser.add_argument("--config", required=True, help="Path to training yaml config")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    result = train_shared_world_model(args.config, seed=args.seed)
    print(f"Saved outputs to: {result['output_dir']}")


if __name__ == "__main__":
    main()
