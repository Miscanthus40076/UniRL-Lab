from pathlib import Path
import argparse
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bidreamer.env_sanity import run_env_sanity


def parse_args():
    parser = argparse.ArgumentParser(description="Check forward/reverse bidreamer env direction sanity")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout_steps", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    config = {
        "seed": int(args.seed),
        "output_dir": args.output_dir or f"outputs/online_bidreamer_from_scratch/seed_{args.seed}/env_sanity",
        "sanity": {
            "random_rollout_steps": int(args.rollout_steps),
            "gif_fps": 20,
        },
    }
    summary = run_env_sanity(config=config, seed=args.seed)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
