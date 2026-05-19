from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from policy.operator_token_probe import run_operator_token_probe


def parse_args():
    parser = argparse.ArgumentParser(description="Train Operator Token Probe V0 from a frozen Capacity V2 Dreamer checkpoint.")
    parser.add_argument("--run-dir", required=True, help="Source exam dir or output dir containing the frozen Dreamer run.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path. Defaults to <run-dir>/policy.ckpt.")
    parser.add_argument("--output-dir", default="outputs/operator_token_probe/seed_0")
    parser.add_argument("--train-steps", type=int, default=3000)
    parser.add_argument("--collect-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--num-tokens", type=int, default=8)
    parser.add_argument("--token-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default=("cuda" if __import__("torch").cuda.is_available() else "cpu"))
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--deterministic", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    config = {
        "policy": {
            "type": "operator_token_probe",
            "checkpoint": {"save": True},
            "operator_token_probe": {
                "device": args.device,
                "allow_cpu_fallback": bool(args.allow_cpu_fallback),
                "source_run_dir": args.run_dir,
                "source_checkpoint": args.checkpoint,
                "output_dir": args.output_dir,
                "collect_steps": int(args.collect_steps),
                "train_steps": int(args.train_steps),
                "batch_size": int(args.batch_size),
                "seq_len": int(args.seq_len),
                "learning_rate": float(args.learning_rate),
                "max_episode_steps": int(args.max_episode_steps),
                "deterministic": bool(args.deterministic),
            },
        },
        "operator_token": {
            "enabled": True,
            "num_tokens": int(args.num_tokens),
            "token_dim": int(args.token_dim),
            "hidden_dim": int(args.hidden_dim),
            "gumbel_temperature": 1.0,
            "straight_through": True,
            "train_on_capacity_only": True,
            "effect_loss_scale": 1.0,
            "inverse_loss_scale": 0.5,
            "prior_loss_scale": 0.1,
            "usage_entropy_scale": 0.01,
            "detach_features": True,
        },
    }
    result = run_operator_token_probe(config=config, exam_dir=None)
    print(f"Saved operator token summary: {result['summary_path']}")
    print(f"Saved operator token checkpoint: {result['checkpoint_path']}")


if __name__ == "__main__":
    main()
