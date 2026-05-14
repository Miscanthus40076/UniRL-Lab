from pathlib import Path
import argparse
import json
import sys

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Compare forward/reverse shared world model metrics")
    parser.add_argument("--metrics", required=True, help="Path to eval_metrics.json")
    return parser.parse_args()


def main():
    args = parse_args()
    metrics_path = Path(args.metrics)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    out_path = metrics_path.parent / "forward_reverse_compare.png"
    keys = ["recon_mse", "reward_mse", "continue_bce", "one_step_obs_pred_mse"]
    forward = [metrics["forward"].get(key, float("nan")) for key in keys]
    reverse = [metrics["reverse"].get(key, float("nan")) for key in keys]
    plt.figure(figsize=(8, 4))
    xs = range(len(keys))
    plt.bar([x - 0.2 for x in xs], forward, width=0.4, label="forward")
    plt.bar([x + 0.2 for x in xs], reverse, width=0.4, label="reverse")
    plt.xticks(list(xs), keys, rotation=20)
    plt.title("Shared World Model Direction Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved comparison plot: {out_path}")


if __name__ == "__main__":
    main()
