from pathlib import Path
import argparse
import sys

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bidreamer.bidirectional_dataset import load_bidirectional_dataset
from src.bidreamer.direction_world_model import DirectionConditionedWorldModel
from src.bidreamer.eval_shared_world_model import evaluate_shared_world_model
from src.bidreamer.train_shared_world_model import _device_from_config, _wm_config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate shared bidreamer world model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    device = _device_from_config(str(config["training"]["device"]))
    action_alignment = str(config["data"].get("action_alignment", "env_step_current"))
    forward_dataset = load_bidirectional_dataset(config["data"]["forward_path"], "forward", action_alignment)
    reverse_dataset = load_bidirectional_dataset(config["data"]["reverse_path"], "reverse", action_alignment)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = DirectionConditionedWorldModel(_wm_config(config, forward_dataset)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    output_dir = Path(args.checkpoint).resolve().parent
    result = evaluate_shared_world_model(
        model=model,
        forward_dataset=forward_dataset,
        reverse_dataset=reverse_dataset,
        batch_size=int(config["training"]["batch_size"]),
        seq_len=int(config["training"]["seq_len"]),
        num_eval_batches=int(config["eval"]["num_eval_batches"]),
        horizons=[int(x) for x in config["eval"]["open_loop_horizons"]],
        device=device,
        output_dir=output_dir,
    )
    print(f"Saved eval metrics: {result['eval_metrics_path']}")


if __name__ == "__main__":
    main()
