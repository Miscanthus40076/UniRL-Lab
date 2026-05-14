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
from src.bidreamer.reverse_latent_memory import ReverseLatentMemory
from src.bidreamer.train_shared_world_model import _device_from_config, _wm_config


def parse_args():
    parser = argparse.ArgumentParser(description="Build reverse latent memory from shared world model")
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
    reverse_dataset = load_bidirectional_dataset(config["data"]["reverse_path"], "reverse", action_alignment)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = DirectionConditionedWorldModel(_wm_config(config, reverse_dataset)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    memory = ReverseLatentMemory.build_from_reverse_dataset(
        model=model,
        dataset=reverse_dataset,
        device=device,
        use_success_only=bool(config["latent_memory"]["use_success_only"]),
        distance_metric=str(config["latent_memory"]["distance_metric"]),
        normalize_features=bool(config["latent_memory"]["normalize_features"]),
        feature_whiten=bool(config["latent_memory"]["feature_whiten"]),
        store_obs=bool(config["data"].get("store_obs_in_reverse_memory", True)),
    )
    output_dir = Path(args.checkpoint).resolve().parent
    memory_path = output_dir / "reverse_latent_memory.pt"
    summary_path = output_dir / "reverse_latent_memory_summary.json"
    memory.save(memory_path)
    memory.save_summary(summary_path)
    print(f"Saved reverse latent memory: {memory_path}")


if __name__ == "__main__":
    main()
