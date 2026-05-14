from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
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
    parser = argparse.ArgumentParser(description="Visualize nearest reverse latent neighbors")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _to_image(obs):
    image = obs.detach().cpu().numpy() if torch.is_tensor(obs) else obs
    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = image.transpose(1, 2, 0)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    return image


def main():
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    device = _device_from_config(str(config["training"]["device"]))
    action_alignment = str(config["data"].get("action_alignment", "env_step_current"))
    forward_dataset = load_bidirectional_dataset(config["data"]["forward_path"], "forward", action_alignment)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = DirectionConditionedWorldModel(_wm_config(config, forward_dataset)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    memory = ReverseLatentMemory.load(args.memory)
    output_dir = Path(args.checkpoint).resolve().parent / "latent_neighbor_vis"
    output_dir.mkdir(parents=True, exist_ok=True)
    limit = int(config["eval"]["num_visualize_neighbors"])
    saved = 0
    with torch.no_grad():
        for episode in forward_dataset.episodes:
            obs_seq = torch.as_tensor(episode["obs"][None, ...], dtype=torch.float32, device=device)
            action_seq = torch.as_tensor(episode["action"][None, ...], dtype=torch.float32, device=device)
            reward_seq = torch.as_tensor(episode["reward"][None, ...], dtype=torch.float32, device=device)
            done_seq = torch.as_tensor(episode["done"][None, ...], dtype=torch.float32, device=device)
            is_first_seq = torch.as_tensor(episode["is_first"][None, ...], dtype=torch.float32, device=device)
            outputs = model.posterior_outputs(
                {"obs": obs_seq, "action": action_seq, "reward": reward_seq, "done": done_seq, "is_first": is_first_seq},
                "forward",
            )
            feats = model.extract_feat(outputs["post"], deterministic=True).squeeze(0)
            for t in range(feats.shape[0]):
                nearest = memory.nearest(feats[t], k=1)[0]
                subgoal = memory.get_subgoal(nearest, subgoal_step=int(config["latent_memory"]["subgoal_step"]))
                if memory.obs is None:
                    continue
                fig, axes = plt.subplots(1, 3, figsize=(9, 3))
                axes[0].imshow(_to_image(episode["obs"][t]))
                axes[0].set_title("forward")
                axes[1].imshow(_to_image(memory.obs[nearest.global_index]))
                axes[1].set_title("nearest reverse")
                axes[2].imshow(_to_image(subgoal["obs"]))
                axes[2].set_title("subgoal reverse")
                for ax in axes:
                    ax.axis("off")
                fig.tight_layout()
                fig.savefig(output_dir / f"neighbor_{saved:03d}.png")
                plt.close(fig)
                saved += 1
                if saved >= limit:
                    print(f"Saved visualizations to: {output_dir}")
                    return
    print(f"Saved visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
