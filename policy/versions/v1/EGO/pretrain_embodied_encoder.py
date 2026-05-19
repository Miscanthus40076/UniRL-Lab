from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .embodied_encoder import EmbodiedEncoder, EmbodiedEncoderConfig, add_prev_action_if_missing, summarize_robot_fields


def train_embodied_encoder_on_batch(
    batch: dict[str, torch.Tensor],
    *,
    action_dim: int,
    config: EmbodiedEncoderConfig,
    train_steps: int = 1000,
    learning_rate: float = 3e-4,
    device: str = "cuda",
) -> tuple[EmbodiedEncoder, dict[str, float]]:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for EmbodiedEncoder pretrain but is unavailable")
    module = EmbodiedEncoder(action_dim=action_dim, config=config).to(device)
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    module.aux_losses(batch)
    optimizer = torch.optim.Adam(module.parameters(), lr=float(learning_rate))
    last_metrics: dict[str, float] = {}
    for _ in range(int(train_steps)):
        losses = module.aux_losses(batch)
        loss = losses.get("aux_loss_total")
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("EmbodiedEncoder pretrain did not produce a finite aux_loss_total")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_metrics = {key: float(value.detach().cpu()) for key, value in losses.items()}
    return module, last_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain EGO EmbodiedEncoder on an existing torch batch.")
    parser.add_argument("--batch-path", required=True, help="Path to a torch-saved batch dict with obs/action and optional robot fields.")
    parser.add_argument("--output", default="embodied_encoder.pt", help="Output checkpoint path.")
    parser.add_argument("--action-dim", type=int, required=True)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embodied-encoder-dim", type=int, default=64)
    parser.add_argument("--embodied-robot-latent-dim", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    batch = torch.load(args.batch_path, map_location="cpu")
    if not isinstance(batch, dict):
        raise ValueError("--batch-path must point to a torch-saved dict")
    config = EmbodiedEncoderConfig(
        use_embodied_encoder=True,
        embodied_encoder_dim=int(args.embodied_encoder_dim),
        embodied_robot_latent_dim=int(args.embodied_robot_latent_dim),
        freeze_embodied_encoder=False,
        detach_embodied_encoder=False,
        embodied_pretrain=True,
    )
    batch = add_prev_action_if_missing(batch)
    field_summary = summarize_robot_fields(batch, config.robot_state_keys)
    print(
        {
            "used_fields": field_summary["used_robot_fields"],
            "missing_fields": field_summary["missing_robot_fields"],
            "fallback_to_action": field_summary["fallback_to_action"],
        }
    )
    if bool(field_summary["fallback_to_action"]):
        print(
            "[EGO EmbodiedEncoder] warning: no explicit proprio/q/gripper/ee_pose fields found; "
            "using shifted action fallback only. This is not full robot self prior pretraining."
        )
    module, metrics = train_embodied_encoder_on_batch(
        batch,
        action_dim=int(args.action_dim),
        config=config,
        train_steps=int(args.train_steps),
        learning_rate=float(args.learning_rate),
        device=str(args.device),
    )
    module.save_checkpoint(Path(args.output))
    print({"output": str(args.output), **metrics})


if __name__ == "__main__":
    main()
