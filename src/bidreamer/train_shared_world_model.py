from __future__ import annotations

from pathlib import Path
import json
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from .bidirectional_dataset import load_bidirectional_dataset
from .direction_world_model import DirectionConditionedWorldModel, DirectionWorldModelConfig
from .eval_shared_world_model import evaluate_shared_world_model


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device_from_config(name: str) -> str:
    if name == "cuda_if_available_else_cpu":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _wm_config(config: dict, dataset) -> DirectionWorldModelConfig:
    model_cfg = config["model"]
    obs_shape = dataset.obs_shape
    is_image = len(obs_shape) == 3
    return DirectionWorldModelConfig(
        obs_dim=None if is_image else int(np.prod(obs_shape)),
        obs_shape=tuple(obs_shape) if is_image else None,
        action_dim=int(model_cfg["action_dim"]),
        encoder_type="cnn" if is_image else "mlp",
        embed_dim=int(model_cfg["embed_dim"]),
        deter_dim=int(model_cfg["deter_dim"]),
        stoch_dim=int(model_cfg["stoch_dim"]),
        stoch_classes=int(model_cfg["stoch_classes"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg.get("num_layers", 2)),
        free_nats=float(model_cfg["free_nats"]),
        kl_balance=float(model_cfg["kl_balance"]),
        rssm_unimix=float(model_cfg["unimix"]),
        contact_num_classes=model_cfg.get("contact_num_classes"),
        predict_grasp=bool(model_cfg.get("predict_grasp", False)),
        separate_continue_heads=bool(model_cfg.get("separate_continue_heads", False)),
    )


def _attach_loss_config(batch: dict, config: dict):
    loss_cfg = config["loss"]
    model_cfg = config["model"]
    batch["_recon_scale"] = float(loss_cfg["recon_scale"])
    batch["_continue_scale"] = float(loss_cfg["continue_scale"])
    batch["_dynamics_scale"] = float(loss_cfg["kl_scale"])
    batch["_representation_scale"] = float(loss_cfg["kl_scale"]) * 0.1
    batch["_grasp_scale"] = float(loss_cfg["grasp_scale"])
    batch["_contact_scale"] = float(loss_cfg["contact_scale"])
    batch["_kl_balance"] = float(model_cfg["kl_balance"])
    batch["_free_nats"] = float(model_cfg["free_nats"])


def _train_eval_metrics(model, dataset, direction, training_cfg, config, device):
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in dataset.iter_eval_batches(
            num_batches=int(config["eval"]["num_eval_batches"]),
            batch_size=int(training_cfg["batch_size"]),
            seq_len=int(training_cfg["seq_len"]),
            device=device,
        ):
            _attach_loss_config(batch, config)
            batch["_reward_scale"] = float(config["loss"][f"reward_scale_{direction}"])
            metrics = model.forward_loss(batch) if direction == "forward" else model.reverse_loss(batch)
            rows.append(
                {
                    key: float(value.detach().cpu())
                    for key, value in metrics.items()
                    if key not in {"outputs", "priority", "direction"}
                }
            )
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def _plot_loss_curves(metrics_history: list[dict], output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    steps = [row["train_step"] for row in metrics_history]
    plt.figure(figsize=(10, 6))
    for key in (
        "forward_total_loss",
        "reverse_total_loss",
        "mixed_total_loss",
        "val_forward_total_loss",
        "val_reverse_total_loss",
    ):
        values = [row[key] for row in metrics_history if key in row]
        if len(values) == len(steps):
            plt.plot(steps, values, marker="o", label=key)
    plt.xlabel("Train step")
    plt.ylabel("Loss")
    plt.title("Shared Bidreamer World Model Losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def train_shared_world_model(config_path: str | Path, seed: int = 0) -> dict:
    config = _load_yaml(config_path)
    device = _device_from_config(str(config["training"]["device"]))
    _set_seed(seed)

    action_alignment = str(config["data"].get("action_alignment", "env_step_current"))
    forward_dataset = load_bidirectional_dataset(config["data"]["forward_path"], "forward", action_alignment=action_alignment)
    reverse_dataset = load_bidirectional_dataset(config["data"]["reverse_path"], "reverse", action_alignment=action_alignment)
    forward_train, forward_val = forward_dataset.split(val_ratio=float(config["data"]["val_ratio"]), seed=seed)
    reverse_train, reverse_val = reverse_dataset.split(val_ratio=float(config["data"]["val_ratio"]), seed=seed)

    model = DirectionConditionedWorldModel(_wm_config(config, forward_train)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    output_dir = Path("outputs") / "bidreamer_shared_world_model" / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_history = []
    train_steps = int(config["training"]["train_steps"])
    batch_size = int(config["training"]["batch_size"])
    seq_len = int(config["training"]["seq_len"])
    eval_interval = int(config["training"]["eval_interval"])

    for step in range(1, train_steps + 1):
        model.train()
        forward_batch = forward_train.sample_batch(batch_size=batch_size, seq_len=seq_len, device=device)
        reverse_batch = reverse_train.sample_batch(batch_size=batch_size, seq_len=seq_len, device=device)
        _attach_loss_config(forward_batch, config)
        _attach_loss_config(reverse_batch, config)
        forward_batch["_reward_scale"] = float(config["loss"]["reward_scale_forward"])
        reverse_batch["_reward_scale"] = float(config["loss"]["reward_scale_reverse"])
        forward_batch["_forward_loss_scale"] = float(config["loss"]["forward_loss_scale"])
        reverse_batch["_reverse_loss_scale"] = float(config["loss"]["reverse_loss_scale"])

        mixed = model.mixed_loss(forward_batch, reverse_batch)
        optimizer.zero_grad()
        mixed["mixed_total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["grad_clip"]))
        optimizer.step()

        if step % eval_interval == 0 or step == 1 or step == train_steps:
            row = {
                "train_step": step,
                "forward_total_loss": float(mixed["forward_total_loss"].detach().cpu()),
                "reverse_total_loss": float(mixed["reverse_total_loss"].detach().cpu()),
                "mixed_total_loss": float(mixed["mixed_total_loss"].detach().cpu()),
                "forward_recon_loss": float(mixed["forward_recon_loss"].detach().cpu()),
                "reverse_recon_loss": float(mixed["reverse_recon_loss"].detach().cpu()),
                "forward_reward_loss": float(mixed["forward_reward_loss"].detach().cpu()),
                "reverse_reward_loss": float(mixed["reverse_reward_loss"].detach().cpu()),
                "forward_continue_loss": float(mixed["forward_continue_loss"].detach().cpu()),
                "reverse_continue_loss": float(mixed["reverse_continue_loss"].detach().cpu()),
                "forward_kl_loss": float(mixed["forward_kl_loss"].detach().cpu()),
                "reverse_kl_loss": float(mixed["reverse_kl_loss"].detach().cpu()),
            }
            forward_val_metrics = _train_eval_metrics(model, forward_val, "forward", config["training"], config, device)
            reverse_val_metrics = _train_eval_metrics(model, reverse_val, "reverse", config["training"], config, device)
            row.update({f"val_forward_{k}": v for k, v in forward_val_metrics.items()})
            row.update({f"val_reverse_{k}": v for k, v in reverse_val_metrics.items()})
            metrics_history.append(row)
            (output_dir / "train_metrics.json").write_text(json.dumps(metrics_history, indent=2) + "\n", encoding="utf-8")
            _plot_loss_curves(metrics_history, output_dir / "loss_curves.png")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "seed": seed,
                    "global_step": step,
                },
                output_dir / "shared_world_model.pt",
            )

    eval_result = evaluate_shared_world_model(
        model=model,
        forward_dataset=forward_val,
        reverse_dataset=reverse_val,
        batch_size=batch_size,
        seq_len=seq_len,
        num_eval_batches=int(config["eval"]["num_eval_batches"]),
        horizons=[int(x) for x in config["eval"]["open_loop_horizons"]],
        device=device,
        output_dir=output_dir,
    )
    return {
        "output_dir": output_dir,
        "metrics_history": metrics_history,
        "eval_result": eval_result,
    }
