from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .direction_world_model import DirectionConditionedWorldModel


def _to_scalar(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(value.detach().cpu())


def _mean_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {key: float(np.mean([row[key] for row in rows if key in row])) for key in keys}


def compute_one_step_metrics(model: DirectionConditionedWorldModel, batch: dict, direction: str) -> dict[str, float]:
    outputs = model.posterior_outputs(batch, direction)
    obs = batch["obs"].float()
    reward = batch["reward"].float()
    done = batch["done"].float()
    obs_target = obs / 255.0 if obs.ndim >= 4 and obs.max() > 2.0 else obs
    recon_mse = F.mse_loss(outputs["obs_pred"], obs_target).detach()
    reward_mse = F.mse_loss(outputs["reward_pred"], reward).detach()
    continue_bce = F.binary_cross_entropy_with_logits(outputs["continue_logit"], 1.0 - done).detach()
    metrics = {
        "recon_mse": _to_scalar(recon_mse),
        "reward_mse": _to_scalar(reward_mse),
        "continue_bce": _to_scalar(continue_bce),
        "one_step_obs_pred_mse": _to_scalar(recon_mse),
    }
    if outputs.get("contact_logits") is not None and "contact_mode" in batch:
        pred = outputs["contact_logits"].argmax(dim=-1)
        metrics["contact_acc"] = _to_scalar((pred == batch["contact_mode"]).float().mean())
    if outputs.get("grasp_logit") is not None and "is_grasping" in batch:
        pred = (torch.sigmoid(outputs["grasp_logit"]) >= 0.5).float()
        metrics["grasp_acc"] = _to_scalar((pred == batch["is_grasping"]).float().mean())
    return metrics


def compute_open_loop_metrics(
    model: DirectionConditionedWorldModel,
    batch: dict,
    direction: str,
    horizons: list[int],
) -> dict[str, float]:
    # The batch action sequence is already Dreamer-aligned. Since posterior state at t
    # already incorporates batch["action"][t], future rollout must start from action[t+1].
    outputs = model.posterior_outputs(batch, direction)
    obs = batch["obs"].float()
    post = outputs["post"]
    metrics: dict[str, float] = {}
    for horizon in horizons:
        errors = []
        valid_count = 0
        batch_size, seq_len = obs.shape[:2]
        for b in range(batch_size):
            for t in range(seq_len - horizon - 1):
                done_window = batch["done"][b, t + 1 : t + horizon + 1]
                if bool(done_window.any()):
                    continue
                init_state = {key: value[b : b + 1, t] for key, value in post.items()}
                action_seq = batch["action"][b : b + 1, t + 1 : t + horizon + 1]
                rollout = model.rollout_open_loop(init_state, action_seq)
                pred = rollout["obs_pred"][:, -1]
                target = obs[b : b + 1, t + horizon]
                target = target / 255.0 if target.ndim >= 4 and target.max() > 2.0 else target
                errors.append(F.mse_loss(pred, target).detach().cpu().item())
                valid_count += 1
        metrics[f"open_loop_obs_pred_mse_h{horizon}"] = float(np.mean(errors)) if errors else float("nan")
        metrics[f"open_loop_valid_sample_count_h{horizon}"] = int(valid_count)
    return metrics


def evaluate_direction(
    model: DirectionConditionedWorldModel,
    dataset,
    direction: str,
    batch_size: int,
    seq_len: int,
    num_eval_batches: int,
    horizons: list[int],
    device: str,
) -> dict[str, float]:
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in dataset.iter_eval_batches(num_batches=num_eval_batches, batch_size=batch_size, seq_len=seq_len, device=device):
            one_step = compute_one_step_metrics(model, batch, direction)
            open_loop = compute_open_loop_metrics(model, batch, direction, horizons)
            row = dict(one_step)
            row.update(open_loop)
            rows.append(row)
    metrics = _mean_dict(rows)
    metrics["num_eval_batches"] = int(num_eval_batches)
    return metrics


def _plot_open_loop(metrics: dict, output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    forward_points = []
    reverse_points = []
    for key, value in metrics.get("forward", {}).items():
        if key.startswith("open_loop_obs_pred_mse_h"):
            forward_points.append((int(key.split("h")[-1]), value))
    for key, value in metrics.get("reverse", {}).items():
        if key.startswith("open_loop_obs_pred_mse_h"):
            reverse_points.append((int(key.split("h")[-1]), value))
    forward_points.sort()
    reverse_points.sort()
    plt.figure(figsize=(6, 4))
    if forward_points:
        xs, ys = zip(*forward_points)
        plt.plot(xs, ys, marker="o", label="forward")
    if reverse_points:
        xs, ys = zip(*reverse_points)
        plt.plot(xs, ys, marker="o", label="reverse")
    plt.xlabel("Horizon")
    plt.ylabel("Open-loop obs pred MSE")
    plt.title("Shared World Model Open-loop Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def evaluate_shared_world_model(
    model: DirectionConditionedWorldModel,
    forward_dataset,
    reverse_dataset,
    batch_size: int,
    seq_len: int,
    num_eval_batches: int,
    horizons: list[int],
    device: str,
    output_dir: str | Path,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "forward": evaluate_direction(model, forward_dataset, "forward", batch_size, seq_len, num_eval_batches, horizons, device),
        "reverse": evaluate_direction(model, reverse_dataset, "reverse", batch_size, seq_len, num_eval_batches, horizons, device),
    }
    eval_path = output_dir / "eval_metrics.json"
    eval_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    plot_path = _plot_open_loop(metrics, output_dir / "open_loop_compare.png")
    return {
        "metrics": metrics,
        "eval_metrics_path": eval_path,
        "plot_path": plot_path,
    }
