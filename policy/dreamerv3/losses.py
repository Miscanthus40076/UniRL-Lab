from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .transforms import symlog, twohot_logprob


@dataclass(slots=True)
class WorldModelLossConfig:
    free_nats: float = 1.0
    recon_scale: float = 1.0
    reward_scale: float = 1.0
    continue_scale: float = 1.0
    dynamics_scale: float = 1.0
    representation_scale: float = 0.1
    grasp_scale: float = 0.2
    contact_scale: float = 0.2
    kl_balance: float = 0.8
    use_symlog_obs: bool = True
    use_symlog_reward: bool = True
    use_twohot_reward: bool = True
    twohot_bins: int = 255
    twohot_low: float = -20.0
    twohot_high: float = 20.0
    context_update_penalty: float = 0.0


def categorical_kl(post: dict[str, torch.Tensor], prior: dict[str, torch.Tensor]) -> torch.Tensor:
    post_log_probs = F.log_softmax(post["logits"], dim=-1)
    prior_log_probs = F.log_softmax(prior["logits"], dim=-1)
    post_probs = post_log_probs.exp()
    kl = (post_probs * (post_log_probs - prior_log_probs)).sum(dim=-1)
    return kl.sum(dim=-1)


def _reduce_except_batch(x: torch.Tensor) -> torch.Tensor:
    if x.ndim <= 1:
        return x
    dims = tuple(range(1, x.ndim))
    return x.mean(dim=dims)


def world_model_loss(outputs: dict, batch: dict, config: WorldModelLossConfig) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    obs = batch["obs"].float()
    reward = batch["reward"].float()
    done = batch["done"].float()
    continue_target = 1.0 - done
    obs_target = obs / 255.0 if obs.ndim >= 4 and obs.max() > 2.0 else obs
    if obs.ndim < 4 and config.use_symlog_obs:
        obs_target = symlog(obs_target)

    recon_error = (outputs["obs_pred"] - obs_target).pow(2)
    recon_loss_per_seq = _reduce_except_batch(recon_error)
    recon_loss = recon_loss_per_seq.mean()
    if config.use_symlog_reward:
        reward_target = symlog(reward)
    else:
        reward_target = reward
    if config.use_twohot_reward:
        reward_logprob = twohot_logprob(
            outputs["reward_logits"],
            reward_target,
            num_bins=config.twohot_bins,
            low=config.twohot_low,
            high=config.twohot_high,
        )
        reward_loss_per_seq = -_reduce_except_batch(reward_logprob)
        reward_loss = reward_loss_per_seq.mean()
    else:
        reward_error = (outputs["reward_pred"] - reward_target).pow(2)
        reward_loss_per_seq = _reduce_except_batch(reward_error)
        reward_loss = reward_loss_per_seq.mean()
    continue_loss_raw = F.binary_cross_entropy_with_logits(
        outputs["continue_logit"],
        continue_target,
        reduction="none",
    )
    continue_loss_per_seq = _reduce_except_batch(continue_loss_raw)
    continue_loss = continue_loss_per_seq.mean()
    post_sg = {key: value.detach() for key, value in outputs["post"].items()}
    prior_sg = {key: value.detach() for key, value in outputs["prior"].items()}
    dynamics_kl_per_seq = torch.clamp(categorical_kl(post_sg, outputs["prior"]), min=float(config.free_nats)).mean(dim=1)
    representation_kl_per_seq = torch.clamp(categorical_kl(outputs["post"], prior_sg), min=float(config.free_nats)).mean(dim=1)
    dynamics_kl = dynamics_kl_per_seq.mean()
    representation_kl = representation_kl_per_seq.mean()
    kl_loss = config.kl_balance * dynamics_kl + (1.0 - config.kl_balance) * representation_kl

    total_per_seq = (
        config.recon_scale * recon_loss_per_seq
        + config.reward_scale * reward_loss_per_seq
        + config.continue_scale * continue_loss_per_seq
        + config.dynamics_scale * dynamics_kl_per_seq
        + config.representation_scale * representation_kl_per_seq
    )
    context_gate = outputs.get("context_gate")
    context_update_loss = None
    if context_gate is not None:
        context_update_per_seq = _reduce_except_batch(context_gate.float())
        context_update_loss = context_update_per_seq.mean()
        total_per_seq = total_per_seq + config.context_update_penalty * context_update_per_seq
    total = total_per_seq.mean()
    metrics = {
        "model_loss": total.detach(),
        "recon_loss": recon_loss.detach(),
        "reward_loss": reward_loss.detach(),
        "continue_loss": continue_loss.detach(),
        "kl_loss": kl_loss.detach(),
        "dynamics_kl_loss": dynamics_kl.detach(),
        "representation_kl_loss": representation_kl.detach(),
        "priority": total_per_seq.detach(),
    }
    if context_update_loss is not None:
        gate = context_gate.detach().float()
        metrics["context_update_loss"] = context_update_loss.detach()
        metrics["context_gate_mean"] = gate.mean()
        metrics["context_gate_std"] = gate.std(unbiased=False)
        metrics["context_gate_min"] = gate.min()
        metrics["context_gate_max"] = gate.max()
        context_delta_norm = outputs.get("context_delta_norm")
        if context_delta_norm is not None:
            metrics["context_delta_norm_mean"] = context_delta_norm.detach().float().mean()
        context = outputs.get("context")
        if context is not None:
            metrics["context_norm_mean"] = torch.linalg.vector_norm(context.detach().float(), dim=-1).mean()

    if outputs.get("grasp_logit") is not None and "is_grasping" in batch:
        grasp_loss = F.binary_cross_entropy_with_logits(outputs["grasp_logit"], batch["is_grasping"].float())
        total = total + config.grasp_scale * grasp_loss
        metrics["model_loss"] = total.detach()
        metrics["grasp_loss"] = grasp_loss.detach()
    if outputs.get("contact_logits") is not None and "contact_mode" in batch:
        contact_loss = F.cross_entropy(
            outputs["contact_logits"].reshape(-1, outputs["contact_logits"].shape[-1]),
            batch["contact_mode"].long().reshape(-1),
        )
        total = total + config.contact_scale * contact_loss
        metrics["model_loss"] = total.detach()
        metrics["contact_loss"] = contact_loss.detach()

    return total, metrics
