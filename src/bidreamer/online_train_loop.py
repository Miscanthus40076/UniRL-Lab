from __future__ import annotations

from pathlib import Path
import json

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from policy.dreamerv3.processor import DreamerV3Processor

from .bidirectional_agent import BidirectionalAgent
from .bidirectional_dataset import normalize_raw_episode
from .bidirectional_env_factory import make_bidirectional_env
from .direction_world_model import DirectionConditionedWorldModel, DirectionWorldModelConfig
from .env_sanity import run_env_sanity
from .online_replay_buffer import EpisodeReplayBuffer
from .reverse_latent_memory import ReverseLatentMemory
from .train_shared_world_model import _attach_loss_config, _device_from_config, _plot_loss_curves, _set_seed


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _infer_model_config(config: dict, observation_example) -> DirectionWorldModelConfig:
    processor = DreamerV3Processor(device="cpu")
    obs_spec = processor.infer_observation_spec(observation_example)
    model_cfg = config["model"]
    is_image = obs_spec.mode == "image"
    obs_shape = tuple(obs_spec.obs_shape) if is_image and obs_spec.obs_shape is not None else None
    obs_dim = None if is_image else int(obs_spec.obs_dim)
    return DirectionWorldModelConfig(
        obs_dim=obs_dim,
        obs_shape=obs_shape,
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
        use_symlog_obs=bool(model_cfg.get("use_symlog_obs", not is_image)),
        use_symlog_reward=bool(model_cfg.get("use_symlog_reward", True)),
        use_twohot_reward=bool(model_cfg.get("use_twohot_reward", True)),
        reward_bins=int(model_cfg.get("reward_bins", 255)),
        reward_low=float(model_cfg.get("reward_low", -20.0)),
        reward_high=float(model_cfg.get("reward_high", 20.0)),
        separate_continue_heads=bool(model_cfg.get("separate_continue_heads", False)),
    )


def _preprocess_obs(obs, processor: DreamerV3Processor, obs_spec):
    return processor.preprocess_obs(obs, obs_spec)


def _attach_online_loss(batch: dict, config: dict, direction: str):
    _attach_loss_config(batch, config)
    batch["_reward_scale"] = float(config["loss"][f"reward_scale_{direction}"])
    batch["_forward_loss_scale"] = float(config["loss"]["forward_loss_scale"])
    batch["_reverse_loss_scale"] = float(config["loss"]["reverse_loss_scale"])


def _random_action(action_dim: int, rng: np.random.Generator):
    return rng.uniform(-1.0, 1.0, size=(action_dim,)).astype(np.float32)


def _collect_one_step_random(env, replay: EpisodeReplayBuffer, obs, processor, obs_spec, rng):
    raw_obs = _preprocess_obs(obs, processor, obs_spec)
    action = _random_action(env.action_dim, rng)
    next_obs, reward, done, info = env.step(action)
    replay.add_step(
        obs=raw_obs,
        action=action,
        reward=reward,
        done=done,
        info=info,
        direction=info["bidreamer_direction"],
    )
    if done:
        next_obs = env.reset()
    return next_obs


def _collect_one_step_policy(env, replay: EpisodeReplayBuffer, obs, processor, obs_spec, agent: BidirectionalAgent, direction: str):
    raw_obs = _preprocess_obs(obs, processor, obs_spec)
    action = agent.act(raw_obs, direction=direction, deterministic=False)
    next_obs, reward, done, info = env.step(action)
    replay.add_step(
        obs=raw_obs,
        action=action,
        reward=reward,
        done=done,
        info=info,
        direction=info["bidreamer_direction"],
    )
    if done:
        next_obs = env.reset()
        agent.reset_policy_state(direction)
    return next_obs


def _eval_world_model(agent: BidirectionalAgent, forward_replay: EpisodeReplayBuffer, reverse_replay: EpisodeReplayBuffer, config: dict, device: str) -> dict:
    batch_size = int(config["training"]["batch_size"])
    seq_len = int(config["training"]["seq_len"])
    forward_batch = forward_replay.sample_batch(batch_size=batch_size, seq_len=seq_len, device=device)
    reverse_batch = reverse_replay.sample_batch(batch_size=batch_size, seq_len=seq_len, device=device)
    _attach_online_loss(forward_batch, config, "forward")
    _attach_online_loss(reverse_batch, config, "reverse")
    agent.model.eval()
    with torch.no_grad():
        mixed = agent.model.mixed_loss(forward_batch, reverse_batch)
    return {
        key: float(value.detach().cpu()) if torch.is_tensor(value) else value
        for key, value in mixed.items()
        if not isinstance(value, dict)
    }


def _save_gif(path: Path, frames, fps: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, format="GIF", fps=fps)


def _to_image(obs):
    image = obs.detach().cpu().numpy() if torch.is_tensor(obs) else np.asarray(obs)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = image.transpose(1, 2, 0)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    return image


def _normalized_replay_episode(replay_episode: dict, direction: str, action_alignment: str = "env_step_current") -> dict | None:
    if not replay_episode["obs"]:
        return None
    success = np.asarray(
        [float(bool(info.get("success") or info.get("is_success"))) for info in replay_episode["info"]],
        dtype=np.float32,
    )
    raw_episode = {
        "obs": np.asarray(replay_episode["obs"], dtype=np.float32),
        "action": np.asarray(replay_episode["action"], dtype=np.float32),
        "reward": np.asarray(replay_episode["reward"], dtype=np.float32),
        "done": np.asarray(replay_episode["done"], dtype=np.float32),
        "is_first": np.concatenate(
            [
                np.asarray([1.0], dtype=np.float32),
                np.zeros(max(len(replay_episode["obs"]) - 1, 0), dtype=np.float32),
            ]
        ),
        "success": success,
        "info": np.asarray(replay_episode["info"], dtype=object),
    }
    return normalize_raw_episode(raw_episode, direction, action_alignment)


def _build_reverse_latent_memory(
    agent: BidirectionalAgent,
    reverse_replay: EpisodeReplayBuffer,
    config: dict,
    device: str,
    output_dir: Path,
):
    memory_cfg = config.get("latent_memory", {})
    memory = ReverseLatentMemory.build_from_reverse_replay(
        model=agent.model,
        replay=reverse_replay,
        device=device,
        action_alignment=str(memory_cfg.get("action_alignment", "env_step_current")),
        use_success_only=bool(memory_cfg.get("use_success_only", True)),
        distance_metric=str(memory_cfg.get("distance_metric", "cosine")),
        normalize_features=bool(memory_cfg.get("normalize_features", True)),
        feature_whiten=bool(memory_cfg.get("feature_whiten", False)),
        store_obs=bool(memory_cfg.get("store_obs", True)),
    )
    memory_path = output_dir / "reverse_latent_memory.pt"
    summary_path = output_dir / "reverse_latent_memory_summary.json"
    memory.save(memory_path)
    memory.save_summary(summary_path)
    return memory, memory.summary()


def _visualize_reverse_latent_neighbors(
    agent: BidirectionalAgent,
    forward_replay: EpisodeReplayBuffer,
    reverse_latent_memory: ReverseLatentMemory,
    output_dir: Path,
    limit: int,
    subgoal_step: int,
    device: str,
):
    if reverse_latent_memory.obs is None:
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    with torch.no_grad():
        for replay_episode in forward_replay.iter_episodes(include_current=True):
            episode = _normalized_replay_episode(replay_episode, "forward")
            if episode is None:
                continue
            batch = {
                "obs": torch.as_tensor(episode["obs"][None, ...], dtype=torch.float32, device=device),
                "action": torch.as_tensor(episode["action"][None, ...], dtype=torch.float32, device=device),
                "reward": torch.as_tensor(episode["reward"][None, ...], dtype=torch.float32, device=device),
                "done": torch.as_tensor(episode["done"][None, ...], dtype=torch.float32, device=device),
                "is_first": torch.as_tensor(episode["is_first"][None, ...], dtype=torch.float32, device=device),
            }
            outputs = agent.model.posterior_outputs(batch, "forward")
            feats = agent.model.extract_feat(outputs["post"], deterministic=True).squeeze(0)
            for t in range(feats.shape[0]):
                nearest = reverse_latent_memory.nearest(feats[t], k=1)[0]
                subgoal = reverse_latent_memory.get_subgoal(nearest, subgoal_step=subgoal_step)
                fig, axes = plt.subplots(1, 3, figsize=(9, 3))
                axes[0].imshow(_to_image(episode["obs"][t]))
                axes[0].set_title("current forward")
                axes[1].imshow(_to_image(reverse_latent_memory.obs[nearest.global_index]))
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
                    return saved
    return saved


def _episode_success(final_info: dict, episode_return: float) -> bool:
    for key in ("success", "is_success"):
        if key in final_info:
            return bool(final_info[key])
    return episode_return > 0.0


def _evaluate_policy(agent: BidirectionalAgent, config: dict, processor: DreamerV3Processor, obs_spec, direction: str, output_dir: Path) -> dict:
    eval_cfg = config.get("eval", {})
    episodes = int(eval_cfg.get("episodes_per_direction", 2))
    max_steps = int(eval_cfg.get("max_episode_steps", 200))
    deterministic = bool(eval_cfg.get("deterministic", True))
    save_gif = bool(eval_cfg.get("save_gif", True))
    gif_fps = int(eval_cfg.get("gif_fps", 20))
    env = make_bidirectional_env(config, direction)
    policy_state = agent.snapshot_policy_state()
    try:
        returns = []
        lengths = []
        successes = []
        action_rows = []
        frames = []
        for episode_idx in range(episodes):
            obs = env.reset()
            agent.reset_policy_state(direction)
            episode_return = 0.0
            final_info = {}
            step_count = 0
            if episode_idx == 0 and save_gif:
                frames.append(env.render())
            for _ in range(max_steps):
                raw_obs = _preprocess_obs(obs, processor, obs_spec)
                action = agent.act(raw_obs, direction=direction, deterministic=deterministic)
                if not np.all(np.isfinite(action)):
                    raise RuntimeError(f"{direction} eval actor produced non-finite action")
                action_rows.append(action.copy())
                obs, reward, done, info = env.step(action)
                episode_return += float(reward)
                final_info = dict(info or {})
                step_count += 1
                if episode_idx == 0 and save_gif:
                    frames.append(env.render())
                if done:
                    break
            returns.append(episode_return)
            lengths.append(step_count)
            successes.append(int(_episode_success(final_info, episode_return)))
        if save_gif and frames:
            _save_gif(output_dir / f"{direction}_eval.gif", frames, fps=gif_fps)
        actions = np.asarray(action_rows, dtype=np.float32) if action_rows else np.zeros((0, env.action_dim), dtype=np.float32)
        return {
            f"{direction}_eval_success_rate": float(np.mean(successes)) if successes else 0.0,
            f"{direction}_eval_return": float(np.mean(returns)) if returns else 0.0,
            f"{direction}_eval_episode_length": float(np.mean(lengths)) if lengths else 0.0,
            f"{direction}_eval_action_abs_mean": float(np.abs(actions).mean()) if actions.size else 0.0,
            f"{direction}_eval_action_abs_max": float(np.abs(actions).max()) if actions.size else 0.0,
            f"{direction}_eval_action_sat_frac": float((np.abs(actions) > 0.99).mean()) if actions.size else 0.0,
            f"{direction}_eval_action_zero_frac": float((np.abs(actions) < 1e-4).mean()) if actions.size else 0.0,
        }
    finally:
        agent.restore_policy_state(policy_state)
        env.close()


def train_online_bidreamer_from_scratch(
    config_path: str | Path,
    seed: int = 0,
    dry_run_steps: int | None = None,
    stage: str = "actor_value_with_prior",
) -> dict:
    if stage not in {"world_model_only", "actor_value_no_prior", "actor_value_with_prior"}:
        raise ValueError(f"Unsupported stage for this phase: {stage}")
    config = _load_yaml(config_path)
    latent_prior_cfg = config.get("latent_prior", {})
    latent_prior_enabled = bool(latent_prior_cfg.get("enabled", False))
    if stage == "actor_value_no_prior" and latent_prior_enabled:
        raise ValueError("Step 4 requires latent_prior.enabled=false")
    if stage == "world_model_only" and latent_prior_enabled:
        raise ValueError("world_model_only stage does not support latent_prior.enabled=true")
    if stage == "actor_value_with_prior" and not latent_prior_enabled:
        raise ValueError("Step 5 requires latent_prior.enabled=true")
    _set_seed(seed)
    device = _device_from_config(str(config["training"]["device"]))
    output_dir = Path("outputs") / "online_bidreamer_from_scratch" / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    sanity_config = dict(config)
    sanity_config["output_dir"] = str(output_dir / "env_sanity")
    sanity = run_env_sanity(config=sanity_config, seed=seed)

    forward_env = make_bidirectional_env(config, "forward")
    reverse_env = make_bidirectional_env(config, "reverse")
    processor = DreamerV3Processor(device=device)
    try:
        forward_obs = forward_env.reset()
        reverse_obs = reverse_env.reset()
        obs_spec = processor.infer_observation_spec(forward_obs)
        model = DirectionConditionedWorldModel(_infer_model_config(config, forward_obs)).to(device)
        agent = BidirectionalAgent(
            model=model,
            learning_rate=float(config["training"]["learning_rate"]),
            grad_clip=float(config["training"]["grad_clip"]),
            actor_lr=float(config["training"].get("actor_lr", 8e-5)),
            value_lr=float(config["training"].get("value_lr", 8e-5)),
            actor_grad_clip=float(config["training"].get("actor_grad_clip", 100.0)),
            value_grad_clip=float(config["training"].get("value_grad_clip", 100.0)),
            imagination_horizon=int(config["training"].get("imagination_horizon", 15)),
            gamma=float(config["training"].get("gamma", 0.99)),
            lambda_=float(config["training"].get("lambda", 0.95)),
            entropy_coef=float(config["training"].get("entropy_coef", 1e-3)),
            actor_hidden_dim=int(config["model"].get("actor_hidden_dim", config["model"]["hidden_dim"])),
            actor_num_layers=int(config["model"].get("actor_num_layers", 2)),
            value_hidden_dim=int(config["model"].get("value_hidden_dim", config["model"]["hidden_dim"])),
            value_num_layers=int(config["model"].get("value_num_layers", 2)),
            min_std=float(config["training"].get("min_std", 0.1)),
            max_std=float(config["training"].get("max_std", 1.0)),
            init_std=float(config["training"].get("init_std", 1.0)),
            imag_last=int(config["training"].get("imag_last", 0)),
            use_twohot_value=bool(config["training"].get("use_twohot_value", True)),
            use_slow_value=bool(config["training"].get("use_slow_value", True)),
            slow_value_rate=float(config["training"].get("slow_value_rate", 0.02)),
            repval_loss=bool(config["training"].get("repval_loss", False)),
            repval_scale=float(config["training"].get("repval_scale", 0.3)),
            use_return_norm=bool(config["training"].get("use_return_norm", True)),
            use_advantage_norm=bool(config["training"].get("use_advantage_norm", True)),
            norm_rate=float(config["training"].get("norm_rate", 0.01)),
            norm_eps=float(config["training"].get("norm_eps", 1e-8)),
            latent_prior_enabled=latent_prior_enabled,
            latent_prior_beta=float(latent_prior_cfg.get("beta", 1.0)),
            latent_prior_horizon=int(latent_prior_cfg.get("horizon", config["training"].get("imagination_horizon", 15))),
            latent_prior_subgoal_step=int(latent_prior_cfg.get("subgoal_step", config.get("latent_memory", {}).get("subgoal_step", 1))),
        )
        forward_replay = EpisodeReplayBuffer(direction="forward", capacity=int(config["replay"]["capacity"]))
        reverse_replay = EpisodeReplayBuffer(direction="reverse", capacity=int(config["replay"]["capacity"]))
        rng = np.random.default_rng(seed)
        total_env_steps = int(dry_run_steps or config["training"]["total_env_steps"])
        warmup_per_direction = int(config["training"]["warmup_env_steps_per_direction"])
        batch_size = int(config["training"]["batch_size"])
        seq_len = int(config["training"]["seq_len"])
        eval_interval = int(config["training"]["eval_interval"])
        latent_prior_start = int(latent_prior_cfg.get("start_after_env_steps", 10000))
        memory_update_interval = int(latent_prior_cfg.get("memory_update_interval", eval_interval))
        latent_prior_vis_limit = int(latent_prior_cfg.get("num_visualize_neighbors", 8))

        for _ in range(warmup_per_direction):
            forward_obs = _collect_one_step_random(forward_env, forward_replay, forward_obs, processor, obs_spec, rng)
        for _ in range(warmup_per_direction):
            reverse_obs = _collect_one_step_random(reverse_env, reverse_replay, reverse_obs, processor, obs_spec, rng)

        agent.reset_policy_state("forward")
        agent.reset_policy_state("reverse")
        reverse_latent_memory = None
        latest_memory_summary = {}
        if latent_prior_enabled:
            reverse_latent_memory, latest_memory_summary = _build_reverse_latent_memory(
                agent=agent,
                reverse_replay=reverse_replay,
                config=config,
                device=device,
                output_dir=output_dir,
            )
            _visualize_reverse_latent_neighbors(
                agent=agent,
                forward_replay=forward_replay,
                reverse_latent_memory=reverse_latent_memory,
                output_dir=output_dir / "latent_neighbor_vis" / "step_000000",
                limit=latent_prior_vis_limit,
                subgoal_step=int(latent_prior_cfg.get("subgoal_step", 1)),
                device=device,
            )

        metrics_history = []
        train_step = 0
        for env_step in range(1, total_env_steps + 1):
            if env_step % 2 == 1:
                if stage == "world_model_only":
                    forward_obs = _collect_one_step_random(forward_env, forward_replay, forward_obs, processor, obs_spec, rng)
                else:
                    forward_obs = _collect_one_step_policy(forward_env, forward_replay, forward_obs, processor, obs_spec, agent, "forward")
            else:
                if stage == "world_model_only":
                    reverse_obs = _collect_one_step_random(reverse_env, reverse_replay, reverse_obs, processor, obs_spec, rng)
                else:
                    reverse_obs = _collect_one_step_policy(reverse_env, reverse_replay, reverse_obs, processor, obs_spec, agent, "reverse")

            if latent_prior_enabled and memory_update_interval > 0 and env_step % memory_update_interval == 0:
                reverse_latent_memory, latest_memory_summary = _build_reverse_latent_memory(
                    agent=agent,
                    reverse_replay=reverse_replay,
                    config=config,
                    device=device,
                    output_dir=output_dir,
                )
                _visualize_reverse_latent_neighbors(
                    agent=agent,
                    forward_replay=forward_replay,
                    reverse_latent_memory=reverse_latent_memory,
                    output_dir=output_dir / "latent_neighbor_vis" / f"step_{env_step:06d}",
                    limit=latent_prior_vis_limit,
                    subgoal_step=int(latent_prior_cfg.get("subgoal_step", 1)),
                    device=device,
                )

            if len(forward_replay._eligible_episodes(seq_len)) == 0 or len(reverse_replay._eligible_episodes(seq_len)) == 0:
                continue

            forward_batch = forward_replay.sample_batch(batch_size=batch_size, seq_len=seq_len, device=device)
            reverse_batch = reverse_replay.sample_batch(batch_size=batch_size, seq_len=seq_len, device=device)
            _attach_online_loss(forward_batch, config, "forward")
            _attach_online_loss(reverse_batch, config, "reverse")
            metrics = agent.train_world_model(forward_batch, reverse_batch)
            if stage != "world_model_only":
                metrics.update(agent.train_reverse_actor_value(reverse_batch))
                forward_prior_memory = None
                if latent_prior_enabled and env_step >= latent_prior_start:
                    forward_prior_memory = reverse_latent_memory
                metrics.update(agent.train_forward_actor_value(forward_batch, reverse_latent_memory=forward_prior_memory))
            train_step += 1
            row = {
                "env_step": env_step,
                "train_step": train_step,
                "forward_replay_size": len(forward_replay),
                "reverse_replay_size": len(reverse_replay),
            }
            if latest_memory_summary:
                row["reverse_memory_num_latents"] = int(latest_memory_summary["num_latents"])
                row["reverse_memory_num_episodes"] = int(latest_memory_summary["num_episodes"])
                row["reverse_memory_fallback_used"] = float(bool(latest_memory_summary.get("fallback_used", False)))
            row.update(metrics)
            metrics_history.append(row)

            if env_step % eval_interval == 0 or env_step == total_env_steps:
                eval_metrics = _eval_world_model(agent, forward_replay, reverse_replay, config, device)
                if stage != "world_model_only":
                    eval_dir = output_dir / "eval" / f"step_{env_step:06d}"
                    eval_metrics.update(_evaluate_policy(agent, config, processor, obs_spec, "forward", eval_dir))
                    eval_metrics.update(_evaluate_policy(agent, config, processor, obs_spec, "reverse", eval_dir))
                row.update({f"eval_{k}": v for k, v in eval_metrics.items()})
                (output_dir / "train_metrics.json").write_text(json.dumps(metrics_history, indent=2) + "\n", encoding="utf-8")
                _plot_loss_curves(metrics_history, output_dir / "loss_curves.png")
                agent.save(output_dir / "shared_world_model.pt")

        if not metrics_history:
            raise RuntimeError("No world model updates were performed during online training")
        (output_dir / "train_metrics.json").write_text(json.dumps(metrics_history, indent=2) + "\n", encoding="utf-8")
        _plot_loss_curves(metrics_history, output_dir / "loss_curves.png")
        agent.save(output_dir / "shared_world_model.pt")
        return {
            "output_dir": output_dir,
            "sanity": sanity,
            "metrics_history": metrics_history,
        }
    finally:
        forward_env.close()
        reverse_env.close()
