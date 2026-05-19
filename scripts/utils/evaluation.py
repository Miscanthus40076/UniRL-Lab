from pathlib import Path
import json

from .media import save_gif


def _episode_success(success_metric, success_threshold, episode_return, final_info):
    if success_metric == "info_success":
        for key in ("success", "is_success"):
            if key in final_info:
                return bool(final_info[key])
        return False
    if success_metric == "episode_return_threshold":
        return float(episode_return) >= float(success_threshold)
    if success_metric == "episode_return_positive":
        return float(episode_return) > float(success_threshold)
    raise ValueError(f"Unsupported eval success metric: {success_metric}")


def run_policy_evaluation(
    env,
    policy,
    output_dir,
    eval_episodes,
    max_episode_steps=None,
    gif_fps=20,
    visualize_episodes=None,
    success_metric="episode_return_positive",
    success_threshold=0.0,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_episodes = int(eval_episodes)
    visualize_episodes = eval_episodes if visualize_episodes is None else int(visualize_episodes)
    visualize_episodes = max(0, min(visualize_episodes, eval_episodes))

    gif_paths = []
    episode_summaries = []
    success_count = 0

    def _preview_transition(next_obs, reward, raw_done):
        preview = getattr(policy, "preview_transition", None)
        if callable(preview):
            payload = preview(next_obs, done=bool(raw_done), reward=float(reward))
            return dict(payload or {})
        return {}

    def _finalize_persistent(action, next_obs, raw_done, info, diagnostics):
        finalize = getattr(env, "finalize_step", None)
        if callable(finalize):
            payload = finalize(
                action=action,
                next_obs=next_obs,
                raw_done=bool(raw_done),
                info=info or {},
                diagnostics=diagnostics,
            )
            if isinstance(payload, dict):
                return payload
        return {"done": bool(raw_done), "reset_reason": ("env_done" if raw_done else "NA")}

    clear_history = getattr(env, "clear_persistent_history", None)
    if callable(clear_history):
        clear_history()

    for episode_idx in range(1, eval_episodes + 1):
        obs = env.reset()
        reset = getattr(policy, "reset", None)
        if callable(reset):
            reset()

        done = False
        step_in_episode = 0
        episode_return = 0.0
        final_info = {}
        frames = [env.render()] if episode_idx <= visualize_episodes else None

        while not done:
            action = policy.act(obs)
            next_obs, reward, raw_done, info = env.step(action)
            diagnostics = _preview_transition(next_obs, reward=reward, raw_done=bool(raw_done))
            persistent_result = _finalize_persistent(action, next_obs, raw_done=bool(raw_done), info=info, diagnostics=diagnostics)
            obs = next_obs
            done = bool(persistent_result.get("done", False))
            episode_return += reward
            step_in_episode += 1
            final_info = info or {}
            if frames is not None:
                frames.append(env.render())

            if max_episode_steps is not None and step_in_episode >= int(max_episode_steps):
                done = True

        success = _episode_success(
            success_metric=success_metric,
            success_threshold=success_threshold,
            episode_return=episode_return,
            final_info=final_info,
        )
        success_count += int(success)

        if frames is not None:
            gif_path = output_dir / f"eval_episode_{episode_idx:02d}.gif"
            save_gif(frames, gif_path, fps=gif_fps)
            gif_paths.append(gif_path)

        episode_summaries.append(
            {
                "episode_index": episode_idx,
                "episode_return": float(episode_return),
                "episode_length": int(step_in_episode),
                "success": bool(success),
            }
        )
        print(
            f"Eval episode {episode_idx} finished with return "
            f"{episode_return:.4f} over {step_in_episode} steps success={int(success)}"
        )

    summary = {
        "eval_episodes": eval_episodes,
        "visualized_episodes": visualize_episodes,
        "success_metric": success_metric,
        "success_threshold": float(success_threshold),
        "success_count": success_count,
        "success_rate": (success_count / eval_episodes) if eval_episodes > 0 else 0.0,
        "episodes": episode_summaries,
        "gif_paths": [str(path) for path in gif_paths],
    }
    getter = getattr(env, "get_persistent_stats", None)
    if callable(getter):
        payload = getter()
        if isinstance(payload, dict):
            summary.update(payload)
    summary_path = output_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"Eval summary: success_rate={summary['success_rate']:.4f} "
        f"({success_count}/{eval_episodes}), visualized={visualize_episodes}"
    )

    return {
        "gif_paths": gif_paths,
        "summary": summary,
        "summary_path": summary_path,
    }
