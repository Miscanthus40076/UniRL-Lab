from pathlib import Path

from .media import save_gif


def run_policy_evaluation(env, policy, output_dir, eval_episodes, max_episode_steps=None, gif_fps=20):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gif_paths = []

    for episode_idx in range(1, int(eval_episodes) + 1):
        obs = env.reset()
        reset = getattr(policy, "reset", None)
        if callable(reset):
            reset()

        done = False
        step_in_episode = 0
        episode_return = 0.0
        frames = [env.render()]

        while not done:
            action = policy.act(obs)
            obs, reward, done, _ = env.step(action)
            episode_return += reward
            step_in_episode += 1
            frames.append(env.render())

            if max_episode_steps is not None and step_in_episode >= int(max_episode_steps):
                done = True

        gif_path = output_dir / f"eval_episode_{episode_idx:02d}.gif"
        save_gif(frames, gif_path, fps=gif_fps)
        gif_paths.append(gif_path)
        print(
            f"Eval episode {episode_idx} finished with return "
            f"{episode_return:.4f} over {step_in_episode} steps"
        )

    return gif_paths
