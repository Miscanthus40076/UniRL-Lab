from .base_env import BaseEnv


class IsaacEnv(BaseEnv):
    """Minimal Isaac/Gymnasium-style wrapper using the project BaseEnv API."""

    def __init__(self, env):
        self.env = env

    def reset(self):
        obs, info = self.env.reset()
        del info
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # TODO: 明确单环境与向量化环境的 done 语义；这里的按位或在 tensor/array 场景下会把上层训练循环变复杂。
        done = terminated | truncated
        return obs, reward, done, info

    @property
    def obs_dim(self):
        # TODO: 这里只覆盖了一维 Box 观测；图像观测或 Dict/Tuple space 需要单独的 shape 适配逻辑。
        return self.env.observation_space.shape[-1]

    @property
    def action_dim(self):
        return self.env.action_space.shape[-1]

    def render(self):
        return self.env.render()

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()
