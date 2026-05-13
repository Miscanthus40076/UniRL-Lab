import numpy as np
from .mlp import MLPNetwork
from .noise import GaussianNoise
from ..base_policy import BasePolicy


class MLPPolicyFolder(BasePolicy):
    def __init__(self, action_dim, obs_dim=None, hidden_size=64, noise_std=0.1):
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.noise_std = noise_std
        self.obs_dim = obs_dim

        # Initialize network and noise
        self.network = None
        if self.obs_dim is not None:
            self.network = MLPNetwork(input_dim=self.obs_dim, output_dim=self.action_dim, hidden_dim=self.hidden_size)
        self.noise = GaussianNoise(action_dim, std=noise_std)
        # TODO: 这个策略只有前向采样，没有 update/save/load；如果它用于 train.py，就需要补成可训练策略或明确标成 baseline。

    def act(self, obs):
        image_obs = self._prepare_image_observation(obs)
        flat_obs = image_obs.reshape(-1)

        if self.network is None or self.obs_dim != flat_obs.shape[0]:
            # TODO: 不要在运行时静默重建网络；obs 维度变化应在 env/policy 初始化阶段校验并尽早失败。
            self.obs_dim = int(flat_obs.shape[0])
            self.network = MLPNetwork(
                input_dim=self.obs_dim,
                output_dim=self.action_dim,
                hidden_dim=self.hidden_size,
            )

        action = self.network.forward(flat_obs)

        # Add noise for exploration
        noisy_action = action + self.noise.sample()

        # Clip to [-1, 1] range
        return np.clip(noisy_action, -1.0, 1.0).astype(np.float32)
