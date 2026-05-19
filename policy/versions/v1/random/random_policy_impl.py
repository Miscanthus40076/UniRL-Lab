import numpy as np

from ..base_policy import BasePolicy


class RandomPolicyFolder(BasePolicy):
    def __init__(self, action_dim, action_low=-1.0, action_high=1.0):
        self.action_dim = action_dim
        self.action_low = float(action_low)
        self.action_high = float(action_high)

    def act(self, obs):
        # The random policy ignores image content, but it still accepts and
        # normalizes multi-camera observations to follow the shared contract.
        # TODO: 随机策略不该为了“共享 contract”强依赖图像预处理；这里需要和真实 env observation contract 一起收敛。
        self._prepare_image_observation(obs)

        return np.random.uniform(
            low=self.action_low,
            high=self.action_high,
            size=(self.action_dim,),
        ).astype(np.float32)
