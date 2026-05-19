import numpy as np


class GaussianNoise:
    def __init__(self, action_dim, std=0.1):
        self.action_dim = action_dim
        self.std = std

    def sample(self):
        return np.random.normal(0, self.std, size=(self.action_dim,)).astype(np.float32)