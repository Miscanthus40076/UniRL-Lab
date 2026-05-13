from abc import ABC, abstractmethod


class BaseEnv(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @property
    @abstractmethod
    def obs_dim(self):
        pass

    @property
    @abstractmethod
    def action_dim(self):
        pass

    @abstractmethod
    def render(self):
        pass

    def close(self):
        pass
