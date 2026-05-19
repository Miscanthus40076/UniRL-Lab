from abc import ABC, abstractmethod
import numpy as np


class BasePolicy(ABC):
    METRICS_SCHEMA = "policy_metrics/v1"

    def _prepare_image_observation(self, obs):
        """Normalize image observations into a camera-first float32 array."""
        # TODO: 这个 helper 实际也在吞向量观测；应拆分 image/vector 两条预处理路径，避免策略接口语义含混。
        if isinstance(obs, (list, tuple)):
            cameras = [np.asarray(camera, dtype=np.float32) for camera in obs]
            if not cameras:
                raise ValueError("Image observation list is empty")
            return np.stack(cameras, axis=0)

        obs_array = np.asarray(obs, dtype=np.float32)

        if obs_array.ndim == 0:
            raise ValueError("Observation must contain image data")

        if obs_array.ndim == 1:
            return obs_array.reshape(1, -1)

        if obs_array.ndim == 2:
            return obs_array[None, ...]

        return obs_array

    def _build_metrics_payload(self, scalars=None, metadata=None):
        scalars = dict(scalars or {})
        metadata = dict(metadata or {})
        return {
            "schema": self.METRICS_SCHEMA,
            "scalars": scalars,
            "metadata": metadata,
        }

    @abstractmethod
    def act(self, obs):
        pass
