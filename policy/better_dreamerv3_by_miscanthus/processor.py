from __future__ import annotations

import numpy as np

from .dreamerv3_model import DreamerObservationSpec


class DreamerV3Processor:
    def __init__(self, device: str = "cpu"):
        self.device = device

    def _looks_like_image_frame(self, item) -> bool:
        arr = np.asarray(item)
        if arr.ndim == 2:
            return True
        if arr.ndim != 3:
            return False
        channel_first = arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4)
        channel_last = arr.shape[-1] in (1, 3, 4)
        return channel_first or channel_last

    def _is_image_observation(self, obs) -> bool:
        if isinstance(obs, (list, tuple)):
            return bool(obs) and all(self._looks_like_image_frame(item) for item in obs)
        arr = np.asarray(obs)
        return arr.ndim >= 2

    def _prepare_image(self, obs) -> np.ndarray:
        if isinstance(obs, (list, tuple)):
            cameras = list(obs)
        else:
            obs_array = np.asarray(obs)
            cameras = list(obs_array) if obs_array.ndim == 4 else [obs]
        frames = []
        for camera in cameras:
            frame = np.asarray(camera, dtype=np.float32)
            if frame.ndim == 2:
                frame = frame[..., None]
            if frame.ndim != 3:
                raise ValueError(f"Unsupported image observation shape: {frame.shape}")
            if frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
                chw = frame
            else:
                chw = np.transpose(frame, (2, 0, 1))
            if chw.max() > 2.0:
                chw = chw / 255.0
            frames.append(chw.astype(np.float32))
        if not frames:
            raise ValueError("Image observation list is empty")
        base_hw = frames[0].shape[1:]
        for frame in frames[1:]:
            if frame.shape[1:] != base_hw:
                raise ValueError("All camera images must share the same height and width")
        return np.concatenate(frames, axis=0)

    def infer_observation_spec(self, obs) -> DreamerObservationSpec:
        if self._is_image_observation(obs):
            image = self._prepare_image(obs)
            return DreamerObservationSpec(mode="image", obs_shape=tuple(int(x) for x in image.shape))
        vector = np.asarray(obs, dtype=np.float32).reshape(-1)
        return DreamerObservationSpec(mode="vector", obs_dim=int(vector.shape[0]))

    def preprocess_obs(self, obs, observation: DreamerObservationSpec) -> np.ndarray:
        if observation.mode == "image":
            prepared = self._prepare_image(obs)
            if tuple(prepared.shape) != tuple(observation.obs_shape):
                raise ValueError(f"Expected image observation shape {observation.obs_shape}, got {prepared.shape}")
            return prepared

        prepared = np.asarray(obs, dtype=np.float32).reshape(-1)
        if prepared.shape[0] != observation.obs_dim:
            raise ValueError(f"Expected vector observation dim {observation.obs_dim}, got {prepared.shape[0]}")
        return prepared

    def postprocess_action(self, action) -> np.ndarray:
        return np.asarray(action, dtype=np.float32).reshape(-1)
