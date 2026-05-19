from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from metaworld.envs.sawyer_peg_insertion_side_v3 import SawyerPegInsertionSideEnvV3


class SawyerPegInsertionSideSparseEnvV3(SawyerPegInsertionSideEnvV3):
    """Forward peg insertion task with strictly sparse environment reward."""

    @staticmethod
    def _success_from_obj_to_target(obj_to_target: float, target_radius: float) -> bool:
        return float(obj_to_target) <= float(target_radius)

    def _dense_terms(
        self,
        action: npt.NDArray[Any],
        obs: npt.NDArray[np.float64],
    ) -> tuple[float, float, float, float, float, float, float, float]:
        original_version = getattr(self, "reward_function_version", "v2")
        try:
            self.reward_function_version = "v2"
            return super().compute_reward(action, obs)
        finally:
            self.reward_function_version = original_version

    def evaluate_state(
        self,
        obs: npt.NDArray[np.float64],
        action: npt.NDArray[np.float32],
    ) -> tuple[float, dict[str, Any]]:
        (
            dense_reward,
            tcp_to_obj,
            tcp_open,
            obj_to_target,
            grasp_reward,
            in_place_reward,
            collision_box_front,
            ip_orig,
        ) = self._dense_terms(action, obs)

        success = float(self._success_from_obj_to_target(obj_to_target, self.TARGET_RADIUS))
        assert self.obj_init_pos is not None
        obj = obs[4:7]
        grasp_success = float(
            tcp_to_obj < 0.02
            and (tcp_open > 0)
            and (obj[2] - 0.01 > self.obj_init_pos[2])
        )
        near_object = float(tcp_to_obj <= 0.03)

        info = {
            "success": success,
            "near_object": near_object,
            "grasp_success": grasp_success,
            "grasp_reward": float(grasp_reward),
            "in_place_reward": float(in_place_reward),
            "collision_box_reward": float(collision_box_front),
            "in_place_raw": float(ip_orig),
            "obj_to_target": float(obj_to_target),
            "dense_reward": float(dense_reward),
            "unscaled_reward": success,
        }
        return success, info

    def compute_reward(
        self,
        action: npt.NDArray[Any],
        obs: npt.NDArray[np.float64],
    ) -> tuple[float, float, float, float, float, float, float, float]:
        (
            dense_reward,
            tcp_to_obj,
            tcp_open,
            obj_to_target,
            grasp_reward,
            in_place_reward,
            collision_box_front,
            ip_orig,
        ) = self._dense_terms(action, obs)
        del dense_reward
        reward = float(self._success_from_obj_to_target(obj_to_target, self.TARGET_RADIUS))
        return (
            reward,
            float(tcp_to_obj),
            float(tcp_open),
            float(obj_to_target),
            float(grasp_reward),
            float(in_place_reward),
            float(collision_box_front),
            float(ip_orig),
        )
