from __future__ import annotations

import numpy as np

from metaworld.envs.sawyer_peg_insertion_side_v3 import SawyerPegInsertionSideEnvV3


class SawyerPegInsertionSideReverseEnvV3(SawyerPegInsertionSideEnvV3):
    """Reverse task for peg_insert_side_v3: unplug then place peg on table."""

    TABLE_HEIGHT_Z = 0.0
    TABLE_CONTACT_Z_THRESHOLD = 0.055
    UNPLUG_DISTANCE_X = 0.15
    UNPLUG_SUCCESS_RADIUS = 0.07
    RELEASE_DISTANCE = 0.06

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._has_been_unplugged = False
        self._inserted_obj_init_pos = None
        self._unplug_target_pos = None

    def reset_model(self):
        super().reset_model()
        assert self._target_pos is not None

        inserted_obj_pos = self._target_pos + np.array([0.1, 0.0, 0.0], dtype=np.float64)
        self._set_obj_xyz(inserted_obj_pos)

        self.obj_init_pos = inserted_obj_pos.copy()
        self._inserted_obj_init_pos = inserted_obj_pos.copy()
        self._unplug_target_pos = inserted_obj_pos + np.array(
            [self.UNPLUG_DISTANCE_X, 0.0, 0.0],
            dtype=np.float64,
        )
        self._has_been_unplugged = False

        return self._get_obs()

    def _sparse_success(self, obs):
        assert self._unplug_target_pos is not None
        obj = obs[4:7]
        tcp = self.tcp_center

        tcp_to_obj = float(np.linalg.norm(obj - tcp))
        obj_to_unplug_target = float(np.linalg.norm(obj - self._unplug_target_pos))

        unplugged_now = obj_to_unplug_target <= self.UNPLUG_SUCCESS_RADIUS
        if unplugged_now:
            self._has_been_unplugged = True

        on_table = obj[2] <= (self.TABLE_HEIGHT_Z + self.TABLE_CONTACT_Z_THRESHOLD)
        released = tcp_to_obj >= self.RELEASE_DISTANCE
        success = bool(self._has_been_unplugged and on_table and released)

        return success, tcp_to_obj, obj_to_unplug_target, float(on_table), float(released)

    def evaluate_state(self, obs, action):
        del action
        success, tcp_to_obj, obj_to_unplug_target, on_table, released = self._sparse_success(obs)
        reward = 1.0 if success else 0.0
        info = {
            "success": float(success),
            "near_object": float(tcp_to_obj <= 0.03),
            "grasp_success": float(tcp_to_obj <= 0.04),
            "grasp_reward": 0.0,
            "in_place_reward": 0.0,
            "obj_to_target": obj_to_unplug_target,
            "obj_on_table": on_table,
            "released": released,
            "has_been_unplugged": float(self._has_been_unplugged),
            "unscaled_reward": reward,
        }
        return reward, info

    def compute_reward(self, action, obs):
        del action
        success, tcp_to_obj, obj_to_unplug_target, on_table, released = self._sparse_success(obs)
        reward = 1.0 if success else 0.0
        return (
            reward,
            tcp_to_obj,
            0.0,
            obj_to_unplug_target,
            float(self._has_been_unplugged),
            on_table,
            released,
            0.0,
        )
