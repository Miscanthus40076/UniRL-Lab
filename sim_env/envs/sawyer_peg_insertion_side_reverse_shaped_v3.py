from __future__ import annotations

import numpy as np

from .sawyer_peg_insertion_side_reverse_v3 import SawyerPegInsertionSideReverseEnvV3


class SawyerPegInsertionSideReverseShapedEnvV3(SawyerPegInsertionSideReverseEnvV3):
    """Reverse peg task with minimal shaping: approach, touch, unplug, then place and release."""

    APPROACH_DISTANCE = 0.12
    TOUCH_DISTANCE = 0.03

    APPROACH_REWARD_SCALE = 0.20
    TOUCH_BONUS_SCALE = 0.20
    UNPLUG_PROGRESS_SCALE = 0.35
    UNPLUG_BONUS_SCALE = 0.20
    TABLE_REWARD_SCALE = 0.10
    RELEASE_REWARD_SCALE = 0.10
    SUCCESS_BONUS_SCALE = 1.00

    def _shaping_terms(self, obs):
        success, tcp_to_obj, obj_to_unplug_target, on_table, released = self._sparse_success(obs)

        approach_reward = float(np.clip(1.0 - (tcp_to_obj / self.APPROACH_DISTANCE), 0.0, 1.0))
        touch_bonus = float(tcp_to_obj <= self.TOUCH_DISTANCE)
        unplug_progress = float(np.clip(1.0 - (obj_to_unplug_target / self.UNPLUG_DISTANCE_X), 0.0, 1.0))
        unplug_bonus = float(self._has_been_unplugged)
        table_reward = float(on_table) * unplug_bonus
        release_reward = float(released) * table_reward

        reward = (
            self.APPROACH_REWARD_SCALE * approach_reward
            + self.TOUCH_BONUS_SCALE * touch_bonus
            + self.UNPLUG_PROGRESS_SCALE * unplug_progress
            + self.UNPLUG_BONUS_SCALE * unplug_bonus
            + self.TABLE_REWARD_SCALE * table_reward
            + self.RELEASE_REWARD_SCALE * release_reward
            + self.SUCCESS_BONUS_SCALE * float(success)
        )

        return {
            "success": bool(success),
            "reward": float(reward),
            "tcp_to_obj": float(tcp_to_obj),
            "obj_to_unplug_target": float(obj_to_unplug_target),
            "on_table": float(on_table),
            "released": float(released),
            "approach_reward": float(approach_reward),
            "touch_bonus": float(touch_bonus),
            "unplug_progress": float(unplug_progress),
            "unplug_bonus": float(unplug_bonus),
            "table_reward": float(table_reward),
            "release_reward": float(release_reward),
        }

    def evaluate_state(self, obs, action):
        del action
        terms = self._shaping_terms(obs)
        info = {
            "success": float(terms["success"]),
            "near_object": float(terms["tcp_to_obj"] <= self.TOUCH_DISTANCE),
            "grasp_success": float(terms["tcp_to_obj"] <= 0.04),
            "grasp_reward": float(terms["touch_bonus"]),
            "in_place_reward": float(terms["unplug_progress"]),
            "obj_to_target": float(terms["obj_to_unplug_target"]),
            "obj_on_table": float(terms["on_table"]),
            "released": float(terms["released"]),
            "has_been_unplugged": float(self._has_been_unplugged),
            "approach_reward": float(terms["approach_reward"]),
            "touch_bonus": float(terms["touch_bonus"]),
            "unplug_progress": float(terms["unplug_progress"]),
            "unplug_bonus": float(terms["unplug_bonus"]),
            "table_reward": float(terms["table_reward"]),
            "release_reward": float(terms["release_reward"]),
            "unscaled_reward": float(terms["reward"]),
        }
        return float(terms["reward"]), info

    def compute_reward(self, action, obs):
        del action
        terms = self._shaping_terms(obs)
        return (
            float(terms["reward"]),
            float(terms["tcp_to_obj"]),
            float(terms["touch_bonus"]),
            float(terms["obj_to_unplug_target"]),
            float(terms["unplug_bonus"]),
            float(terms["on_table"]),
            float(terms["released"]),
            float(terms["unplug_progress"]),
        )
