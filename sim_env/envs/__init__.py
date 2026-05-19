from .base_env import BaseEnv
from .isaac_env import IsaacEnv

try:
    from .dm_control_env import DMControlEnv
    from .ball_in_cup_take_out_env import BallInCupTakeOutEnv
except ModuleNotFoundError:
    DMControlEnv = None
    BallInCupTakeOutEnv = None

try:
    from .metaworld_env import MetaWorldEnv
except ModuleNotFoundError:
    MetaWorldEnv = None

try:
    from .sawyer_peg_insertion_side_reverse_v3 import SawyerPegInsertionSideReverseEnvV3
except ModuleNotFoundError:
    SawyerPegInsertionSideReverseEnvV3 = None

try:
    from .sawyer_peg_insertion_side_sparse_v3 import SawyerPegInsertionSideSparseEnvV3
except ModuleNotFoundError:
    SawyerPegInsertionSideSparseEnvV3 = None

try:
    from .sawyer_peg_insertion_side_reverse_shaped_v3 import SawyerPegInsertionSideReverseShapedEnvV3
except ModuleNotFoundError:
    SawyerPegInsertionSideReverseShapedEnvV3 = None

try:
    from .peg_insert_side_reverse_sparse import (
        PegInsertSideReverseSparseStandaloneEnv,
        SawyerPegInsertionSideReverseSparseStandaloneV1,
    )
except ModuleNotFoundError:
    PegInsertSideReverseSparseStandaloneEnv = None
    SawyerPegInsertionSideReverseSparseStandaloneV1 = None

__all__ = [
    "BaseEnv",
    "DMControlEnv",
    "BallInCupTakeOutEnv",
    "MetaWorldEnv",
    "PegInsertSideReverseSparseStandaloneEnv",
    "SawyerPegInsertionSideReverseSparseStandaloneV1",
    "SawyerPegInsertionSideSparseEnvV3",
    "SawyerPegInsertionSideReverseEnvV3",
    "SawyerPegInsertionSideReverseShapedEnvV3",
    "IsaacEnv",
]
