from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .action_penalty import ActionPenaltyConfig
from .event_dynamics import DreamerEventDynamicsConfig
from .operator_intrinsic_reward import OperatorIntrinsicRewardConfig
from .thick_context import ThickContextConfig


@dataclass(slots=True)
class DreamerObservationSpec:
    mode: str
    obs_dim: int | None = None
    obs_shape: tuple[int, int, int] | None = None

    def asdict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class DreamerAuxConfig:
    predict_grasp: bool = False
    contact_num_classes: int | None = None


@dataclass(slots=True)
class DreamerV3ModelConfig:
    action_dim: int
    observation: DreamerObservationSpec
    device: str = "cpu"
    embed_dim: int = 128
    proprio_input_dim: int = 4
    proprio_embed_dim: int = 32
    proprio_loss_weight: float = 1.0
    deter_dim: int = 128
    stoch_dim: int = 32
    stoch_classes: int = 32
    hidden_dim: int = 256
    num_layers: int = 2
    actor_hidden_dim: int = 256
    actor_num_layers: int = 2
    value_hidden_dim: int = 256
    value_num_layers: int = 2
    world_model_lr: float = 3e-4
    actor_lr: float = 8e-5
    value_lr: float = 8e-5
    grad_clip: float = 100.0
    actor_grad_clip: float = 100.0
    value_grad_clip: float = 100.0
    gamma: float = 0.99
    lambda_: float = 0.95
    imagination_horizon: int = 15
    entropy_coef: float = 3e-4
    min_std: float = 0.1
    max_std: float = 1.0
    init_std: float = 1.0
    imag_last: int = 0
    use_symlog_obs: bool = True
    use_symlog_reward: bool = True
    use_twohot_reward: bool = True
    use_twohot_value: bool = True
    free_nats: float = 1.0
    kl_balance: float = 0.8
    twohot_bins: int = 255
    twohot_low: float = -20.0
    twohot_high: float = 20.0
    rssm_unimix: float = 0.01
    use_slow_value: bool = True
    slow_value_rate: float = 0.02
    repval_loss: bool = True
    repval_scale: float = 0.3
    use_return_norm: bool = True
    use_advantage_norm: bool = True
    use_actor_return_scale: bool = True
    actor_return_scale_decay: float = 0.99
    actor_return_scale_q_low: float = 0.05
    actor_return_scale_q_high: float = 0.95
    actor_return_scale_min: float = 1.0
    slow_gain_reward_scale: float = 1.0
    external_slow_gain_reward_scale: float = 1.0
    v8_confirm_reward_scale: float = 1.0
    v8_memory_size: int = 2048
    v8_memory_horizon: int = 200
    v8_min_memory: int = 32
    v8_slow_summary_window: int = 20
    v8_credit_window: int = 50
    v8_memory_attention_dim: int = 64
    v8_memory_context_dim: int = 64
    v8_pose_bias_weight: float = 1.0
    v8_pose_sigma: float = 0.05
    v8_confirm_pose_sigma: float = 0.03
    v8_pose_conf_threshold: float = 0.8
    v8_visual_change_ema_decay: float = 0.99
    v8_visual_change_threshold: float = 1.0
    v8_milestone_enabled: bool = False
    v8_proprio_bin: float = 0.03
    v8_visual_signature_dim: int = 16
    norm_rate: float = 0.01
    norm_eps: float = 1e-8
    aux: DreamerAuxConfig = field(default_factory=DreamerAuxConfig)
    thick_context: ThickContextConfig = field(default_factory=ThickContextConfig)
    event_dynamics: DreamerEventDynamicsConfig = field(default_factory=DreamerEventDynamicsConfig)
    operator_intrinsic_reward: OperatorIntrinsicRewardConfig = field(default_factory=OperatorIntrinsicRewardConfig)
    action_penalty: ActionPenaltyConfig = field(default_factory=ActionPenaltyConfig)

    @property
    def encoder_type(self) -> str:
        return "cnn" if self.observation.mode == "image" else "mlp"

    @property
    def base_feat_dim(self) -> int:
        return int(self.deter_dim + self.stoch_dim * self.stoch_classes)

    @property
    def augmented_feat_dim(self) -> int:
        if not self.thick_context.enabled:
            return self.base_feat_dim
        return int(self.base_feat_dim + self.thick_context.context_dim)

    def asdict(self) -> dict:
        data = asdict(self)
        data["observation"] = self.observation.asdict()
        return data
