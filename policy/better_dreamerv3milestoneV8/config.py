from __future__ import annotations

from dataclasses import asdict, dataclass, field
import torch

from .action_penalty import ActionPenaltyConfig
from .event_dynamics import DreamerEventDynamicsConfig
from .operator_intrinsic_reward import OperatorIntrinsicRewardConfig
from .thick_context import ThickContextConfig

from .dreamerv3_model import DreamerAuxConfig, DreamerObservationSpec, DreamerV3ModelConfig


@dataclass(slots=True)
class DreamerV3CheckpointConfig:
    load: bool = False
    path: str | None = None
    save: bool = False


@dataclass(slots=True)
class DreamerV3PolicyConfig:
    device: str = "cuda"
    allow_cpu_fallback: bool = False
    batch_size: int = 16
    seq_len: int = 32
    warmup_steps: int = 1000
    train_every: int = 1
    train_ratio: float = 1.0
    max_updates_per_step: int = 0
    replay_capacity: int = 100000
    priority_replay: bool = False
    priority_exponent: float = 0.8
    priority_uniform_mix: float = 0.1
    priority_initial: float = 1.0

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

    gamma: float = 0.99
    lambda_: float = 0.95
    imagination_horizon: int = 15
    entropy_coef: float = 3e-4

    grad_clip: float = 100.0
    actor_grad_clip: float = 100.0
    value_grad_clip: float = 100.0
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

    aux_predict_grasp: bool = False
    aux_contact_num_classes: int | None = None
    thick_context: ThickContextConfig = field(default_factory=ThickContextConfig)
    event_dynamics: DreamerEventDynamicsConfig = field(default_factory=DreamerEventDynamicsConfig)
    operator_intrinsic_reward: OperatorIntrinsicRewardConfig = field(default_factory=OperatorIntrinsicRewardConfig)
    action_penalty: ActionPenaltyConfig = field(default_factory=ActionPenaltyConfig)

    def validate(self):
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be 'cpu' or 'cuda'")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.seq_len <= 1:
            raise ValueError("seq_len must be > 1")
        if self.train_every <= 0:
            raise ValueError("train_every must be positive")
        if self.train_ratio < 0.0:
            raise ValueError("train_ratio must be >= 0")
        if self.replay_capacity <= 0:
            raise ValueError("replay_capacity must be positive")
        if self.max_std <= self.min_std:
            raise ValueError("max_std must be greater than min_std")
        if self.stoch_dim <= 0 or self.stoch_classes <= 0:
            raise ValueError("stoch_dim and stoch_classes must be positive")
        if self.aux_contact_num_classes is not None and self.aux_contact_num_classes <= 1:
            raise ValueError("aux_contact_num_classes must be > 1 when provided")
        self.thick_context.validate()
        self.event_dynamics.validate()
        self.operator_intrinsic_reward.validate()
        self.action_penalty.validate()
        if self.event_dynamics.enabled and not self.thick_context.enabled:
            raise ValueError("event_dynamics.enabled=true requires thick_context.enabled=true")
        if self.operator_intrinsic_reward.enabled and self.operator_intrinsic_reward.capacity_only:
            if not self.event_dynamics.enabled or not self.event_dynamics.capacity_enabled:
                raise ValueError(
                    "operator_intrinsic_reward.capacity_only=true requires event_dynamics.enabled=true "
                    "and event_dynamics.capacity_enabled=true"
                )

    def asdict(self) -> dict:
        return asdict(self)


def build_dreamerv3_policy_config(policy_config: dict | None) -> tuple[DreamerV3PolicyConfig, DreamerV3CheckpointConfig]:
    policy_config = dict(policy_config or {})
    raw = dict(policy_config.get("dreamerv3", {}))
    aux = dict(raw.pop("aux", {}))
    thick_context = ThickContextConfig(**dict(raw.pop("thick_context", {})))
    event_dynamics = DreamerEventDynamicsConfig(**dict(raw.pop("event_dynamics", {})))
    operator_intrinsic_reward = OperatorIntrinsicRewardConfig(**dict(policy_config.get("operator_intrinsic_reward", {})))
    action_penalty = ActionPenaltyConfig(**dict(policy_config.get("action_penalty", raw.pop("action_penalty", {}))))
    if "predict_grasp" in aux:
        raw["aux_predict_grasp"] = aux["predict_grasp"]
    if "contact_num_classes" in aux:
        raw["aux_contact_num_classes"] = aux["contact_num_classes"]
    raw["thick_context"] = thick_context
    raw["event_dynamics"] = event_dynamics
    raw["operator_intrinsic_reward"] = operator_intrinsic_reward
    raw["action_penalty"] = action_penalty

    cfg = DreamerV3PolicyConfig(**raw)
    cfg.validate()
    if cfg.device == "cuda" and not torch.cuda.is_available():
        if cfg.allow_cpu_fallback:
            print("[DreamerV3] CUDA unavailable, falling back to CPU because allow_cpu_fallback=true")
            cfg.device = "cpu"
        else:
            raise RuntimeError(
                "DreamerV3 policy is configured with device='cuda' but CUDA is unavailable. "
                "Set policy.dreamerv3.allow_cpu_fallback=true only for debug fallback."
            )

    checkpoint_raw = dict(policy_config.get("checkpoint", {}))
    checkpoint = DreamerV3CheckpointConfig(
        load=bool(checkpoint_raw.get("load", False)),
        path=checkpoint_raw.get("path"),
        save=bool(checkpoint_raw.get("save", False)),
    )
    return cfg, checkpoint


def build_dreamerv3_model_config(
    cfg: DreamerV3PolicyConfig,
    action_dim: int,
    observation: DreamerObservationSpec,
) -> DreamerV3ModelConfig:
    return DreamerV3ModelConfig(
        action_dim=int(action_dim),
        observation=observation,
        device=cfg.device,
        embed_dim=cfg.embed_dim,
        proprio_input_dim=cfg.proprio_input_dim,
        proprio_embed_dim=cfg.proprio_embed_dim,
        proprio_loss_weight=cfg.proprio_loss_weight,
        deter_dim=cfg.deter_dim,
        stoch_dim=cfg.stoch_dim,
        stoch_classes=cfg.stoch_classes,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        actor_hidden_dim=cfg.actor_hidden_dim,
        actor_num_layers=cfg.actor_num_layers,
        value_hidden_dim=cfg.value_hidden_dim,
        value_num_layers=cfg.value_num_layers,
        world_model_lr=cfg.world_model_lr,
        actor_lr=cfg.actor_lr,
        value_lr=cfg.value_lr,
        grad_clip=cfg.grad_clip,
        actor_grad_clip=cfg.actor_grad_clip,
        value_grad_clip=cfg.value_grad_clip,
        gamma=cfg.gamma,
        lambda_=cfg.lambda_,
        imagination_horizon=cfg.imagination_horizon,
        entropy_coef=cfg.entropy_coef,
        min_std=cfg.min_std,
        max_std=cfg.max_std,
        init_std=cfg.init_std,
        imag_last=cfg.imag_last,
        use_symlog_obs=cfg.use_symlog_obs,
        use_symlog_reward=cfg.use_symlog_reward,
        use_twohot_reward=cfg.use_twohot_reward,
        use_twohot_value=cfg.use_twohot_value,
        free_nats=cfg.free_nats,
        kl_balance=cfg.kl_balance,
        twohot_bins=cfg.twohot_bins,
        twohot_low=cfg.twohot_low,
        twohot_high=cfg.twohot_high,
        rssm_unimix=cfg.rssm_unimix,
        use_slow_value=cfg.use_slow_value,
        slow_value_rate=cfg.slow_value_rate,
        repval_loss=cfg.repval_loss,
        repval_scale=cfg.repval_scale,
        use_return_norm=cfg.use_return_norm,
        use_advantage_norm=cfg.use_advantage_norm,
        use_actor_return_scale=cfg.use_actor_return_scale,
        actor_return_scale_decay=cfg.actor_return_scale_decay,
        actor_return_scale_q_low=cfg.actor_return_scale_q_low,
        actor_return_scale_q_high=cfg.actor_return_scale_q_high,
        actor_return_scale_min=cfg.actor_return_scale_min,
        slow_gain_reward_scale=cfg.slow_gain_reward_scale,
        external_slow_gain_reward_scale=cfg.external_slow_gain_reward_scale,
        v8_confirm_reward_scale=cfg.v8_confirm_reward_scale,
        v8_memory_size=cfg.v8_memory_size,
        v8_memory_horizon=cfg.v8_memory_horizon,
        v8_min_memory=cfg.v8_min_memory,
        v8_slow_summary_window=cfg.v8_slow_summary_window,
        v8_credit_window=cfg.v8_credit_window,
        v8_memory_attention_dim=cfg.v8_memory_attention_dim,
        v8_memory_context_dim=cfg.v8_memory_context_dim,
        v8_pose_bias_weight=cfg.v8_pose_bias_weight,
        v8_pose_sigma=cfg.v8_pose_sigma,
        v8_confirm_pose_sigma=cfg.v8_confirm_pose_sigma,
        v8_pose_conf_threshold=cfg.v8_pose_conf_threshold,
        v8_visual_change_ema_decay=cfg.v8_visual_change_ema_decay,
        v8_visual_change_threshold=cfg.v8_visual_change_threshold,
        v8_milestone_enabled=cfg.v8_milestone_enabled,
        v8_proprio_bin=cfg.v8_proprio_bin,
        v8_visual_signature_dim=cfg.v8_visual_signature_dim,
        norm_rate=cfg.norm_rate,
        norm_eps=cfg.norm_eps,
        aux=DreamerAuxConfig(
            predict_grasp=cfg.aux_predict_grasp,
            contact_num_classes=cfg.aux_contact_num_classes,
        ),
        thick_context=cfg.thick_context,
        event_dynamics=cfg.event_dynamics,
        operator_intrinsic_reward=cfg.operator_intrinsic_reward,
        action_penalty=cfg.action_penalty,
    )
