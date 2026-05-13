from __future__ import annotations

from dataclasses import asdict, dataclass, field


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
    entropy_coef: float = 1e-3
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
    norm_rate: float = 0.01
    norm_eps: float = 1e-8
    aux: DreamerAuxConfig = field(default_factory=DreamerAuxConfig)

    @property
    def encoder_type(self) -> str:
        return "cnn" if self.observation.mode == "image" else "mlp"

    def asdict(self) -> dict:
        data = asdict(self)
        data["observation"] = self.observation.asdict()
        return data
