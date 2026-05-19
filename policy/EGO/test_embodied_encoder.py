from __future__ import annotations

import torch

from .dreamerv3_model import DreamerObservationSpec, DreamerV3ModelConfig
from .embodied_encoder import EmbodiedEncoder, EmbodiedEncoderConfig, add_prev_action_if_missing
from .replay_buffer import EpisodeReplayBuffer
from .world_model import DreamerV3WorldModel, DreamerV3WorldModelConfig


def _batch(batch_size: int = 2, seq_len: int = 4, obs_dim: int = 5, action_dim: int = 3):
    return {
        "obs": torch.randn(batch_size, seq_len, obs_dim),
        "action": torch.randn(batch_size, seq_len, action_dim),
        "reward": torch.randn(batch_size, seq_len),
        "done": torch.zeros(batch_size, seq_len),
        "is_first": torch.zeros(batch_size, seq_len),
        "robot_state": torch.randn(batch_size, seq_len, action_dim),
    }


def _obs_action_batch(batch_size: int = 2, seq_len: int = 4, obs_dim: int = 5, action_dim: int = 3):
    return {
        "obs": torch.randn(batch_size, seq_len, obs_dim),
        "action": torch.randn(batch_size, seq_len, action_dim),
        "is_first": torch.zeros(batch_size, seq_len),
    }


def test_embodied_encoder_forward_shapes():
    config = EmbodiedEncoderConfig(use_embodied_encoder=True, embodied_encoder_dim=7, embodied_robot_latent_dim=11)
    module = EmbodiedEncoder(action_dim=3, config=config)
    out = module(_batch(action_dim=3))
    assert out["e_self"].shape == (2, 4, 7)
    assert out["z_robot"].shape == (2, 4, 11)
    assert "robot_state" in out["used_keys"]
    assert out["diagnostics"]["fallback_to_action"] is False
    assert out["diagnostics"]["has_proprio"] is True


def test_embodied_encoder_action_only_fallback():
    config = EmbodiedEncoderConfig(use_embodied_encoder=True, embodied_encoder_dim=7, embodied_robot_latent_dim=11)
    module = EmbodiedEncoder(action_dim=3, config=config)
    out = module(_obs_action_batch(action_dim=3))
    assert out["e_self"].shape == (2, 4, 7)
    assert out["diagnostics"]["fallback_to_action"] is True
    assert out["diagnostics"]["used_robot_fields"] == ["prev_action"]


def test_prev_action_shift_respects_is_first():
    batch = _obs_action_batch(batch_size=1, seq_len=4, action_dim=2)
    batch["action"] = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])
    batch["is_first"] = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    enriched = add_prev_action_if_missing(batch)
    expected = torch.tensor([[[0.0, 0.0], [1.0, 2.0], [0.0, 0.0], [5.0, 6.0]]])
    assert torch.allclose(enriched["prev_action"], expected)


def test_embodied_encoder_aux_loss_computes():
    config = EmbodiedEncoderConfig(use_embodied_encoder=True)
    module = EmbodiedEncoder(action_dim=3, config=config)
    losses = module.aux_losses(_batch(action_dim=3))
    assert "aux_loss_total" in losses
    assert torch.isfinite(losses["aux_loss_total"])


def test_dreamer_path_unchanged_when_disabled():
    config = DreamerV3WorldModelConfig(
        obs_dim=5,
        action_dim=3,
        encoder_type="mlp",
        embed_dim=8,
        deter_dim=8,
        stoch_dim=2,
        stoch_classes=4,
        hidden_dim=16,
        embodied_encoder=EmbodiedEncoderConfig(use_embodied_encoder=False),
    )
    model = DreamerV3WorldModel(config)
    batch = _batch(obs_dim=5, action_dim=3)
    encoded = model.encode_observation(batch)
    expected = model._seq_apply(model.encoder, batch["obs"])
    assert torch.allclose(encoded["embed"], expected)
    assert encoded["e_self"] is None


def test_dreamer_embodied_embed_enters_rssm():
    config = DreamerV3WorldModelConfig(
        obs_dim=5,
        action_dim=3,
        encoder_type="mlp",
        embed_dim=8,
        deter_dim=8,
        stoch_dim=2,
        stoch_classes=4,
        hidden_dim=16,
        embodied_encoder=EmbodiedEncoderConfig(
            use_embodied_encoder=True,
            embodied_encoder_dim=6,
            embodied_robot_latent_dim=6,
            freeze_embodied_encoder=False,
            detach_embodied_encoder=False,
        ),
    )
    model = DreamerV3WorldModel(config)
    outputs = model(_batch(obs_dim=5, action_dim=3))
    assert outputs["embed"].shape == (2, 4, 8)
    assert outputs["post"]["h"].shape[:2] == (2, 4)
    assert outputs["e_self"].shape == (2, 4, 6)
    assert outputs["embodied_diagnostics"]["fallback_to_action"] is False


def test_replay_carries_optional_robot_fields():
    replay = EpisodeReplayBuffer(capacity=10)
    for index in range(4):
        replay.add_step(
            obs=torch.randn(5).numpy(),
            action=torch.randn(3).numpy(),
            reward=0.0,
            done=False,
            robot_fields={
                "robot_state": torch.ones(3).numpy() * index,
                "gripper": torch.ones(1).numpy() * index,
            },
        )
    batch = replay.sample_batch(batch_size=1, seq_len=3)
    assert "robot_state" in batch
    assert "gripper" in batch
    assert batch["robot_state"].shape[-1] == 3


def test_model_config_exposes_embodied_encoder_defaults():
    config = DreamerV3ModelConfig(
        action_dim=3,
        observation=DreamerObservationSpec(mode="vector", obs_dim=5),
    )
    assert config.embodied_encoder.use_embodied_encoder is False
