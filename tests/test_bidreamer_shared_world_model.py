from pathlib import Path
import tempfile

import numpy as np
import torch

from src.bidreamer.bidirectional_agent import BidirectionalAgent
from src.bidreamer.bidirectional_dataset import BidirectionalEpisodeDataset
from src.bidreamer.direction_world_model import DirectionConditionedWorldModel, DirectionWorldModelConfig
from src.bidreamer.eval_shared_world_model import compute_open_loop_metrics
from src.bidreamer.online_replay_buffer import EpisodeReplayBuffer
from src.bidreamer.reverse_latent_memory import NearestNeighborResult, ReverseLatentMemory


def _make_model():
    cfg = DirectionWorldModelConfig(
        obs_dim=6,
        action_dim=2,
        encoder_type="mlp",
        embed_dim=16,
        deter_dim=12,
        stoch_dim=4,
        stoch_classes=4,
        hidden_dim=32,
        free_nats=1.0,
        kl_balance=0.8,
    )
    return DirectionConditionedWorldModel(cfg)


def _make_batch(batch_size: int = 2, seq_len: int = 5):
    batch = {
        "obs": torch.randn(batch_size, seq_len, 6),
        "action": torch.randn(batch_size, seq_len, 2),
        "reward": torch.randn(batch_size, seq_len),
        "done": torch.zeros(batch_size, seq_len),
        "is_first": torch.zeros(batch_size, seq_len),
    }
    batch["is_first"][:, 0] = 1.0
    return batch


def _save_flat_npz(path: Path):
    obs = np.arange(30, dtype=np.float32).reshape(5, 6)
    action = np.arange(10, dtype=np.float32).reshape(5, 2)
    reward = np.arange(5, dtype=np.float32)
    done = np.asarray([0, 0, 1, 0, 1], dtype=np.float32)
    episode_id = np.asarray([0, 0, 0, 1, 1], dtype=np.int64)
    np.savez(path, obs=obs, action=action, reward=reward, done=done, episode_id=episode_id)


def test_forward_loss_runs():
    model = _make_model()
    metrics = model.forward_loss(_make_batch())
    assert "total_loss" in metrics
    assert metrics["direction"] == "forward"


def test_reverse_loss_runs():
    model = _make_model()
    metrics = model.reverse_loss(_make_batch())
    assert "total_loss" in metrics
    assert metrics["direction"] == "reverse"


def test_direction_uses_separate_reward_heads():
    model = _make_model()
    batch = _make_batch(batch_size=1, seq_len=4)
    with torch.no_grad():
        for param in model.forward_reward_head.parameters():
            param.zero_()
        for param in model.reverse_reward_head.parameters():
            param.zero_()
        model.reverse_reward_head.net[-1].bias[0] = 5.0
    forward_outputs = model.posterior_outputs(batch, "forward")
    reverse_outputs = model.posterior_outputs(batch, "reverse")
    assert not torch.allclose(forward_outputs["reward_pred"], reverse_outputs["reward_pred"])


def test_mixed_loss_backward():
    model = _make_model()
    mixed = model.mixed_loss(_make_batch(), _make_batch())
    mixed["mixed_total_loss"].backward()
    has_grad = any(param.grad is not None for param in model.parameters() if param.requires_grad)
    assert has_grad


def test_extract_feat_deterministic_uses_probs_not_sample():
    model = _make_model()
    batch = _make_batch(batch_size=1, seq_len=3)
    outputs = model.posterior_outputs(batch, "forward")
    sampled = model.extract_feat(outputs["post"], deterministic=False)
    deterministic = model.extract_feat(outputs["post"], deterministic=True)
    assert sampled.shape == deterministic.shape
    assert not torch.allclose(sampled, deterministic)


def test_dataset_sequence_not_cross_episode():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "data.npz"
        _save_flat_npz(path)
        dataset = BidirectionalEpisodeDataset(path, "forward", action_alignment="env_step_current")
        assert len(dataset.episodes) == 2
        batch = dataset.sample_batch(batch_size=4, seq_len=2, device="cpu")
        assert batch["obs"].shape[1] == 2


def test_action_alignment_env_step_current():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "data.npz"
        _save_flat_npz(path)
        dataset = BidirectionalEpisodeDataset(path, "forward", action_alignment="env_step_current")
        episode = dataset.episodes[0]
        assert np.allclose(episode["action"][0], np.zeros(2, dtype=np.float32))
        assert np.allclose(episode["action"][1], np.asarray([0.0, 1.0], dtype=np.float32))
        assert float(episode["reward"][0]) == 0.0
        assert float(episode["reward"][1]) == 0.0


def test_action_alignment_dreamer_prev():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "data.npz"
        _save_flat_npz(path)
        dataset = BidirectionalEpisodeDataset(path, "forward", action_alignment="dreamer_prev")
        episode = dataset.episodes[0]
        assert np.allclose(episode["action"][0], np.asarray([0.0, 1.0], dtype=np.float32))
        assert float(episode["reward"][0]) == 0.0
        assert float(episode["reward"][1]) == 1.0


def test_open_loop_eval_without_posterior_correction():
    model = _make_model()
    batch = _make_batch(batch_size=1, seq_len=6)
    metrics = compute_open_loop_metrics(model, batch, "forward", horizons=[1, 3])
    assert "open_loop_obs_pred_mse_h1" in metrics
    assert "open_loop_valid_sample_count_h3" in metrics


def test_reverse_latent_memory_build_nearest_and_subgoal():
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.6, 0.4],
            [0.0, 1.0],
            [0.1, 0.9],
            [0.2, 0.8],
        ]
    )
    memory = ReverseLatentMemory(
        features=torch.nn.functional.normalize(features, dim=-1),
        episode_ids=torch.tensor([0, 0, 0, 1, 1, 1]),
        time_indices=torch.tensor([0, 1, 2, 0, 1, 2]),
        obs=torch.randn(6, 6),
        action=torch.randn(6, 2),
        reward=torch.randn(6),
        done=torch.zeros(6),
        success=torch.ones(6),
        distance_metric="cosine",
        normalize_features=True,
    )
    nearest = memory.nearest(torch.tensor([0.0, 1.0]), k=1)[0]
    assert nearest.episode_id == 1
    subgoal = memory.get_subgoal(nearest, subgoal_step=1)
    assert subgoal["episode_id"] == 1
    assert subgoal["time_index"] == 0


def test_subgoal_does_not_cross_episode():
    features = torch.nn.functional.normalize(torch.randn(6, 4), dim=-1)
    memory = ReverseLatentMemory(
        features=features,
        episode_ids=torch.tensor([0, 0, 0, 1, 1, 1]),
        time_indices=torch.tensor([0, 1, 2, 0, 1, 2]),
        distance_metric="cosine",
        normalize_features=True,
    )
    nearest = NearestNeighborResult(global_index=3, episode_id=1, time_index=0, distance=0.0)
    subgoal = memory.get_subgoal(nearest, subgoal_step=2)
    assert subgoal["episode_id"] == 1
    assert subgoal["time_index"] == 0


def test_reverse_latent_memory_save_load():
    features = torch.nn.functional.normalize(torch.randn(4, 3), dim=-1)
    memory = ReverseLatentMemory(
        features=features,
        episode_ids=torch.tensor([0, 0, 1, 1]),
        time_indices=torch.tensor([0, 1, 0, 1]),
        distance_metric="cosine",
        normalize_features=True,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "memory.pt"
        memory.save(path)
        loaded = ReverseLatentMemory.load(path)
        assert torch.allclose(memory.features, loaded.features)
        assert loaded.normalize_features is True


def _fill_replay(buffer: EpisodeReplayBuffer):
    for step in range(3):
        buffer.add_step(
            obs=np.full((6,), step, dtype=np.float32),
            action=np.asarray([step, step + 0.5], dtype=np.float32),
            reward=float(step + 1),
            done=bool(step == 2),
            info={"t": step},
            direction=buffer.direction,
        )
    for step in range(3):
        buffer.add_step(
            obs=np.full((6,), step + 10, dtype=np.float32),
            action=np.asarray([step + 10, step + 10.5], dtype=np.float32),
            reward=float(step + 11),
            done=bool(step == 2),
            info={"t": step},
            direction=buffer.direction,
        )


def test_forward_replay_rejects_reverse_direction():
    replay = EpisodeReplayBuffer(direction="forward")
    try:
        replay.add_step(
            obs=np.zeros(6, dtype=np.float32),
            action=np.zeros(2, dtype=np.float32),
            reward=0.0,
            done=False,
            info={},
            direction="reverse",
        )
        assert False, "expected ValueError"
    except ValueError:
        assert True


def test_reverse_replay_rejects_forward_direction():
    replay = EpisodeReplayBuffer(direction="reverse")
    try:
        replay.add_step(
            obs=np.zeros(6, dtype=np.float32),
            action=np.zeros(2, dtype=np.float32),
            reward=0.0,
            done=False,
            info={},
            direction="forward",
        )
        assert False, "expected ValueError"
    except ValueError:
        assert True


def test_replay_sample_batch_direction_and_alignment():
    replay = EpisodeReplayBuffer(direction="forward")
    _fill_replay(replay)
    batch = replay.sample_batch(batch_size=8, seq_len=3, device="cpu")
    assert batch["direction"] == "forward"
    assert torch.allclose(batch["action"][:, 0], torch.zeros_like(batch["action"][:, 0]))
    assert torch.allclose(batch["reward"][:, 0], torch.zeros_like(batch["reward"][:, 0]))
    assert torch.allclose(batch["done"][:, 0], torch.zeros_like(batch["done"][:, 0]))
    assert torch.all(batch["is_first"][:, 0] == 1.0)
    expected_action_t1 = torch.tensor([0.0, 0.5], dtype=torch.float32)
    assert any(torch.allclose(row, expected_action_t1) for row in batch["action"][:, 1])
    expected_reward_t1 = torch.tensor(1.0, dtype=torch.float32)
    assert any(torch.allclose(value, expected_reward_t1) for value in batch["reward"][:, 1])
    expected_done_t1 = torch.tensor(0.0, dtype=torch.float32)
    assert any(torch.allclose(value, expected_done_t1) for value in batch["done"][:, 1])


def test_replay_sample_sequence_does_not_cross_episode():
    replay = EpisodeReplayBuffer(direction="reverse")
    _fill_replay(replay)
    batch = replay.sample_batch(batch_size=6, seq_len=3, device="cpu")
    assert batch["direction"] == "reverse"
    assert torch.all(batch["is_first"][:, 0] == 1.0)
    # Episode-local values should be either all small or all large; mixed windows would cross episodes.
    obs0 = batch["obs"][:, 0, 0]
    obs2 = batch["obs"][:, 2, 0]
    diffs = obs2 - obs0
    assert all(float(diff.item()) == 2.0 for diff in diffs)


def test_replay_export_npz():
    replay = EpisodeReplayBuffer(direction="forward")
    _fill_replay(replay)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "forward_replay_export.npz"
        out = replay.export_npz(path)
        assert out.exists()
        with np.load(out, allow_pickle=True) as data:
            assert "obs" in data.files
            assert "direction" in data.files
            assert data["direction"][0] == "forward"


def test_bidirectional_agent_rejects_wrong_batch_directions():
    model = _make_model()
    agent = BidirectionalAgent(model, learning_rate=1e-3, grad_clip=100.0)
    forward_batch = _make_batch()
    reverse_batch = _make_batch()
    forward_batch["direction"] = "reverse"
    reverse_batch["direction"] = "forward"
    try:
        agent.train_world_model(forward_batch, reverse_batch)
        assert False, "expected ValueError"
    except ValueError:
        assert True


def test_bidirectional_agent_mixed_total_matches_sum():
    model = _make_model()
    agent = BidirectionalAgent(model, learning_rate=1e-3, grad_clip=100.0)
    forward_batch = _make_batch()
    reverse_batch = _make_batch()
    forward_batch["direction"] = "forward"
    reverse_batch["direction"] = "reverse"
    metrics = agent.train_world_model(forward_batch, reverse_batch)
    assert abs(metrics["mixed_total_loss"] - (metrics["forward_total_loss"] + metrics["reverse_total_loss"])) < 1e-5


def test_bidirectional_agent_actor_value_updates_run():
    model = _make_model()
    agent = BidirectionalAgent(
        model,
        learning_rate=1e-3,
        grad_clip=100.0,
        imagination_horizon=4,
        use_slow_value=False,
    )
    forward_batch = _make_batch()
    reverse_batch = _make_batch()
    forward_batch["direction"] = "forward"
    reverse_batch["direction"] = "reverse"
    forward_metrics = agent.train_forward_actor_value(forward_batch)
    reverse_metrics = agent.train_reverse_actor_value(reverse_batch)
    assert np.isfinite(forward_metrics["forward_actor_loss"])
    assert np.isfinite(forward_metrics["forward_value_loss"])
    assert np.isfinite(reverse_metrics["reverse_actor_loss"])
    assert np.isfinite(reverse_metrics["reverse_value_loss"])


def test_bidirectional_agent_actor_value_rejects_wrong_direction():
    model = _make_model()
    agent = BidirectionalAgent(model, learning_rate=1e-3, grad_clip=100.0, imagination_horizon=4, use_slow_value=False)
    bad_batch = _make_batch()
    bad_batch["direction"] = "reverse"
    try:
        agent.train_forward_actor_value(bad_batch)
        assert False, "expected ValueError"
    except ValueError:
        assert True


def test_bidirectional_agent_act_returns_finite_action():
    model = _make_model()
    agent = BidirectionalAgent(model, learning_rate=1e-3, grad_clip=100.0, imagination_horizon=4, use_slow_value=False)
    obs = np.zeros(6, dtype=np.float32)
    action = agent.act(obs, direction="forward", deterministic=False)
    assert action.shape == (2,)
    assert np.isfinite(action).all()


def test_reverse_latent_memory_build_from_replay():
    model = _make_model()
    replay = EpisodeReplayBuffer(direction="reverse")
    for step in range(3):
        replay.add_step(
            obs=np.full((6,), step, dtype=np.float32),
            action=np.asarray([step, step + 0.5], dtype=np.float32),
            reward=float(step + 1),
            done=bool(step == 2),
            info={"success": True, "is_success": True},
            direction="reverse",
        )
    memory = ReverseLatentMemory.build_from_reverse_replay(
        model=model,
        replay=replay,
        device="cpu",
        action_alignment="env_step_current",
        use_success_only=True,
        distance_metric="cosine",
        normalize_features=True,
        feature_whiten=False,
    )
    assert len(memory.features) > 0
    assert memory.summary()["num_latents"] > 0


def test_forward_latent_prior_disabled_before_memory():
    model = _make_model()
    agent = BidirectionalAgent(
        model,
        learning_rate=1e-3,
        grad_clip=100.0,
        imagination_horizon=4,
        use_slow_value=False,
        latent_prior_enabled=True,
        latent_prior_beta=1.0,
        latent_prior_horizon=2,
        latent_prior_subgoal_step=1,
    )
    forward_batch = _make_batch()
    forward_batch["direction"] = "forward"
    metrics = agent.train_forward_actor_value(forward_batch, reverse_latent_memory=None)
    assert metrics["latent_prior_loss"] == 0.0
    assert metrics["latent_prior_active"] == 0.0


def test_forward_latent_prior_updates_actor_only():
    model = _make_model()
    agent = BidirectionalAgent(
        model,
        learning_rate=1e-3,
        grad_clip=100.0,
        imagination_horizon=4,
        use_slow_value=False,
        latent_prior_enabled=True,
        latent_prior_beta=1.0,
        latent_prior_horizon=2,
        latent_prior_subgoal_step=1,
    )
    forward_batch = _make_batch()
    reverse_batch = _make_batch()
    forward_batch["direction"] = "forward"
    reverse_batch["direction"] = "reverse"
    with torch.no_grad():
        outputs = model.posterior_outputs(reverse_batch, "reverse")
        features = model.extract_feat(outputs["post"], deterministic=True).reshape(-1, model.feat_dim).cpu()
    memory = ReverseLatentMemory(
        features=features,
        episode_ids=torch.zeros(features.shape[0], dtype=torch.long),
        time_indices=torch.arange(features.shape[0], dtype=torch.long),
        obs=reverse_batch["obs"].reshape(-1, reverse_batch["obs"].shape[-1]).cpu(),
        action=reverse_batch["action"].reshape(-1, reverse_batch["action"].shape[-1]).cpu(),
        reward=reverse_batch["reward"].reshape(-1).cpu(),
        done=reverse_batch["done"].reshape(-1).cpu(),
        success=torch.ones(features.shape[0]),
        distance_metric="l2",
        normalize_features=False,
    )
    before = [param.detach().clone() for param in model.parameters()]
    metrics = agent.train_forward_actor_value(forward_batch, reverse_latent_memory=memory)
    after = [param.detach().clone() for param in model.parameters()]
    assert np.isfinite(metrics["latent_prior_loss"])
    assert metrics["latent_prior_active"] == 1.0
    assert np.isfinite(metrics["mean_prior_weight"])
    assert np.isfinite(metrics["latent_subgoal_distance"])
    for before_param, after_param in zip(before, after):
        assert torch.allclose(before_param, after_param)
