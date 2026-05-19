from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

import numpy as np

from policy.dreamerv3.gate_video_recorder import maybe_record_gate_eval_videos, record_gate_eval_videos


ROOT = Path(__file__).resolve().parents[1]


class DummyEnv:
    def __init__(self, episode_length: int = 4, cameras: int = 3):
        self.episode_length = int(episode_length)
        self.cameras = int(cameras)
        self._step = 0

    def _obs(self):
        frames = []
        for camera_index in range(self.cameras):
            frame = np.full((8, 8, 3), fill_value=(camera_index + 1) * 40 + self._step, dtype=np.uint8)
            frames.append(frame)
        return frames

    def reset(self):
        self._step = 0
        return self._obs()

    def step(self, action):
        del action
        self._step += 1
        done = self._step >= self.episode_length
        return self._obs(), float(self._step), bool(done), {}

    def render(self):
        return np.full((8, 8, 3), fill_value=120, dtype=np.uint8)


class DummyPolicy:
    def __init__(self, diagnostics: list[dict[str, object]]):
        self._diagnostics = list(diagnostics)
        self._index = 0

    def reset(self):
        self._index = 0

    def act(self, obs, deterministic: bool = False):
        del obs, deterministic
        return np.zeros(2, dtype=np.float32)

    def get_gate_diagnostics(self):
        if not self._diagnostics:
            return {}
        record = self._diagnostics[min(self._index, len(self._diagnostics) - 1)]
        self._index += 1
        return dict(record)


def _full_config(enabled: bool = True, *, event_enabled: bool = False, capacity_enabled: bool = False) -> dict:
    return {
        "gate_video": {
            "enabled": enabled,
            "interval": 1000,
            "episodes": 2,
            "max_episode_steps": 4,
            "deterministic": True,
            "save_gif": True,
            "gif_fps": 10,
            "overlay_metrics": False,
            "output_subdir": "gate_videos",
        },
        "policy": {
            "type": "dreamerv3",
            "name": "thick_event_lite_dreamer",
            "dreamerv3": {
                "thick_context": {"enabled": True},
                "event_dynamics": {
                    "enabled": event_enabled,
                    "capacity_enabled": capacity_enabled,
                    "capacity_mode": "soft_topk_st",
                    "capacity_ratio": 0.25,
                    "capacity_temperature": 1.0,
                    "capacity_min_k": 1,
                    "capacity_eval_hard": True,
                    "capacity_use_valid_mask": True,
                    "event_sparsity_inside_capacity_only": True,
                    "capacity_v2_detach_event_input": True,
                    "capacity_v2_detach_event_target": True,
                    "capacity_v2_use_sigmoid_gate_multiplier": False,
                    "capacity_v2_freeze_backbone_for_event_loss": True,
                },
            },
        },
    }


def _diagnostics_sequence():
    return [
        {
            "context_gate_type": "l0_st",
            "context_gate": 0.10,
            "context_change_mask": 0.0,
            "context_change_mask_source": "context_gate",
            "context_change_mask_mode": "topk",
            "context_change_mask_top_percent": 0.10,
            "context_l0_open_prob": 0.12,
            "context_gate_hard": 0.0,
            "context_gate_soft": 0.15,
            "context_gate_logit": -0.4,
            "context_update_loss_raw": 0.12,
            "context_update_loss_scaled": 0.0009,
            "context_delta_norm": 0.20,
            "event_gate": 0.10,
            "event_logit": -2.0,
            "event_logit_sigmoid": 0.1192029,
            "event_capacity_gate": 0.0,
            "event_capacity_enabled": 1.0,
            "event_capacity_mode": "soft_topk_st",
            "event_capacity_ratio": 0.25,
            "event_capacity_k": 1.0,
            "capacity_v2_detach_event_input": 1.0,
            "capacity_v2_detach_event_target": 1.0,
            "capacity_v2_use_sigmoid_gate_multiplier": 0.0,
            "event_loss_updates_backbone": 0.0,
            "event_residual_norm": 0.20,
            "event_prediction_error": 0.10,
            "pred_next_feat_error_ordinary_only": 1.00,
            "pred_next_feat_error_event_mixed": 0.90,
            "operator_reward": 0.00,
            "operator_token_id": 0.0,
            "operator_token_novelty": 1.00,
            "operator_rarity": 1.00,
            "operator_effect_gain": 0.01,
            "operator_controllability_gain": 0.02,
            "operator_valid_effect": 1.0,
            "operator_valid_control": 1.0,
            "event_intensity": 0.10,
            "milestone_reward": 0.00,
            "milestone_trigger": 0.0,
            "running_best_intensity": 0.15,
            "habituation_factor": 1.00,
            "token_repeat_decay": 1.00,
            "operator_token_perplexity_online": 3.0,
            "operator_num_active_tokens_online": 3.0,
            "operator_token_count_min": 1.0,
            "operator_token_count_max": 3.0,
        },
        {
            "context_gate_type": "l0_st",
            "context_gate": 0.20,
            "context_change_mask": 1.0,
            "context_change_mask_source": "context_gate",
            "context_change_mask_mode": "topk",
            "context_change_mask_top_percent": 0.10,
            "context_l0_open_prob": 0.18,
            "context_gate_hard": 0.0,
            "context_gate_soft": 0.20,
            "context_gate_logit": -0.2,
            "context_update_loss_raw": 0.18,
            "context_update_loss_scaled": 0.00135,
            "context_delta_norm": 0.30,
            "event_gate": 0.20,
            "event_logit": -1.0,
            "event_logit_sigmoid": 0.2689414,
            "event_capacity_gate": 1.0,
            "event_capacity_enabled": 1.0,
            "event_capacity_mode": "soft_topk_st",
            "event_capacity_ratio": 0.25,
            "event_capacity_k": 1.0,
            "capacity_v2_detach_event_input": 1.0,
            "capacity_v2_detach_event_target": 1.0,
            "capacity_v2_use_sigmoid_gate_multiplier": 0.0,
            "event_loss_updates_backbone": 0.0,
            "event_residual_norm": 0.40,
            "event_prediction_error": 0.20,
            "pred_next_feat_error_ordinary_only": 2.00,
            "pred_next_feat_error_event_mixed": 1.50,
            "operator_reward": 0.25,
            "operator_token_id": 1.0,
            "operator_token_novelty": 0.80,
            "operator_rarity": 0.80,
            "operator_effect_gain": 0.05,
            "operator_controllability_gain": 0.04,
            "operator_valid_effect": 1.0,
            "operator_valid_control": 1.0,
            "event_intensity": 0.30,
            "milestone_reward": 0.12,
            "milestone_trigger": 1.0,
            "running_best_intensity": 0.25,
            "habituation_factor": 0.80,
            "token_repeat_decay": 1.00,
            "operator_token_perplexity_online": 3.2,
            "operator_num_active_tokens_online": 3.0,
            "operator_token_count_min": 1.0,
            "operator_token_count_max": 4.0,
        },
        {
            "context_gate_type": "l0_st",
            "context_gate": 0.30,
            "context_change_mask": 1.0,
            "context_change_mask_source": "context_gate",
            "context_change_mask_mode": "topk",
            "context_change_mask_top_percent": 0.10,
            "context_l0_open_prob": 0.36,
            "context_gate_hard": 1.0,
            "context_gate_soft": 0.60,
            "context_gate_logit": 0.6,
            "context_update_loss_raw": 0.36,
            "context_update_loss_scaled": 0.0027,
            "context_delta_norm": 0.40,
            "event_gate": 0.90,
            "event_logit": 1.2,
            "event_logit_sigmoid": 0.7685248,
            "event_capacity_gate": 1.0,
            "event_capacity_enabled": 1.0,
            "event_capacity_mode": "soft_topk_st",
            "event_capacity_ratio": 0.25,
            "event_capacity_k": 1.0,
            "capacity_v2_detach_event_input": 1.0,
            "capacity_v2_detach_event_target": 1.0,
            "capacity_v2_use_sigmoid_gate_multiplier": 0.0,
            "event_loss_updates_backbone": 0.0,
            "event_residual_norm": 0.90,
            "event_prediction_error": 0.90,
            "pred_next_feat_error_ordinary_only": 3.00,
            "pred_next_feat_error_event_mixed": 2.00,
            "operator_reward": 0.55,
            "operator_token_id": 2.0,
            "operator_token_novelty": 0.60,
            "operator_rarity": 0.60,
            "operator_effect_gain": 0.20,
            "operator_controllability_gain": 0.10,
            "operator_valid_effect": 1.0,
            "operator_valid_control": 1.0,
            "event_intensity": 0.90,
            "milestone_reward": 0.40,
            "milestone_trigger": 1.0,
            "running_best_intensity": 0.60,
            "habituation_factor": 0.65,
            "token_repeat_decay": 0.50,
            "operator_token_perplexity_online": 3.4,
            "operator_num_active_tokens_online": 4.0,
            "operator_token_count_min": 1.0,
            "operator_token_count_max": 5.0,
        },
        {
            "context_gate_type": "l0_st",
            "context_gate": 0.40,
            "context_change_mask": 0.0,
            "context_change_mask_source": "context_gate",
            "context_change_mask_mode": "topk",
            "context_change_mask_top_percent": 0.10,
            "context_l0_open_prob": 0.28,
            "context_gate_hard": 1.0,
            "context_gate_soft": 0.55,
            "context_gate_logit": 0.2,
            "context_update_loss_raw": 0.28,
            "context_update_loss_scaled": 0.0021,
            "context_delta_norm": 0.50,
            "event_gate": 0.10,
            "event_logit": -1.5,
            "event_logit_sigmoid": 0.1824255,
            "event_capacity_gate": 0.0,
            "event_capacity_enabled": 1.0,
            "event_capacity_mode": "soft_topk_st",
            "event_capacity_ratio": 0.25,
            "event_capacity_k": 1.0,
            "capacity_v2_detach_event_input": 1.0,
            "capacity_v2_detach_event_target": 1.0,
            "capacity_v2_use_sigmoid_gate_multiplier": 0.0,
            "event_loss_updates_backbone": 0.0,
            "event_residual_norm": 0.10,
            "event_prediction_error": 0.30,
            "pred_next_feat_error_ordinary_only": 4.00,
            "pred_next_feat_error_event_mixed": 3.50,
            "operator_reward": 0.05,
            "operator_token_id": 2.0,
            "operator_token_novelty": 0.45,
            "operator_rarity": 0.45,
            "operator_effect_gain": 0.02,
            "operator_controllability_gain": 0.01,
            "operator_valid_effect": 1.0,
            "operator_valid_control": 1.0,
            "event_intensity": 0.20,
            "milestone_reward": 0.00,
            "milestone_trigger": 0.0,
            "running_best_intensity": 0.60,
            "habituation_factor": 0.45,
            "token_repeat_decay": 0.25,
            "operator_token_perplexity_online": 3.1,
            "operator_num_active_tokens_online": 3.0,
            "operator_token_count_min": 1.0,
            "operator_token_count_max": 6.0,
        },
    ]


def test_gate_video_disabled_does_not_create_directory(tmp_path: Path):
    policy = DummyPolicy(_diagnostics_sequence())
    env = DummyEnv()
    config = _full_config(enabled=False)
    output_dir = tmp_path / "outputs"
    result = maybe_record_gate_eval_videos(policy, env, output_dir, global_step=1000, config=config, device="cpu")
    assert result is None
    assert not output_dir.exists()


def test_gate_video_enabled_creates_step_directory_and_summary(tmp_path: Path):
    policy = DummyPolicy(_diagnostics_sequence())
    env = DummyEnv()
    config = _full_config(enabled=True)
    output_dir = tmp_path / "outputs"
    result = maybe_record_gate_eval_videos(policy, env, output_dir, global_step=1000, config=config, device="cpu")
    assert result is not None
    step_dir = output_dir / "step_001000"
    assert step_dir.exists()
    assert (step_dir / "gate_eval_summary.json").exists()
    assert (step_dir / "gate_eval_frames.jsonl").exists()
    assert (step_dir / "episode_000.gif").exists()
    assert (step_dir / "episode_000_gate_curves.png").exists()


def test_gate_video_enabled_without_event_gate_still_records_context_only(tmp_path: Path):
    policy = DummyPolicy(
        [
            {"context_gate": 0.1, "context_delta_norm": 0.2},
            {"context_gate": 0.2, "context_delta_norm": 0.3},
        ]
    )
    env = DummyEnv(episode_length=2, cameras=1)
    config = _full_config(enabled=True, event_enabled=False)
    output_dir = tmp_path / "outputs"
    result = maybe_record_gate_eval_videos(policy, env, output_dir, global_step=1000, config=config, device="cpu")
    assert result is not None
    summary = result["summary"]
    assert summary["mean_context_gate"] != "NA"
    assert summary["mean_event_gate"] == "NA"


def test_summary_contains_required_metrics_and_checks(tmp_path: Path):
    policy = DummyPolicy(_diagnostics_sequence())
    env = DummyEnv()
    config = _full_config(enabled=True, event_enabled=True, capacity_enabled=True)
    result = record_gate_eval_videos(policy, env, tmp_path, global_step=2000, config=config, device="cpu")
    summary = json.loads(result["summary_path"].read_text(encoding="utf-8"))
    required = {
        "context_gate_type",
        "context_change_mask_mean",
        "context_change_mask_nonzero_ratio",
        "context_change_mask_source",
        "context_change_mask_mode",
        "context_change_mask_top_percent",
        "event_capacity_enabled",
        "event_capacity_mode",
        "event_capacity_ratio",
        "capacity_v2_detach_event_input",
        "capacity_v2_detach_event_target",
        "capacity_v2_use_sigmoid_gate_multiplier",
        "event_loss_updates_backbone",
        "event_capacity_k_mean",
        "event_capacity_gate_mean",
        "event_capacity_gate_std",
        "event_capacity_gate_min",
        "event_capacity_gate_max",
        "event_capacity_nonzero_ratio",
        "context_l0_open_prob",
        "context_gate_hard_mean",
        "context_gate_soft_mean",
        "context_gate_logit_mean",
        "context_gate_logit_std",
        "context_update_loss_raw",
        "context_update_loss_scaled",
        "operator_intrinsic_reward_mean",
        "operator_intrinsic_reward_std",
        "operator_intrinsic_reward_max",
        "operator_intrinsic_reward_nonzero_ratio",
        "operator_token_novelty_mean",
        "operator_rarity_mean",
        "operator_effect_gain_mean",
        "operator_controllability_gain_mean",
        "operator_valid_effect_ratio",
        "operator_valid_control_ratio",
        "event_intensity_mean",
        "event_intensity_std",
        "event_intensity_max",
        "milestone_reward_mean",
        "milestone_trigger_count",
        "running_best_intensity",
        "habituation_factor_mean",
        "token_repeat_decay_mean",
        "milestone_nonzero_ratio",
        "operator_token_perplexity_online",
        "operator_num_active_tokens_online",
        "operator_token_count_min",
        "operator_token_count_max",
        "mean_context_gate",
        "mean_event_gate",
        "event_gate_mean",
        "event_gate_std",
        "event_logit_mean",
        "event_logit_std",
        "event_logit_sigmoid_mean",
        "event_logit_top_mean",
        "event_logit_normal_mean",
        "event_logit_sigmoid_capacity_mean",
        "event_logit_sigmoid_outside_capacity_mean",
        "event_gate_threshold_dynamic",
        "event_gate_dynamic_peak_ratio",
        "event_gate_top10_peak_ratio",
        "event_gate_capacity_mean",
        "event_gate_outside_capacity_mean",
        "event_gate_masked_mean",
        "event_gate_unmasked_mean",
        "ordinary_error_all_mean",
        "mixed_error_all_mean",
        "ordinary_error_capacity_mean",
        "mixed_error_capacity_mean",
        "relative_prediction_improvement_capacity",
        "ordinary_error_masked_mean",
        "mixed_error_masked_mean",
        "relative_prediction_improvement_all",
        "relative_prediction_improvement_masked",
        "event_residual_norm_capacity_mean",
        "event_residual_norm_outside_capacity_mean",
        "event_residual_norm_masked_mean",
        "event_residual_norm_unmasked_mean",
        "event_mixed_error_mean",
        "ordinary_only_error_mean",
        "relative_prediction_improvement",
        "event_gate_at_top_error_mean",
        "event_gate_at_normal_steps_mean",
        "event_intensity_top_mean",
        "event_intensity_normal_mean",
        "checks",
    }
    assert required.issubset(summary.keys())
    assert {"gate_finite", "sparse_peak_pattern", "dynamic_peak_pattern", "top10_peak_pattern"}.issubset(
        summary["checks"].keys()
    )
    assert summary["context_gate_type"] == "l0_st"
    assert summary["context_change_mask_source"] == "context_gate"
    assert summary["context_change_mask_mode"] == "topk"
    assert summary["event_capacity_mode"] == "soft_topk_st"
    assert summary["capacity_v2_detach_event_input"] is True
    assert summary["capacity_v2_detach_event_target"] is True
    assert summary["capacity_v2_use_sigmoid_gate_multiplier"] is False
    assert summary["event_loss_updates_backbone"] is False


def test_relative_prediction_improvement_is_computed_correctly(tmp_path: Path):
    policy = DummyPolicy(_diagnostics_sequence())
    env = DummyEnv(episode_length=4, cameras=1)
    config = _full_config(enabled=True, event_enabled=True)
    result = record_gate_eval_videos(policy, env, tmp_path, global_step=3000, config=config, device="cpu")
    summary = result["summary"]
    ordinary = float(summary["ordinary_only_error_mean"])
    mixed = float(summary["event_mixed_error_mean"])
    expected = (ordinary - mixed) / max(ordinary, 1e-8)
    assert abs(float(summary["relative_prediction_improvement"]) - expected) < 1e-8


def test_event_gate_at_top_error_mean_is_computed_correctly(tmp_path: Path):
    policy = DummyPolicy(_diagnostics_sequence())
    env = DummyEnv(episode_length=4, cameras=1)
    config = _full_config(enabled=True, event_enabled=True)
    result = record_gate_eval_videos(policy, env, tmp_path, global_step=4000, config=config, device="cpu")
    summary = result["summary"]
    assert float(summary["event_gate_at_top_error_mean"]) == 0.9
    assert float(summary["event_gate_at_normal_steps_mean"]) < 0.5


def test_dynamic_and_top10_peak_ratios_are_computed(tmp_path: Path):
    policy = DummyPolicy(_diagnostics_sequence())
    env = DummyEnv(episode_length=4, cameras=1)
    config = _full_config(enabled=True, event_enabled=True)
    result = record_gate_eval_videos(policy, env, tmp_path, global_step=4500, config=config, device="cpu")
    summary = result["summary"]
    assert abs(float(summary["event_gate_threshold_dynamic"]) - (float(summary["event_gate_mean"]) + float(summary["event_gate_std"]))) < 1e-8
    assert float(summary["event_gate_dynamic_peak_ratio"]) > 0.0
    assert abs(float(summary["event_gate_top10_peak_ratio"]) - 0.25) < 1e-8
    assert summary["checks"]["dynamic_peak_pattern"] is True
    assert summary["checks"]["top10_peak_pattern"] is False


def test_milestone_metrics_are_computed(tmp_path: Path):
    policy = DummyPolicy(_diagnostics_sequence())
    env = DummyEnv(episode_length=4, cameras=1)
    config = _full_config(enabled=True, event_enabled=True, capacity_enabled=True)
    result = record_gate_eval_videos(policy, env, tmp_path, global_step=4550, config=config, device="cpu")
    summary = result["summary"]
    assert float(summary["milestone_reward_mean"]) > 0.0
    assert float(summary["milestone_trigger_count"]) == 4.0
    assert float(summary["milestone_nonzero_ratio"]) == 0.5
    assert float(summary["running_best_intensity"]) > 0.0
    assert float(summary["habituation_factor_mean"]) < 1.0
    assert float(summary["token_repeat_decay_mean"]) < 1.0
    assert float(summary["event_intensity_top_mean"]) > float(summary["event_intensity_normal_mean"])


def test_masked_event_metrics_are_computed(tmp_path: Path):
    policy = DummyPolicy(_diagnostics_sequence())
    env = DummyEnv(episode_length=4, cameras=1)
    config = _full_config(enabled=True, event_enabled=True)
    result = record_gate_eval_videos(policy, env, tmp_path, global_step=4600, config=config, device="cpu")
    summary = result["summary"]
    assert float(summary["context_change_mask_mean"]) == 0.5
    assert float(summary["context_change_mask_nonzero_ratio"]) == 0.5
    assert float(summary["event_gate_masked_mean"]) > float(summary["event_gate_unmasked_mean"])
    assert float(summary["mixed_error_masked_mean"]) < float(summary["ordinary_error_masked_mean"])
    assert float(summary["relative_prediction_improvement_masked"]) > 0.0
    assert float(summary["event_residual_norm_masked_mean"]) > float(summary["event_residual_norm_unmasked_mean"])


def test_capacity_metrics_are_computed(tmp_path: Path):
    policy = DummyPolicy(_diagnostics_sequence())
    env = DummyEnv(episode_length=4, cameras=1)
    config = _full_config(enabled=True, event_enabled=True, capacity_enabled=True)
    result = record_gate_eval_videos(policy, env, tmp_path, global_step=4700, config=config, device="cpu")
    summary = result["summary"]
    assert summary["event_capacity_enabled"] == 1.0
    assert summary["event_capacity_mode"] == "soft_topk_st"
    assert float(summary["event_capacity_nonzero_ratio"]) > 0.0
    assert float(summary["event_gate_capacity_mean"]) > float(summary["event_gate_outside_capacity_mean"])
    assert float(summary["event_logit_sigmoid_capacity_mean"]) > float(summary["event_logit_sigmoid_outside_capacity_mean"])
    assert float(summary["mixed_error_capacity_mean"]) < float(summary["ordinary_error_capacity_mean"])
    assert float(summary["relative_prediction_improvement_capacity"]) > 0.0
    assert float(summary["event_logit_top_mean"]) > float(summary["event_logit_normal_mean"])
    assert float(summary["event_residual_norm_capacity_mean"]) > float(summary["event_residual_norm_outside_capacity_mean"])


def test_missing_gate_metrics_do_not_crash_and_write_na(tmp_path: Path):
    policy = DummyPolicy(
        [
            {"context_gate": 0.1, "context_delta_norm": 0.2},
            {"context_gate": 0.2, "context_delta_norm": 0.3},
        ]
    )
    env = DummyEnv(episode_length=2, cameras=1)
    config = _full_config(enabled=True)
    result = record_gate_eval_videos(policy, env, tmp_path, global_step=5000, config=config, device="cpu")
    summary = result["summary"]
    assert summary["mean_event_gate"] == "NA"
    assert summary["event_prediction_error_mean"] == "NA"
    assert summary["context_l0_open_prob"] == "NA"
    assert summary["context_change_mask_mean"] == "NA"
    lines = (result["step_dir"] / "gate_eval_frames.jsonl").read_text(encoding="utf-8").strip().splitlines()
    first = json.loads(lines[0])
    assert first["event_gate"] == "NA"


def test_help_runs_for_gate_video_module():
    result = subprocess.run(
        [sys.executable, "-m", "policy.dreamerv3.gate_video_recorder", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()
