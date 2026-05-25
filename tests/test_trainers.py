from pathlib import Path

import pytest

from scripts.train import BaseExamTrainApp, load_exam_train_app, render_exam_train_template
from scripts.utils.exam import trainer_name, validate_exam_config
from src.bidreamer.bidirectional_env_factory import _direction_env_config
from src.trainers.factory import build_trainer
from src.trainers.online_policy import OnlinePolicyTrainer


def _standard_config() -> dict:
    return {
        "train": {
            "seed": 0,
            "total_steps": 10,
            "log_interval": 1,
            "record_interval": 1,
            "save_interval": 0,
            "eval_interval": 0,
            "eval_episodes": 0,
            "multi_task": False,
        },
        "env": {
            "type": "dmcontrol",
            "name": "cartpole_swingup",
            "domain_name": "cartpole",
            "task_name": "swingup",
        },
        "policy": {
            "type": "random",
            "checkpoint": {
                "save": False,
            },
        },
    }


def _bidirectional_config() -> dict:
    return {
        "train": {
            "seed": 0,
            "trainer": "bidirectional",
            "bidirectional": {
                "enabled": True,
                "forward_env": {
                    "name": "ball_in_cup_catch",
                    "domain_name": "ball_in_cup",
                    "task_name": "catch",
                },
                "reverse_env": {
                    "name": "ball_in_cup_release",
                    "domain_name": "ball_in_cup",
                    "task_name": "release",
                    "render": {
                        "enabled": True,
                    },
                },
            },
        },
        "env": {
            "type": "dmcontrol",
            "observation": {
                "type": "image",
                "num_cams": 1,
            },
            "action": {
                "clip": True,
                "normalize": True,
            },
            "render": {
                "enabled": False,
                "height": 64,
                "width": 64,
                "camera_id": 0,
                "backend_priority": ["egl", "osmesa"],
                "allow_software_render_fallback": True,
            },
        },
        "replay": {
            "capacity": 128,
        },
        "model": {
            "action_dim": 2,
            "embed_dim": 32,
            "deter_dim": 32,
            "stoch_dim": 8,
            "stoch_classes": 8,
            "hidden_dim": 64,
            "free_nats": 1.0,
            "kl_balance": 0.8,
            "unimix": 0.01,
        },
        "training": {
            "total_env_steps": 10,
            "warmup_env_steps_per_direction": 1,
            "batch_size": 2,
            "seq_len": 2,
            "eval_interval": 5,
            "learning_rate": 3.0e-4,
            "grad_clip": 100.0,
            "device": "cpu",
        },
        "loss": {
            "recon_scale": 1.0,
            "reward_scale_forward": 1.0,
            "reward_scale_reverse": 1.0,
            "continue_scale": 1.0,
            "kl_scale": 1.0,
            "contact_scale": 0.0,
            "grasp_scale": 0.0,
            "forward_loss_scale": 1.0,
            "reverse_loss_scale": 1.0,
        },
    }


def _ref_dreamerv3_config() -> dict:
    config = _standard_config()
    config["train"]["trainer"] = "ref_dreamerv3"
    config["train"]["total_steps"] = 10
    config["train"]["max_episode_steps"] = 100
    config["policy"] = {
        "type": "ref_dreamerv3",
        "ref_dreamerv3": {
            "configs": ["defaults", "debug"],
            "overrides": {
                "jax": {
                    "platform": "cpu",
                    "prealloc": False,
                },
                "run": {
                    "envs": 1,
                    "debug": True,
                },
            },
        },
    }
    config["wrappers"] = [
        {
            "type": "simer_to_embodied",
            "version": "v1",
            "obs_key": "observation",
        }
    ]
    return config


def test_standard_config_uses_online_policy_trainer():
    config = _standard_config()
    validate_exam_config(config)
    trainer = build_trainer(config=config, exam_dir=Path("/tmp/exam"), exam_name="standard_exam")
    assert trainer_name(config) == "online_policy"
    assert trainer.__class__.__name__ == "OnlinePolicyTrainer"


def test_bidirectional_config_uses_bidirectional_trainer():
    config = _bidirectional_config()
    validate_exam_config(config)
    trainer = build_trainer(config=config, exam_dir=Path("/tmp/exam"), exam_name="bidirectional_exam")
    assert trainer_name(config) == "bidirectional"
    assert trainer.__class__.__name__ == "BidirectionalTrainer"


def test_ref_dreamerv3_config_uses_ref_trainer():
    config = _ref_dreamerv3_config()
    validate_exam_config(config)
    trainer = build_trainer(config=config, exam_dir=Path("/tmp/exam"), exam_name="ref_dreamerv3_exam")
    assert trainer_name(config) == "ref_dreamerv3"
    assert trainer.__class__.__name__ == "RefDreamerV3Trainer"


def test_bidirectional_env_resolution_reads_new_train_block():
    config = _bidirectional_config()
    reverse_env = _direction_env_config(config, "reverse")
    assert reverse_env["task_name"] == "release"
    assert reverse_env["name"] == "ball_in_cup_release"
    assert reverse_env["render"]["enabled"] is True
    assert reverse_env["render"]["width"] == 64


def test_bidirectional_validation_requires_reverse_env():
    config = _bidirectional_config()
    del config["train"]["bidirectional"]["reverse_env"]
    with pytest.raises(ValueError, match="reverse_env"):
        validate_exam_config(config)


def test_exam_template_inherits_main_train_entry():
    source = render_exam_train_template()
    assert "from scripts.train import BaseExamTrainApp" in source
    assert "class ExamTrain(BaseExamTrainApp):" in source


def test_existing_exam_train_entry_loads_from_main_loader():
    app = load_exam_train_app("test_dmcontrol_random")
    assert isinstance(app, BaseExamTrainApp)
    assert app.exam_name == "test_dmcontrol_random"
    trainer = app.build_trainer(app.load_config())
    assert trainer.__class__.__name__ == "OnlinePolicyTrainer"


def test_gate_video_output_stays_inside_exam_output_dir():
    config = _standard_config()
    config["gate_video"] = {"output_subdir": "gate_videos"}
    exam_dir = Path("/tmp/example_exam")
    trainer = OnlinePolicyTrainer(config=config, exam_dir=exam_dir, exam_name="example_exam")

    gate_dir = trainer._gate_video_output_root(exam_dir / "output")

    assert gate_dir == exam_dir / "output" / "gate_videos"
