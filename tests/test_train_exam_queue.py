from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

from scripts.train_exam_queue import (
    build_exam_entry,
    build_initial_state,
    discover_pending_exams,
    exam_has_started_training,
    exam_progress_snapshot,
    render_status,
    stop_queue,
    write_json,
)


def _write_exam(exam_dir: Path, config: dict):
    exam_dir.mkdir(parents=True, exist_ok=True)
    (exam_dir / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    (exam_dir / "train.py").write_text("class Dummy:\n    pass\n", encoding="utf-8")


def test_exam_has_started_training_detects_output_markers(tmp_path: Path):
    exam_dir = tmp_path / "demo_exam"
    exam_dir.mkdir()
    output_dir = exam_dir / "output"
    output_dir.mkdir()
    assert exam_has_started_training(exam_dir) is False
    (output_dir / "train.log").write_text("started\n", encoding="utf-8")
    assert exam_has_started_training(exam_dir) is True


def test_discover_pending_exams_filters_started(tmp_path: Path, monkeypatch):
    exam_root = tmp_path / "exam"
    pending = exam_root / "pending_exam"
    started = exam_root / "started_exam"
    doc = exam_root / "doc"
    doc.mkdir(parents=True)

    config = {
        "train": {"total_steps": 10},
        "env": {"type": "dmcontrol", "name": "cartpole", "domain_name": "cartpole", "task_name": "swingup"},
        "policy": {"type": "random", "checkpoint": {"save": False}},
    }
    _write_exam(pending, config)
    _write_exam(started, config)
    (started / "output").mkdir()
    (started / "output" / "run_manifest.json").write_text("{}", encoding="utf-8")

    found = discover_pending_exams(exam_root)
    assert [path.name for path in found] == ["pending_exam"]


def test_exam_progress_snapshot_reads_online_policy_metrics(tmp_path: Path, monkeypatch):
    project_root = tmp_path
    exam_root = project_root / "exam"
    exam_dir = exam_root / "demo_exam"
    exam_dir.mkdir(parents=True)
    config = """
train:
  total_steps: 100
  log_interval: 10
  record_interval: 10
  save_interval: 0
  eval_interval: 0
  eval_episodes: 0
  multi_task: false
env:
  type: dmcontrol
  name: cartpole_swingup
  domain_name: cartpole
  task_name: swingup
policy:
  type: random
  checkpoint:
    save: false
"""
    (exam_dir / "config.yaml").write_text(config, encoding="utf-8")
    (exam_dir / "train.py").write_text("from scripts.train import BaseExamTrainApp\n", encoding="utf-8")
    output_dir = exam_dir / "output"
    output_dir.mkdir()
    (output_dir / "run_manifest.json").write_text('{"trainer": "online_policy"}\n', encoding="utf-8")
    (output_dir / "metrics.csv").write_text(
        "step,episode_index,episode_return,episode_length,fps\n"
        "10,0,1.0,10,5.0\n"
        "40,1,2.0,20,4.0\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("scripts.train_exam_queue.EXAM_ROOT", exam_root)
    entry = build_exam_entry(exam_dir)
    progress = exam_progress_snapshot(entry)
    assert progress["current_units"] == 40
    assert progress["total_units"] == 100
    assert progress["eta_seconds"] == 15.0


def test_render_status_includes_queue_eta(tmp_path: Path, monkeypatch):
    project_root = tmp_path
    exam_root = project_root / "exam"
    running_exam = exam_root / "running_exam"
    queued_exam = exam_root / "queued_exam"

    common_config = """
train:
  total_steps: 100
  log_interval: 10
  record_interval: 10
  save_interval: 0
  eval_interval: 0
  eval_episodes: 0
  multi_task: false
env:
  type: dmcontrol
  name: cartpole_swingup
  domain_name: cartpole
  task_name: swingup
policy:
  type: random
  checkpoint:
    save: false
"""
    for exam_dir in (running_exam, queued_exam):
        exam_dir.mkdir(parents=True)
        (exam_dir / "config.yaml").write_text(common_config, encoding="utf-8")
        (exam_dir / "train.py").write_text("from scripts.train import BaseExamTrainApp\n", encoding="utf-8")
    output_dir = running_exam / "output"
    output_dir.mkdir()
    (output_dir / "run_manifest.json").write_text('{"trainer": "online_policy"}\n', encoding="utf-8")
    (output_dir / "metrics.csv").write_text(
        "step,episode_index,episode_return,episode_length,fps\n"
        "50,0,1.0,10,10.0\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("scripts.train_exam_queue.EXAM_ROOT", exam_root)
    state = build_initial_state([running_exam, queued_exam])
    state["status"] = "running"
    state["worker_pid"] = 999999
    state["exams"][0]["status"] = "running"
    state["exams"][0]["started_at"] = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
    text = render_status(state)
    assert "queue_eta=" in text
    assert "running_exam" in text
    assert "queued_exam" in text


def test_stop_queue_marks_stale_running_worker_as_stopped(tmp_path: Path):
    state_path = tmp_path / "state.json"
    state = {
        "status": "running",
        "worker_pid": 424242,
        "current_index": 0,
        "exams": [
            {
                "exam_name": "demo_exam",
                "exam_dir": str(tmp_path / "demo_exam"),
                "status": "running",
                "started_at": None,
                "finished_at": None,
                "returncode": None,
                "duration_seconds": None,
            }
        ],
    }
    write_json(state_path, state)
    assert stop_queue(state_path) == 0
    updated = json.loads(state_path.read_text(encoding="utf-8"))
    assert updated["status"] == "stopped"
    assert updated["exams"][0]["status"] == "stopped"


def test_stop_queue_signals_worker_group(tmp_path: Path, monkeypatch):
    state_path = tmp_path / "state.json"
    state = {
        "status": "running",
        "worker_pid": 12345,
        "current_index": 0,
        "exams": [
            {
                "exam_name": "demo_exam",
                "exam_dir": str(tmp_path / "demo_exam"),
                "status": "running",
                "started_at": None,
                "finished_at": None,
                "returncode": None,
                "duration_seconds": None,
            }
        ],
    }
    write_json(state_path, state)
    signals = []

    monkeypatch.setattr("scripts.train_exam_queue.is_process_alive", lambda pid: True)
    monkeypatch.setattr("scripts.train_exam_queue.os.killpg", lambda pid, sig: signals.append((pid, sig)))

    assert stop_queue(state_path) == 0
    updated = json.loads(state_path.read_text(encoding="utf-8"))
    assert updated["status"] == "stopping"
    assert signals and signals[0][0] == 12345
