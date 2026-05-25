from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from scripts.utils import load_config, multi_task_enabled, output_dir_for_task, resolve_env_tasks, train_cfg, trainer_name


EXAM_ROOT = PROJECT_ROOT / "exam"
QUEUE_ROOT = EXAM_ROOT / ".train_queue"
STATE_PATH = QUEUE_ROOT / "state.json"
WORKER_LOG_PATH = QUEUE_ROOT / "worker.log"

ONLINE_POLICY_TRAINERS = {"online_policy", "default", "single_policy"}
BIDIRECTIONAL_TRAINERS = {"bidirectional", "bidirectional_online", "bidreamer"}
REF_DREAMERV3_TRAINERS = {"ref_dreamerv3", "dreamerv3_ref"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_iso8601(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp_path.replace(path)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_last_non_empty_line(path: Path) -> str | None:
    if not path.exists():
        return None
    last = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                last = stripped
    return last


def is_process_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "unknown"
    seconds = max(int(round(seconds)), 0)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    return f"{minutes:d}m{secs:02d}s"


def _load_exam_runtime_config(exam_name: str) -> dict:
    config, _ = load_config(EXAM_ROOT, exam_name)
    return config


def exam_training_markers(exam_dir: Path) -> list[Path]:
    output_dir = exam_dir / "output"
    return [
        output_dir / "run_manifest.json",
        output_dir / "metrics.csv",
        output_dir / "policy_metrics.jsonl",
        output_dir / "train_metrics.json",
        output_dir / "train.log",
    ]


def exam_has_started_training(exam_dir: Path) -> bool:
    output_dir = exam_dir / "output"
    for marker in exam_training_markers(exam_dir):
        if marker.exists():
            return True
    if output_dir.exists():
        for child in output_dir.iterdir():
            if child.name.startswith("."):
                continue
            return True
    return False


def discover_pending_exams(exam_root: Path = EXAM_ROOT) -> list[Path]:
    candidates = []
    for exam_dir in sorted(exam_root.iterdir()):
        if not exam_dir.is_dir() or exam_dir.name.startswith(".") or exam_dir.name == "doc":
            continue
        if not (exam_dir / "config.yaml").exists():
            continue
        if not (exam_dir / "train.py").exists():
            continue
        if exam_has_started_training(exam_dir):
            continue
        config_mtime = (exam_dir / "config.yaml").stat().st_mtime
        candidates.append((config_mtime, exam_dir.name, exam_dir))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in candidates]


def exam_total_units(config: dict) -> tuple[str, int]:
    name = str(trainer_name(config)).strip().lower()
    if name in ONLINE_POLICY_TRAINERS:
        return "step", int(train_cfg(config).get("total_steps", 0))
    if name in BIDIRECTIONAL_TRAINERS:
        training_cfg = config.get("training", {})
        return "env_step", int(training_cfg.get("total_env_steps", 0))
    if name in REF_DREAMERV3_TRAINERS:
        return "step", int(train_cfg(config).get("total_steps", 0))
    return "unit", 0


def build_exam_entry(exam_dir: Path) -> dict:
    config = _load_exam_runtime_config(exam_dir.name)
    unit_name, total_units = exam_total_units(config)
    return {
        "exam_name": exam_dir.name,
        "exam_dir": str(exam_dir.resolve()),
        "trainer": trainer_name(config),
        "unit_name": unit_name,
        "total_units": total_units,
        "status": "queued",
        "started_at": None,
        "finished_at": None,
        "returncode": None,
        "duration_seconds": None,
        "log_path": str((exam_dir / "output" / "train.log").resolve()),
    }


def build_initial_state(exam_dirs: list[Path]) -> dict:
    created_at = utc_now_iso()
    exams = [build_exam_entry(exam_dir) for exam_dir in exam_dirs]
    return {
        "schema": "exam_train_queue/v1",
        "created_at": created_at,
        "started_at": None,
        "updated_at": created_at,
        "finished_at": None,
        "status": "queued",
        "worker_pid": None,
        "worker_log_path": str(WORKER_LOG_PATH.resolve()),
        "current_index": None,
        "snapshot_count": len(exams),
        "cold_loaded": True,
        "continue_on_error": False,
        "exams": exams,
    }


def load_state(state_path: Path = STATE_PATH) -> dict:
    if not state_path.exists():
        raise FileNotFoundError(f"Queue state not found: {state_path}")
    return read_json(state_path)


def save_state(state: dict, state_path: Path = STATE_PATH):
    state["updated_at"] = utc_now_iso()
    write_json(state_path, state)


def _queue_running(state: dict) -> bool:
    return state.get("status") == "running" and is_process_alive(state.get("worker_pid"))


def ensure_startable(state_path: Path = STATE_PATH):
    if not state_path.exists():
        return
    state = load_state(state_path)
    if _queue_running(state):
        raise RuntimeError(
            f"Queue already running with pid={state.get('worker_pid')}. "
            f"Use `python {Path(__file__).name} status` to inspect it."
        )


def _state_summary_counts(state: dict) -> tuple[int, int, int]:
    completed = sum(1 for exam in state["exams"] if exam["status"] == "completed")
    failed = sum(1 for exam in state["exams"] if exam["status"] == "failed")
    stopped = sum(1 for exam in state["exams"] if exam["status"] == "stopped")
    queued = sum(1 for exam in state["exams"] if exam["status"] == "queued")
    return completed, failed, stopped, queued


def _online_task_progress(task_output_dir: Path, total_steps: int) -> dict:
    metrics_path = task_output_dir / "metrics.csv"
    step = 0
    fps = None
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as handle:
            header = None
            last_values = None
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                if header is None:
                    header = stripped.split(",")
                    continue
                values = stripped.split(",")
                if len(values) == len(header):
                    last_values = dict(zip(header, values))
            if last_values:
                step = int(float(last_values.get("step", 0) or 0))
                fps_value = last_values.get("fps")
                fps = float(fps_value) if fps_value not in (None, "") else None
    step = min(max(step, 0), total_steps)
    eta_seconds = None
    if fps and fps > 0:
        eta_seconds = max(total_steps - step, 0) / fps
    return {
        "current_units": step,
        "total_units": total_steps,
        "eta_seconds": eta_seconds,
        "rate_units_per_second": fps,
        "complete": step >= total_steps,
        "started": metrics_path.exists() or (task_output_dir / "run_manifest.json").exists(),
    }


def _bidirectional_progress(output_dir: Path, total_steps: int, started_at: str | None) -> dict:
    metrics_path = output_dir / "train_metrics.json"
    step = 0
    started = metrics_path.exists() or (output_dir / "run_manifest.json").exists()
    if metrics_path.exists():
        rows = json.loads(metrics_path.read_text(encoding="utf-8"))
        if rows:
            step = int(rows[-1].get("env_step", 0))
    step = min(max(step, 0), total_steps)
    eta_seconds = None
    rate = None
    started_dt = parse_iso8601(started_at)
    if started_dt and step > 0:
        elapsed = max((datetime.now(timezone.utc) - started_dt).total_seconds(), 1e-6)
        rate = step / elapsed
        if rate > 0:
            eta_seconds = max(total_steps - step, 0) / rate
    return {
        "current_units": step,
        "total_units": total_steps,
        "eta_seconds": eta_seconds,
        "rate_units_per_second": rate,
        "complete": step >= total_steps,
        "started": started,
    }


def exam_progress_snapshot(exam_entry: dict) -> dict:
    exam_name = exam_entry["exam_name"]
    exam_dir = Path(exam_entry["exam_dir"])
    config = _load_exam_runtime_config(exam_name)
    trainer = str(trainer_name(config)).strip().lower()

    if trainer in ONLINE_POLICY_TRAINERS:
        tasks = resolve_env_tasks(config)
        total_per_task = int(train_cfg(config).get("total_steps", 0))
        task_rows = []
        current_units = 0
        total_units = total_per_task * max(len(tasks), 1)
        current_eta = None
        current_rate = None
        active_task_name = None
        for task in tasks:
            task_output_dir = output_dir_for_task(exam_dir, config, task["name"])
            task_progress = _online_task_progress(task_output_dir, total_per_task)
            task_rows.append({"task_name": task["name"], **task_progress})
            current_units += task_progress["current_units"]
            if active_task_name is None and task_progress["started"] and not task_progress["complete"]:
                active_task_name = task["name"]
                current_eta = task_progress["eta_seconds"]
                current_rate = task_progress["rate_units_per_second"]
        return {
            "current_units": current_units,
            "total_units": total_units,
            "eta_seconds": current_eta,
            "rate_units_per_second": current_rate,
            "active_task_name": active_task_name,
            "task_rows": task_rows,
        }

    if trainer in BIDIRECTIONAL_TRAINERS:
        unit_name, total_units = exam_total_units(config)
        output_dir = exam_dir / "output"
        progress = _bidirectional_progress(output_dir, total_units, exam_entry.get("started_at"))
        progress["unit_name"] = unit_name
        return progress

    return {
        "current_units": 0,
        "total_units": int(exam_entry.get("total_units", 0) or 0),
        "eta_seconds": None,
        "rate_units_per_second": None,
        "active_task_name": None,
        "task_rows": [],
    }


def _render_exam_line(index: int, exam: dict, progress: dict) -> str:
    status = exam["status"]
    current_units = int(progress.get("current_units", 0) or 0)
    total_units = int(progress.get("total_units", exam.get("total_units", 0)) or 0)
    pct = 0.0 if total_units <= 0 else 100.0 * current_units / total_units
    eta_text = format_duration(progress.get("eta_seconds"))
    extras = []
    if progress.get("active_task_name"):
        extras.append(f"task={progress['active_task_name']}")
    if exam.get("returncode") not in (None, 0):
        extras.append(f"rc={exam['returncode']}")
    if exam.get("duration_seconds"):
        extras.append(f"dur={format_duration(exam['duration_seconds'])}")
    suffix = f" [{' '.join(extras)}]" if extras else ""
    return f"{index:02d}. {status:<9} {exam['exam_name']:<48} {current_units:>7}/{total_units:<7} {pct:6.2f}% eta={eta_text}{suffix}"


def render_status(state: dict) -> str:
    completed, failed, stopped, queued = _state_summary_counts(state)
    running = _queue_running(state)
    lines = [
        f"queue_status={state['status']}",
        f"worker_pid={state.get('worker_pid')} alive={running}",
        f"snapshot_count={state.get('snapshot_count', len(state.get('exams', [])))} completed={completed} failed={failed} stopped={stopped} queued={queued}",
        f"created_at={state.get('created_at')}",
        f"updated_at={state.get('updated_at')}",
    ]

    progress_rows = []
    current_eta = None
    completed_rates = []
    running_rate = None
    remaining_units = 0
    for exam in state["exams"]:
        progress = exam_progress_snapshot(exam)
        progress_rows.append((exam, progress))
        if exam["status"] == "completed" and exam.get("duration_seconds") and progress.get("total_units"):
            duration = float(exam["duration_seconds"])
            total_units = float(progress["total_units"])
            if duration > 0 and total_units > 0:
                completed_rates.append(duration / total_units)
        elif exam["status"] == "running":
            current_eta = progress.get("eta_seconds")
            if progress.get("rate_units_per_second") and progress["rate_units_per_second"] > 0:
                running_rate = 1.0 / float(progress["rate_units_per_second"])
        elif exam["status"] == "queued":
            remaining_units += int(progress.get("total_units", exam.get("total_units", 0)) or 0)

    queue_eta = current_eta
    seconds_per_unit = None
    if completed_rates:
        seconds_per_unit = sum(completed_rates) / len(completed_rates)
    elif running_rate is not None:
        seconds_per_unit = running_rate
    if remaining_units > 0 and seconds_per_unit is not None:
        queued_eta = remaining_units * seconds_per_unit
        queue_eta = (queue_eta or 0.0) + queued_eta if queue_eta is not None else queued_eta

    lines.append(f"queue_eta={format_duration(queue_eta)}")
    lines.append("")
    lines.append("exams:")
    for index, (exam, progress) in enumerate(progress_rows, start=1):
        lines.append(_render_exam_line(index, exam, progress))
    return "\n".join(lines)


def _train_command(exam_name: str) -> list[str]:
    return [sys.executable, str(PROJECT_ROOT / "scripts" / "train.py"), exam_name]


def _mark_state_stopped(state: dict, *, current_index: int | None, returncode: int | None):
    now = utc_now_iso()
    state["status"] = "stopped"
    state["finished_at"] = now
    state["current_index"] = current_index
    if current_index is None:
        return
    if current_index < 0 or current_index >= len(state["exams"]):
        return
    exam = state["exams"][current_index]
    if exam["status"] == "running":
        exam["status"] = "stopped"
        exam["finished_at"] = now
        if returncode is not None:
            exam["returncode"] = int(returncode)


def run_worker(state_path: Path = STATE_PATH) -> int:
    state = load_state(state_path)
    state["status"] = "running"
    state["started_at"] = state.get("started_at") or utc_now_iso()
    state["worker_pid"] = os.getpid()
    save_state(state, state_path)

    stop_requested = {"flag": False}
    current_process = {"proc": None}
    current_index = {"value": None}

    # Forward stop signals to the active training child, then flush queue state.
    def _handle_stop(_signum, _frame):
        stop_requested["flag"] = True
        proc = current_process["proc"]
        if proc is not None and proc.poll() is None:
            proc.terminate()

    prev_sigterm = signal.signal(signal.SIGTERM, _handle_stop)
    prev_sigint = signal.signal(signal.SIGINT, _handle_stop)
    try:
        for index, exam in enumerate(state["exams"]):
            if stop_requested["flag"]:
                _mark_state_stopped(state, current_index=current_index["value"], returncode=None)
                save_state(state, state_path)
                return 0
            if exam["status"] == "completed":
                continue
            current_index["value"] = index
            state["current_index"] = index
            exam["status"] = "running"
            exam["started_at"] = utc_now_iso()
            exam["returncode"] = None
            save_state(state, state_path)

            exam_dir = Path(exam["exam_dir"])
            output_dir = exam_dir / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            log_path = output_dir / "train.log"
            command = _train_command(exam["exam_name"])
            start_time = time.time()
            with log_path.open("a", encoding="utf-8") as log_handle:
                log_handle.write(f"\n=== queue worker start {utc_now_iso()} ===\n")
                log_handle.write(f"command: {' '.join(command)}\n")
                log_handle.flush()
                process = subprocess.Popen(
                    command,
                    cwd=str(PROJECT_ROOT),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                current_process["proc"] = process
                while True:
                    returncode = process.poll()
                    if returncode is not None:
                        break
                    if stop_requested["flag"]:
                        process.terminate()
                    time.sleep(1.0)
                current_process["proc"] = None

            exam["finished_at"] = utc_now_iso()
            exam["duration_seconds"] = max(time.time() - start_time, 0.0)
            exam["returncode"] = int(returncode)
            if stop_requested["flag"]:
                _mark_state_stopped(state, current_index=index, returncode=returncode)
                exam["duration_seconds"] = max(time.time() - start_time, 0.0)
                save_state(state, state_path)
                return 0
            if returncode == 0:
                exam["status"] = "completed"
                save_state(state, state_path)
                continue

            exam["status"] = "failed"
            state["status"] = "failed"
            state["finished_at"] = utc_now_iso()
            state["current_index"] = index
            save_state(state, state_path)
            return returncode

        state["status"] = "completed"
        state["finished_at"] = utc_now_iso()
        state["current_index"] = None
        save_state(state, state_path)
        return 0
    finally:
        signal.signal(signal.SIGTERM, prev_sigterm)
        signal.signal(signal.SIGINT, prev_sigint)


def stop_queue(state_path: Path = STATE_PATH) -> int:
    if not state_path.exists():
        print(f"Queue state not found: {state_path}")
        return 1

    state = load_state(state_path)
    worker_pid = state.get("worker_pid")
    if not is_process_alive(worker_pid):
        if state.get("status") in {"running", "stopping"}:
            _mark_state_stopped(state, current_index=state.get("current_index"), returncode=None)
            save_state(state, state_path)
        print("Queue worker is not running.")
        return 0

    state["status"] = "stopping"
    save_state(state, state_path)
    try:
        os.killpg(int(worker_pid), signal.SIGTERM)
    except ProcessLookupError:
        _mark_state_stopped(state, current_index=state.get("current_index"), returncode=None)
        save_state(state, state_path)
    print(f"Stop signal sent to worker pid={worker_pid}.")
    print(f"State file: {state_path}")
    return 0


def start_queue(state_path: Path = STATE_PATH, foreground: bool = False) -> int:
    ensure_startable(state_path)
    exam_dirs = discover_pending_exams(EXAM_ROOT)
    if not exam_dirs:
        print("No pending exams found.")
        return 0

    state = build_initial_state(exam_dirs)
    save_state(state, state_path)

    if foreground:
        print(f"Loaded {len(exam_dirs)} pending exams. Running in foreground.")
        return run_worker(state_path)

    QUEUE_ROOT.mkdir(parents=True, exist_ok=True)
    with WORKER_LOG_PATH.open("a", encoding="utf-8") as worker_log:
        worker = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "_worker", "--state-path", str(state_path)],
            cwd=str(PROJECT_ROOT),
            stdout=worker_log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    state = load_state(state_path)
    state["worker_pid"] = int(worker.pid)
    state["status"] = "running"
    state["started_at"] = state.get("started_at") or utc_now_iso()
    save_state(state, state_path)
    print(f"Loaded {len(exam_dirs)} pending exams.")
    print(f"Background worker pid: {worker.pid}")
    print(f"State file: {state_path}")
    print(f"Worker log: {WORKER_LOG_PATH}")
    return 0


def show_status(state_path: Path = STATE_PATH) -> int:
    if not state_path.exists():
        print(f"Queue state not found: {state_path}")
        return 1
    state = load_state(state_path)
    print(render_status(state))
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Cold-load exam queue runner for sequential background training")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser("start", help="Snapshot pending exams once and start sequential training")
    start_parser.add_argument("--foreground", action="store_true", help="Run the worker in foreground instead of daemonizing")

    status_parser = subparsers.add_parser("status", help="Show queue progress and ETA")
    status_parser.add_argument("--state-path", default=str(STATE_PATH), help="Queue state JSON path")

    stop_parser = subparsers.add_parser("stop", help="Stop the background queue and its current training child")
    stop_parser.add_argument("--state-path", default=str(STATE_PATH), help="Queue state JSON path")

    worker_parser = subparsers.add_parser("_worker", help=argparse.SUPPRESS)
    worker_parser.add_argument("--state-path", default=str(STATE_PATH))

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "start":
        return start_queue(foreground=bool(args.foreground))
    if args.command == "status":
        return show_status(Path(args.state_path))
    if args.command == "stop":
        return stop_queue(Path(args.state_path))
    if args.command == "_worker":
        return run_worker(Path(args.state_path))
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
