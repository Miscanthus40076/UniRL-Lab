from __future__ import annotations

from pathlib import Path
import argparse
import importlib.util
import os
import re
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

if __name__ == "__main__":
    sys.modules.setdefault("scripts.train", sys.modules[__name__])

from src.trainers import build_trainer as build_root_trainer
from utils import validate_exam_config


EXAM_ROOT = PROJECT_ROOT / "exam"
TRAIN_FILENAME = "train.py"
CONFIG_FILENAME = "config.yaml"


class BaseExamTrainApp:
    def __init__(self, exam_dir: str | Path):
        self.exam_dir = Path(exam_dir).resolve()
        self.exam_name = self.exam_dir.name

    @classmethod
    def from_source_file(cls, source_file: str | Path) -> "BaseExamTrainApp":
        return cls(Path(source_file).resolve().parent)

    @property
    def config_path(self) -> Path:
        return self.exam_dir / CONFIG_FILENAME

    def load_config(self) -> dict:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Exam config not found: {self.config_path}")
        with self.config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        if not isinstance(config, dict):
            raise TypeError(f"Exam config must be a mapping: {self.config_path}")
        return config

    def build_default_trainer(self, config: dict):
        return build_root_trainer(config=config, exam_dir=self.exam_dir, exam_name=self.exam_name)

    def build_trainer(self, config: dict):
        raise NotImplementedError("Exam train.py must implement build_trainer(config)")

    def run(self):
        config = self.load_config()
        validate_exam_config(config)
        trainer = self.build_trainer(config)
        return trainer.run()


def render_exam_train_template() -> str:
    return """from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import BaseExamTrainApp


class ExamTrain(BaseExamTrainApp):
    def build_trainer(self, config: dict):
        return self.build_default_trainer(config)


def main():
    return ExamTrain.from_source_file(__file__).run()


if __name__ == "__main__":
    main()
"""


def _exam_train_path(exam_name: str) -> Path:
    return EXAM_ROOT / exam_name / TRAIN_FILENAME


def _module_name_for_exam(exam_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", str(exam_name)).strip("_")
    return f"exam_train_{slug or 'exam'}"


def _load_exam_train_module(exam_name: str):
    train_path = _exam_train_path(exam_name)
    if not train_path.exists():
        raise FileNotFoundError(
            f"Exam train entry not found: {train_path}. "
            "Each exam must provide its own train.py that inherits from scripts/train.py."
        )

    module_name = _module_name_for_exam(exam_name)
    spec = importlib.util.spec_from_file_location(module_name, train_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create module spec for exam train entry: {train_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_exam_train_class(module):
    explicit = getattr(module, "EXAM_TRAIN_CLASS", None)
    if explicit is not None:
        if not isinstance(explicit, type) or not issubclass(explicit, BaseExamTrainApp):
            raise TypeError("EXAM_TRAIN_CLASS must be a BaseExamTrainApp subclass")
        return explicit

    named = getattr(module, "ExamTrain", None)
    if isinstance(named, type) and issubclass(named, BaseExamTrainApp) and named is not BaseExamTrainApp:
        return named

    candidates = []
    for value in vars(module).values():
        if isinstance(value, type) and issubclass(value, BaseExamTrainApp) and value is not BaseExamTrainApp:
            candidates.append(value)

    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ImportError("Exam train.py must define an ExamTrain subclass of BaseExamTrainApp")
    raise ImportError("Exam train.py defines multiple BaseExamTrainApp subclasses; set EXAM_TRAIN_CLASS explicitly")


def load_exam_train_app(exam_name: str) -> BaseExamTrainApp:
    module = _load_exam_train_module(exam_name)
    train_cls = _resolve_exam_train_class(module)
    return train_cls(EXAM_ROOT / exam_name)


def train(exam_name: str):
    return load_exam_train_app(exam_name).run()


def parse_args():
    parser = argparse.ArgumentParser(description="Train an exam from exam/<name>/train.py")
    parser.add_argument("exam_name", help="Name of the exam directory under exam/")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.exam_name)
