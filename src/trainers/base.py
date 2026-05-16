from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import json


class Trainer(ABC):
    trainer_name = "base"

    def __init__(
        self,
        config: dict,
        exam_dir: str | Path,
        exam_name: str | None = None,
        seed: int | None = None,
        output_dir: str | Path | None = None,
    ):
        self.config = dict(config)
        self.exam_dir = Path(exam_dir)
        self.exam_name = str(exam_name or self.exam_dir.name)
        self.seed = int(seed if seed is not None else self.config.get("train", {}).get("seed", 0))
        self._output_dir_override = Path(output_dir) if output_dir is not None else None

    @property
    def output_dir(self) -> Path:
        if self._output_dir_override is not None:
            return self._output_dir_override
        return self.default_output_dir()

    def default_output_dir(self) -> Path:
        return self.exam_dir / "output"

    def ensure_output_dir(self) -> Path:
        output_dir = self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def write_json(self, path: str | Path, payload: dict) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    @abstractmethod
    def run(self) -> dict:
        raise NotImplementedError
