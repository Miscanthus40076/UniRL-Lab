from __future__ import annotations

from .online_policy import OnlinePolicyTrainer


STANDARD_TRAINERS = {"online_policy", "default", "single_policy"}
BIDIRECTIONAL_TRAINERS = {"bidirectional", "bidirectional_online", "bidreamer"}


def _normalized_trainer_name(config: dict) -> str:
    from scripts.utils import trainer_name

    return str(trainer_name(config)).strip().lower()


def build_trainer(config: dict, exam_dir, exam_name: str | None = None):
    name = _normalized_trainer_name(config)
    if name in STANDARD_TRAINERS:
        return OnlinePolicyTrainer(config=config, exam_dir=exam_dir, exam_name=exam_name)
    if name in BIDIRECTIONAL_TRAINERS:
        from src.bidreamer.online_train_loop import BidirectionalTrainer

        return BidirectionalTrainer(config=config, exam_dir=exam_dir, exam_name=exam_name)
    raise ValueError(f"Unsupported trainer: {name}")
