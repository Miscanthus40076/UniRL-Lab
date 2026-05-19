from __future__ import annotations

import torch

from policy.dreamerv3_milestoneV3.operator_intrinsic_reward import (
    DCTEventElement,
    DCTSequenceMilestoneMemory,
    DCTTemporalSequenceBuffer,
)


def _sig(values):
    tensor = torch.as_tensor(values, dtype=torch.float32)
    return tensor / torch.clamp(torch.linalg.vector_norm(tensor), min=1e-8)


def _event(token: int, step: int, signature=None):
    sig = _sig(signature or [1.0, 0.0, 0.0, 0.0])
    return DCTEventElement(token_id=token, signature=sig, step=step, element_key=(token, tuple(sig.tolist())))


def _sequence_signature(events):
    parts = []
    for event in events:
        parts.append(torch.nn.functional.one_hot(torch.tensor(event.token_id), num_classes=4).float())
        parts.append(event.signature.float())
    sig = torch.cat(parts)
    return sig / torch.clamp(torch.linalg.vector_norm(sig), min=1e-8)


def test_temporal_buffer_keeps_order_and_capacity():
    buffer = DCTTemporalSequenceBuffer(capacity=2)
    buffer.append(_event(0, 0))
    buffer.append(_event(1, 3))
    buffer.append(_event(2, 6))
    assert [event.token_id for event in buffer.previous_events()] == [1, 2]


def test_first_bigram_inserted_then_seen_rejected():
    memory = DCTSequenceMilestoneMemory(capacity=4, seen_enabled=True)
    sig = _sequence_signature([_event(0, 0), _event(1, 3)])
    first = memory.evaluate(sig, [0, 1], seen_threshold=0.2, active_threshold=0.2, update_state=True)
    second = memory.evaluate(sig, [0, 1], seen_threshold=0.2, active_threshold=0.2, update_state=True)
    assert first["accepted"] is True
    assert first["reason"] == "sequence_new_inserted"
    assert second["accepted"] is False
    assert second["reason"] == "sequence_already_seen"


def test_similar_trigram_second_time_rejected():
    memory = DCTSequenceMilestoneMemory(capacity=4, seen_enabled=True)
    sig_a = _sequence_signature([_event(0, 0), _event(1, 3), _event(2, 6)])
    sig_b = sig_a + 1e-4
    sig_b = sig_b / torch.clamp(torch.linalg.vector_norm(sig_b), min=1e-8)
    assert memory.evaluate(sig_a, [0, 1, 2], 0.2, 0.2, True)["accepted"] is True
    result = memory.evaluate(sig_b, [0, 1, 2], 0.2, 0.2, True)
    assert result["accepted"] is False
    assert result["reason"] == "sequence_already_seen"


def test_seen_memory_blocks_repeat_after_active_reset():
    memory = DCTSequenceMilestoneMemory(capacity=1, seen_enabled=True)
    sig = _sequence_signature([_event(0, 0), _event(1, 3)])
    assert memory.evaluate(sig, [0, 1], 0.2, 0.2, True)["accepted"] is True
    memory.reset_active()
    result = memory.evaluate(sig, [0, 1], 0.2, 0.2, True)
    assert result["accepted"] is False
    assert result["reason"] == "sequence_already_seen"


def test_high_priority_preempts_low_priority():
    memory = DCTSequenceMilestoneMemory(capacity=1, seen_enabled=False, preemption_enabled=True, preemption_margin=0.05)
    high_sig = _sequence_signature([_event(0, 0), _event(1, 3)])
    memory.active.append({"signature": _sequence_signature([_event(0, 0), _event(0, 3)]), "priority": 0.0})
    result = memory.evaluate(high_sig, [0, 1], 0.0, -1.0, True)
    assert result["accepted"] is True
    assert result["reason"] == "sequence_preempted_low_priority"
    assert memory.counts()["preemption_count"] == 1


def test_low_priority_rejected_when_active_full():
    memory = DCTSequenceMilestoneMemory(capacity=1, seen_enabled=True, preemption_enabled=True, preemption_margin=10.0)
    sig_a = _sequence_signature([_event(0, 0), _event(1, 3)])
    sig_b = _sequence_signature([_event(2, 0), _event(3, 3)])
    assert memory.evaluate(sig_a, [0, 1], 0.0, 0.0, True)["accepted"] is True
    result = memory.evaluate(sig_b, [2, 3], 0.0, 0.0, True)
    assert result["accepted"] is False
    assert result["reason"] == "sequence_rejected_low_priority"


def test_no_privileged_labels_in_sequence_reward_source():
    source = open("policy/dreamerv3_milestoneV3/operator_intrinsic_reward.py", encoding="utf-8").read().lower()
    for forbidden in (".get(\"success", ".get('success", "contact_mode", "alignment", "insertion_depth", "ball_in_cup"):
        assert forbidden not in source
