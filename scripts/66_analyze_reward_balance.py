from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_STEPS = [100, 200, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]


def _load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _merged_metrics(row: dict) -> dict:
    merged = {}
    merged.update(row.get("train_scalars", {}))
    policy_metrics = row.get("policy_metrics", {})
    merged.update(policy_metrics.get("scalars", {}))
    merged.update(row.get("logged_metrics", {}))
    merged.update(row.get("persistent_stats", {}))
    step_metrics = row.get("persistent_step_metrics", {})
    for key, value in step_metrics.items():
        if key != "reset_reason":
            merged[f"step/{key}"] = value
        else:
            merged["step/reset_reason"] = value
    merged["step"] = row.get("step")
    return merged


def _fmt(value) -> str:
    if value is None:
        return "NA"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, bool)):
        return str(value)
    try:
        value = float(value)
    except Exception:
        return str(value)
    if abs(value) >= 1000 or (0 < abs(value) < 1e-4):
        return f"{value:.3e}"
    return f"{value:.6f}"


def _print_final_summary(metrics: dict) -> None:
    print("FINAL")
    keys = [
        "bad_numeric_count",
        "wm/event_replay_head_used",
        "wm/high_gain_label_used",
        "event/old_event_gate_used_for_reward",
        "wm/fast_error",
        "wm/with_slow_error",
        "wm/slow_gain",
        "wm/distill_loss",
        "wm/fast_pred_loss",
        "wm/slow_update_penalty",
        "wm/slow_gate_rate",
        "wm/local_slow_gain_mean",
        "wm/local_slow_gain_positive_rate",
        "wm/slow_gain_reward_mean",
        "wm/slow_gain_reward_nonzero_rate",
        "actor/slow_gain_intrinsic_mean",
        "actor/slow_gain_intrinsic_nonzero_rate",
        "actor/env_reward_pred_mean",
        "actor/total_imagined_reward_mean",
        "intrinsic/slow_gain_reward_mean",
        "intrinsic/slow_gain_reward_nonzero_rate",
        "intrinsic/slow_gain_reward_when_slow_gain_top20",
        "intrinsic/slow_gain_reward_when_slow_gain_not_top20",
        "wm/slow_gain_reward_corr",
        "action_penalty_mean",
        "actor/action_norm",
        "actor/action_abs_mean",
        "actor/action_std",
        "env/success_rate",
        "reset_count",
        "reset_reason_max_lifetime",
        "reset_reason_env_done",
        "reset_reason_stale",
        "lifetime_step_mean",
        "lifetime_step_max",
        "step/reset_reason",
    ]
    for key in keys:
        print(f"{key}: {_fmt(metrics.get(key))}")


def _print_step_table(rows: list[dict], steps: list[int]) -> None:
    merged_rows = {row["step"]: _merged_metrics(row) for row in rows}
    headers = [
        "step",
        "corr",
        "top20",
        "not_top20",
        "actor_intr",
        "env_pred",
        "total_imag",
        "action_pen",
        "slow_gate",
        "distill",
        "action_norm",
    ]
    print("\nTIMELINE")
    print("\t".join(headers))
    for step in steps:
        metrics = merged_rows.get(step)
        if not metrics:
            continue
        values = [
            step,
            metrics.get("wm/slow_gain_reward_corr"),
            metrics.get("intrinsic/slow_gain_reward_when_slow_gain_top20"),
            metrics.get("intrinsic/slow_gain_reward_when_slow_gain_not_top20"),
            metrics.get("actor/slow_gain_intrinsic_mean"),
            metrics.get("actor/env_reward_pred_mean"),
            metrics.get("actor/total_imagined_reward_mean"),
            metrics.get("action_penalty_mean"),
            metrics.get("wm/slow_gate_rate"),
            metrics.get("wm/distill_loss"),
            metrics.get("actor/action_norm"),
        ]
        print("\t".join(_fmt(v) for v in values))


def _print_simple_diagnosis(rows: list[dict]) -> None:
    merged = [_merged_metrics(row) for row in rows]
    corr_values = [m["wm/slow_gain_reward_corr"] for m in merged if m.get("wm/slow_gain_reward_corr") is not None]
    pos_align = sum(
        1
        for m in merged
        if (m.get("intrinsic/slow_gain_reward_when_slow_gain_top20") is not None)
        and (m.get("intrinsic/slow_gain_reward_when_slow_gain_not_top20") is not None)
        and (m["intrinsic/slow_gain_reward_when_slow_gain_top20"] > m["intrinsic/slow_gain_reward_when_slow_gain_not_top20"])
    )
    total_align = sum(
        1
        for m in merged
        if (m.get("intrinsic/slow_gain_reward_when_slow_gain_top20") is not None)
        and (m.get("intrinsic/slow_gain_reward_when_slow_gain_not_top20") is not None)
    )
    pos_corr = sum(1 for v in corr_values if v > 0)
    neg_corr = sum(1 for v in corr_values if v < 0)
    print("\nDIAGNOSIS")
    print(f"positive_corr_steps: {pos_corr}/{len(corr_values)}")
    print(f"negative_corr_steps: {neg_corr}/{len(corr_values)}")
    print(f"top20_reward_gt_not_top20_steps: {pos_align}/{total_align}")
    if corr_values:
        print(f"corr_min: {_fmt(min(corr_values))}")
        print(f"corr_max: {_fmt(max(corr_values))}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze reward balance from policy_metrics.jsonl.")
    parser.add_argument("policy_metrics", type=Path)
    parser.add_argument("--steps", type=int, nargs="*", default=DEFAULT_STEPS)
    args = parser.parse_args()

    rows = _load_rows(args.policy_metrics)
    if not rows:
        raise SystemExit("No rows found in policy_metrics.jsonl")
    final_metrics = _merged_metrics(rows[-1])
    _print_final_summary(final_metrics)
    _print_step_table(rows, args.steps)
    _print_simple_diagnosis(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
