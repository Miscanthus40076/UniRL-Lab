from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np


def save_line(xs, ys, path, xlabel, ylabel, title):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys, linewidth=1.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def _smooth_values(values, window: int):
    if window <= 1:
        return np.asarray(values, dtype=np.float32)
    arr = np.asarray(values, dtype=np.float32)
    out = np.empty_like(arr, dtype=np.float32)
    for idx in range(len(arr)):
        left = max(0, idx - window + 1)
        window_values = arr[left : idx + 1]
        finite = window_values[np.isfinite(window_values)]
        out[idx] = np.nan if finite.size == 0 else finite.mean()
    return out


def plot_csv(rows, out_dir, x_key="step"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    numeric = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)):
                numeric.setdefault(key, []).append(float(value))
    # TODO: 这里按列独立收集数值会丢失“缺失值在哪一行”的位置信息；稀疏 metrics 会导致 x/y 对齐语义不准确。

    x_series = numeric.get(x_key)
    saved = []

    for key, values in numeric.items():
        if key == x_key:
            continue

        if x_series and len(x_series) == len(values):
            xs = x_series
            xlabel = x_key
        else:
            xs = list(range(1, len(values) + 1))
            xlabel = "record"

        saved.append(
            save_line(
                xs=xs,
                ys=values,
                path=out_dir / f"{key}.png",
                xlabel=xlabel,
                ylabel=key,
                title=key,
            )
        )

    return saved


def _append_numeric_point(series, row, x_key):
    x_value = row.get(x_key)
    for key, value in row.items():
        if key == x_key:
            continue
        if isinstance(value, (int, float)):
            bucket = series.setdefault(key, {"xs": [], "ys": []})
            bucket["ys"].append(float(value))
            if isinstance(x_value, (int, float)):
                bucket["xs"].append(float(x_value))
            else:
                bucket["xs"].append(float(len(bucket["ys"])))


def _load_policy_metric_rows(path: Path, x_key: str):
    rows = []
    series = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            train_scalars = record.get("train_scalars", {})
            if not isinstance(train_scalars, dict):
                raise TypeError("train_scalars must be a dict")
            _append_numeric_point(series, train_scalars, x_key)

            policy_metrics = record.get("policy_metrics", {})
            if not isinstance(policy_metrics, dict):
                raise TypeError("policy_metrics must be a dict")
            policy_scalars = policy_metrics.get("scalars", {})
            if not isinstance(policy_scalars, dict):
                raise TypeError("policy_metrics.scalars must be a dict")
            logged_metrics = record.get("logged_metrics", {})
            if logged_metrics is None:
                logged_metrics = {}
            if not isinstance(logged_metrics, dict):
                raise TypeError("logged_metrics must be a dict when present")
            persistent_stats = record.get("persistent_stats", {})
            if persistent_stats is None:
                persistent_stats = {}
            if not isinstance(persistent_stats, dict):
                raise TypeError("persistent_stats must be a dict when present")
            persistent_step_metrics = record.get("persistent_step_metrics", {})
            if persistent_step_metrics is None:
                persistent_step_metrics = {}
            if not isinstance(persistent_step_metrics, dict):
                raise TypeError("persistent_step_metrics must be a dict when present")

            merged_row = {x_key: train_scalars.get(x_key, record.get(x_key))}
            merged_row.update(train_scalars)
            merged_row.update(policy_scalars)
            merged_row.update(logged_metrics)
            merged_row.update(persistent_stats)
            merged_row.update(persistent_step_metrics)
            rows.append(merged_row)

            prefixed_row = {x_key: merged_row.get(x_key)}
            prefixed_row.update({f"policy/{key}": value for key, value in policy_scalars.items()})
            _append_numeric_point(series, prefixed_row, x_key)
    return rows, series


def _series_from_rows(rows, key, x_key):
    xs = []
    ys = []
    for index, row in enumerate(rows, start=1):
        value = row.get(key)
        if not isinstance(value, (int, float)):
            continue
        xs.append(float(row.get(x_key, index)) if isinstance(row.get(x_key), (int, float)) else float(index))
        ys.append(float(value))
    return xs, ys


def _first_available(rows, aliases, x_key):
    for key in aliases:
        xs, ys = _series_from_rows(rows, key, x_key)
        if ys:
            return key, xs, ys
    return None, [], []


def _plot_subplot(ax, rows, x_key, aliases_by_label, smooth: int):
    plotted = False
    for label, aliases in aliases_by_label:
        _, xs, ys = _first_available(rows, aliases, x_key)
        if not ys:
            continue
        ax.plot(xs, _smooth_values(ys, smooth), linewidth=1.5, label=label)
        plotted = True
    if plotted:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No available metrics", ha="center", va="center", transform=ax.transAxes)
    ax.grid(alpha=0.25)


def _save_fast_slow_diagnostics(rows, out_dir, x_key="step", smooth: int = 10):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # This figure checks the hypothesized collapse chain:
    # slow_gain ~= slow_penalty -> slow gate oscillation rises -> milestone novelty drops
    # -> actor/environment activity falls -> behavior degenerates toward immobility.
    fig, axes = plt.subplots(5, 1, figsize=(13, 18), sharex=True)

    axes[0].set_title("Fast/Slow Prediction Competition")
    _plot_subplot(
        axes[0],
        rows,
        x_key,
        [
            ("wm/fast_error", ["wm/fast_error"]),
            ("wm/with_slow_error", ["wm/with_slow_error"]),
            ("wm/teacher_error", ["wm/teacher_error", "wm/with_slow_error"]),
            ("wm/slow_gain", ["wm/slow_gain"]),
            ("wm/slow_update_penalty", ["wm/slow_update_penalty", "wm/slow_penalty", "context_update_loss_scaled"]),
        ],
        smooth=smooth,
    )

    axes[1].set_title("Removed Slow Rate Limit")
    _plot_subplot(
        axes[1],
        rows,
        x_key,
        [
            ("wm/slow_rate_limit_loss_raw", ["wm/slow_rate_limit_loss_raw"]),
            ("wm/slow_rate_limit_loss_weighted", ["wm/slow_rate_limit_loss_weighted"]),
            ("wm/slow_gate_rate", ["wm/slow_gate_rate", "context_gate_mean"]),
        ],
        smooth=smooth,
    )

    axes[2].set_title("Distillation")
    _plot_subplot(
        axes[2],
        rows,
        x_key,
        [
            ("wm/fast_pred_loss", ["wm/fast_pred_loss"]),
            ("wm/fast_pred_loss_weighted", ["wm/fast_pred_loss_weighted"]),
            ("wm/distill_loss", ["wm/distill_loss"]),
            ("wm/distill_loss_weighted", ["wm/distill_loss_weighted"]),
            ("wm/teacher_pred_loss", ["wm/teacher_pred_loss", "prediction_loss"]),
        ],
        smooth=smooth,
    )

    axes[3].set_title("Event / Milestone")
    _plot_subplot(
        axes[3],
        rows,
        x_key,
        [
            ("event/event_gate_mean", ["event/event_gate_mean", "event_gate_mean"]),
            ("intrinsic/milestone_new_rate", ["intrinsic/milestone_new_rate", "milestone_nonzero_ratio"]),
            (
                "intrinsic/milestone_reward_nonzero_rate",
                ["intrinsic/milestone_reward_nonzero_rate", "milestone_nonzero_ratio"],
            ),
            (
                "intrinsic/sequence_reward_mean",
                ["intrinsic/sequence_reward_mean", "dct_v3_sequence_reward_mean", "dct_sequence_milestone_component_mean"],
            ),
            (
                "intrinsic/sequence_reward_nonzero_rate",
                ["intrinsic/sequence_reward_nonzero_rate", "dct_v3_sequence_reward_nonzero_ratio"],
            ),
        ],
        smooth=smooth,
    )

    axes[4].set_title("Actor / Environment Behavior")
    _plot_subplot(
        axes[4],
        rows,
        x_key,
        [
            ("actor/action_norm", ["actor/action_norm", "action_norm_mean"]),
            ("actor/action_abs_mean", ["actor/action_abs_mean"]),
            ("actor/action_std", ["actor/action_std"]),
            ("actor/action_entropy", ["actor/action_entropy", "entropy"]),
            ("env/contact_rate", ["env/contact_rate"]),
            ("env/object_motion", ["env/object_motion"]),
            ("env/success_rate", ["env/success_rate"]),
        ],
        smooth=smooth,
    )
    axes[4].set_xlabel(x_key)

    out_path = out_dir / "fast_slow_diagnostics.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _save_fast_slow_critical_region(rows, out_dir, x_key="step", smooth: int = 10):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    panels = [
        ("wm/slow_gain", ["wm/slow_gain"]),
        ("wm/slow_penalty", ["wm/slow_penalty", "context_update_loss_scaled"]),
        ("wm/slow_gate_rate", ["wm/slow_gate_rate", "context_gate_mean"]),
        ("actor/action_norm", ["actor/action_norm", "action_norm_mean"]),
    ]
    for ax, (label, aliases) in zip(axes, panels):
        ax.set_title(label)
        _, xs, ys = _first_available(rows, aliases, x_key)
        if ys:
            ax.plot(xs, _smooth_values(ys, smooth), linewidth=1.5, label=label)
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No available metrics", ha="center", va="center", transform=ax.transAxes)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel(x_key)

    out_path = out_dir / "fast_slow_critical_region.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _save_slow_gain_reward_alignment(rows, out_dir, x_key="step", smooth: int = 10):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(13, 15), sharex=True)

    axes[0].set_title("Local Slow Gain")
    _plot_subplot(
        axes[0],
        rows,
        x_key,
        [
            ("wm/local_slow_gain_mean", ["wm/local_slow_gain_mean"]),
            ("wm/local_slow_gain_std", ["wm/local_slow_gain_std"]),
            ("wm/positive_slow_gain_rate", ["wm/positive_slow_gain_rate", "wm/local_slow_gain_positive_rate"]),
            ("wm/local_slow_gain_top20_mean", ["wm/local_slow_gain_top20_mean"]),
        ],
        smooth=smooth,
    )

    axes[1].set_title("Intrinsic Reward Alignment")
    _plot_subplot(
        axes[1],
        rows,
        x_key,
        [
            ("intrinsic/reward_when_slow_gain_top20", ["intrinsic/reward_when_slow_gain_top20"]),
            ("intrinsic/reward_when_slow_gain_not_top20", ["intrinsic/reward_when_slow_gain_not_top20"]),
            (
                "intrinsic/reward_nonzero_rate_when_slow_gain_top20",
                ["intrinsic/reward_nonzero_rate_when_slow_gain_top20"],
            ),
            (
                "intrinsic/reward_nonzero_rate_when_slow_gain_not_top20",
                ["intrinsic/reward_nonzero_rate_when_slow_gain_not_top20"],
            ),
        ],
        smooth=smooth,
    )

    axes[2].set_title("Milestone Reward Alignment")
    _plot_subplot(
        axes[2],
        rows,
        x_key,
        [
            ("intrinsic/milestone_reward_when_slow_gain_top20", ["intrinsic/milestone_reward_when_slow_gain_top20"]),
            (
                "intrinsic/milestone_reward_when_slow_gain_not_top20",
                ["intrinsic/milestone_reward_when_slow_gain_not_top20"],
            ),
            (
                "intrinsic/milestone_nonzero_rate_when_slow_gain_top20",
                ["intrinsic/milestone_nonzero_rate_when_slow_gain_top20"],
            ),
            (
                "intrinsic/milestone_nonzero_rate_when_slow_gain_not_top20",
                ["intrinsic/milestone_nonzero_rate_when_slow_gain_not_top20"],
            ),
        ],
        smooth=smooth,
    )

    axes[3].set_title("Sequence Reward Alignment")
    _plot_subplot(
        axes[3],
        rows,
        x_key,
        [
            ("intrinsic/sequence_reward_when_slow_gain_top20", ["intrinsic/sequence_reward_when_slow_gain_top20"]),
            (
                "intrinsic/sequence_reward_when_slow_gain_not_top20",
                ["intrinsic/sequence_reward_when_slow_gain_not_top20"],
            ),
            (
                "intrinsic/sequence_nonzero_rate_when_slow_gain_top20",
                ["intrinsic/sequence_nonzero_rate_when_slow_gain_top20"],
            ),
            (
                "intrinsic/sequence_nonzero_rate_when_slow_gain_not_top20",
                ["intrinsic/sequence_nonzero_rate_when_slow_gain_not_top20"],
            ),
        ],
        smooth=smooth,
    )
    axes[3].set_xlabel(x_key)

    out_path = out_dir / "slow_gain_reward_alignment.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _save_fast_slow_diagnostics_v5(rows, out_dir, x_key="step", smooth: int = 10):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(5, 1, figsize=(13, 18), sharex=True)

    axes[0].set_title("Prediction / Distillation")
    _plot_subplot(
        axes[0],
        rows,
        x_key,
        [
            ("wm/fast_error", ["wm/fast_error"]),
            ("wm/with_slow_error", ["wm/with_slow_error"]),
            ("wm/teacher_error", ["wm/teacher_error", "wm/with_slow_error"]),
            ("wm/slow_gain", ["wm/slow_gain"]),
            ("wm/distill_loss", ["wm/distill_loss"]),
            ("wm/event_replay_loss", ["wm/event_replay_loss"]),
        ],
        smooth=smooth,
    )

    axes[1].set_title("Gain Selector")
    _plot_subplot(
        axes[1],
        rows,
        x_key,
        [
            ("wm/gain_score_mean", ["wm/gain_score_mean"]),
            ("wm/gain_score_max", ["wm/gain_score_max"]),
            ("event/gain_candidate_rate", ["event/gain_candidate_rate"]),
            ("event/stable_gain_event_rate", ["event/stable_gain_event_rate"]),
        ],
        smooth=smooth,
    )

    axes[2].set_title("Stable Gain Segments")
    _plot_subplot(
        axes[2],
        rows,
        x_key,
        [
            ("event/stable_gain_segment_count", ["event/stable_gain_segment_count"]),
            ("event/stable_gain_segment_mean_len", ["event/stable_gain_segment_mean_len"]),
            ("event/gain_event_signature_count", ["event/gain_event_signature_count"]),
        ],
        smooth=smooth,
    )

    axes[3].set_title("Reward Alignment")
    _plot_subplot(
        axes[3],
        rows,
        x_key,
        [
            ("intrinsic/reward_when_slow_gain_top20", ["intrinsic/reward_when_slow_gain_top20"]),
            ("intrinsic/reward_when_slow_gain_not_top20", ["intrinsic/reward_when_slow_gain_not_top20"]),
            ("intrinsic/event_replay_reward_mean", ["intrinsic/event_replay_reward_mean"]),
            ("actor/event_replay_intrinsic_prob_mean", ["actor/event_replay_intrinsic_prob_mean", "actor/event_replay_intrinsic_mean"]),
            ("intrinsic/sequence_reward_when_slow_gain_top20", ["intrinsic/sequence_reward_when_slow_gain_top20"]),
            ("intrinsic/sequence_reward_when_slow_gain_not_top20", ["intrinsic/sequence_reward_when_slow_gain_not_top20"]),
        ],
        smooth=smooth,
    )

    axes[4].set_title("Actor / Env")
    _plot_subplot(
        axes[4],
        rows,
        x_key,
        [
            ("actor/action_norm", ["actor/action_norm", "action_norm_mean"]),
            ("actor/action_abs_mean", ["actor/action_abs_mean"]),
            ("actor/action_std", ["actor/action_std"]),
            ("event/old_event_gate_mean", ["event/old_event_gate_mean"]),
            ("env/contact_rate", ["env/contact_rate"]),
            ("env/object_motion", ["env/object_motion"]),
            ("env/success_rate", ["env/success_rate"]),
        ],
        smooth=smooth,
    )
    axes[4].set_xlabel(x_key)

    out_path = out_dir / "fast_slow_diagnostics_v5.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _save_event_gate_replacement_v5(rows, out_dir, x_key="step", smooth: int = 10):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    axes[0].set_title("Old Event Gate Legacy")
    _plot_subplot(
        axes[0],
        rows,
        x_key,
        [
            ("event/old_event_gate_mean", ["event/old_event_gate_mean"]),
            ("event/old_event_gate_saturation_rate", ["event/old_event_gate_saturation_rate"]),
            ("event/old_event_gate_used_for_milestone", ["event/old_event_gate_used_for_milestone"]),
        ],
        smooth=smooth,
    )

    axes[1].set_title("Gain Event Selector")
    _plot_subplot(
        axes[1],
        rows,
        x_key,
        [
            ("event/gain_candidate_rate", ["event/gain_candidate_rate"]),
            ("event/stable_gain_event_rate", ["event/stable_gain_event_rate"]),
        ],
        smooth=smooth,
    )

    axes[2].set_title("Milestone Source Split")
    _plot_subplot(
        axes[2],
        rows,
        x_key,
        [
            ("intrinsic/gain_event_milestone_reward_mean", ["intrinsic/gain_event_milestone_reward_mean"]),
            ("intrinsic/gain_sequence_reward_mean", ["intrinsic/gain_sequence_reward_mean"]),
            ("intrinsic/legacy_event_milestone_reward_mean", ["intrinsic/legacy_event_milestone_reward_mean"]),
        ],
        smooth=smooth,
    )
    axes[2].set_xlabel(x_key)

    out_path = out_dir / "event_gate_replacement_v5.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _save_event_replay_head_diagnostics(rows, out_dir, x_key="step", smooth: int = 10):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(5, 1, figsize=(13, 18), sharex=True)

    axes[0].set_title("Slow Gain Target")
    _plot_subplot(
        axes[0],
        rows,
        x_key,
        [
            ("wm/local_slow_gain_mean", ["wm/local_slow_gain_mean"]),
            ("wm/local_slow_gain_top20_mean", ["wm/local_slow_gain_top20_mean"]),
            ("wm/high_gain_label_rate", ["wm/high_gain_label_rate"]),
            ("wm/event_replay_target_mean", ["wm/event_replay_target_mean"]),
            ("wm/event_replay_target_nonzero_rate", ["wm/event_replay_target_nonzero_rate"]),
        ],
        smooth=smooth,
    )

    axes[1].set_title("Event Replay Classification")
    _plot_subplot(
        axes[1],
        rows,
        x_key,
        [
            ("wm/event_replay_cls_loss", ["wm/event_replay_cls_loss"]),
            ("wm/event_replay_high_gain_prob_top20_mean", ["wm/event_replay_high_gain_prob_top20_mean"]),
            ("wm/event_replay_high_gain_prob_not_top20_mean", ["wm/event_replay_high_gain_prob_not_top20_mean"]),
            ("wm/event_replay_high_gain_acc", ["wm/event_replay_high_gain_acc"]),
            ("wm/event_replay_high_gain_auc", ["wm/event_replay_high_gain_auc"]),
        ],
        smooth=smooth,
    )

    axes[2].set_title("Event Replay Value Auxiliary")
    _plot_subplot(
        axes[2],
        rows,
        x_key,
        [
            ("wm/event_replay_value_loss", ["wm/event_replay_value_loss"]),
            ("wm/event_replay_target_pred_corr", ["wm/event_replay_target_pred_corr"]),
            ("wm/event_replay_value_loss_weighted", ["wm/event_replay_value_loss_weighted"]),
        ],
        smooth=smooth,
    )

    axes[3].set_title("Reward Alignment")
    _plot_subplot(
        axes[3],
        rows,
        x_key,
        [
            ("intrinsic/event_replay_reward_when_slow_gain_top20", ["intrinsic/event_replay_reward_when_slow_gain_top20"]),
            ("intrinsic/event_replay_reward_when_slow_gain_not_top20", ["intrinsic/event_replay_reward_when_slow_gain_not_top20"]),
            ("wm/slow_gain_event_replay_reward_corr", ["wm/slow_gain_event_replay_reward_corr"]),
            ("intrinsic/reward_when_slow_gain_top20", ["intrinsic/reward_when_slow_gain_top20"]),
            ("intrinsic/reward_when_slow_gain_not_top20", ["intrinsic/reward_when_slow_gain_not_top20"]),
        ],
        smooth=smooth,
    )

    axes[4].set_title("Actor / Env Behavior")
    _plot_subplot(
        axes[4],
        rows,
        x_key,
        [
            ("actor/event_replay_intrinsic_prob_mean", ["actor/event_replay_intrinsic_prob_mean", "actor/event_replay_intrinsic_mean"]),
            ("actor/action_norm", ["actor/action_norm", "action_norm_mean"]),
            ("actor/action_abs_mean", ["actor/action_abs_mean"]),
            ("actor/action_std", ["actor/action_std"]),
            ("env/contact_rate", ["env/contact_rate"]),
            ("env/object_motion", ["env/object_motion"]),
            ("env/success_rate", ["env/success_rate"]),
        ],
        smooth=smooth,
    )
    axes[4].set_xlabel(x_key)

    out_path = out_dir / "event_replay_head_diagnostics.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _save_slow_gain_reward_diagnostics_v52(rows, out_dir, x_key="step", smooth: int = 10):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(5, 1, figsize=(13, 18), sharex=True)

    axes[0].set_title("Local Slow Gain")
    _plot_subplot(
        axes[0],
        rows,
        x_key,
        [
            ("wm/local_slow_gain_mean", ["wm/local_slow_gain_mean"]),
            ("wm/local_slow_gain_top20_mean", ["wm/local_slow_gain_top20_mean"]),
            ("wm/local_slow_gain_positive_rate", ["wm/local_slow_gain_positive_rate"]),
            ("wm/slow_gain_reward_mean", ["wm/slow_gain_reward_mean"]),
            ("wm/slow_gain_reward_nonzero_rate", ["wm/slow_gain_reward_nonzero_rate"]),
        ],
        smooth=smooth,
    )

    axes[1].set_title("Prediction / Distillation")
    _plot_subplot(
        axes[1],
        rows,
        x_key,
        [
            ("wm/fast_error", ["wm/fast_error"]),
            ("wm/with_slow_error", ["wm/with_slow_error"]),
            ("wm/slow_gain", ["wm/slow_gain"]),
            ("wm/distill_loss", ["wm/distill_loss"]),
            ("wm/slow_gate_rate", ["wm/slow_gate_rate"]),
        ],
        smooth=smooth,
    )

    axes[2].set_title("Actor Reward")
    _plot_subplot(
        axes[2],
        rows,
        x_key,
        [
            ("actor/slow_gain_intrinsic_mean", ["actor/slow_gain_intrinsic_mean"]),
            ("actor/env_reward_pred_mean", ["actor/env_reward_pred_mean"]),
            ("actor/total_imagined_reward_mean", ["actor/total_imagined_reward_mean"]),
            ("actor/slow_gain_reward_scale", ["actor/slow_gain_reward_scale"]),
        ],
        smooth=smooth,
    )

    axes[3].set_title("Reward Alignment Sanity Check")
    _plot_subplot(
        axes[3],
        rows,
        x_key,
        [
            ("intrinsic/slow_gain_reward_when_slow_gain_top20", ["intrinsic/slow_gain_reward_when_slow_gain_top20"]),
            ("intrinsic/slow_gain_reward_when_slow_gain_not_top20", ["intrinsic/slow_gain_reward_when_slow_gain_not_top20"]),
            ("intrinsic/reward_when_slow_gain_top20", ["intrinsic/reward_when_slow_gain_top20"]),
            ("intrinsic/reward_when_slow_gain_not_top20", ["intrinsic/reward_when_slow_gain_not_top20"]),
            ("wm/slow_gain_reward_corr", ["wm/slow_gain_reward_corr"]),
        ],
        smooth=smooth,
    )

    axes[4].set_title("Behavior")
    _plot_subplot(
        axes[4],
        rows,
        x_key,
        [
            ("actor/action_norm", ["actor/action_norm", "action_norm_mean"]),
            ("actor/action_abs_mean", ["actor/action_abs_mean"]),
            ("actor/action_std", ["actor/action_std"]),
            ("env/contact_rate", ["env/contact_rate"]),
            ("env/object_motion", ["env/object_motion"]),
            ("env/success_rate", ["env/success_rate"]),
        ],
        smooth=smooth,
    )
    axes[4].set_xlabel(x_key)

    out_path = out_dir / "slow_gain_reward_diagnostics_v52.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _save_reward_scale_balance_v52(rows, out_dir, x_key="step", smooth: int = 10):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(5, 1, figsize=(13, 18), sharex=True)

    axes[0].set_title("Reward Scale Balance")
    _plot_subplot(
        axes[0],
        rows,
        x_key,
        [
            ("actor/slow_gain_intrinsic_mean", ["actor/slow_gain_intrinsic_mean"]),
            ("actor/slow_gain_scaled_intrinsic_mean", ["actor/slow_gain_scaled_intrinsic_mean"]),
            ("abs(actor/env_reward_pred_mean)", ["actor/env_reward_pred_abs_mean"]),
            ("action_penalty_mean", ["action_penalty_mean"]),
        ],
        smooth=smooth,
    )

    axes[1].set_title("Unscaled Ratios")
    _plot_subplot(
        axes[1],
        rows,
        x_key,
        [
            ("actor/slow_gain_to_env_abs_ratio", ["actor/slow_gain_to_env_abs_ratio"]),
            ("actor/slow_gain_to_action_penalty_ratio", ["actor/slow_gain_to_action_penalty_ratio"]),
            ("actor/slow_gain_to_negative_terms_ratio", ["actor/slow_gain_to_negative_terms_ratio"]),
        ],
        smooth=smooth,
    )

    axes[2].set_title("Scaled Ratios")
    _plot_subplot(
        axes[2],
        rows,
        x_key,
        [
            ("actor/slow_gain_scaled_to_env_abs_ratio", ["actor/slow_gain_scaled_to_env_abs_ratio"]),
            ("actor/slow_gain_scaled_to_action_penalty_ratio", ["actor/slow_gain_scaled_to_action_penalty_ratio"]),
            ("actor/slow_gain_scaled_to_negative_terms_ratio", ["actor/slow_gain_scaled_to_negative_terms_ratio"]),
        ],
        smooth=smooth,
    )

    axes[3].set_title("Reward Alignment")
    _plot_subplot(
        axes[3],
        rows,
        x_key,
        [
            ("wm/slow_gain_reward_corr", ["wm/slow_gain_reward_corr"]),
            ("intrinsic/slow_gain_reward_when_slow_gain_top20", ["intrinsic/slow_gain_reward_when_slow_gain_top20"]),
            ("intrinsic/slow_gain_reward_when_slow_gain_not_top20", ["intrinsic/slow_gain_reward_when_slow_gain_not_top20"]),
        ],
        smooth=smooth,
    )

    axes[4].set_title("Behavior")
    _plot_subplot(
        axes[4],
        rows,
        x_key,
        [
            ("actor/action_norm", ["actor/action_norm", "action_norm_mean"]),
            ("env/contact_rate", ["env/contact_rate"]),
            ("env/object_motion", ["env/object_motion"]),
            ("env/success_rate", ["env/success_rate"]),
        ],
        smooth=smooth,
    )
    axes[4].set_xlabel(x_key)

    out_path = out_dir / "reward_scale_balance_v52.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _save_self_motion_external_bridge_v7(rows, out_dir, x_key="step", smooth: int = 10):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(5, 1, figsize=(13, 18), sharex=True)

    axes[0].set_title("Self-Motion Residual")
    _plot_subplot(
        axes[0],
        rows,
        x_key,
        [
            ("wm/actual_delta_norm_mean", ["wm/actual_delta_norm_mean"]),
            ("wm/self_delta_pred_norm_mean", ["wm/self_delta_pred_norm_mean"]),
            ("wm/external_residual_norm_mean", ["wm/external_residual_norm_mean"]),
            ("wm/self_motion_explained_ratio", ["wm/self_motion_explained_ratio"]),
            ("wm/self_motion_loss", ["wm/self_motion_loss"]),
        ],
        smooth=smooth,
    )

    axes[1].set_title("External Gate")
    _plot_subplot(
        axes[1],
        rows,
        x_key,
        [
            ("wm/external_effect_score_mean", ["wm/external_effect_score_mean"]),
            ("wm/external_effect_score_nonzero_rate", ["wm/external_effect_score_nonzero_rate"]),
            ("wm/external_effect_score_max", ["wm/external_effect_score_max"]),
        ],
        smooth=smooth,
    )

    axes[2].set_title("Reward Comparison")
    _plot_subplot(
        axes[2],
        rows,
        x_key,
        [
            ("wm/raw_slow_gain_reward_mean", ["wm/raw_slow_gain_reward_mean"]),
            ("wm/external_slow_gain_reward_mean", ["wm/external_slow_gain_reward_mean"]),
            ("actor/raw_slow_gain_intrinsic_mean", ["actor/raw_slow_gain_intrinsic_mean"]),
            ("actor/external_slow_gain_intrinsic_mean", ["actor/external_slow_gain_intrinsic_mean"]),
            ("actor/external_to_raw_intrinsic_ratio", ["actor/external_to_raw_intrinsic_ratio"]),
        ],
        smooth=smooth,
    )

    axes[3].set_title("Reward Balance")
    _plot_subplot(
        axes[3],
        rows,
        x_key,
        [
            ("actor/external_slow_gain_to_env_abs_ratio", ["actor/external_slow_gain_to_env_abs_ratio"]),
            (
                "actor/external_slow_gain_to_action_penalty_ratio",
                ["actor/external_slow_gain_to_action_penalty_ratio"],
            ),
            ("actor/intrinsic_to_env_abs_ratio", ["actor/intrinsic_to_env_abs_ratio"]),
            ("actor/intrinsic_to_action_penalty_ratio", ["actor/intrinsic_to_action_penalty_ratio"]),
        ],
        smooth=smooth,
    )

    axes[4].set_title("Behavior")
    _plot_subplot(
        axes[4],
        rows,
        x_key,
        [
            ("actor/action_norm", ["actor/action_norm", "action_norm_mean"]),
            ("actor/action_saturation_rate", ["actor/action_saturation_rate"]),
            ("env/contact_rate", ["env/contact_rate"]),
            ("env/object_motion", ["env/object_motion"]),
            ("env/success_rate", ["env/success_rate"]),
        ],
        smooth=smooth,
    )
    axes[4].set_xlabel(x_key)

    out_path = out_dir / "self_motion_external_bridge_v7.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _save_external_bridge_object_diagnostics_v7(rows, out_dir, x_key="step", smooth: int = 10):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(5, 1, figsize=(13, 18), sharex=True)

    axes[0].set_title("No Contact")
    _plot_subplot(
        axes[0],
        rows,
        x_key,
        [
            ("intrinsic/raw_slow_gain_reward_when_no_contact", ["intrinsic/raw_slow_gain_reward_when_no_contact"]),
            (
                "intrinsic/external_slow_gain_reward_when_no_contact",
                ["intrinsic/external_slow_gain_reward_when_no_contact"],
            ),
        ],
        smooth=smooth,
    )

    axes[1].set_title("Object Static")
    _plot_subplot(
        axes[1],
        rows,
        x_key,
        [
            (
                "intrinsic/raw_slow_gain_reward_when_object_static",
                ["intrinsic/raw_slow_gain_reward_when_object_static"],
            ),
            (
                "intrinsic/external_slow_gain_reward_when_object_static",
                ["intrinsic/external_slow_gain_reward_when_object_static"],
            ),
        ],
        smooth=smooth,
    )

    axes[2].set_title("Hand High / Object Static")
    _plot_subplot(
        axes[2],
        rows,
        x_key,
        [
            (
                "intrinsic/raw_slow_gain_reward_when_hand_high_object_static",
                ["intrinsic/raw_slow_gain_reward_when_hand_high_object_static"],
            ),
            (
                "intrinsic/external_slow_gain_reward_when_hand_high_object_static",
                ["intrinsic/external_slow_gain_reward_when_hand_high_object_static"],
            ),
        ],
        smooth=smooth,
    )

    axes[3].set_title("Object Moving")
    _plot_subplot(
        axes[3],
        rows,
        x_key,
        [
            (
                "intrinsic/raw_slow_gain_reward_when_object_moving",
                ["intrinsic/raw_slow_gain_reward_when_object_moving"],
            ),
            (
                "intrinsic/external_slow_gain_reward_when_object_moving",
                ["intrinsic/external_slow_gain_reward_when_object_moving"],
            ),
        ],
        smooth=smooth,
    )

    axes[4].set_title("Object/Contact Outcome")
    _plot_subplot(
        axes[4],
        rows,
        x_key,
        [
            ("env/object_motion", ["env/object_motion"]),
            ("env/contact_rate", ["env/contact_rate"]),
            ("env/success_rate", ["env/success_rate"]),
        ],
        smooth=smooth,
    )
    axes[4].set_xlabel(x_key)

    out_path = out_dir / "external_bridge_object_diagnostics_v7.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _save_rssm_pose_probe_diagnostics(rows, out_dir, x_key="step", smooth: int = 10):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(5, 1, figsize=(13, 18), sharex=True)

    axes[0].set_title("Hand Z Probe")
    _plot_subplot(
        axes[0],
        rows,
        x_key,
        [
            ("probe/encoder_hand_z_corr", ["probe/encoder_hand_z_corr"]),
            ("probe/rssm_hand_z_corr", ["probe/rssm_hand_z_corr"]),
            ("probe/encoder_hand_z_nmse", ["probe/encoder_hand_z_nmse"]),
            ("probe/rssm_hand_z_nmse", ["probe/rssm_hand_z_nmse"]),
        ],
        smooth=smooth,
    )

    axes[1].set_title("Hand Pos / Gripper")
    _plot_subplot(
        axes[1],
        rows,
        x_key,
        [
            ("probe/encoder_hand_pos_mse", ["probe/encoder_hand_pos_mse"]),
            ("probe/rssm_hand_pos_mse", ["probe/rssm_hand_pos_mse"]),
            ("probe/encoder_gripper_mse", ["probe/encoder_gripper_mse"]),
            ("probe/rssm_gripper_mse", ["probe/rssm_gripper_mse"]),
        ],
        smooth=smooth,
    )

    axes[2].set_title("Object Z Probe")
    _plot_subplot(
        axes[2],
        rows,
        x_key,
        [
            ("probe/encoder_object_z_corr", ["probe/encoder_object_z_corr"]),
            ("probe/rssm_object_z_corr", ["probe/rssm_object_z_corr"]),
            ("probe/encoder_object_z_nmse", ["probe/encoder_object_z_nmse"]),
            ("probe/rssm_object_z_nmse", ["probe/rssm_object_z_nmse"]),
        ],
        smooth=smooth,
    )

    axes[3].set_title("Hand-Object Distance")
    _plot_subplot(
        axes[3],
        rows,
        x_key,
        [
            ("probe/encoder_hand_object_distance_corr", ["probe/encoder_hand_object_distance_corr"]),
            ("probe/rssm_hand_object_distance_corr", ["probe/rssm_hand_object_distance_corr"]),
            ("probe/encoder_hand_object_distance_mse", ["probe/encoder_hand_object_distance_mse"]),
            ("probe/rssm_hand_object_distance_mse", ["probe/rssm_hand_object_distance_mse"]),
        ],
        smooth=smooth,
    )

    axes[4].set_title("Behavior Context")
    _plot_subplot(
        axes[4],
        rows,
        x_key,
        [
            ("actor/action_norm", ["actor/action_norm", "action_norm_mean"]),
            ("env/hand_z", ["env/hand_z"]),
            ("env/object_z", ["env/object_z"]),
            ("env/hand_object_distance", ["env/hand_object_distance"]),
            ("env/success_rate", ["env/success_rate"]),
        ],
        smooth=smooth,
    )
    axes[4].set_xlabel(x_key)

    out_path = out_dir / "rssm_pose_probe_diagnostics.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _save_proprio_rssm_diagnostics(rows, out_dir, x_key="step", smooth: int = 10):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(5, 1, figsize=(13, 18), sharex=True)

    axes[0].set_title("Proprio Auxiliary Loss")
    _plot_subplot(
        axes[0],
        rows,
        x_key,
        [
            ("wm/proprio_loss", ["wm/proprio_loss"]),
            ("wm/proprio_loss_weighted", ["wm/proprio_loss_weighted"]),
            ("wm/proprio_hand_pos_mse", ["wm/proprio_hand_pos_mse"]),
            ("wm/proprio_hand_z_mse", ["wm/proprio_hand_z_mse"]),
        ],
        smooth=smooth,
    )

    axes[1].set_title("Hand Z Probe")
    _plot_subplot(
        axes[1],
        rows,
        x_key,
        [
            ("probe/encoder_hand_z_corr", ["probe/encoder_hand_z_corr"]),
            ("probe/rssm_hand_z_corr", ["probe/rssm_hand_z_corr"]),
            ("probe/encoder_hand_z_nmse", ["probe/encoder_hand_z_nmse"]),
            ("probe/rssm_hand_z_nmse", ["probe/rssm_hand_z_nmse"]),
        ],
        smooth=smooth,
    )

    axes[2].set_title("Hand/Object Relation Probe")
    _plot_subplot(
        axes[2],
        rows,
        x_key,
        [
            ("probe/encoder_hand_object_distance_corr", ["probe/encoder_hand_object_distance_corr"]),
            ("probe/rssm_hand_object_distance_corr", ["probe/rssm_hand_object_distance_corr"]),
            ("probe/encoder_hand_object_distance_mse", ["probe/encoder_hand_object_distance_mse"]),
            ("probe/rssm_hand_object_distance_mse", ["probe/rssm_hand_object_distance_mse"]),
        ],
        smooth=smooth,
    )

    axes[3].set_title("Behavior / World Model Context")
    _plot_subplot(
        axes[3],
        rows,
        x_key,
        [
            ("actor/action_norm", ["actor/action_norm", "action_norm_mean"]),
            ("wm/slow_gain_reward_mean", ["wm/slow_gain_reward_mean", "intrinsic/slow_gain_reward_mean"]),
            ("wm/slow_gate_rate", ["wm/slow_gate_rate"]),
            ("wm/distill_loss", ["wm/distill_loss"]),
            ("env/success_rate", ["env/success_rate"]),
        ],
        smooth=smooth,
    )

    axes[4].set_title("Env Pose Context")
    _plot_subplot(
        axes[4],
        rows,
        x_key,
        [
            ("env/hand_z", ["env/hand_z"]),
            ("env/object_z", ["env/object_z"]),
            ("env/hand_object_distance", ["env/hand_object_distance"]),
        ],
        smooth=smooth,
    )
    axes[4].set_xlabel(x_key)

    out_path = out_dir / "proprio_rssm_diagnostics.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _save_v8_from_v7_memory_attention_reobservation(rows, out_dir, x_key="step", smooth: int = 10):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(6, 1, figsize=(13, 22), sharex=True)

    axes[0].set_title("V7 Bridge Retained")
    _plot_subplot(axes[0], rows, x_key, [
        ("wm/self_motion_loss", ["wm/self_motion_loss"]),
        ("wm/self_motion_explained_ratio", ["wm/self_motion_explained_ratio"]),
        ("wm/external_effect_score_mean", ["wm/external_effect_score_mean"]),
        ("wm/external_slow_gain_reward_mean", ["wm/external_slow_gain_reward_mean"]),
    ], smooth=smooth)

    axes[1].set_title("Memory")
    _plot_subplot(axes[1], rows, x_key, [
        ("wm/v8_memory_size", ["wm/v8_memory_size"]),
        ("wm/v8_memory_valid_rate", ["wm/v8_memory_valid_rate"]),
        ("wm/v8_same_lifetime_query_rate", ["wm/v8_same_lifetime_query_rate"]),
    ], smooth=smooth)

    axes[2].set_title("Attention")
    _plot_subplot(axes[2], rows, x_key, [
        ("wm/v8_attention_entropy", ["wm/v8_attention_entropy"]),
        ("wm/v8_attention_top1_weight", ["wm/v8_attention_top1_weight"]),
        ("wm/v8_attention_effective_memory_count", ["wm/v8_attention_effective_memory_count"]),
        ("wm/v8_selected_pose_dist", ["wm/v8_selected_pose_dist"]),
    ], smooth=smooth)

    axes[3].set_title("Confirmation")
    _plot_subplot(axes[3], rows, x_key, [
        ("wm/v8_pose_confidence_mean", ["wm/v8_pose_confidence_mean"]),
        ("wm/v8_pose_confirm_pass_rate", ["wm/v8_pose_confirm_pass_rate"]),
        ("wm/v8_visual_change_norm_mean", ["wm/v8_visual_change_norm_mean"]),
        ("wm/v8_visual_confirm_pass_rate", ["wm/v8_visual_confirm_pass_rate"]),
        ("wm/v8_confirmed_external_change_rate", ["wm/v8_confirmed_external_change_rate"]),
    ], smooth=smooth)

    axes[4].set_title("Reward")
    _plot_subplot(axes[4], rows, x_key, [
        ("wm/raw_slow_gain_reward_mean", ["wm/raw_slow_gain_reward_mean"]),
        ("wm/external_slow_gain_reward_mean", ["wm/external_slow_gain_reward_mean"]),
        ("wm/v8_recent_external_slow_sequence_score_mean", ["wm/v8_recent_external_slow_sequence_score_mean"]),
        ("wm/v8_confirm_reward_mean", ["wm/v8_confirm_reward_mean"]),
        ("actor/v8_confirm_intrinsic_mean", ["actor/v8_confirm_intrinsic_mean"]),
        ("actor/v8_confirm_to_env_abs_ratio", ["actor/v8_confirm_to_env_abs_ratio"]),
        ("actor/v8_confirm_to_action_penalty_ratio", ["actor/v8_confirm_to_action_penalty_ratio"]),
    ], smooth=smooth)

    axes[5].set_title("Behavior")
    _plot_subplot(axes[5], rows, x_key, [
        ("actor/action_norm", ["actor/action_norm", "action_norm_mean"]),
        ("actor/action_saturation_rate", ["actor/action_saturation_rate"]),
        ("env/contact_rate", ["env/contact_rate"]),
        ("env/object_motion", ["env/object_motion"]),
        ("env/success_rate", ["env/success_rate"]),
    ], smooth=smooth)
    axes[5].set_xlabel(x_key)

    out_path = out_dir / "v8_from_v7_memory_attention_reobservation.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _save_v8_from_v7_condition_diagnostics(rows, out_dir, x_key="step", smooth: int = 10):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(5, 1, figsize=(13, 18), sharex=True)

    axes[0].set_title("No Contact")
    _plot_subplot(axes[0], rows, x_key, [
        ("intrinsic/raw_slow_gain_reward_when_no_contact", ["intrinsic/raw_slow_gain_reward_when_no_contact"]),
        ("intrinsic/external_slow_gain_reward_when_no_contact", ["intrinsic/external_slow_gain_reward_when_no_contact"]),
        ("intrinsic/v8_confirm_reward_when_no_contact", ["intrinsic/v8_confirm_reward_when_no_contact"]),
        ("intrinsic/no_contact_v8_to_external_ratio", ["intrinsic/no_contact_v8_to_external_ratio"]),
    ], smooth=smooth)

    axes[1].set_title("Object Static")
    _plot_subplot(axes[1], rows, x_key, [
        ("intrinsic/raw_slow_gain_reward_when_object_static", ["intrinsic/raw_slow_gain_reward_when_object_static"]),
        ("intrinsic/external_slow_gain_reward_when_object_static", ["intrinsic/external_slow_gain_reward_when_object_static"]),
        ("intrinsic/v8_confirm_reward_when_object_static", ["intrinsic/v8_confirm_reward_when_object_static"]),
        ("intrinsic/object_static_v8_to_external_ratio", ["intrinsic/object_static_v8_to_external_ratio"]),
    ], smooth=smooth)

    axes[2].set_title("Hand High / Object Static")
    _plot_subplot(axes[2], rows, x_key, [
        ("intrinsic/raw_slow_gain_reward_when_hand_high_object_static", ["intrinsic/raw_slow_gain_reward_when_hand_high_object_static"]),
        ("intrinsic/external_slow_gain_reward_when_hand_high_object_static", ["intrinsic/external_slow_gain_reward_when_hand_high_object_static"]),
        ("intrinsic/v8_confirm_reward_when_hand_high_object_static", ["intrinsic/v8_confirm_reward_when_hand_high_object_static"]),
        ("intrinsic/hand_high_object_static_v8_to_external_ratio", ["intrinsic/hand_high_object_static_v8_to_external_ratio"]),
    ], smooth=smooth)

    axes[3].set_title("Object Moving")
    _plot_subplot(axes[3], rows, x_key, [
        ("intrinsic/raw_slow_gain_reward_when_object_moving", ["intrinsic/raw_slow_gain_reward_when_object_moving"]),
        ("intrinsic/external_slow_gain_reward_when_object_moving", ["intrinsic/external_slow_gain_reward_when_object_moving"]),
        ("intrinsic/v8_confirm_reward_when_object_moving", ["intrinsic/v8_confirm_reward_when_object_moving"]),
        ("intrinsic/object_moving_v8_to_external_ratio", ["intrinsic/object_moving_v8_to_external_ratio"]),
    ], smooth=smooth)

    axes[4].set_title("Operator / Outcome")
    _plot_subplot(axes[4], rows, x_key, [
        ("wm/v8_operator_candidate_rate", ["wm/v8_operator_candidate_rate"]),
        ("wm/v8_operator_memory_age_mean", ["wm/v8_operator_memory_age_mean"]),
        ("env/object_motion", ["env/object_motion"]),
        ("env/success_rate", ["env/success_rate"]),
    ], smooth=smooth)
    axes[4].set_xlabel(x_key)

    out_path = out_dir / "v8_from_v7_condition_diagnostics.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_policy_metrics_jsonl(path, out_dir, x_key="step", smooth: int = 10):
    path = Path(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, series = _load_policy_metric_rows(path, x_key)

    saved = []
    for key, values in series.items():
        saved.append(
            save_line(
                xs=values["xs"],
                ys=values["ys"],
                path=out_dir / f"{key}.png",
                xlabel=x_key,
                ylabel=key,
                title=key,
            )
        )
    if rows:
        saved.append(_save_fast_slow_diagnostics(rows, out_dir, x_key=x_key, smooth=int(smooth)))
        saved.append(_save_fast_slow_critical_region(rows, out_dir, x_key=x_key, smooth=int(smooth)))
        saved.append(_save_slow_gain_reward_alignment(rows, out_dir, x_key=x_key, smooth=int(smooth)))
        saved.append(_save_fast_slow_diagnostics_v5(rows, out_dir, x_key=x_key, smooth=int(smooth)))
        saved.append(_save_event_gate_replacement_v5(rows, out_dir, x_key=x_key, smooth=int(smooth)))
        saved.append(_save_event_replay_head_diagnostics(rows, out_dir, x_key=x_key, smooth=int(smooth)))
        saved.append(_save_slow_gain_reward_diagnostics_v52(rows, out_dir, x_key=x_key, smooth=int(smooth)))
        saved.append(_save_reward_scale_balance_v52(rows, out_dir, x_key=x_key, smooth=int(smooth)))
        saved.append(_save_self_motion_external_bridge_v7(rows, out_dir, x_key=x_key, smooth=int(smooth)))
        saved.append(_save_external_bridge_object_diagnostics_v7(rows, out_dir, x_key=x_key, smooth=int(smooth)))
        saved.append(_save_rssm_pose_probe_diagnostics(rows, out_dir, x_key=x_key, smooth=int(smooth)))
        saved.append(_save_proprio_rssm_diagnostics(rows, out_dir, x_key=x_key, smooth=int(smooth)))
        saved.append(_save_v8_from_v7_memory_attention_reobservation(rows, out_dir, x_key=x_key, smooth=int(smooth)))
        saved.append(_save_v8_from_v7_condition_diagnostics(rows, out_dir, x_key=x_key, smooth=int(smooth)))
    return saved


def _parse_args():
    parser = argparse.ArgumentParser(description="Plot policy_metrics.jsonl and fast/slow diagnostics.")
    parser.add_argument("jsonl_path", help="Path to policy_metrics.jsonl")
    parser.add_argument("out_dir", help="Output directory for plots")
    parser.add_argument("--x-key", default="step", help="X-axis key, default: step")
    parser.add_argument("--smooth", type=int, default=10, help="Rolling mean window, default: 10")
    return parser.parse_args()


def main():
    args = _parse_args()
    paths = plot_policy_metrics_jsonl(args.jsonl_path, args.out_dir, x_key=args.x_key, smooth=args.smooth)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
