from pathlib import Path
import json

import matplotlib.pyplot as plt


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


def plot_policy_metrics_jsonl(path, out_dir, x_key="step"):
    path = Path(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
            merged_row = {x_key: train_scalars.get(x_key, record.get(x_key))}
            merged_row.update({f"policy/{key}": value for key, value in policy_scalars.items()})
            _append_numeric_point(series, merged_row, x_key)

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
    return saved
