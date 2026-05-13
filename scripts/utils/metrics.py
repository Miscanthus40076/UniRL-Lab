from pathlib import Path
import csv
import json


def scalar(value):
    if isinstance(value, (int, float)):
        return float(value)

    item = getattr(value, "item", None)
    if callable(item):
        value = item()
        if isinstance(value, (int, float)):
            return float(value)

    raise TypeError(f"Metric value must be scalar-convertible, got {type(value)!r}")


def _norm_scalar_dict(metrics, field_name="metrics"):
    if metrics is None:
        return {}
    if not isinstance(metrics, dict):
        raise TypeError(f"{field_name} must be a dict, got {type(metrics)!r}")

    out = {}
    for key, value in metrics.items():
        if not isinstance(key, str):
            raise TypeError(f"Metric key must be str, got {type(key)!r}")
        out[key] = scalar(value)
    return out


def normalize_policy_metrics(payload):
    if payload is None:
        return {
            "schema": "policy_metrics/v1",
            "scalars": {},
            "metadata": {},
        }
    if not isinstance(payload, dict):
        raise TypeError(f"Policy metrics payload must be a dict, got {type(payload)!r}")

    if "schema" not in payload and "scalars" not in payload and "metadata" not in payload:
        return {
            "schema": "policy_metrics/v1",
            "scalars": _norm_scalar_dict(payload, field_name="metrics"),
            "metadata": {},
        }

    schema = payload.get("schema", "policy_metrics/v1")
    if schema != "policy_metrics/v1":
        raise ValueError(f"Unsupported policy metrics schema: {schema}")

    scalars = _norm_scalar_dict(payload.get("scalars", {}), field_name="scalars")
    metadata = payload.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise TypeError(f"Policy metadata must be a dict, got {type(metadata)!r}")

    return {
        "schema": schema,
        "scalars": scalars,
        "metadata": metadata,
    }


def metrics_row_from_payload(payload):
    return normalize_policy_metrics(payload)["scalars"]


def norm_metrics(metrics):
    return metrics_row_from_payload(metrics)


class CsvLog:
    def __init__(self, path):
        self.path = Path(path)
        self.rows = []
        self.keys = []

    def add(self, row):
        clean = {}
        for key, value in row.items():
            clean[key] = float(value) if isinstance(value, (int, float)) else value
            if key not in self.keys:
                self.keys.append(key)

        self.rows.append(clean)
        self.flush()

    def flush(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # TODO: 这里每次 add 都全量重写整个 CSV；长训练会带来明显 I/O 和内存开销，需要改成 append 或分批 flush。
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.keys)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)


class JsonlLog:
    def __init__(self, path):
        self.path = Path(path)

    def add(self, payload):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
