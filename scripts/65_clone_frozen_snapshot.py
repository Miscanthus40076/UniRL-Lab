from __future__ import annotations

import argparse
from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _snapshot_root(kind: str) -> Path:
    if kind == "policy":
        return PROJECT_ROOT / "policy" / "versions"
    if kind == "env":
        return PROJECT_ROOT / "sim_env" / "envs" / "versions"
    raise ValueError(f"Unsupported kind: {kind}")


def clone_snapshot(kind: str, src_version: str, dst_version: str, component: str | None = None):
    root = _snapshot_root(kind)
    src_root = root / src_version
    dst_root = root / dst_version
    if not src_root.exists():
        raise FileNotFoundError(f"Source version not found: {src_root}")
    dst_root.mkdir(parents=True, exist_ok=True)

    if component is None:
        targets = [path for path in src_root.iterdir() if path.name != "__pycache__"]
    else:
        targets = [src_root / component]

    for src in targets:
        if not src.exists():
            raise FileNotFoundError(f"Component not found: {src}")
        dst = dst_root / src.name
        if dst.exists():
            raise FileExistsError(f"Destination already exists: {dst}")
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(src, dst)
        print(f"cloned {src} -> {dst}")


def parse_args():
    parser = argparse.ArgumentParser(description="Clone a frozen policy/env snapshot into a new version directory.")
    parser.add_argument("--kind", choices=["policy", "env"], required=True)
    parser.add_argument("--src-version", required=True)
    parser.add_argument("--dst-version", required=True)
    parser.add_argument("--component", default=None, help="Optional single component name to clone.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    clone_snapshot(
        kind=args.kind,
        src_version=args.src_version,
        dst_version=args.dst_version,
        component=args.component,
    )

