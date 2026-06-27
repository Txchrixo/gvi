"""Evaluation helpers for trained student models and dataset health."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from gvi.training.dataset_builder import dataset_stats


def evaluate_dataset(dataset_root: Path) -> dict[str, object]:
    dataset_root = Path(dataset_root)
    stats = dataset_stats(dataset_root)
    warnings: list[str] = []
    splits = stats.get("splits", {})
    if isinstance(splits, dict):
        train = splits.get("train", {}).get("images", 0) if isinstance(splits.get("train"), dict) else 0
        val = splits.get("val", {}).get("images", 0) if isinstance(splits.get("val"), dict) else 0
        if train < 50:
            warnings.append("Prototype dataset: add at least 50 training images.")
        if val < 10:
            warnings.append("Validation set is small; metrics will be unstable.")
    return {"stats": stats, "warnings": warnings}


def evaluate_yolo(dataset_root: Path, model: Path, imgsz: int = 640, split: str = "val", dry_run: bool = False) -> dict[str, object]:
    cmd = ["yolo", "segment", "val", f"model={model}", f"data={dataset_root / 'data.yaml'}", f"imgsz={imgsz}", f"split={split}"]
    if dry_run:
        return {"dry_run": True, "command": cmd}
    proc = subprocess.run(cmd, check=False)
    return {"ok": proc.returncode == 0, "returncode": proc.returncode, "command": cmd}


def read_results_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"path": str(path), "warning": "Could not parse results JSON"}
