"""YOLO training launcher."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TrainConfig:
    dataset_root: Path
    model: str = "yolo11n-seg.pt"
    epochs: int = 80
    imgsz: int = 640
    batch: int = 8
    device: str | None = None
    workers: int = 4
    project: Path | None = None
    name: str = "gvi_platformer_yolo"
    dry_run: bool = False

    def __post_init__(self) -> None:
        # Accept str paths from programmatic callers, not just typer-converted Paths.
        # Resolve to absolute: Ultralytics resolves a relative project= against its
        # own global runs_dir setting, not the process cwd, which silently relocates
        # output outside the dataset folder.
        self.dataset_root = Path(self.dataset_root).resolve()
        if self.project is not None:
            self.project = Path(self.project).resolve()

    def command(self) -> list[str]:
        data_yaml = self.dataset_root / "data.yaml"
        cmd = [
            "yolo",
            "segment",
            "train",
            f"model={self.model}",
            f"data={data_yaml}",
            f"epochs={self.epochs}",
            f"imgsz={self.imgsz}",
            f"batch={self.batch}",
            f"workers={self.workers}",
            f"project={self.project or (self.dataset_root / 'runs')}",
            f"name={self.name}",
        ]
        if self.device:
            cmd.append(f"device={self.device}")
        return cmd


def train_yolo(config: TrainConfig) -> dict[str, object]:
    data_yaml = config.dataset_root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing {data_yaml}. Run: gvi dataset init ...")
    cmd = config.command()
    if config.dry_run:
        return {"dry_run": True, "command": cmd}
    try:
        import ultralytics  # noqa: F401  # type: ignore
    except Exception as exc:
        raise RuntimeError("Install training deps first: python -m pip install -e '.[training]'") from exc
    proc = subprocess.run(cmd, check=False)
    ok = proc.returncode == 0
    return {
        "ok": ok,
        "returncode": proc.returncode,
        "command": cmd,
        "expected_best": str((config.project or (config.dataset_root / 'runs')) / config.name / "weights" / "best.pt"),
    }
