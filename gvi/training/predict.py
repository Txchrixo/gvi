"""Run trained student predictions and write GVI annotation files."""
from __future__ import annotations

from pathlib import Path

from gvi.training.autolabel import YoloTeacherBackend
from gvi.training.overlay import draw_overlay
from gvi.training.taxonomy import Taxonomy


def predict_image(image_path: Path, model_path: Path, out_dir: Path, taxonomy_name: str = "platformer", conf: float = 0.25) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    taxonomy = Taxonomy.load(taxonomy_name)
    backend = YoloTeacherBackend(model_path, conf=conf)
    ann = backend.label_image(image_path, taxonomy)
    ann_path = out_dir / f"{image_path.stem}_predictions.json"
    overlay = out_dir / f"{image_path.stem}_overlay.jpg"
    ann.write_json(ann_path)
    draw_overlay(ann, overlay)
    return {"predictions": str(ann_path), "overlay": str(overlay)}
