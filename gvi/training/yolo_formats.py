"""YOLO format helpers."""
from __future__ import annotations

import shutil
from pathlib import Path

from gvi.training.annotations import AnnotationFile
from gvi.training.taxonomy import Taxonomy


def export_gvi_annotations_to_yolo(dataset_root: Path, taxonomy: Taxonomy | None = None) -> int:
    """Export annotations/gvi/*.json into labels/{split}/*.txt.

    Returns the number of label files written.
    """
    dataset_root = Path(dataset_root).resolve()
    taxonomy = taxonomy or Taxonomy.load(dataset_root / "classes.json")
    count = 0
    for ann_path in (dataset_root / "annotations" / "gvi").glob("*.json"):
        ann = AnnotationFile.read_json(ann_path)
        split = ann.split if ann.split in {"train", "val", "test"} else "train"
        label_path = dataset_root / "labels" / split / f"{Path(ann.image_path).stem}.txt"
        ann.write_yolo_seg(label_path, taxonomy.class_to_id)
        # Ensure image exists in matching split for YOLO.
        img_src = Path(ann.image_path)
        img_dest = dataset_root / "images" / split / img_src.name
        if img_src.exists() and img_src.resolve() != img_dest.resolve():
            img_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_src, img_dest)
        count += 1
    return count
