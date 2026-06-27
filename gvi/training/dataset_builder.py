"""Dataset creation and health checks for YOLO segmentation training."""
from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

from gvi.training.annotations import AnnotationFile
from gvi.training.taxonomy import DatasetConfig, Taxonomy


DATASET_DIRS = [
    "raw",
    "images/train",
    "images/val",
    "images/test",
    "labels/train",
    "labels/val",
    "labels/test",
    "annotations/gvi",
    "annotations/coco",
    "autolabel",
    "predictions",
    "review",
    "overlays",
    "models",
    "runs",
    "exports/godot",
]


def init_dataset(dataset_root: Path, taxonomy_name: str = "platformer", force: bool = False) -> dict[str, Path]:
    """Create a GVI training dataset scaffold."""
    dataset_root = Path(dataset_root).resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)
    for rel in DATASET_DIRS:
        (dataset_root / rel).mkdir(parents=True, exist_ok=True)
    taxonomy = Taxonomy.load(taxonomy_name)
    taxonomy.write_yolo_data_yaml(dataset_root)
    taxonomy.write_json(dataset_root / "classes.json")
    config = DatasetConfig(dataset_root=dataset_root, taxonomy_name=taxonomy.name)
    (dataset_root / "gvi_dataset_config.json").write_text(config.model_dump_json(indent=2), encoding="utf-8")
    readme = dataset_root / "README_DATASET.md"
    if force or not readme.exists():
        readme.write_text(_dataset_readme(taxonomy), encoding="utf-8")
    card = dataset_root / "DATASET_CARD.md"
    if force or not card.exists():
        card.write_text(_dataset_card(taxonomy), encoding="utf-8")
    return {
        "dataset_root": dataset_root,
        "data_yaml": dataset_root / "data.yaml",
        "classes_json": dataset_root / "classes.json",
        "dataset_card": card,
    }


def _dataset_readme(taxonomy: Taxonomy) -> str:
    classes = ", ".join(c.name for c in taxonomy.classes)
    return f"""# GVI training dataset

Task: YOLO segmentation for Godot scene reconstruction.

Classes: {classes}

Recommended workflow:

```bash
gvi dataset init --type {taxonomy.name} ./dataset
gvi autolabel ./dataset/raw --dataset ./dataset --backend heuristic --classes platform ladder spike door enemy pickup
gvi dataset stats ./dataset
gvi review ./dataset
gvi train ./dataset --model yolo11n-seg.pt --epochs 80 --imgsz 640
```

Folders:

- `raw/`: unlabelled screenshots, PSD renders or mockups.
- `images/{{train,val,test}}`: training images.
- `labels/{{train,val,test}}`: YOLO segmentation labels.
- `annotations/gvi`: rich GVI JSON labels.
- `autolabel/`: teacher outputs before human correction.
- `review/`: active learning review files.
- `overlays/`: visual QA overlays.
- `models/`: trained weights.
"""


def _dataset_card(taxonomy: Taxonomy) -> str:
    lines = "\n".join([f"- `{c.name}`: {c.description}" for c in taxonomy.classes])
    return f"""# Dataset Card — GVI {taxonomy.name}

## Goal
Train a student segmentation model that recognizes 2D game / UI elements and
converts them into reliable Godot scene IR objects.

## Classes
{lines}

## Sources
Document every source you collect. Respect game licenses and avoid shipping
third-party copyrighted assets in public model demos unless you have permission.
Use synthetic examples and your own assets for public releases.

## Annotation rules
- Use tight masks for interactive elements.
- Use full visible object masks for enemies, pickups and doors.
- Use collision-relevant masks for platforms, walls and hazards.
- Put ambiguous elements in `needs_review` instead of guessing.

## Quality gates
- Minimum 50 images for prototype.
- 200+ corrected images before a demo model.
- Track mAP, per-class recall, false positives and GVI fidelity error.
"""


def ingest_raw_images(dataset_root: Path, source: Path, split: bool = False, seed: int = 42) -> list[Path]:
    """Copy images into raw/ or split them into train/val/test image folders."""
    dataset_root = Path(dataset_root).resolve()
    source = Path(source).resolve()
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    images = [p for p in (source.rglob("*") if source.is_dir() else [source]) if p.suffix.lower() in exts]
    copied: list[Path] = []
    random.Random(seed).shuffle(images)
    for idx, img in enumerate(images):
        if split:
            r = idx / max(len(images), 1)
            part = "train" if r < 0.8 else "val" if r < 0.95 else "test"
            dest = dataset_root / "images" / part / img.name
        else:
            dest = dataset_root / "raw" / img.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, dest)
        copied.append(dest)
    return copied


def dataset_stats(dataset_root: Path) -> dict[str, object]:
    dataset_root = Path(dataset_root).resolve()
    stats: dict[str, object] = {"dataset_root": str(dataset_root), "splits": {}, "classes": {}}
    class_counts: Counter[str] = Counter()
    for split in ["train", "val", "test"]:
        images = list((dataset_root / "images" / split).glob("*"))
        labels = list((dataset_root / "labels" / split).glob("*.txt"))
        stats["splits"][split] = {"images": len(images), "labels": len(labels)}  # type: ignore[index]
    classes = {}
    classes_json = dataset_root / "classes.json"
    if classes_json.exists():
        data = json.loads(classes_json.read_text(encoding="utf-8"))
        classes = {str(c["id"]): c["name"] for c in data.get("classes", [])}
    for label_file in (dataset_root / "labels").rglob("*.txt"):
        for line in label_file.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if parts:
                class_counts[classes.get(parts[0], parts[0])] += 1
    stats["classes"] = dict(class_counts)
    raw_count = len([p for p in (dataset_root / "raw").glob("*") if p.is_file()])
    stats["raw_images"] = raw_count
    return stats


def make_empty_annotation(image_path: Path, split: str = "unassigned") -> AnnotationFile:
    with Image.open(image_path) as im:
        w, h = im.size
    return AnnotationFile(image_path=image_path, width=w, height=h, split=split)  # type: ignore[arg-type]
