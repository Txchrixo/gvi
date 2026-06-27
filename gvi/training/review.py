"""Review and labelling tool helpers."""
from __future__ import annotations

from pathlib import Path


def write_cvat_quickstart(dataset_root: Path) -> Path:
    path = dataset_root / "review" / "CVAT_QUICKSTART.md"
    text = f"""# CVAT quickstart for this GVI dataset

1. Start CVAT with Docker from the official CVAT repository or hosted CVAT.
2. Create a new segmentation task.
3. Upload images from:

```text
{(dataset_root / 'images' / 'train').as_posix()}
{(dataset_root / 'raw').as_posix()}
```

4. Import labels using YOLO/COCO if available.
5. Correct masks/classes.
6. Export as YOLO segmentation or COCO.
7. Place corrected labels back into `labels/train`, `labels/val`, `labels/test`.

GVI review file:

```text
{(dataset_root / 'review' / 'review.json').as_posix()}
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
