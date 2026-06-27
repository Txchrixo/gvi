"""Tiny synthetic 2D platformer dataset generator.

This is not a replacement for real data. It lets you verify the training path,
YOLO format and review UI before collecting hundreds of examples.
"""
from __future__ import annotations

import random
import uuid
from pathlib import Path

from PIL import Image, ImageDraw

from gvi.training.annotations import AnnotationFile, AnnotationObject, Polygon
from gvi.training.overlay import draw_overlay
from gvi.training.taxonomy import Taxonomy


def generate_synthetic_platformer(dataset_root: Path, count: int = 50, split: str = "train", seed: int = 42) -> dict[str, int]:
    rng = random.Random(seed)
    dataset_root = Path(dataset_root)
    taxonomy = Taxonomy.load(dataset_root / "classes.json" if (dataset_root / "classes.json").exists() else "platformer")
    class_to_id = taxonomy.class_to_id
    img_dir = dataset_root / "images" / split
    lab_dir = dataset_root / "labels" / split
    ann_dir = dataset_root / "annotations" / "gvi"
    overlay_dir = dataset_root / "overlays"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(count):
        w, h = 640, 360
        im = Image.new("RGB", (w, h), (26, 29, 82))
        draw = ImageDraw.Draw(im)
        objects: list[AnnotationObject] = []
        # Background wall stripes.
        for x in range(0, w, 90):
            draw.rectangle([x, 0, x + 8, h], fill=(58, 65, 190))
        # Platforms.
        for _ in range(rng.randint(4, 8)):
            px = rng.randint(20, w - 180)
            py = rng.randint(80, h - 60)
            pw = rng.randint(80, 180)
            ph = rng.randint(14, 30)
            draw.rectangle([px, py, px + pw, py + ph], fill=(7, 7, 22), outline=(230, 240, 255), width=2)
            objects.append(_obj("platform", px, py, pw, ph, 0.98))
        # Ladders.
        for _ in range(rng.randint(1, 3)):
            lx = rng.randint(30, w - 50)
            ly = rng.randint(60, h - 120)
            lh = rng.randint(60, 120)
            draw.line([lx, ly, lx, ly + lh], fill=(230, 240, 255), width=3)
            draw.line([lx + 25, ly, lx + 25, ly + lh], fill=(230, 240, 255), width=3)
            for yy in range(ly + 8, ly + lh, 12):
                draw.line([lx, yy, lx + 25, yy], fill=(230, 240, 255), width=2)
            objects.append(_obj("ladder", lx, ly, 25, lh, 0.98))
        # Spikes.
        for _ in range(rng.randint(1, 4)):
            sx = rng.randint(40, w - 40)
            sy = rng.randint(h // 2, h - 30)
            pts = [(sx, sy + 24), (sx + 14, sy), (sx + 28, sy + 24)]
            draw.polygon(pts, fill=(245, 245, 255), outline=(5, 5, 15))
            objects.append(_obj("spike", sx, sy, 28, 24, 0.98, pts))
        # Door.
        dx = rng.randint(20, w - 60)
        dy = h - 70
        draw.rectangle([dx, dy, dx + 32, dy + 58], fill=(8, 8, 45), outline=(240, 240, 255), width=2)
        objects.append(_obj("door", dx, dy, 32, 58, 0.98))
        # Enemies / pickups.
        for _ in range(rng.randint(1, 3)):
            ex = rng.randint(50, w - 50)
            ey = rng.randint(60, h - 80)
            draw.rectangle([ex, ey, ex + 32, ey + 24], fill=(255, 116, 47), outline=(250, 250, 250), width=2)
            objects.append(_obj("enemy", ex, ey, 32, 24, 0.97))
        for _ in range(rng.randint(1, 3)):
            cx = rng.randint(40, w - 40)
            cy = rng.randint(30, h - 90)
            draw.ellipse([cx, cy, cx + 12, cy + 12], fill=(155, 170, 255), outline=(245, 245, 255))
            objects.append(_obj("pickup", cx, cy, 12, 12, 0.97))
        image_path = img_dir / f"synthetic_{idx:04d}.png"
        im.save(image_path)
        ann = AnnotationFile(image_path=image_path, width=w, height=h, split=split, objects=objects)  # type: ignore[arg-type]
        ann.write_json(ann_dir / f"synthetic_{idx:04d}.json")
        ann.write_yolo_seg(lab_dir / f"synthetic_{idx:04d}.txt", class_to_id)
        draw_overlay(ann, overlay_dir / f"synthetic_{idx:04d}_overlay.jpg")
    return {"generated": count}


def _obj(cls: str, x: int, y: int, w: int, h: int, conf: float, points=None) -> AnnotationObject:
    if points is None:
        points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    return AnnotationObject(
        id=f"syn_{uuid.uuid4().hex[:8]}",
        class_name=cls,
        bbox_xywh=(float(x), float(y), float(w), float(h)),
        polygon=Polygon(points=[(float(px), float(py)) for px, py in points]),
        confidence=conf,
        source="synthetic",
    )
