"""Annotation schemas used by GVI training and auto-label loops."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class Polygon(BaseModel):
    """A polygon represented as pixel coordinates [[x, y], ...]."""

    points: list[tuple[float, float]] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.points) >= 3


class AnnotationObject(BaseModel):
    """One object predicted or annotated in an image."""

    id: str
    class_name: str
    class_id: int | None = None
    bbox_xywh: tuple[float, float, float, float]
    polygon: Polygon | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = "human"
    godot_candidates: list[str] = Field(default_factory=list)
    needs_review: bool = False
    review_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("bbox_xywh")
    @classmethod
    def _bbox_positive(cls, value: tuple[float, float, float, float]):
        x, y, w, h = value
        if w <= 0 or h <= 0:
            raise ValueError("bbox width and height must be positive")
        return value

    def yolo_seg_line(self, image_width: int, image_height: int, fallback_box: bool = True) -> str:
        """Return one YOLO segmentation line.

        YOLO segmentation labels are: class x1 y1 x2 y2 ... normalized. If a
        polygon is unavailable and fallback_box is true, the rectangle bbox is
        converted to a 4-point polygon so training can still begin.
        """
        if self.class_id is None:
            raise ValueError(f"class_id missing for {self.class_name}")
        points: list[tuple[float, float]] = []
        if self.polygon and self.polygon.is_valid:
            points = self.polygon.points
        elif fallback_box:
            x, y, w, h = self.bbox_xywh
            points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        else:
            raise ValueError("polygon missing")
        values: list[str] = [str(self.class_id)]
        for px, py in points:
            nx = min(max(px / max(image_width, 1), 0.0), 1.0)
            ny = min(max(py / max(image_height, 1), 0.0), 1.0)
            values.append(f"{nx:.6f}")
            values.append(f"{ny:.6f}")
        return " ".join(values)


class AnnotationFile(BaseModel):
    """GVI JSON annotation file for one source image."""

    image_path: Path
    width: int
    height: int
    split: Literal["train", "val", "test", "raw", "unassigned"] = "unassigned"
    objects: list[AnnotationObject] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def write_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def read_json(cls, path: Path) -> "AnnotationFile":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def write_yolo_seg(self, label_path: Path, class_to_id: dict[str, int]) -> Path:
        label_path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        for obj in self.objects:
            if obj.class_name not in class_to_id:
                continue
            obj.class_id = class_to_id[obj.class_name]
            lines.append(obj.yolo_seg_line(self.width, self.height))
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return label_path


class ReviewItem(BaseModel):
    id: str
    image_path: Path
    object_id: str | None = None
    class_name: str | None = None
    confidence: float | None = None
    reason: str
    priority: Literal["low", "medium", "high", "critical"] = "medium"
    suggested_actions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReviewFile(BaseModel):
    dataset_root: Path
    items: list[ReviewItem] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)

    def write_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path
