"""Active-learning utilities: select the most useful samples to correct."""
from __future__ import annotations

from pathlib import Path

from gvi.training.annotations import AnnotationFile, ReviewFile, ReviewItem


def build_review_file(dataset_root: Path, threshold: float = 0.72, max_items: int = 200) -> ReviewFile:
    dataset_root = Path(dataset_root).resolve()
    items: list[ReviewItem] = []
    for ann_path in sorted((dataset_root / "annotations" / "gvi").glob("*.json")):
        ann = AnnotationFile.read_json(ann_path)
        for obj in ann.objects:
            if obj.needs_review or obj.confidence < threshold:
                priority = "critical" if obj.confidence < 0.35 else "high" if obj.confidence < 0.55 else "medium"
                items.append(
                    ReviewItem(
                        id=f"review_{len(items)+1:05d}",
                        image_path=ann.image_path,
                        object_id=obj.id,
                        class_name=obj.class_name,
                        confidence=obj.confidence,
                        reason=obj.review_reason or f"confidence below threshold {threshold}",
                        priority=priority,  # type: ignore[arg-type]
                        suggested_actions=["verify_class", "adjust_mask", "confirm_godot_node"],
                    )
                )
    items.sort(key=lambda item: (item.confidence if item.confidence is not None else 1.0))
    items = items[:max_items]
    review = ReviewFile(
        dataset_root=dataset_root,
        items=items,
        summary={"threshold": threshold, "items": len(items), "strategy": "low_confidence_first"},
    )
    review.write_json(dataset_root / "review" / "review.json")
    return review
