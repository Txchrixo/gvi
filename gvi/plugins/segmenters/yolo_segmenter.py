"""YOLO-based semantic detector.

Uses Ultralytics YOLO (default ``yolov8n``) for instance segmentation /
object detection. Provides per-element class labels (person, car, chair,
book, ...) that downstream plugins and exporters can use to pick more
appropriate Godot node types (CharacterBody2D for people, etc.).

If a YOLO-seg model is available it returns polygon masks; otherwise
falls back to the bounding-box variants. Weights are auto-downloaded
and cached in ``~/.cache/gvi/models/yolo/``.
"""
from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path

import numpy as np

from gvi.core.plugin import Capability, Plugin, PluginContext
from gvi.core.types import (
    AssetProfile,
    CapabilityType,
    DetectedElement,
    PipelineStepResult,
    SegmentationResult,
)

_YOLO_MODEL_FILES = {
    "yolov8n": "yolov8n-seg.pt",  # nano with segmentation
    "yolov8s": "yolov8s-seg.pt",
    "yolov11n": "yolo11n-seg.pt",
}

# COCO-80 -> friendly semantic class for Godot node picking.
_YOLO_TO_GODOT_CLASS = {
    "person": "character",
    "bicycle": "vehicle",
    "car": "vehicle",
    "motorcycle": "vehicle",
    "airplane": "vehicle",
    "bus": "vehicle",
    "train": "vehicle",
    "truck": "vehicle",
    "boat": "vehicle",
    "traffic light": "prop",
    "fire hydrant": "prop",
    "stop sign": "prop",
    "parking meter": "prop",
    "bench": "prop",
    "bird": "character",
    "cat": "character",
    "dog": "character",
    "horse": "character",
    "sheep": "character",
    "cow": "character",
    "elephant": "character",
    "bear": "character",
    "zebra": "character",
    "giraffe": "character",
    "backpack": "prop",
    "umbrella": "prop",
    "handbag": "prop",
    "tie": "prop",
    "suitcase": "prop",
    "frisbee": "prop",
    "skis": "prop",
    "snowboard": "prop",
    "sports ball": "prop",
    "kite": "prop",
    "baseball bat": "prop",
    "baseball glove": "prop",
    "skateboard": "prop",
    "surfboard": "prop",
    "tennis racket": "prop",
    "bottle": "prop",
    "wine glass": "prop",
    "cup": "prop",
    "fork": "prop",
    "knife": "prop",
    "spoon": "prop",
    "bowl": "prop",
    "banana": "prop",
    "apple": "prop",
    "sandwich": "prop",
    "orange": "prop",
    "broccoli": "prop",
    "carrot": "prop",
    "hot dog": "prop",
    "pizza": "prop",
    "donut": "prop",
    "cake": "prop",
    "chair": "prop",
    "couch": "prop",
    "potted plant": "prop",
    "bed": "prop",
    "dining table": "prop",
    "toilet": "prop",
    "tv": "prop",
    "laptop": "prop",
    "mouse": "prop",
    "remote": "prop",
    "keyboard": "prop",
    "cell phone": "prop",
    "microwave": "prop",
    "oven": "prop",
    "toaster": "prop",
    "sink": "prop",
    "refrigerator": "prop",
    "book": "prop",
    "clock": "prop",
    "vase": "prop",
    "scissors": "prop",
    "teddy bear": "prop",
    "hair drier": "prop",
    "toothbrush": "prop",
}


class YOLOSemanticSegmenter(Plugin):
    def __init__(self) -> None:
        self._model = None
        self._loaded_key: str | None = None
        self._last_init_error: str | None = None
        self._try_init("yolov8n")

    def _try_init(self, key: str) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # noqa: BLE001
            self._last_init_error = f"ultralytics missing: {exc}"
            return
        cache_dir = Path.home() / ".cache" / "gvi" / "models" / "yolo"
        cache_dir.mkdir(parents=True, exist_ok=True)
        weight_name = _YOLO_MODEL_FILES.get(key, "yolov8n-seg.pt")
        try:
            self._model = YOLO(weight_name)
            self._loaded_key = key
            self._last_init_error = None
        except Exception as exc:  # noqa: BLE001
            self._last_init_error = f"YOLO load failed: {exc}"
            self._model = None

    def capability(self) -> Capability:
        return Capability(
            id="segmenter.yolo",
            type=CapabilityType.SEMANTIC,
            name="YOLO semantic instance segmenter",
            supported_mime_types=["image/png", "image/jpeg", "image/webp", "image/bmp"],
            supported_extensions=[".png", ".jpg", ".jpeg", ".webp", ".bmp"],
            priority=88,
            provides=["segmentation", "semantic_labels", "elements", "masks"],
            tags={"gpu", "cpu", "ml", "semantic"},
        )

    def run(self, payload, ctx: PluginContext) -> PipelineStepResult:
        profile: AssetProfile = payload.get("profile") if isinstance(payload, dict) else payload
        if not profile or not profile.path:
            return PipelineStepResult(name="yolo_segmenter", ok=False, errors=["No profile or image path"])

        opts = ctx.options
        requested = opts.get("semantic_model", "yolov8n")
        if requested != self._loaded_key:
            self._try_init(requested)

        if self._model is None:
            return PipelineStepResult(
                name="yolo_segmenter",
                ok=False,
                warnings=[f"YOLO unavailable ({self._last_init_error}); skipping semantic pass."],
            )

        start = time.perf_counter()
        out_dir = Path(ctx.work_dir) / "assets"
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            import cv2
            from PIL import Image

            img = cv2.imread(str(profile.path), cv2.IMREAD_UNCHANGED)
            if img is None:
                return PipelineStepResult(name="yolo_segmenter", ok=False, errors=[f"Could not read image: {profile.path}"])
            h, w = img.shape[:2]

            results = self._model.predict(source=str(profile.path), verbose=False, retina_masks=True, conf=0.25, iou=0.5)
            if not results:
                return PipelineStepResult(name="yolo_segmenter", ok=False, warnings=["YOLO returned no results"])

            r = results[0]
            names = r.names if hasattr(r, "names") else self._model.names
            boxes = getattr(r, "boxes", None)
            masks = getattr(r, "masks", None)

            elements: list[DetectedElement] = []
            n = 0
            if boxes is not None and getattr(boxes, "xyxy", None) is not None:
                xyxy = boxes.xyxy
                cls = boxes.cls
                conf = boxes.conf
                if hasattr(xyxy, "cpu"):
                    xyxy = xyxy.cpu().numpy()
                    cls = cls.cpu().numpy()
                    conf = conf.cpu().numpy()
                mask_data = None
                if masks is not None and getattr(masks, "data", None) is not None:
                    mask_data = masks.data
                    if hasattr(mask_data, "cpu"):
                        mask_data = mask_data.cpu().numpy()

                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = [int(v) for v in xyxy[i]]
                    cw, ch = x2 - x1, y2 - y1
                    if cw < 4 or ch < 4:
                        continue
                    cls_id = int(cls[i])
                    cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(names[cls_id])
                    godot_class = _YOLO_TO_GODOT_CLASS.get(cls_name, "sprite")

                    # Mask if available
                    mask = None
                    asset_path = None
                    if mask_data is not None and i < mask_data.shape[0]:
                        m = mask_data[i].astype(np.uint8)
                        if m.shape != (h, w):
                            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                        mask = m
                        element_id = f"yolo_{uuid.uuid4().hex[:8]}"
                        roi = img[y1:y2, x1:x2]
                        if roi.ndim == 2:
                            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGRA)
                        elif roi.shape[2] == 3:
                            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2BGRA)
                        roi_mask = mask[y1:y2, x1:x2]
                        roi_rgba = roi.copy()
                        roi_rgba[:, :, 3] = roi_mask
                        pil_img = Image.fromarray(roi_rgba)
                        bbox = pil_img.getbbox()
                        if bbox:
                            trimmed = pil_img.crop(bbox)
                            tx, ty, tx2, ty2 = bbox
                            asset_path = out_dir / f"{element_id}.png"
                            trimmed.save(asset_path)
                            x1 += tx
                            y1 += ty
                            cw, ch = tx2 - tx, ty2 - ty

                    elements.append(
                        DetectedElement(
                            id=f"yolo_{uuid.uuid4().hex[:8]}",
                            element_type=godot_class,
                            bounds=(x1, y1, cw, ch),
                            confidence=float(conf[i]),
                            asset_path=asset_path,
                            semantic_class=cls_name,
                            z_index=200 + i,
                            source="yolo",
                            metadata={"yolo_class": cls_name, "yolo_conf": float(conf[i])},
                        )
                    )
                    n += 1

            # Merge into existing segmentation if present
            existing: SegmentationResult | None = payload.get("segmentation") if isinstance(payload, dict) else None
            if existing is not None:
                ids = {e.id for e in existing.elements}
                for e in elements:
                    if e.id not in ids:
                        existing.elements.append(e)
                existing.sort_and_index()
                segmentation = existing
            else:
                segmentation = SegmentationResult(
                    elements=elements,
                    num_elements=len(elements),
                    coverage_ratio=0.0,
                    method_used=f"yolo_{self._loaded_key}",
                ).sort_and_index()

            elapsed = (time.perf_counter() - start) * 1000
            return PipelineStepResult(
                name="yolo_segmenter",
                data={**payload, "segmentation": segmentation, "semantic_labels": [e.semantic_class for e in elements if e.semantic_class]},
                metrics={"yolo_time_ms": elapsed, "yolo_detections": float(n)},
                artifacts={"assets_dir": out_dir},
            )

        except Exception as exc:  # noqa: BLE001
            return PipelineStepResult(name="yolo_segmenter", ok=False, warnings=[f"YOLO inference failed: {exc}"])