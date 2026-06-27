"""Auto-labeling backends.

Best-practice workflow: teacher models produce first-pass labels, humans correct
labels, then a small YOLO student model learns the domain.
"""
from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image

from gvi.training.annotations import AnnotationFile, AnnotationObject, Polygon
from gvi.training.overlay import draw_overlay
from gvi.training.rules import score_object
from gvi.training.taxonomy import Taxonomy


class TeacherBackend(ABC):
    id: str

    @abstractmethod
    def label_image(self, image_path: Path, taxonomy: Taxonomy, selected_classes: list[str] | None = None) -> AnnotationFile:
        raise NotImplementedError


class HeuristicTeacherBackend(TeacherBackend):
    """Dependency-light teacher for prototyping on geometric 2D assets.

    It uses OpenCV contours when available and falls back to one image-level box.
    It is not meant to be perfect; it bootstraps the full dataset loop offline.
    """

    id = "heuristic"

    def label_image(self, image_path: Path, taxonomy: Taxonomy, selected_classes: list[str] | None = None) -> AnnotationFile:
        with Image.open(image_path) as im:
            width, height = im.size
        objects: list[AnnotationObject] = []
        allowed = selected_classes or [c.name for c in taxonomy.classes]
        default_class = allowed[0] if allowed else taxonomy.classes[0].name
        try:
            import cv2
            import numpy as np

            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError("cv2 could not read image")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Robust for bright/dark separated level screenshots.
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(blur, 40, 120)
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_area = max(24, int(width * height * 0.0003))
            for c in sorted(contours, key=cv2.contourArea, reverse=True)[:300]:
                area = cv2.contourArea(c)
                if area < min_area:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                if w < 5 or h < 5:
                    continue
                cls = _guess_class_from_geometry(default_class, allowed, x, y, w, h, width, height)
                pts = c.reshape(-1, 2)
                # Simplify polygon to keep YOLO labels readable.
                eps = 0.01 * cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
                poly_points = [(float(px), float(py)) for px, py in approx[:80]]
                obj = AnnotationObject(
                    id=f"auto_{uuid.uuid4().hex[:8]}",
                    class_name=cls,
                    bbox_xywh=(float(x), float(y), float(w), float(h)),
                    polygon=Polygon(points=poly_points) if len(poly_points) >= 3 else None,
                    confidence=0.45,
                    source="heuristic",
                    needs_review=True,
                    review_reason="Heuristic bootstrap label; should be reviewed.",
                )
                decision = score_object(obj, width, height)
                obj.godot_candidates = decision.godot_candidates
                obj.confidence = min(0.65, decision.final_confidence)
                objects.append(obj)
        except Exception:
            # Last-resort object so pipeline still writes files.
            obj = AnnotationObject(
                id=f"auto_{uuid.uuid4().hex[:8]}",
                class_name=default_class,
                bbox_xywh=(0.0, 0.0, float(width), float(height)),
                confidence=0.20,
                source="fallback",
                needs_review=True,
                review_reason="No vision backend available; image-level placeholder.",
            )
            obj.godot_candidates = ["Sprite2D"]
            objects.append(obj)
        return AnnotationFile(image_path=image_path, width=width, height=height, split="unassigned", objects=objects)


class YoloTeacherBackend(TeacherBackend):
    """Use an existing YOLO model as a teacher to create GVI annotations."""

    id = "yolo"

    def __init__(self, model_path: str | Path = "yolo11n-seg.pt", conf: float = 0.25):
        self.model_path = str(model_path)
        self.conf = conf

    def label_image(self, image_path: Path, taxonomy: Taxonomy, selected_classes: list[str] | None = None) -> AnnotationFile:
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:
            raise RuntimeError("Install training deps first: python -m pip install -e '.[training]'") from exc
        with Image.open(image_path) as im:
            width, height = im.size
        model = YOLO(self.model_path)
        result = model.predict(str(image_path), conf=self.conf, verbose=False, retina_masks=True)[0]
        objects: list[AnnotationObject] = []
        names = getattr(result, "names", {})
        boxes = getattr(result, "boxes", None)
        masks = getattr(result, "masks", None)
        xyxy = boxes.xyxy.cpu().numpy() if boxes is not None and boxes.xyxy is not None else []
        cls = boxes.cls.cpu().numpy() if boxes is not None and boxes.cls is not None else []
        confs = boxes.conf.cpu().numpy() if boxes is not None and boxes.conf is not None else []
        polys = []
        if masks is not None and getattr(masks, "xy", None) is not None:
            polys = masks.xy
        allowed = set(selected_classes or taxonomy.class_to_id.keys())
        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = [float(v) for v in box]
            model_name = names.get(int(cls[i]), str(int(cls[i]))) if isinstance(names, dict) else str(int(cls[i]))
            class_name = model_name if model_name in allowed else _closest_taxonomy_class(model_name, taxonomy, allowed)
            points = []
            if i < len(polys):
                points = [(float(px), float(py)) for px, py in polys[i][:100]]
            obj = AnnotationObject(
                id=f"yolo_{uuid.uuid4().hex[:8]}",
                class_name=class_name,
                bbox_xywh=(x1, y1, max(x2 - x1, 1.0), max(y2 - y1, 1.0)),
                polygon=Polygon(points=points) if len(points) >= 3 else None,
                confidence=float(confs[i]) if i < len(confs) else 0.5,
                source=f"yolo:{self.model_path}",
            )
            dec = score_object(obj, width, height)
            obj.godot_candidates = dec.godot_candidates
            obj.needs_review = dec.needs_review
            obj.review_reason = dec.reason
            objects.append(obj)
        return AnnotationFile(image_path=image_path, width=width, height=height, split="unassigned", objects=objects)


class GroundedSamTeacherBackend(TeacherBackend):
    """Adapter placeholder for Grounding DINO + SAM/SAM2.

    This intentionally provides a clean integration point without vendoring huge
    model repos. Users can install a Grounded-SAM2 implementation or point this
    backend to a local API. The project remains reusable and legally clean.
    """

    id = "grounded-sam"

    def __init__(self, endpoint: str | None = None):
        self.endpoint = endpoint

    def label_image(self, image_path: Path, taxonomy: Taxonomy, selected_classes: list[str] | None = None) -> AnnotationFile:
        if self.endpoint:
            return self._label_via_endpoint(image_path, taxonomy, selected_classes)
        raise RuntimeError(
            "Grounded-SAM backend is configured but no local endpoint was provided. "
            "Use --backend heuristic first, --backend yolo with a model, or run a Grounded-SAM2 server "
            "and pass --teacher-endpoint. See docs/TRAINING.md."
        )

    def _label_via_endpoint(self, image_path: Path, taxonomy: Taxonomy, selected_classes: list[str] | None) -> AnnotationFile:
        try:
            import requests  # type: ignore
        except Exception as exc:
            raise RuntimeError("Install requests or use another backend.") from exc
        prompts = taxonomy.prompt_list(selected_classes)
        with image_path.open("rb") as f:
            resp = requests.post(self.endpoint, files={"image": f}, data={"prompts": json.dumps(prompts)}, timeout=180)
        resp.raise_for_status()
        payload = resp.json()
        with Image.open(image_path) as im:
            width, height = im.size
        objects: list[AnnotationObject] = []
        for item in payload.get("objects", []):
            cls = item.get("class_name") or item.get("label") or (selected_classes or [taxonomy.classes[0].name])[0]
            bbox = item.get("bbox_xywh") or item.get("bbox") or [0, 0, width, height]
            points = item.get("polygon") or []
            obj = AnnotationObject(
                id=item.get("id", f"gsam_{uuid.uuid4().hex[:8]}"),
                class_name=cls,
                bbox_xywh=tuple(float(v) for v in bbox[:4]),  # type: ignore[arg-type]
                polygon=Polygon(points=[tuple(p) for p in points]) if len(points) >= 3 else None,
                confidence=float(item.get("confidence", 0.65)),
                source="grounded-sam:endpoint",
                needs_review=float(item.get("confidence", 0.65)) < 0.75,
            )
            dec = score_object(obj, width, height)
            obj.godot_candidates = dec.godot_candidates
            obj.needs_review = obj.needs_review or dec.needs_review
            obj.review_reason = dec.reason
            objects.append(obj)
        return AnnotationFile(image_path=image_path, width=width, height=height, objects=objects)


def _guess_class_from_geometry(default_class: str, allowed: list[str], x: int, y: int, w: int, h: int, iw: int, ih: int) -> str:
    aspect = w / max(h, 1)
    lower_half = y > ih * 0.45
    if "ladder" in allowed and h > 1.7 * w and h > ih * 0.08:
        return "ladder"
    if "spike" in allowed and 0.6 <= aspect <= 1.6 and w * h < iw * ih * 0.02:
        return "spike"
    if "platform" in allowed and (aspect > 1.6 or lower_half):
        return "platform"
    if "door" in allowed and h > 1.4 * w and lower_half:
        return "door"
    return default_class


def _closest_taxonomy_class(model_name: str, taxonomy: Taxonomy, allowed: set[str]) -> str:
    # Simple conservative mapper; domain-specific remapping belongs in taxonomy/rules.
    name = model_name.lower().replace(" ", "_")
    aliases = {
        "person": "enemy",
        "character": "enemy",
        "light": "light",
        "coin": "coin",
        "button": "ui_button",
        "text": "text",
    }
    cand = aliases.get(name, name)
    if cand in allowed:
        return cand
    return next(iter(allowed)) if allowed else taxonomy.classes[0].name


def get_backend(backend: str, model: str | None = None, endpoint: str | None = None, conf: float = 0.25) -> TeacherBackend:
    backend = backend.lower().replace("_", "-")
    if backend in {"heuristic", "opencv"}:
        return HeuristicTeacherBackend()
    if backend == "yolo":
        return YoloTeacherBackend(model or "yolo11n-seg.pt", conf=conf)
    if backend in {"huggingface", "hf", "grounding-dino", "dino"}:
        # Imported lazily: pulls in transformers/torch only when actually used.
        from gvi.training.hf_teacher import HuggingFaceTeacherBackend

        kwargs: dict[str, object] = {}
        if model:
            kwargs["dino_model"] = model
        if conf:
            kwargs["box_threshold"] = conf
        return HuggingFaceTeacherBackend(**kwargs)  # type: ignore[arg-type]
    if backend in {"grounded-sam", "grounded-sam2", "sam2"}:
        return GroundedSamTeacherBackend(endpoint=endpoint)
    raise ValueError(f"Unknown teacher backend: {backend}")


def autolabel_directory(
    source_dir: Path,
    dataset_root: Path,
    taxonomy_name: str = "platformer",
    backend: str = "heuristic",
    classes: list[str] | None = None,
    model: str | None = None,
    teacher_endpoint: str | None = None,
    conf: float = 0.25,
    write_yolo: bool = True,
    split: str = "train",
) -> dict[str, object]:
    dataset_root = Path(dataset_root).resolve()
    source_dir = Path(source_dir)
    taxonomy = Taxonomy.load(dataset_root / "classes.json" if (dataset_root / "classes.json").exists() else taxonomy_name)
    teacher = get_backend(backend, model=model, endpoint=teacher_endpoint, conf=conf)
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    images = [p for p in (source_dir.rglob("*") if source_dir.is_dir() else [source_dir]) if p.suffix.lower() in exts]
    written: list[str] = []
    for image_path in images:
        ann = teacher.label_image(image_path, taxonomy, selected_classes=classes)
        ann.split = split  # type: ignore[assignment]
        ann_path = dataset_root / "annotations" / "gvi" / f"{image_path.stem}.json"
        ann.write_json(ann_path)
        overlay = dataset_root / "overlays" / f"{image_path.stem}_overlay.jpg"
        draw_overlay(ann, overlay)
        if write_yolo:
            label_path = dataset_root / "labels" / split / f"{image_path.stem}.txt"
            ann.write_yolo_seg(label_path, taxonomy.class_to_id)
            # Copy image into the YOLO image split.
            target_image = dataset_root / "images" / split / image_path.name
            target_image.parent.mkdir(parents=True, exist_ok=True)
            if image_path.resolve() != target_image.resolve():
                target_image.write_bytes(image_path.read_bytes())
        written.append(str(ann_path))
    return {"backend": teacher.id, "images": len(images), "annotations": written, "dataset_root": str(dataset_root)}
