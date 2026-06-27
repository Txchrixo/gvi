"""Hugging Face teacher backend: Grounding DINO (open-vocabulary detection)
+ SAM 2 (precise masks), running locally via `transformers`.

This is the backend that finally closes the "grounded-sam" gap that previous
versions only stubbed out behind an HTTP endpoint. Instead of requiring the
user to stand up a separate Grounded-SAM2 server, it downloads and runs the
models locally through Hugging Face, which is the workflow the project's own
training docs recommend (teacher -> human correction -> YOLO student).

Design contract (must match gvi/training/autolabel.py::TeacherBackend):
  - subclass TeacherBackend, set `id`
  - implement label_image(image_path, taxonomy, selected_classes) -> AnnotationFile
  - every produced AnnotationObject carries: id, class_name, bbox_xywh,
    optional polygon, confidence, source, godot_candidates, needs_review,
    review_reason. godot_candidates/needs_review/review_reason are filled by
    score_object() exactly like the YOLO and heuristic backends do.

Why two models:
  - Grounding DINO turns a *text prompt* ("ladder", "orange enemy", ...) into
    bounding boxes. This is what lets the teacher detect arbitrary game/UI
    classes that no COCO-pretrained model knows.
  - SAM 2 turns each of those boxes into a precise segmentation mask, which is
    what YOLO-seg training actually needs (polygons, not just rectangles).

Heavy dependencies (`transformers`, `torch`) are imported lazily inside
methods so that merely importing this module — or the whole `gvi.training`
package — never forces them to be installed. Missing deps raise a clear,
actionable RuntimeError, exactly like the YOLO backend does.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from PIL import Image

from gvi.training.annotations import AnnotationFile, AnnotationObject, Polygon
from gvi.training.rules import score_object
from gvi.training.taxonomy import Taxonomy


# Default model checkpoints. Kept here (not hard-coded in the class body) so a
# user can override them via the constructor without editing source.
DEFAULT_DINO_MODEL = "IDEA-Research/grounding-dino-base"
DEFAULT_SAM2_MODEL = "facebook/sam2-hiera-small"


class HuggingFaceTeacherBackend:
    """Grounding DINO + SAM 2 teacher, run locally through transformers.

    Not registered as an ABC subclass import-time dependency; it duck-types the
    TeacherBackend interface (same `id` attribute + `label_image` signature) so
    this module has zero hard dependency on anything heavy at import time.
    """

    id = "huggingface"

    def __init__(
        self,
        dino_model: str = DEFAULT_DINO_MODEL,
        sam2_model: str = DEFAULT_SAM2_MODEL,
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
        device: str | None = None,
        use_sam2: bool = True,
        review_below: float = 0.75,
    ) -> None:
        self.dino_model = dino_model
        self.sam2_model = sam2_model
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device
        self.use_sam2 = use_sam2
        self.review_below = review_below
        # Lazily-initialized handles (loaded once, reused across images).
        self._dino_processor: Any = None
        self._dino_model: Any = None
        self._sam2_predictor: Any = None
        self._torch: Any = None
        self._resolved_device: str | None = None

    # ------------------------------------------------------------------ deps
    def _ensure_torch(self) -> Any:
        if self._torch is None:
            try:
                import torch  # type: ignore
            except Exception as exc:  # pragma: no cover - env-dependent
                raise RuntimeError(
                    "The Hugging Face teacher backend needs PyTorch and transformers. "
                    "Install with: python -m pip install -e '.[hf]'  "
                    "(or: python -m pip install transformers torch torchvision)."
                ) from exc
            self._torch = torch
            if self.device:
                self._resolved_device = self.device
            else:
                self._resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._torch

    def _ensure_dino(self) -> None:
        if self._dino_model is not None:
            return
        torch = self._ensure_torch()
        try:
            from transformers import (  # type: ignore
                AutoProcessor,
                AutoModelForZeroShotObjectDetection,
            )
        except Exception as exc:
            raise RuntimeError(
                "transformers is required for the Hugging Face teacher backend. "
                "Install with: python -m pip install -e '.[hf]'."
            ) from exc
        self._dino_processor = AutoProcessor.from_pretrained(self.dino_model)
        self._dino_model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(self.dino_model)
            .to(self._resolved_device)
            .eval()
        )

    def _ensure_sam2(self) -> None:
        if not self.use_sam2 or self._sam2_predictor is not None:
            return
        self._ensure_torch()
        # SAM 2 ships through transformers as Sam2Model/Sam2Processor in recent
        # versions. If unavailable, we degrade gracefully to box-only masks
        # rather than crashing the whole run.
        try:
            from transformers import Sam2Processor, Sam2Model  # type: ignore

            self._sam2_processor = Sam2Processor.from_pretrained(self.sam2_model)
            self._sam2_model = Sam2Model.from_pretrained(self.sam2_model).to(self._resolved_device).eval()
            self._sam2_predictor = "transformers"
        except Exception:
            # Fall back: try the standalone `sam2` package if the user installed it.
            try:
                from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

                self._sam2_predictor = SAM2ImagePredictor.from_pretrained(self.sam2_model)
            except Exception:
                # No SAM2 available; masks will fall back to box polygons.
                self._sam2_predictor = None
                self.use_sam2 = False

    # ------------------------------------------------------------------ prompt
    @staticmethod
    def _build_prompt(taxonomy: Taxonomy, selected_classes: list[str] | None) -> tuple[str, list[tuple[str, str]]]:
        """Build the Grounding DINO text prompt and a (phrase -> class_name) map.

        Grounding DINO expects lowercase phrases separated by " . ". We expand
        each taxonomy class into its prompt phrases (from the taxonomy config)
        and remember which phrase maps back to which canonical class so the
        detected label can be normalized to a real taxonomy class.
        """
        phrase_to_class: list[tuple[str, str]] = []
        seen: set[str] = set()
        allowed = set(selected_classes) if selected_classes else None
        for cls in taxonomy.classes:
            if allowed and cls.name not in allowed:
                continue
            phrases = cls.prompts or [cls.name]
            for phrase in phrases:
                p = phrase.strip().lower()
                if p and p not in seen:
                    seen.add(p)
                    phrase_to_class.append((p, cls.name))
        if not phrase_to_class:
            # Defensive: at least prompt with the first class.
            first = taxonomy.classes[0].name
            phrase_to_class.append((first.lower(), first))
        prompt_text = " . ".join(p for p, _ in phrase_to_class) + " ."
        return prompt_text, phrase_to_class

    @staticmethod
    def _match_phrase_to_class(detected_phrase: str, phrase_to_class: list[tuple[str, str]], taxonomy: Taxonomy) -> str:
        """Map a Grounding DINO output phrase back to a canonical class name.

        DINO may return a substring or a concatenation of prompt words, so we
        match by longest-overlap against the known phrases before falling back.
        """
        d = (detected_phrase or "").strip().lower()
        if not d:
            return phrase_to_class[0][1]
        # Exact phrase match first.
        for phrase, cls in phrase_to_class:
            if phrase == d:
                return cls
        # Substring / containment match (DINO often returns partial phrases).
        best_cls = None
        best_len = 0
        for phrase, cls in phrase_to_class:
            if phrase in d or d in phrase:
                if len(phrase) > best_len:
                    best_len = len(phrase)
                    best_cls = cls
        if best_cls:
            return best_cls
        # Token-overlap fallback.
        d_tokens = set(d.replace(".", " ").split())
        for phrase, cls in phrase_to_class:
            if d_tokens & set(phrase.split()):
                return cls
        return phrase_to_class[0][1]

    # ------------------------------------------------------------------ core
    def label_image(
        self,
        image_path: Path,
        taxonomy: Taxonomy,
        selected_classes: list[str] | None = None,
    ) -> AnnotationFile:
        image_path = Path(image_path)
        self._ensure_dino()
        self._ensure_sam2()
        torch = self._torch

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        prompt_text, phrase_to_class = self._build_prompt(taxonomy, selected_classes)

        # --- Grounding DINO: text prompt -> boxes -----------------------------
        inputs = self._dino_processor(images=image, text=prompt_text, return_tensors="pt").to(self._resolved_device)
        with torch.no_grad():
            outputs = self._dino_model(**inputs)
        results = self._dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(height, width)],
        )[0]

        boxes = results.get("boxes")
        scores = results.get("scores")
        labels = results.get("labels") or results.get("text_labels") or []
        boxes_list = boxes.cpu().tolist() if boxes is not None else []
        scores_list = scores.cpu().tolist() if scores is not None else []

        # --- SAM 2: boxes -> masks (optional) ---------------------------------
        masks_list = self._segment_boxes(image, boxes_list) if (self.use_sam2 and boxes_list) else [None] * len(boxes_list)

        objects: list[AnnotationObject] = []
        for i, box in enumerate(boxes_list):
            x1, y1, x2, y2 = [float(v) for v in box]
            w = max(x2 - x1, 1.0)
            h = max(y2 - y1, 1.0)
            detected_phrase = labels[i] if i < len(labels) else ""
            class_name = self._match_phrase_to_class(str(detected_phrase), phrase_to_class, taxonomy)
            confidence = float(scores_list[i]) if i < len(scores_list) else 0.5

            polygon = None
            mask = masks_list[i] if i < len(masks_list) else None
            if mask is not None:
                points = _mask_to_polygon(mask)
                if len(points) >= 3:
                    polygon = Polygon(points=points)

            obj = AnnotationObject(
                id=f"hf_{uuid.uuid4().hex[:8]}",
                class_name=class_name,
                bbox_xywh=(x1, y1, w, h),
                polygon=polygon,
                confidence=confidence,
                source=f"huggingface:dino+sam2" if polygon is not None else "huggingface:dino",
                needs_review=confidence < self.review_below,
                review_reason=(
                    f"Teacher confidence {confidence:.2f} below {self.review_below:.2f}."
                    if confidence < self.review_below
                    else None
                ),
            )
            # Business rules: refine confidence, set godot node candidates, and
            # possibly flag needs_review — exactly like the other backends.
            decision = score_object(obj, width, height)
            obj.godot_candidates = decision.godot_candidates
            obj.needs_review = obj.needs_review or decision.needs_review
            if decision.reason and not obj.review_reason:
                obj.review_reason = decision.reason
            objects.append(obj)

        return AnnotationFile(
            image_path=image_path,
            width=width,
            height=height,
            split="unassigned",
            objects=objects,
            metadata={
                "teacher": "huggingface",
                "dino_model": self.dino_model,
                "sam2_model": self.sam2_model if self.use_sam2 else None,
                "device": self._resolved_device,
                "prompt": prompt_text,
            },
        )

    # ------------------------------------------------------------------ sam2
    def _segment_boxes(self, image: Image.Image, boxes_xyxy: list[list[float]]) -> list[Any]:
        """Return a list of boolean/float masks (numpy arrays) aligned with boxes.

        Supports two SAM2 access paths: the transformers Sam2Model API and the
        standalone `sam2` SAM2ImagePredictor. Any failure degrades to None masks
        (callers then fall back to the rectangle as a 4-point polygon).
        """
        import numpy as np

        if self._sam2_predictor is None:
            return [None] * len(boxes_xyxy)

        try:
            if self._sam2_predictor == "transformers":
                torch = self._torch
                inputs = self._sam2_processor(
                    images=image,
                    input_boxes=[[list(map(float, b)) for b in boxes_xyxy]],
                    return_tensors="pt",
                ).to(self._resolved_device)
                with torch.no_grad():
                    outputs = self._sam2_model(**inputs)
                masks = self._sam2_processor.post_process_masks(
                    outputs.pred_masks.cpu(),
                    inputs["original_sizes"].cpu(),
                )[0]
                out: list[Any] = []
                for m in masks:
                    arr = m.numpy() if hasattr(m, "numpy") else np.asarray(m)
                    # arr shape can be (num_preds, H, W); take the best mask.
                    if arr.ndim == 3:
                        arr = arr[0]
                    out.append(arr > 0.5)
                return out
            else:
                # standalone sam2 predictor
                predictor = self._sam2_predictor
                predictor.set_image(np.array(image))
                out = []
                for b in boxes_xyxy:
                    m, _scores, _ = predictor.predict(box=np.array(b, dtype=float), multimask_output=False)
                    arr = m[0] if getattr(m, "ndim", 2) == 3 else m
                    out.append(np.asarray(arr) > 0.5)
                return out
        except Exception:
            return [None] * len(boxes_xyxy)


def _mask_to_polygon(mask: Any, max_points: int = 100) -> list[tuple[float, float]]:
    """Convert a boolean mask to a simplified polygon (pixel coords).

    Uses OpenCV's largest external contour and Douglas-Peucker simplification so
    the resulting YOLO-seg label is compact. Returns [] if no contour is found.
    """
    try:
        import cv2
        import numpy as np

        m = np.asarray(mask).astype("uint8")
        if m.ndim != 2 or m.sum() == 0:
            return []
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        c = max(contours, key=cv2.contourArea)
        eps = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        pts = [(float(px), float(py)) for px, py in approx[:max_points]]
        return pts if len(pts) >= 3 else []
    except Exception:
        return []
