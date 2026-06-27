"""OCR text extractor powered by EasyOCR.

Detects text regions in images, recognises characters, and emits
DetectedElement records so the Godot exporter can render them as
real Label / RichTextLabel nodes (not just textures).

Model weights are downloaded automatically on first use and cached in
``~/.cache/gvi/models/easyocr/``.
"""
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from gvi.core.plugin import Capability, Plugin, PluginContext
from gvi.core.types import (
    AssetProfile,
    CapabilityType,
    DetectedElement,
    PipelineStepResult,
    SegmentationResult,
)

_EASYOCR_HOME_OVERRIDE = os.environ.setdefault("EASYOCR_MODULE_PATH", str(Path.home() / ".cache" / "gvi" / "models" / "easyocr"))


class OCRExtractorPlugin(Plugin):
    """EasyOCR-based OCR with auto-downloaded model weights."""

    def __init__(self) -> None:
        self._reader = None
        self._languages: list[str] = ["en"]
        self._last_init_error: str | None = None
        self._try_init()

    def _try_init(self) -> None:
        try:
            import easyocr  # type: ignore
        except Exception as exc:  # noqa: BLE001
            self._last_init_error = f"easyocr not installed: {exc}"
            return
        try:
            cache_dir = Path.home() / ".cache" / "gvi" / "models" / "easyocr"
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["EASYOCR_MODULE_PATH"] = str(cache_dir)
            self._reader = easyocr.Reader(self._languages, gpu=False, model_storage_directory=str(cache_dir), user_network_directory=str(cache_dir), download_enabled=True)
            self._last_init_error = None
        except Exception as exc:  # noqa: BLE001
            self._last_init_error = f"easyocr init failed: {exc}"

    # ---------------------------------------------------------------- capability
    def capability(self) -> Capability:
        return Capability(
            id="ocr.easyocr",
            type=CapabilityType.OCR,
            name="EasyOCR text extractor",
            supported_mime_types=["image/png", "image/jpeg", "image/webp", "image/bmp"],
            supported_extensions=[".png", ".jpg", ".jpeg", ".webp", ".bmp"],
            priority=85,
            provides=["text_elements", "ocr"],
            tags={"cpu", "ml", "ocr"},
        )

    # ------------------------------------------------------------------ run
    def run(self, payload, ctx: PluginContext) -> PipelineStepResult:
        profile: AssetProfile = payload.get("profile") if isinstance(payload, dict) else payload
        if not profile or not profile.path:
            return PipelineStepResult(name="ocr_easyocr", ok=False, errors=["No profile or image path"])

        if self._reader is None:
            return PipelineStepResult(
                name="ocr_easyocr",
                ok=False,
                warnings=[f"OCR unavailable ({self._last_init_error}); skipping text extraction."],
            )

        start = time.perf_counter()
        out_dir = Path(ctx.work_dir) / "assets"
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            import cv2
            from PIL import Image, ImageDraw

            img = cv2.imread(str(profile.path), cv2.IMREAD_UNCHANGED)
            if img is None:
                return PipelineStepResult(name="ocr_easyocr", ok=False, errors=[f"Could not read image: {profile.path}"])

            h, w = img.shape[:2]
            bgr = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # EasyOCR expects BGR or RGB; numpy array works.
            results = self._reader.readtext(bgr, detail=1, paragraph=False)

            text_elements: list[DetectedElement] = []
            for idx, (box, text, conf) in enumerate(results):
                text = (text or "").strip()
                if not text or conf < 0.20:
                    continue
                xs = [int(p[0]) for p in box]
                ys = [int(p[1]) for p in box]
                x1, y1 = max(0, min(xs)), max(0, min(ys))
                x2, y2 = min(w, max(xs)), min(h, max(ys))
                cw, ch = x2 - x1, y2 - y1
                if cw < 4 or ch < 4:
                    continue

                # Crop a tight transparent PNG of just the text region so the
                # exporter can still fall back to a TextureRect+Label combo.
                element_id = f"ocr_{uuid.uuid4().hex[:8]}"
                crop_bgr = bgr[y1:y2, x1:x2]
                rgba = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)
                # Build mask from non-white pixels (text glyphs)
                gray_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                rgba[:, :, 3] = mask
                asset_path = out_dir / f"{element_id}.png"
                Image.fromarray(rgba).save(asset_path)

                text_elements.append(
                    DetectedElement(
                        id=element_id,
                        element_type="text",
                        bounds=(int(x1), int(y1), int(cw), int(ch)),
                        rotated_rect={
                            "quad": [[int(p[0]), int(p[1])] for p in box],
                        },
                        confidence=float(conf),
                        asset_path=asset_path,
                        text_content=text,
                        text_language=self._languages[0],
                        z_index=100 + idx,
                        source="ocr",
                        metadata={"ocr_confidence": float(conf), "polygon": [[int(p[0]), int(p[1])] for p in box]},
                    )
                )

            # If segmentation already exists, append the text elements to it; otherwise create a fresh one.
            existing: SegmentationResult | None = payload.get("segmentation") if isinstance(payload, dict) else None
            if existing is not None:
                # Avoid duplicates by id.
                ids = {e.id for e in existing.elements}
                for t in text_elements:
                    if t.id not in ids:
                        existing.elements.append(t)
                existing.sort_and_index()
                segmentation = existing
            else:
                segmentation = SegmentationResult(
                    elements=text_elements,
                    num_elements=len(text_elements),
                    coverage_ratio=0.0,
                    method_used="ocr_easyocr",
                    diagnostics={"num_text_regions": len(text_elements)},
                ).sort_and_index()

            elapsed = (time.perf_counter() - start) * 1000
            return PipelineStepResult(
                name="ocr_easyocr",
                data={**payload, "segmentation": segmentation, "text_elements": text_elements},
                metrics={
                    "ocr_time_ms": elapsed,
                    "num_text_regions": float(len(text_elements)),
                },
                artifacts={"assets_dir": out_dir},
                warnings=[] if text_elements else ["OCR found no text in the image."],
            )

        except Exception as exc:  # noqa: BLE001
            return PipelineStepResult(
                name="ocr_easyocr",
                ok=False,
                warnings=[f"OCR failed: {exc}"],
            )