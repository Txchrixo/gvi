"""SAM 2 (Segment Anything Model 2) segmenter with native model weight download.

Uses the official Ultralytics SAM 2 implementation, which downloads the
pretrained weights automatically on first use and caches them in
``~/.cache/gvi/models/sam2/``.

Supports the four official SAM 2 sizes:
- sam2_hiera_tiny     (~74 MB, fastest)
- sam2_hiera_small    (~162 MB, balanced)
- sam2_hiera_base_plus (~200 MB, high quality)
- sam2_hiera_large    (~855 MB, best quality)

On any failure (no torch, no weights, GPU/CPU mismatch), it gracefully
falls back to the OpenCV segmenter and reports the reason.
"""
from __future__ import annotations

import shutil
import time
import uuid
import warnings
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

_SAM2_MODEL_FILES = {
    "sam2_hiera_tiny": "sam2_hiera_tiny.pt",
    "sam2_hiera_small": "sam2_hiera_small.pt",
    "sam2_hiera_base_plus": "sam2_hiera_base_plus.pt",
    "sam2_hiera_large": "sam2_hiera_large.pt",
}


class SAM2Segmenter(Plugin):
    """Native SAM 2 segmenter with weight download + mask generation."""

    def __init__(self) -> None:
        self._model = None
        self._overrides: dict | None = None
        self._last_init_error: str | None = None
        self._loaded_model_key: str | None = None
        self._device: str = "cpu"
        self._try_init("sam2_hiera_small")

    # ------------------------------------------------------------- lifecycle
    def _try_init(self, model_key: str) -> None:
        try:
            from ultralytics import SAM  # type: ignore
            import torch
        except Exception as exc:  # noqa: BLE001
            self._last_init_error = f"SAM 2 deps missing: {exc}"
            return

        cache_dir = Path.home() / ".cache" / "gvi" / "models" / "sam2"
        cache_dir.mkdir(parents=True, exist_ok=True)
        weight_name = _SAM2_MODEL_FILES[model_key]
        weight_path = cache_dir / weight_name

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._device = device
            # Ultralytics will auto-download if file is missing.
            self._model = SAM(weight_name)
            self._overrides = {"device": device, "verbose": False, "save": False}
            self._loaded_model_key = model_key
            self._last_init_error = None
            # If the weight was downloaded somewhere else, mirror it into our cache.
            if weight_path.exists() is False:
                for src in (Path.cwd() / weight_name, Path.home() / weight_name):
                    if src.exists():
                        try:
                            shutil.copy2(src, weight_path)
                        except Exception:  # noqa: BLE001
                            pass
        except Exception as exc:  # noqa: BLE001
            self._last_init_error = f"SAM 2 init failed: {exc}"
            self._model = None

    # ---------------------------------------------------------------- capability
    def capability(self) -> Capability:
        return Capability(
            id="segmenter.sam2",
            type=CapabilityType.SEGMENTER,
            name="SAM 2 (Segment Anything 2) native segmenter",
            supported_mime_types=["image/png", "image/jpeg", "image/webp", "image/bmp"],
            supported_extensions=[".png", ".jpg", ".jpeg", ".webp", ".bmp"],
            priority=92,
            provides=["segmentation", "elements", "masks", "sam2"],
            tags={"gpu", "cpu", "ml", "zero-shot", "high-quality"},
        )

    # ------------------------------------------------------------------ run
    def run(self, payload, ctx: PluginContext) -> PipelineStepResult:
        profile: AssetProfile = payload.get("profile") if isinstance(payload, dict) else payload
        if not profile or not profile.path:
            return PipelineStepResult(name="sam2_segmenter", ok=False, errors=["No profile or image path"])

        opts = ctx.options
        requested = opts.get("sam2_model", "sam2_hiera_small")
        if self._loaded_model_key != requested:
            self._try_init(requested)

        if self._model is None:
            warnings.warn(f"SAM 2 unavailable ({self._last_init_error}); falling back to OpenCV.")
            from gvi.plugins.segmenters.opencv_segmenter import OpenCVSegmenter
            return OpenCVSegmenter().run(payload, ctx)

        start = time.perf_counter()
        img_path = str(profile.path)
        output_dir = Path(ctx.work_dir) / "assets"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            import cv2
            from PIL import Image

            img_bgr = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img_bgr is None:
                return PipelineStepResult(name="sam2_segmenter", ok=False, errors=[f"Could not read image: {img_path}"])

            # Ultralytics SAM accepts a path or numpy array.
            results = self._model.predict(
                source=img_path,
                device=self._device,
                retina_masks=True,
                imgsz=1024,
                conf=0.25,
                iou=0.5,
                verbose=False,
            )

            elements: list[DetectedElement] = []
            if results:
                res = results[0]
                masks = getattr(res, "masks", None)
                boxes = getattr(res, "boxes", None)
                has_masks = masks is not None and getattr(masks, "data", None) is not None

                if has_masks:
                    data = masks.data
                    if hasattr(data, "cpu"):
                        data = data.cpu().numpy()
                    else:
                        data = np.asarray(data)
                    # data shape: (N, H, W) on model's input size — resize to original.
                    orig_h, orig_w = img_bgr.shape[:2]
                    n = data.shape[0]
                    scores = np.ones(n, dtype=np.float32)
                    if boxes is not None and getattr(boxes, "conf", None) is not None:
                        try:
                            conf = boxes.conf
                            if hasattr(conf, "cpu"):
                                conf = conf.cpu().numpy()
                            scores = np.asarray(conf)
                        except Exception:  # noqa: BLE001
                            pass
                    for idx in range(n):
                        mask_small = data[idx].astype(np.uint8)
                        if mask_small.shape != (orig_h, orig_w):
                            mask_small = cv2.resize(mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                        if int(np.sum(mask_small > 0)) < 16:
                            continue
                        elem = self._mask_to_element(img_bgr, mask_small, float(scores[idx]), output_dir)
                        if elem:
                            elements.append(elem)
                else:
                    # No masks? Try point-grid fallback.
                    elements = self._point_grid_segment(img_bgr, output_dir)

            elements.sort(key=lambda e: (e.bounds[1], e.bounds[0]))
            elements = elements[: int(opts.get("max_elements", 500))]

            total_area = sum(e.bounds[2] * e.bounds[3] for e in elements)
            h, w = img_bgr.shape[:2]
            coverage = min(1.0, total_area / (h * w)) if elements else 0.0

            # Background preservation layer if requested.
            full_elements: list[DetectedElement] = []
            if opts.get("include_background", True) and opts.get("background_mode", "source") != "none":
                bg_path = output_dir / "background_source.png"
                Image.fromarray(img_bgr if img_bgr.shape[2] == 4 else cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGBA)).save(bg_path)
                full_elements.append(
                    DetectedElement(
                        id="background_source",
                        element_type="background",
                        bounds=(0, 0, w, h),
                        asset_path=bg_path,
                        z_index=-1000,
                        locked=True,
                        confidence=1.0,
                        metadata={"source": "preservation_layer", "editable": False},
                    )
                )
            full_elements.extend(elements)

            result = SegmentationResult(
                elements=full_elements,
                num_elements=len(full_elements),
                coverage_ratio=coverage,
                method_used=f"sam2_{self._loaded_model_key}",
                processing_time_ms=(time.perf_counter() - start) * 1000,
                diagnostics={
                    "image_size": (w, h),
                    "device": self._device,
                    "model": self._loaded_model_key,
                    "raw_masks": len(elements),
                },
            ).sort_and_index()

            return PipelineStepResult(
                name="sam2_segmenter",
                data={**payload, "segmentation": result},
                metrics={
                    "processing_time_ms": result.processing_time_ms,
                    "num_elements": float(result.num_elements),
                    "coverage_ratio": result.coverage_ratio,
                },
                artifacts={"assets_dir": output_dir},
            )

        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"SAM 2 inference failed ({exc}); falling back to OpenCV.")
            from gvi.plugins.segmenters.opencv_segmenter import OpenCVSegmenter
            return OpenCVSegmenter().run(payload, ctx)

    # ------------------------------------------------------------- helpers
    def _mask_to_element(self, img_bgr, mask, score, output_dir) -> DetectedElement | None:
        import cv2
        from PIL import Image

        h, w = img_bgr.shape[:2]
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        x, y = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())
        cw, ch = x2 - x + 1, y2 - y + 1
        if cw < 4 or ch < 4:
            return None

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        rrect = cv2.minAreaRect(contour)
        pts = cv2.boxPoints(rrect).astype(np.int32)

        element_id = f"elem_{uuid.uuid4().hex[:8]}"
        roi_bgr = img_bgr[y:y + ch, x:x + cw]
        roi_mask = mask[y:y + ch, x:x + cw]
        if roi_bgr.ndim == 2:
            rgba = cv2.cvtColor(roi_bgr, cv2.COLOR_GRAY2BGRA)
        elif roi_bgr.shape[2] == 3:
            rgba = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2BGRA)
        else:
            rgba = roi_bgr.copy()
        rgba[:, :, 3] = roi_mask
        pil_img = Image.fromarray(rgba)
        bbox = pil_img.getbbox()
        if bbox:
            trimmed = pil_img.crop(bbox)
            trim_x, trim_y, trim_x2, trim_y2 = bbox
            new_x, new_y = x + trim_x, y + trim_y
            new_w, new_h = trim_x2 - trim_x, trim_y2 - trim_y
        else:
            trimmed = pil_img
            new_x, new_y, new_w, new_h = x, y, cw, ch

        asset_path = output_dir / f"{element_id}.png"
        trimmed.save(asset_path, "PNG")

        area = int(np.sum(roi_mask > 0))
        extent = area / max(new_w * new_h, 1)
        aspect = max(new_w, new_h) / max(min(new_w, new_h), 1)
        if extent > 0.25 or aspect > 4.0 and extent > 0.20:
            etype = "panel"
        elif aspect > 1.8 and extent > 0.35:
            etype = "frame"
        elif extent > 0.65 and 0.6 <= aspect <= 1.8:
            etype = "button"
        else:
            etype = "sprite"

        return DetectedElement(
            id=element_id,
            element_type=etype,
            bounds=(new_x, new_y, new_w, new_h),
            rotated_rect={
                "center": [float(rrect[0][0]), float(rrect[0][1])],
                "size": [float(rrect[1][0]), float(rrect[1][1])],
                "angle_deg": float(rrect[2]),
                "quad": pts.tolist(),
            },
            confidence=float(score),
            asset_path=asset_path,
            source="sam2",
            metadata={"area": area, "extent": float(extent), "aspect_ratio": float(aspect), "sam_score": float(score)},
        )

    def _point_grid_segment(self, img_bgr, output_dir) -> list[DetectedElement]:
        import cv2
        from PIL import Image

        h, w = img_bgr.shape[:2]
        points_per_side = 32
        elements: list[DetectedElement] = []
        try:
            res = self._model.predict(
                source=img_bgr,
                device=self._device,
                points_per_side=points_per_side,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,
                verbose=False,
            )
            if not res:
                return []
            r = res[0]
            masks = getattr(r, "masks", None)
            if masks is None or getattr(masks, "data", None) is None:
                return []
            data = masks.data
            if hasattr(data, "cpu"):
                data = data.cpu().numpy()
            else:
                data = np.asarray(data)
            for idx in range(data.shape[0]):
                mask_small = data[idx].astype(np.uint8)
                if mask_small.shape != (h, w):
                    mask_small = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
                if int(np.sum(mask_small > 0)) < 100:
                    continue
                elem = self._mask_to_element(img_bgr, mask_small, 0.9, output_dir)
                if elem:
                    elements.append(elem)
        except Exception:  # noqa: BLE001
            return []
        return elements