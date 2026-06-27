"""OpenCV-based deterministic segmenter -- v1.1 (over-segmentation fix).

This is the workhorse that always works (no ML, no GPU). The actual
computer-vision algorithm lives in `_opencv_core.py`, which has zero
dependency on pydantic/gvi.core so it can be unit-tested in isolation.
This file is a thin adapter: it feeds the plugin's image/options into the
core, then wraps the result into the pydantic DetectedElement/
SegmentationResult types the rest of the pipeline expects.

v1.1 change log (see docs/CHANGELOG_v1.1.md for the full writeup and
before/after benchmark numbers):
  - Replaced "always run every strategy and IoU-dedupe the leftovers" with
    scene-complexity-adaptive strategy selection.
  - Added a per-candidate plausibility filter (solidity + internal color
    homogeneity) that rejects texture/noise fragments before they are even
    saved as PNGs.
  - Added color+proximity-aware region merging (union-find), which is what
    actually fixes "157 boxes for one toy" -> "30 boxes", not just IoU
    dedup of near-identical boxes.
  - Added optional GrabCut boundary refinement for flattened (non-alpha)
    photographic content.
  - Added homography-based rectification for rotated quads (tilted frames),
    instead of a wasteful/skewed axis-aligned crop.
  - Added an adaptive element budget by scene type instead of a flat 500.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from gvi.core.plugin import Capability, Plugin, PluginContext
from gvi.core.types import (
    AssetProfile,
    CapabilityType,
    DetectedElement,
    PipelineStepResult,
    SegmentationResult,
)
from gvi.plugins.segmenters._opencv_core import run_segmentation, to_bgr


class OpenCVSegmenter(Plugin):
    def capability(self) -> Capability:
        return Capability(
            id="segmenter.opencv",
            type=CapabilityType.SEGMENTER,
            name="OpenCV adaptive deterministic segmenter (v1.1)",
            supported_mime_types=["image/png", "image/jpeg", "image/webp", "image/bmp"],
            supported_extensions=[".png", ".jpg", ".jpeg", ".webp", ".bmp"],
            priority=80,
            provides=["segmentation", "elements", "masks"],
            tags={"cpu", "fast", "deterministic"},
        )

    def run(self, payload, ctx: PluginContext) -> PipelineStepResult:
        profile: AssetProfile = payload.get("profile") if isinstance(payload, dict) else payload
        if not profile or not profile.path:
            return PipelineStepResult(name="opencv_segmenter", ok=False, errors=["No profile or image path"])

        start = time.perf_counter()
        output_dir = Path(ctx.work_dir) / "assets"
        output_dir.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(profile.path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return PipelineStepResult(name="opencv_segmenter", ok=False, errors=[f"Could not read image: {profile.path}"])

        h, w = img.shape[:2]
        has_alpha = img.ndim == 3 and img.shape[2] == 4
        opts = ctx.options
        max_elements = opts.get("max_elements")  # None => adaptive budget in core
        if max_elements == 500 or max_elements == 200:
            # Legacy callers/configs still pass the old flat defaults; treat
            # them as "no opinion" so the new adaptive budget applies.
            max_elements = None
        min_area_ratio = float(opts.get("min_area_ratio", 0.0001))
        merge_overlaps = bool(opts.get("merge_overlaps", True))

        core_result = run_segmentation(
            img,
            output_dir,
            preset=opts.get("preset", "balanced"),
            min_area_ratio=min_area_ratio,
            max_elements=max_elements,
            merge_overlaps=merge_overlaps,
        )

        candidates = [
            DetectedElement(
                id=c.id,
                element_type=c.element_type,
                bounds=c.bounds,
                rotated_rect=c.rotated_rect,
                confidence=c.confidence,
                asset_path=c.asset_path,
                source="opencv",
                metadata={**c.metadata, "rectified": c.rectified},
            )
            for c in core_result["elements"]
        ]

        elements: list[DetectedElement] = []
        if opts.get("include_background", True) and opts.get("background_mode", "source") != "none":
            bg_path = self._save_background(profile.path, output_dir, img, opts.get("background_mode", "source"))
            if bg_path:
                elements.append(
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
        elements.extend(candidates)

        scene = core_result["scene"]
        result = SegmentationResult(
            elements=elements,
            num_elements=len(elements),
            coverage_ratio=core_result["coverage_ratio"],
            method_used=f"opencv_v1_1_{scene['label']}",
            processing_time_ms=(time.perf_counter() - start) * 1000,
            diagnostics={
                "image_size": (w, h),
                "has_alpha": has_alpha,
                "scene_label": scene["label"],
                "scene_complexity": scene["complexity"],
                "unique_colors_64": scene["unique_colors_64"],
                "editable_elements": len(candidates),
                "background_preserved": any(e.element_type == "background" for e in elements),
                "raw_candidates": core_result["num_raw_candidates"],
                "after_plausibility_filter": core_result["num_after_plausibility"],
                "after_merge": core_result["num_after_merge"],
                "element_budget": core_result["max_elements_budget"],
            },
        ).sort_and_index()

        artifacts = {"assets_dir": output_dir}
        warnings: list[str] = []
        if opts.get("make_debug_overlay", False):
            overlay = self._write_debug_overlay(profile.path, result, Path(ctx.work_dir) / "segmentation_overlay.png")
            artifacts["debug_overlay"] = overlay
        if len(candidates) == 0:
            warnings.append("No editable foreground elements detected; source preservation layer still gives faithful output.")

        return PipelineStepResult(
            name="opencv_segmenter",
            data={**payload, "segmentation": result} if isinstance(payload, dict) else result,
            metrics={
                "processing_time_ms": result.processing_time_ms,
                "num_elements": float(result.num_elements),
                "editable_elements": float(len(candidates)),
                "coverage_ratio": result.coverage_ratio,
                "raw_candidates": float(core_result["num_raw_candidates"]),
                "fragmentation_reduction_ratio": float(
                    1.0 - (len(candidates) / max(1, core_result["num_raw_candidates"]))
                ),
            },
            artifacts=artifacts,
            warnings=warnings,
        )

    def _save_background(self, source_path, output_dir, img, mode) -> Path | None:
        bg_path = output_dir / "background_source.png"
        if mode == "source":
            pil = Image.open(source_path).convert("RGBA")
            pil.save(bg_path)
        elif mode == "transparent":
            h, w = img.shape[:2]
            Image.new("RGBA", (w, h), (0, 0, 0, 0)).save(bg_path)
        elif mode == "solid":
            bgr = to_bgr(img)
            rgb = tuple(int(v) for v in bgr.reshape(-1, 3).mean(axis=0)[::-1])
            h, w = img.shape[:2]
            Image.new("RGBA", (w, h), rgb + (255,)).save(bg_path)
        else:
            return None
        return bg_path

    def _write_debug_overlay(self, source_path, segmentation, output_path) -> Path:
        img = Image.open(source_path).convert("RGBA")
        draw = ImageDraw.Draw(img)
        for elem in segmentation.elements:
            if elem.element_type == "background":
                continue
            x, y, w, h = elem.bounds
            draw.rectangle((x, y, x + w, y + h), outline=(255, 0, 0, 220), width=2)
            draw.text((x + 2, y + 2), elem.element_type, fill=(255, 0, 0, 220))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)
        return output_path
