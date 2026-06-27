"""Scene quality validator (SSIM, PSNR, IoU, geometry, texture-limit checks)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from gvi.core.plugin import Capability, Plugin, PluginContext
from gvi.core.types import AssetProfile, CapabilityType, DetectedElement, PipelineStepResult, SegmentationResult, ValidationResult
from gvi.plugins.validators._godot_headless import HeadlessValidationResult, scaffold_minimal_project, validate_project_headless


def validate_project_headless_for_scene(scene_path: Path) -> HeadlessValidationResult:
    """Best-effort real-engine check: scaffold a throwaway project.godot next
    to the generated scene (if one isn't already there) and try to import it
    with a real Godot 4 binary. Returns available=False (not an exception)
    when no Godot binary can be found -- see _godot_headless.py docstring.
    """
    try:
        scaffold_minimal_project(scene_path.parent)
        return validate_project_headless(scene_path.parent)
    except Exception as exc:  # never let a best-effort check break the pipeline
        return HeadlessValidationResult(available=True, ok=False, errors=[f"Unexpected error: {exc}"])


class SceneValidator(Plugin):
    GODOT_MAX_TEXTURE_SIZE = 16384

    def capability(self) -> Capability:
        return Capability(
            id="validator.scene",
            type=CapabilityType.VALIDATOR,
            name="Scene Quality Validator",
            priority=100,
            requires=["scene", "segmentation"],
            provides=["validation", "ssim", "psnr", "iou"],
            tags={"quality", "metrics"},
        )

    def run(self, payload: dict[str, Any], ctx: PluginContext) -> PipelineStepResult:
        profile: AssetProfile | None = payload.get("profile")
        segmentation: SegmentationResult | None = payload.get("segmentation")
        if not profile or not segmentation:
            return PipelineStepResult(name="scene_validator", ok=False, warnings=["Missing profile or segmentation"])

        result = self.validate_segmentation(profile, segmentation)

        # NOTE: found by actually running the live pipeline end-to-end, not
        # by reading the code -- the headless Godot hook was originally
        # wired only into validate_scene_file() (the standalone `gvi
        # validate` CLI path), which run() never calls. run() has its own
        # validate_segmentation() path operating on in-memory data, so the
        # headless check has to be invoked here too, against the scene file
        # the exporter step just wrote to request.output_dir / "scene.tscn".
        request_obj = payload.get("request")
        if request_obj is not None:
            scene_path = Path(request_obj.output_dir) / "scene.tscn"
            if scene_path.exists():
                headless = validate_project_headless_for_scene(scene_path)
                result.godot_headless = {
                    "available": headless.available,
                    "ok": headless.ok,
                    "binary_used": headless.binary_used,
                    "skipped_reason": headless.skipped_reason,
                    "errors": headless.errors,
                }
                if headless.available and not headless.ok:
                    result.errors.extend(f"Godot headless import error: {e}" for e in (headless.errors or ["unknown failure"]))
                elif not headless.available:
                    result.warnings.append(
                        "Godot headless validation skipped (no Godot 4 binary on PATH) -- "
                        "only geometry/SSIM/IoU checks ran."
                    )
                result.is_valid = result.is_valid and not (headless.available and not headless.ok)

        manifest_path = payload.get("request").output_dir / "validation.json" if payload.get("request") else None
        artifacts = {}
        if manifest_path:
            manifest_path.write_text(json.dumps(result.model_dump(), indent=2, default=str), encoding="utf-8")
            artifacts["validation"] = manifest_path
        return PipelineStepResult(
            name="scene_validator",
            data={**payload, "validation": result},
            metrics=result.metrics,
            warnings=result.warnings,
            errors=result.errors,
            artifacts=artifacts,
        )

    def validate_segmentation(self, profile: AssetProfile, segmentation: SegmentationResult) -> ValidationResult:
        result = ValidationResult()
        self._validate_geometry(segmentation.elements, profile.width or 0, profile.height or 0, result)
        self._validate_texture_limits(segmentation.elements, result)
        self._detect_overlaps(segmentation.elements, result)
        if profile.path and profile.path.exists() and (profile.extension or "").lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            self._compute_visual_fidelity(profile.path, segmentation.elements, result)
        result.coverage_ratio = segmentation.coverage_ratio
        result.is_valid = len(result.errors) == 0 and (result.ssim_score is None or result.ssim_score >= 0.70)
        result.metrics = {
            "ssim": result.ssim_score or 0.0,
            "psnr": result.psnr_score or 0.0,
            "iou": result.iou_score or 0.0,
            "coverage_ratio": result.coverage_ratio or 0.0,
            "num_overlaps": float(len(result.overlap_violations)),
            "num_out_of_bounds": float(len(result.out_of_bounds)),
            "num_errors": float(len(result.errors)),
        }
        return result

    def validate_scene_file(self, scene_path: Path, source_image: Path | None = None) -> ValidationResult:
        manifest_path = scene_path.with_name("manifest.json")
        result = ValidationResult()
        if not scene_path.exists():
            result.errors.append(f"Scene file does not exist: {scene_path}")
            result.metrics = {"num_errors": 1.0}
            return result
        if not manifest_path.exists():
            result.errors.append(f"Manifest not found next to scene: {manifest_path}")
            result.metrics = {"num_errors": 1.0}
            return result
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        elements = self._elements_from_manifest(data, scene_path.parent)
        canvas = data.get("canvas", {})
        self._validate_geometry(elements, int(canvas.get("width") or 0), int(canvas.get("height") or 0), result)
        self._validate_texture_limits(elements, result)
        self._detect_overlaps(elements, result)
        src = source_image or Path(data.get("source") or "")
        if src and src.exists():
            self._compute_visual_fidelity(src, elements, result)

        headless = validate_project_headless_for_scene(scene_path)
        result.godot_headless = {
            "available": headless.available,
            "ok": headless.ok,
            "binary_used": headless.binary_used,
            "skipped_reason": headless.skipped_reason,
            "errors": headless.errors,
        }
        if headless.available and not headless.ok:
            result.errors.extend(f"Godot headless import error: {e}" for e in (headless.errors or ["unknown failure"]))
        elif not headless.available:
            result.warnings.append(
                "Godot headless validation skipped (no Godot 4 binary on PATH) -- "
                "only the structural .tscn grammar checker ran. " + (headless.skipped_reason or "")
            )

        result.is_valid = len(result.errors) == 0 and (result.ssim_score is None or result.ssim_score >= 0.70)
        result.metrics = {
            "ssim": result.ssim_score or 0.0,
            "psnr": result.psnr_score or 0.0,
            "iou": result.iou_score or 0.0,
            "num_overlaps": float(len(result.overlap_violations)),
            "num_out_of_bounds": float(len(result.out_of_bounds)),
            "num_errors": float(len(result.errors)),
        }
        return result

    def _elements_from_manifest(self, data: dict[str, Any], base_dir: Path) -> list[DetectedElement]:
        assets = data.get("assets", {})
        elements: list[DetectedElement] = []
        for node in data.get("nodes", []):
            bounds = node.get("bounds") or {}
            asset = assets.get(node.get("id"))
            asset_path = None
            if asset:
                asset_path = base_dir / asset.replace("res://", "")
            metadata = node.get("metadata", {}) or {}
            elem_type = "background" if metadata.get("source") == "preservation_layer" else str(node.get("type", "sprite"))
            elements.append(DetectedElement(
                id=node.get("id", "node"),
                element_type=elem_type,
                bounds=(int(bounds.get("x", 0)), int(bounds.get("y", 0)), int(bounds.get("width", 0)), int(bounds.get("height", 0))),
                asset_path=asset_path,
                z_index=int(node.get("z_index", 0)),
                locked=bool(metadata.get("locked", False)),
                metadata=metadata,
            ))
        return elements

    def _validate_geometry(self, elements: list[DetectedElement], img_w: int, img_h: int, result: ValidationResult) -> None:
        if img_w <= 0 or img_h <= 0:
            result.warnings.append("Image/canvas dimensions unknown; geometry validation limited.")
            return
        for elem in elements:
            x, y, w, h = elem.bounds
            if w <= 0 or h <= 0:
                result.warnings.append(f"Element {elem.id} has empty bounds")
            if x < 0 or y < 0 or x + w > img_w + 1 or y + h > img_h + 1:
                result.out_of_bounds.append(elem.id)
                result.warnings.append(f"Element {elem.id} out of bounds: ({x}, {y}, {w}, {h}) vs ({img_w}, {img_h})")

    def _compute_visual_fidelity(self, source_path: Path, elements: list[DetectedElement], result: ValidationResult) -> None:
        try:
            import cv2
            from skimage.metrics import peak_signal_noise_ratio, structural_similarity
            src_raw = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
            src = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            if src is None:
                result.warnings.append(f"Could not load source image: {source_path}")
                return
            reconstruction, coverage_mask = self._reconstruct(src.shape, elements)
            src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            recon_gray = cv2.cvtColor(reconstruction, cv2.COLOR_BGR2GRAY)
            result.ssim_score = float(structural_similarity(src_gray, recon_gray))
            import warnings as _warnings
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore", RuntimeWarning)
                psnr_value = float(peak_signal_noise_ratio(src, reconstruction))
            if not np.isfinite(psnr_value):
                psnr_value = 99.0
            result.psnr_score = psnr_value
            if src_raw is not None and src_raw.ndim == 3 and src_raw.shape[2] == 4:
                src_mask = src_raw[:, :, 3] > 1
            else:
                src_mask = np.ones(coverage_mask.shape, dtype=bool)
            result.iou_score = float(np.logical_and(src_mask, coverage_mask).sum() / max(np.logical_or(src_mask, coverage_mask).sum(), 1))
        except ImportError as exc:
            result.warnings.append(f"skimage/OpenCV unavailable for fidelity metrics: {exc}")
        except Exception as exc:
            result.warnings.append(f"Visual fidelity computation failed: {exc}")

    def _reconstruct(self, shape, elements):
        import cv2
        h, w = shape[:2]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        coverage = np.zeros((h, w), dtype=bool)
        for elem in sorted(elements, key=lambda e: e.z_index):
            if not elem.asset_path or not elem.asset_path.exists():
                continue
            asset = cv2.imread(str(elem.asset_path), cv2.IMREAD_UNCHANGED)
            if asset is None:
                continue
            x, y, ew, eh = elem.bounds
            if x >= w or y >= h:
                continue
            if ew > 0 and eh > 0 and (asset.shape[1] != ew or asset.shape[0] != eh):
                asset = cv2.resize(asset, (ew, eh), interpolation=cv2.INTER_AREA)
            roi_h = min(asset.shape[0], h - max(0, y))
            roi_w = min(asset.shape[1], w - max(0, x))
            if roi_h <= 0 or roi_w <= 0:
                continue
            yy, xx = max(0, y), max(0, x)
            asset_crop = asset[:roi_h, :roi_w]
            if asset_crop.ndim == 3 and asset_crop.shape[2] == 4:
                alpha = asset_crop[:, :, 3:4].astype(np.float32) / 255.0
                fg = asset_crop[:, :, :3].astype(np.float32)
                bg = canvas[yy:yy+roi_h, xx:xx+roi_w].astype(np.float32)
                canvas[yy:yy+roi_h, xx:xx+roi_w] = (fg * alpha + bg * (1.0 - alpha)).astype(np.uint8)
                coverage[yy:yy+roi_h, xx:xx+roi_w] |= (alpha[:, :, 0] > 0.01)
            else:
                canvas[yy:yy+roi_h, xx:xx+roi_w] = asset_crop[:roi_h, :roi_w, :3] if asset_crop.ndim == 3 else cv2.cvtColor(asset_crop, cv2.COLOR_GRAY2BGR)
                coverage[yy:yy+roi_h, xx:xx+roi_w] = True
        return canvas, coverage

    def _validate_texture_limits(self, elements: list[DetectedElement], result: ValidationResult) -> None:
        try:
            from PIL import Image
            for elem in elements:
                if not elem.asset_path or not elem.asset_path.exists():
                    continue
                with Image.open(elem.asset_path) as img:
                    if img.width > self.GODOT_MAX_TEXTURE_SIZE or img.height > self.GODOT_MAX_TEXTURE_SIZE:
                        result.texture_limit_violations.append(elem.id)
                        result.errors.append(f"Element {elem.id} texture {img.width}x{img.height} exceeds Godot limit {self.GODOT_MAX_TEXTURE_SIZE}")
        except Exception as exc:  # noqa: BLE001
            result.warnings.append(f"Texture limit validation skipped: {exc}")

    def _detect_overlaps(self, elements: list[DetectedElement], result: ValidationResult) -> None:
        editable = [e for e in elements if e.element_type != "background" and not e.locked]
        for i in range(len(editable)):
            for j in range(i + 1, len(editable)):
                iou = self._box_iou(editable[i].bounds, editable[j].bounds)
                if iou > 0.25:
                    result.overlap_violations.append({"element_a": editable[i].id, "element_b": editable[j].id, "iou": float(iou)})
        if result.overlap_violations:
            result.warnings.append(f"Found {len(result.overlap_violations)} significant editable-layer overlaps")

    def _box_iou(self, a, b) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        xi1, yi1 = max(ax, bx), max(ay, by)
        xi2, yi2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        inter = (xi2 - xi1) * (yi2 - yi1)
        union = aw * ah + bw * bh - inter
        return inter / max(union, 1)