"""Pipeline planner — picks the optimal sequence of plugins for each request.

The planner is deterministic and capability-aware. It understands:
- The asset kind (raster / vector / PSD / PDF / SVG)
- The target (which Godot component is desired)
- The preset (fast / balanced / fidelity / semantic / tilemap)
- Available segmenters and post-processors
"""
from __future__ import annotations

from dataclasses import dataclass

from gvi.core.registry import PluginRegistry
from gvi.core.types import AssetProfile, ConversionRequest, TargetKind


@dataclass(frozen=True)
class PipelinePlan:
    id: str
    steps: tuple[str, ...]
    reason: str
    parallel_groups: tuple[tuple[str, ...], ...] = ()


class PipelinePlanner:
    """Deterministic, capability-based planner."""

    def __init__(self, registry: PluginRegistry) -> None:
        self.registry = registry

    def plan(self, profile: AssetProfile, request: ConversionRequest) -> PipelinePlan:
        ext = (profile.extension or "").lower()
        preset = request.preset
        target = request.target
        steps: list[str] = []
        reason_parts: list[str] = []

        # ---- 1) Detector (always first) -------------------------------------
        # NOTE: The orchestrator runs detector.asset_probe internally before the
        # planner executes. We do not add it here to avoid running it twice with
        # incompatible payload types.

        # ---- 2) Parser (if structured) ---------------------------------------
        if ext in {".svg", ".svgz"} or profile.mime_type == "image/svg+xml":
            steps.append("parser.svg")
            reason_parts.append("SVG parser keeps vector/text structure and rasterizes a fidelity background.")
        elif ext == ".pdf":
            steps.append("parser.pdf")
            reason_parts.append("PDF parser rasterizes the requested page for image reconstruction.")
        elif ext in {".psd", ".psb"}:
            steps.append("parser.psd")
            reason_parts.append("PSD parser extracts the composite plus named layers when psd-tools is installed.")

        # ---- 3) Analyzers (raster heuristics + spritesheet detection) ------
        if profile.kind.value == "raster" or ext in {".pdf", ".psd", ".psb"}:
            steps.append("analyzer.raster_heuristics")
            reason_parts.append("Raster heuristics detect text/UI/pixel-art and set semantic hints.")
            if self.registry.has("analyzer.spritesheet"):
                steps.append("analyzer.spritesheet")
                reason_parts.append("Spritesheet analyzer detects grid layouts for optimal slicing.")

        # ---- 4) Semantic detection (YOLO) -----------------------------------
        if (
            request.options.semantic_detection
            and target != TargetKind.GODOT_TILEMAP
            and self.registry.has("segmenter.yolo")
        ):
            steps.append("segmenter.yolo")
            reason_parts.append("YOLO semantic detector labels each candidate with class (person/button/panel/...).")

        # ---- 5) Primary segmentation ----------------------------------------
        steps.extend(self._primary_segmenter_steps(profile, request))

        # ---- 6) OCR (text extraction) ---------------------------------------
        if (
            request.options.include_text_elements
            and target in {TargetKind.GODOT_CONTROL, TargetKind.GODOT_RICHTEXT, TargetKind.GODOT_THEME}
            and self.registry.has("ocr.easyocr")
        ):
            steps.append("ocr.easyocr")
            reason_parts.append("OCR extracts readable text so labels are editable Godot nodes.")

        # ---- 7) Post-processing --------------------------------------------
        if self.registry.has("postprocess.hierarchy"):
            steps.append("postprocess.hierarchy")
            reason_parts.append("Hierarchy builder groups children under logical parents (frames contain sprites).")
        if self.registry.has("postprocess.theme") and target == TargetKind.GODOT_THEME:
            steps.append("postprocess.theme")
            reason_parts.append("Theme extractor builds a Godot Theme resource from dominant colors and fonts.")
        if self.registry.has("postprocess.tilemap") and target == TargetKind.GODOT_TILEMAP:
            steps.append("postprocess.tilemap")
            reason_parts.append("Tilemap builder slices the source image into a real TileSet with autotiling + physics.")

        # ---- 8) Exporters --------------------------------------------------
        steps.append("exporter.godot.scene")
        if target == TargetKind.GODOT_TILEMAP and self.registry.has("exporter.godot.tileset"):
            steps.append("exporter.godot.tileset")
            reason_parts.append("TileSet exporter writes a real .tres TileSet resource (with physics/navigation if requested).")

        # ---- 9) Validation --------------------------------------------------
        if self.registry.has("validator.scene"):
            steps.append("validator.scene")

        return PipelinePlan(
            id=" -> ".join(steps),
            steps=tuple(steps),
            reason=" ".join(reason_parts),
        )

    # ------------------------------------------------------------------ helpers
    def _primary_segmenter_steps(self, profile: AssetProfile, request: ConversionRequest) -> list[str]:
        """Decide between SAM 2, OpenCV, or both."""
        steps: list[str] = []
        preset = request.preset
        opts = request.options

        if opts.sam2_enabled or preset == "fidelity":
            if self.registry.has("segmenter.sam2"):
                steps.append("segmenter.sam2")
            elif self.registry.has("segmenter.opencv"):
                steps.append("segmenter.opencv")
        elif preset == "semantic":
            # YOLO already covered, OpenCV adds shape refinement
            if self.registry.has("segmenter.opencv"):
                steps.append("segmenter.opencv")
        elif preset == "tilemap":
            # Tilemap builder slices; no segmentation needed for this route
            pass
        else:
            if self.registry.has("segmenter.opencv"):
                steps.append("segmenter.opencv")

        if not steps and self.registry.has("segmenter.opencv"):
            steps.append("segmenter.opencv")

        return steps

    def rank_candidates(self, profile: AssetProfile, request: ConversionRequest) -> list[tuple[str, float]]:
        ranked: list[tuple[str, float]] = []
        for plugin in self.registry.all():
            score = plugin.can_handle(profile, request)
            if score:
                ranked.append((plugin.capability().id, score))
        return sorted(ranked, key=lambda item: item[1], reverse=True)