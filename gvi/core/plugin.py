"""Plugin base class + capability model."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable

from pydantic import BaseModel, Field

from gvi.core.types import (
    AssetProfile,
    CapabilityType,
    ConversionRequest,
    PipelineStepResult,
    TargetKind,
)


class Capability(BaseModel):
    id: str
    type: CapabilityType
    name: str
    supported_mime_types: list[str] = Field(default_factory=list)
    supported_extensions: list[str] = Field(default_factory=list)
    supported_targets: list[TargetKind | str] = Field(default_factory=list)
    requires: list[str] = Field(default_factory=list)
    provides: list[str] = Field(default_factory=list)
    priority: int = 50
    cost: float = 1.0
    quality: float = 0.5
    tags: set[str] = Field(default_factory=set)


class PluginContext(BaseModel):
    work_dir: str
    cache_dir: str
    options: dict[str, Any] = Field(default_factory=dict)


class Plugin(ABC):
    """Base class for every GVI plugin."""

    @abstractmethod
    def capability(self) -> Capability: ...

    def can_handle(self, profile: AssetProfile, request: ConversionRequest | None = None) -> float:
        cap = self.capability()
        score = 0.0
        if profile.mime_type and profile.mime_type in cap.supported_mime_types:
            score += 0.4
        if profile.extension and profile.extension.lower() in cap.supported_extensions:
            score += 0.3
        if request:
            tgt = request.target if request.target in cap.supported_targets else request.target.value
            if tgt in cap.supported_targets:
                score += 0.3
        return min(score, 1.0)

    @abstractmethod
    def run(self, payload: Any, ctx: PluginContext) -> PipelineStepResult: ...


def load_builtin_plugins() -> Iterable[Plugin]:
    """Lazy-load every built-in plugin so missing optional deps don't crash import."""
    from gvi.plugins.detectors.asset_probe import AssetProbePlugin
    from gvi.plugins.analyzers.raster_heuristics import RasterHeuristicsPlugin
    from gvi.plugins.analyzers.spritesheet_analyzer import SpritesheetAnalyzer

    plugins: list[Plugin] = [
        AssetProbePlugin(),
        RasterHeuristicsPlugin(),
        SpritesheetAnalyzer(),
    ]

    # Parsers
    try:
        from gvi.plugins.parsers.svg_parser import SVGParserPlugin
        plugins.append(SVGParserPlugin())
    except Exception:
        pass
    try:
        from gvi.plugins.parsers.pdf_parser import PDFParserPlugin
        plugins.append(PDFParserPlugin())
    except Exception:
        pass
    try:
        from gvi.plugins.parsers.psd_parser import PSDParserPlugin
        plugins.append(PSDParserPlugin())
    except Exception:
        pass

    # OCR
    try:
        from gvi.ocr.text_extractor import OCRExtractorPlugin
        plugins.append(OCRExtractorPlugin())
    except Exception:
        pass

    # Semantic
    try:
        from gvi.plugins.segmenters.yolo_segmenter import YOLOSemanticSegmenter
        plugins.append(YOLOSemanticSegmenter())
    except Exception:
        pass

    # Segmenters
    from gvi.plugins.segmenters.opencv_segmenter import OpenCVSegmenter
    plugins.append(OpenCVSegmenter())

    try:
        from gvi.plugins.segmenters.sam2_segmenter import SAM2Segmenter
        plugins.append(SAM2Segmenter())
    except Exception:
        pass

    # Post-processing
    try:
        from gvi.postprocess.hierarchy_builder import HierarchyBuilder
        plugins.append(HierarchyBuilder())
    except Exception:
        pass
    try:
        from gvi.postprocess.theme_extractor import ThemeExtractor
        plugins.append(ThemeExtractor())
    except Exception:
        pass
    try:
        from gvi.postprocess.tilemap_builder import TilemapBuilder
        plugins.append(TilemapBuilder())
    except Exception:
        pass

    # Exporters
    from gvi.plugins.exporters.godot_scene import GodotSceneExporter
    plugins.append(GodotSceneExporter())

    try:
        from gvi.plugins.exporters.godot_tileset import GodotTilesetExporter
        plugins.append(GodotTilesetExporter())
    except Exception:
        pass

    # Validators
    from gvi.plugins.validators.scene_validator import SceneValidator
    plugins.append(SceneValidator())

    return plugins