"""Shared types and pydantic schemas used across the GVI pipeline."""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class AssetKind(str, Enum):
    RASTER = "raster"
    VECTOR = "vector"
    DOCUMENT = "document"
    DESIGN = "design"
    PSD = "psd"
    FIGMA = "figma"
    SPRITESHEET = "spritesheet"
    TILEMAP_SOURCE = "tilemap_source"
    UNKNOWN = "unknown"


class TargetKind(str, Enum):
    """Supported Godot 4 output targets."""

    GODOT_CONTROL = "godot.control"
    GODOT_NODE2D = "godot.node2d"
    GODOT_SPRITE2D = "godot.sprite2d"
    GODOT_TILEMAP = "godot.tilemap"
    GODOT_RICHTEXT = "godot.richtext"
    GODOT_THEME = "godot.theme"
    GODOT_ANIMATION = "godot.animation"


class CapabilityType(str, Enum):
    DETECTOR = "detector"
    PARSER = "parser"
    ANALYZER = "analyzer"
    SEMANTIC = "semantic"
    SEGMENTER = "segmenter"
    OCR = "ocr"
    TRANSFORMER = "transformer"
    POSTPROCESSOR = "postprocessor"
    EXPORTER = "exporter"
    VALIDATOR = "validator"


class AssetProfile(BaseModel):
    """Discovered metadata about an input asset."""

    path: Path | None = None
    mime_type: str | None = None
    extension: str | None = None
    kind: AssetKind = AssetKind.UNKNOWN

    width: int | None = None
    height: int | None = None
    channels: int | None = None
    dpi: tuple[int, int] | None = None
    page_count: int | None = None

    has_alpha: bool = False
    has_text: bool | None = None
    has_ui_layout: bool | None = None
    has_pixel_art: bool | None = None
    has_vector_data: bool | None = None
    has_layers: bool | None = None
    is_spritesheet: bool | None = None
    is_tilemap_source: bool | None = None

    quality_score: float | None = Field(default=None, ge=0, le=1)

    # Semantic predictions from analyzer (UI-like / character-like / sprite-like)
    semantic_hint: str | None = None
    scene_kind: str | None = None

    notes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversionOptions(BaseModel):
    """Advanced knobs for the pipeline."""

    # Background / source preservation
    include_background: bool = True
    background_mode: Literal["source", "solid", "transparent", "none"] = "source"

    # Segmentation
    max_elements: int = 500
    min_area_ratio: float = Field(default=0.0001, ge=0.0, le=1.0)
    merge_overlaps: bool = True
    include_text_elements: bool = True
    text_detection_mode: Literal["easyocr", "auto", "none"] = "auto"
    # v1.1: defaults to False. yolov8n-seg is COCO-pretrained (80 real-world
    # classes: person, car, dog, bottle...). For GVI's actual target content
    # -- game sprites, UI mockups, product/icon cutouts -- essentially none
    # of those classes ever fire, so running it by default only costs a
    # multi-second model-download/inference attempt for ~zero useful
    # detections. Flip it on explicitly if your input is an actual photo of
    # real-world scenes/objects that overlap with COCO's classes.
    semantic_detection: bool = False
    semantic_model: Literal["yolov8n", "yolov8s", "yolov11n"] = "yolov8n"

    # SAM2
    sam2_enabled: bool = False
    sam2_model: Literal["sam2_hiera_tiny", "sam2_hiera_small", "sam2_hiera_base_plus", "sam2_hiera_large"] = "sam2_hiera_small"
    sam2_points_per_side: int = 32

    # Output
    copy_assets: bool = True
    make_debug_overlay: bool = False
    export_manifest: bool = True
    godot_version: Literal["4.x", "3.x"] = "4.x"
    texture_filter: Literal["inherit", "nearest", "linear"] = "inherit"
    preserve_pixel_art: bool = True
    center_sprites: bool = True

    # Tilemap
    tilemap_tile_size: int = 32
    tilemap_physics: bool = True
    tilemap_navigation: bool = False
    tilemap_autotile: bool = True
    tilemap_auto_detect_grid: bool = True

    # Theme
    theme_extract_colors: bool = True
    theme_extract_fonts: bool = True

    # Animation
    animation_frame_size: int | None = None
    animation_frames: int | None = None
    animation_fps: int = 12

    # Input options
    pdf_page: int = Field(default=0, ge=0)
    pdf_dpi: int = Field(default=160, ge=72, le=600)
    svg_rasterize: bool = True

    # Behavior
    fail_fast: bool = False
    warnings_as_errors: bool = False

    # Cache
    models_dir: Path = Field(default_factory=lambda: Path.home() / ".cache" / "gvi" / "models")


class ConversionRequest(BaseModel):
    """A user request to convert an asset."""

    input_path: Path
    target: TargetKind
    output_dir: Path
    preset: Literal["fast", "balanced", "fidelity", "semantic", "tilemap"] = "balanced"
    user_hints: dict[str, Any] = Field(default_factory=dict)
    options: ConversionOptions = Field(default_factory=ConversionOptions)

    @field_validator("input_path")
    @classmethod
    def _input_path_exists(cls, value: Path) -> Path:
        if not value.exists():
            raise ValueError(f"Input asset does not exist: {value}")
        return value


class PipelineStepResult(BaseModel):
    """One plugin execution result inside a pipeline."""

    name: str
    ok: bool = True
    data: Any = None
    artifacts: dict[str, Path] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class ConversionResult(BaseModel):
    """Final conversion result."""

    request: ConversionRequest
    profile: AssetProfile
    pipeline_id: str
    output_dir: Path
    artifacts: dict[str, Path] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


class DetectedElement(BaseModel):
    """A detected visual element extracted from an image."""

    id: str
    element_type: str  # sprite, text, frame, panel, button, background, character, ...
    bounds: tuple[int, int, int, int]  # x, y, w, h
    rotated_rect: dict[str, Any] | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    mask_path: Path | None = None
    asset_path: Path | None = None
    text_content: str | None = None
    text_language: str | None = None
    z_index: int = 0
    parent_id: str | None = None
    locked: bool = False
    semantic_class: str | None = None  # YOLO class if known
    source: str | None = None  # which step produced it
    metadata: dict[str, Any] = Field(default_factory=dict)


class SegmentationResult(BaseModel):
    """Result of a segmentation operation."""

    elements: list[DetectedElement] = Field(default_factory=list)
    num_elements: int = 0
    coverage_ratio: float = 0.0
    method_used: str = ""
    processing_time_ms: float = 0.0
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    def sort_and_index(self) -> "SegmentationResult":
        self.elements.sort(key=lambda e: (e.z_index, e.bounds[1], e.bounds[0], e.bounds[2] * e.bounds[3]))
        for index, elem in enumerate(self.elements):
            elem.z_index = index
        self.num_elements = len(self.elements)
        return self


class ValidationResult(BaseModel):
    """Result of scene validation."""

    is_valid: bool = False
    ssim_score: float | None = None
    psnr_score: float | None = None
    iou_score: float | None = None
    coverage_ratio: float | None = None
    overlap_violations: list[dict[str, Any]] = Field(default_factory=list)
    out_of_bounds: list[str] = Field(default_factory=list)
    texture_limit_violations: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    godot_headless: dict[str, Any] | None = None  # see _godot_headless.py; None when not attempted


class BenchmarkResult(BaseModel):
    """Result of a benchmark run."""

    test_name: str
    dataset_size: int
    avg_processing_time_ms: float = 0.0
    avg_ssim: float = 0.0
    avg_psnr: float = 0.0
    avg_iou: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    method: str = ""
    details: list[dict[str, Any]] = Field(default_factory=list)