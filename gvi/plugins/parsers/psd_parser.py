"""PSD parser — extracts the composite plus named layers with hierarchy."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from gvi.core.plugin import Capability, Plugin, PluginContext
from gvi.core.types import AssetKind, AssetProfile, CapabilityType, DetectedElement, PipelineStepResult, SegmentationResult


class PSDParserPlugin(Plugin):
    def capability(self) -> Capability:
        return Capability(
            id="parser.psd",
            type=CapabilityType.PARSER,
            name="Advanced PSD/PSB parser",
            supported_extensions=[".psd", ".psb"],
            priority=92,
            provides=["raster_profile", "segmentation", "ir_hints"],
            tags={"design", "layered"},
        )

    def run(self, payload: dict[str, Any], ctx: PluginContext) -> PipelineStepResult:
        profile: AssetProfile = payload["profile"]
        if not profile.path:
            return PipelineStepResult(name="psd_parser", ok=False, errors=["Missing PSD path"])

        out_dir = Path(ctx.work_dir) / "assets"
        out_dir.mkdir(parents=True, exist_ok=True)
        warnings: list[str] = []

        try:
            from psd_tools import PSDImage
        except ImportError:
            return PipelineStepResult(
                name="psd_parser", ok=False,
                errors=["psd-tools is required for PSD import. Install with: python -m pip install 'gvi[psd]'"],
            )

        psd = PSDImage.open(profile.path)
        composite_path = out_dir / f"{profile.path.stem}_composite.png"
        psd.composite().save(composite_path)

        elements: list[DetectedElement] = [DetectedElement(
            id="background_source",
            element_type="background",
            bounds=(0, 0, int(psd.width), int(psd.height)),
            asset_path=composite_path,
            z_index=-1000,
            locked=True,
            source="psd_composite",
            metadata={"source": "psd_composite"},
        )]

        idx = 0

        def walk_layers(container, parent_id: str | None = None) -> None:
            nonlocal idx
            for layer in container:
                if layer.is_group():
                    group_id = f"psd_group_{idx:04d}"
                    idx += 1
                    # group placeholder so hierarchy is preserved
                    elements.append(DetectedElement(
                        id=group_id,
                        element_type="frame",
                        bounds=(0, 0, int(psd.width), int(psd.height)),
                        z_index=200 + idx,
                        confidence=0.5,
                        parent_id=parent_id,
                        source="psd",
                        metadata={"layer_name": layer.name, "layer_kind": "group"},
                    ))
                    walk_layers(layer, parent_id=group_id)
                    continue
                if not layer.is_visible():
                    continue
                bbox = layer.bbox
                if bbox.width <= 0 or bbox.height <= 0:
                    continue
                try:
                    img = layer.composite()
                    if img is None:
                        continue
                    safe = self._safe(layer.name)
                    asset_path = out_dir / f"layer_{idx:04d}_{safe}.png"
                    img.save(asset_path)
                    kind = "type" if getattr(layer, "kind", "") == "type" else "sprite"
                    text_content = None
                    try:
                        engine = getattr(layer, "engine_dict", None)
                        if engine and "Editor" in str(engine):
                            text_content = str(engine.get("Editor", {}).get("Text", ""))
                    except Exception:  # noqa: BLE001
                        pass
                    elements.append(DetectedElement(
                        id=f"psd_layer_{idx:04d}",
                        element_type="text" if kind == "type" else "sprite",
                        bounds=(int(bbox.x1), int(bbox.y1), int(bbox.width), int(bbox.height)),
                        asset_path=asset_path,
                        z_index=idx,
                        confidence=0.95,
                        parent_id=parent_id,
                        text_content=text_content,
                        source="psd",
                        metadata={
                            "layer_name": layer.name,
                            "layer_kind": kind,
                            "opacity": float(getattr(layer, "opacity", 1.0) or 1.0),
                            "visible": True,
                        },
                    ))
                    idx += 1
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"Could not extract PSD layer {layer.name!r}: {exc}")

        walk_layers(list(psd))

        raster_profile = profile.model_copy(deep=True)
        raster_profile.path = composite_path
        raster_profile.kind = AssetKind.RASTER
        raster_profile.extension = ".png"
        raster_profile.mime_type = "image/png"
        raster_profile.width = int(psd.width)
        raster_profile.height = int(psd.height)
        raster_profile.has_alpha = True
        raster_profile.metadata.update({"source_psd": str(profile.path), "num_layers_extracted": idx})

        segmentation = SegmentationResult(
            elements=elements,
            num_elements=len(elements),
            coverage_ratio=1.0,
            method_used="psd_layers",
            diagnostics={"num_layers_extracted": idx},
        ).sort_and_index()

        return PipelineStepResult(
            name="psd_parser",
            data={**payload, "profile": raster_profile, "segmentation": segmentation, "parsed_asset": composite_path},
            artifacts={"parsed_asset": composite_path, "assets_dir": out_dir},
            metrics={"num_layers": float(idx)},
            warnings=warnings,
        )

    def _safe(self, name: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name or "layer")[:64]