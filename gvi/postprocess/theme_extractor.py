"""Theme extractor — pulls dominant colors and font hints into a Godot Theme palette.

The extracted theme is consumed by the Godot exporter which writes a real
``theme.tres`` file alongside the scene.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from gvi.core.plugin import Capability, Plugin, PluginContext
from gvi.core.types import AssetProfile, CapabilityType, PipelineStepResult


class ThemeExtractor(Plugin):
    def capability(self) -> Capability:
        return Capability(
            id="postprocess.theme",
            type=CapabilityType.POSTPROCESSOR,
            name="Godot theme color/font extractor",
            supported_mime_types=["image/png", "image/jpeg", "image/webp", "image/bmp"],
            supported_extensions=[".png", ".jpg", ".jpeg", ".webp", ".bmp"],
            priority=65,
            provides=["theme"],
        )

    def run(self, payload: dict[str, Any], ctx: PluginContext) -> PipelineStepResult:
        profile: AssetProfile | None = payload.get("profile")
        if not profile or not profile.path:
            return PipelineStepResult(name="theme", ok=False, warnings=["No profile for theme extraction"])

        try:
            img = Image.open(profile.path).convert("RGB")
            small = img.resize((128, 128))
            colors = small.getcolors(maxcolors=128 * 128)
            if not colors:
                return PipelineStepResult(name="theme", ok=False, warnings=["Theme extractor: too few colors"])

            counts = Counter()
            for count, rgb in colors:
                # Quantize to 16-step buckets for stable palettes.
                q = tuple((c // 16) * 16 for c in rgb)
                counts[q] += count
            palette = counts.most_common(16)
            palette_hex = ["#%02x%02x%02x" % rgb for rgb, _ in palette]

            # Determine dominant family from background luminance.
            lum = np.mean([sum(rgb) / 3 for rgb, _ in palette[:3]])
            scheme = "dark" if lum < 90 else "light"

            # Estimate font: count near-horizontal strokes via aspect ratio of text-shaped elements.
            seg = payload.get("segmentation")
            font_hints: list[dict[str, Any]] = []
            if seg is not None:
                for elem in seg.elements:
                    if elem.element_type == "text" and elem.metadata.get("ocr_confidence"):
                        x, y, w, h = elem.bounds
                        if h >= 8 and h <= 96:
                            font_hints.append({"size_px": int(h), "weight": "regular", "family": "ThemeDefault"})

            output_dir = Path(ctx.work_dir)
            theme_data = {
                "scheme": scheme,
                "palette": palette_hex,
                "fonts": font_hints,
                "background_luminance": float(lum),
            }
            (output_dir / "theme.json").write_text(__import__("json").dumps(theme_data, indent=2), encoding="utf-8")
            return PipelineStepResult(
                name="theme",
                data={**payload, "theme": theme_data},
                artifacts={"theme_json": output_dir / "theme.json"},
                metrics={"palette_size": float(len(palette_hex))},
            )

        except Exception as exc:  # noqa: BLE001
            return PipelineStepResult(name="theme", ok=False, warnings=[f"Theme extractor failed: {exc}"])