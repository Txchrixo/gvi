"""Raster heuristics — classifies a raster image into UI / sprite / photo / pixel-art
and detects text presence and edge density.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageStat

from gvi.core.plugin import Capability, Plugin, PluginContext
from gvi.core.types import AssetProfile, CapabilityType, PipelineStepResult


class RasterHeuristicsPlugin(Plugin):
    def capability(self) -> Capability:
        return Capability(
            id="analyzer.raster_heuristics",
            type=CapabilityType.ANALYZER,
            name="Raster heuristics: text/UI/pixel-art/quality",
            supported_mime_types=["image/png", "image/jpeg", "image/webp", "image/bmp"],
            supported_extensions=[".png", ".jpg", ".jpeg", ".webp", ".bmp"],
            priority=80,
            provides=["raster_features"],
        )

    def run(self, payload, ctx: PluginContext) -> PipelineStepResult:
        profile: AssetProfile = payload if isinstance(payload, AssetProfile) else payload["profile"]
        if not profile.path:
            return PipelineStepResult(name="raster_heuristics", ok=False, warnings=["No path"])

        try:
            img = cv2.imread(str(profile.path), cv2.IMREAD_UNCHANGED)
            if img is None:
                return PipelineStepResult(name="raster_heuristics", ok=False, warnings=["OpenCV could not read image"])

            h, w = img.shape[:2]
            bgr = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if bgr.shape[2] == 4:
                bgr = bgr[:, :, :3]

            scale = min(1.0, 768 / max(w, h))
            small = cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA) if scale < 1 else bgr
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 60, 140)
            edge_density = float(np.mean(edges > 0))

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rects = 0
            text_like = 0
            for c in contours[:1000]:
                cx, cy, cw, ch = cv2.boundingRect(c)
                area = cv2.contourArea(c)
                if cw > 20 and ch > 8:
                    rects += 1
                if 3 <= ch <= 60 and 3 <= cw <= 220 and area / max(cw * ch, 1) < 0.65:
                    text_like += 1

            with Image.open(profile.path) as pil:
                rgb = pil.convert("RGB")
                pal_img = rgb.resize((min(160, pil.width), min(160, pil.height)))
                colors = pal_img.getcolors(maxcolors=8192) or []
                palette_count = len(colors)
                stat = ImageStat.Stat(rgb.resize((min(256, pil.width), min(256, pil.height))))
                channel_std = float(sum(stat.stddev) / max(len(stat.stddev), 1))

            profile.has_pixel_art = bool((profile.width or w) <= 512 and palette_count < 256)
            profile.has_ui_layout = bool(rects >= 4 or (edge_density > 0.05 and channel_std < 70))
            profile.has_text = bool(text_like >= 8 or (profile.has_ui_layout and edge_density > 0.035))

            # Decide semantic hint
            if profile.has_ui_layout and text_like >= 8:
                profile.semantic_hint = "ui"
                profile.scene_kind = "ui_flat"
            elif profile.has_pixel_art:
                profile.semantic_hint = "pixel_art"
                profile.scene_kind = "pixel_art"
            elif edge_density > 0.13:
                profile.semantic_hint = "object"
                profile.scene_kind = "high_contrast"
            else:
                profile.semantic_hint = "scene"
                profile.scene_kind = "mixed"

            profile.quality_score = max(0.0, min(1.0, 1.0 - edge_density * 0.35))
            profile.metadata.update({
                "edge_density": edge_density,
                "rect_candidates": rects,
                "text_like_components": text_like,
                "palette_count": palette_count,
                "channel_stddev": channel_std,
                "recommended_segmentation": "ui" if profile.has_ui_layout else "object",
            })

            return PipelineStepResult(name="raster_heuristics", data=profile, metrics={"edge_density": edge_density})

        except Exception as exc:  # noqa: BLE001
            return PipelineStepResult(name="raster_heuristics", ok=False, warnings=[f"Heuristics failed: {exc}"])