"""Spritesheet analyzer — detects tiled/grid spritesheets and frame layout.

Sets ``is_spritesheet`` and computes ``frame_size`` if it can detect a grid.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from gvi.core.plugin import Capability, Plugin, PluginContext
from gvi.core.types import AssetProfile, AssetKind, CapabilityType, PipelineStepResult


class SpritesheetAnalyzer(Plugin):
    def capability(self) -> Capability:
        return Capability(
            id="analyzer.spritesheet",
            type=CapabilityType.ANALYZER,
            name="Spritesheet grid detector",
            supported_mime_types=["image/png", "image/jpeg", "image/webp", "image/bmp"],
            supported_extensions=[".png", ".jpg", ".jpeg", ".webp", ".bmp"],
            priority=60,
            provides=["spritesheet_features"],
        )

    def run(self, payload, ctx: PluginContext) -> PipelineStepResult:
        profile: AssetProfile = payload if isinstance(payload, AssetProfile) else payload["profile"]
        if not profile.path:
            return PipelineStepResult(name="spritesheet", ok=False, warnings=["No path"])

        try:
            with Image.open(profile.path) as img:
                w, h = img.size
                alpha = img.convert("RGBA").split()[-1] if img.mode in {"RGBA", "LA"} else None
                if alpha is None:
                    return PipelineStepResult(name="spritesheet", data=profile)
                a = np.array(alpha)

                # Score candidate grid sizes by looking for alpha gaps at multiples of size.
                best_score = 0.0
                best_size = 0
                for size in (8, 16, 24, 32, 48, 64, 96, 128):
                    score = self._score_grid(a, size)
                    if score > best_score:
                        best_score = score
                        best_size = size

                if best_score > 0.4 and best_size > 0:
                    profile.is_spritesheet = True
                    profile.kind = AssetKind.SPRITESHEET
                    profile.metadata.update(
                        {
                            "spritesheet_frame_size": best_size,
                            "spritesheet_cols": w // best_size,
                            "spritesheet_rows": h // best_size,
                            "spritesheet_score": float(best_score),
                        }
                    )
                return PipelineStepResult(
                    name="spritesheet",
                    data=profile,
                    metrics={"spritesheet_score": float(best_score)},
                )
        except Exception as exc:  # noqa: BLE001
            return PipelineStepResult(name="spritesheet", ok=False, warnings=[f"Spritesheet analyzer failed: {exc}"])

    def _score_grid(self, alpha: np.ndarray, size: int) -> float:
        """Higher score = more aligned alpha boundaries at this grid size."""
        h, w = alpha.shape
        if size >= min(h, w):
            return 0.0
        # Compute row/col "alpha minima" — borders between tiles tend to be transparent.
        row_means = alpha.mean(axis=1)
        col_means = alpha.mean(axis=0)
        # Count how many row/col positions matching the grid have a low alpha dip.
        dips_row = sum(1 for r in range(size, h, size) if row_means[r] < row_means.mean() * 0.5)
        dips_col = sum(1 for c in range(size, w, size) if col_means[c] < col_means.mean() * 0.5)
        total = (h // size) + (w // size)
        if total <= 0:
            return 0.0
        return min(1.0, (dips_row + dips_col) / total)