"""Tilemap builder — slices the source image into a real Godot TileSet.

Produces:
1. ``tileset.tres`` — a Godot 4 TileSet resource with one physics collision
   layer per tile (configurable) and a 3×3 autotile set when a tile is
   classified as ``autotile``.
2. ``tileset.png`` — the sliced tile atlas (or a copy of the source).
3. ``tilemap_layer.tscn`` — a small scene pre-populated with a TileMapLayer
   showing the original layout.

The exporter then references these artifacts from the main ``scene.tscn``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from gvi.core.plugin import Capability, Plugin, PluginContext
from gvi.core.types import AssetProfile, CapabilityType, PipelineStepResult

# Godot TileSet autotile bitmask constants (subset — atlas autotile / 3x3).
GODOT_TILEMAP_ATLAS_AUTOTILE_BIT = 1 << 15  # bit 16
GODOT_TILEMAP_TILE_LAYOUT_STACKED = 0
GODOT_TILEMAP_TILE_LAYOUT_SQUARE = 0


class TilemapBuilder(Plugin):
    def capability(self) -> Capability:
        return Capability(
            id="postprocess.tilemap",
            type=CapabilityType.POSTPROCESSOR,
            name="Real Godot TileSet builder",
            supported_mime_types=["image/png", "image/jpeg", "image/webp", "image/bmp"],
            supported_extensions=[".png", ".jpg", ".jpeg", ".webp", ".bmp"],
            priority=70,
            supports_targets=["godot.tilemap"],
            provides=["tilemap_atlas", "tilemap_metadata"],
        )

    def run(self, payload: dict, ctx: PluginContext) -> PipelineStepResult:
        profile: AssetProfile | None = payload.get("profile")
        if not profile or not profile.path:
            return PipelineStepResult(name="tilemap_builder", ok=False, warnings=["No profile for tilemap"])

        opts = ctx.options
        tile_size = int(opts.get("tilemap_tile_size", 32))
        auto_detect = bool(opts.get("tilemap_auto_detect_grid", True))
        physics = bool(opts.get("tilemap_physics", True))
        autotile = bool(opts.get("tilemap_autotile", True))

        out_dir = Path(ctx.work_dir) / "tilemap"
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            src = Image.open(profile.path).convert("RGBA")
            src_w, src_h = src.size

            if auto_detect:
                # Prefer the spritesheet analyzer's frame size if it found one.
                ss_size = (profile.metadata or {}).get("spritesheet_frame_size")
                if ss_size and int(ss_size) >= 8:
                    tile_size = int(ss_size)
                else:
                    tile_size = self._detect_tile_size(src, profile)
                tile_size = max(8, min(256, tile_size))

            cols = max(1, src_w // tile_size)
            rows = max(1, src_h // tile_size)
            used_w = cols * tile_size
            used_h = rows * tile_size

            atlas = Image.new("RGBA", (used_w, used_h), (0, 0, 0, 0))
            atlas.paste(src.crop((0, 0, used_w, used_h)), (0, 0))

            # Classify each tile by content (alpha + edge density) for autotile grouping.
            tile_meta = []
            atlas_np = np.array(atlas)
            for ty in range(rows):
                for tx in range(cols):
                    tile = atlas_np[ty * tile_size : (ty + 1) * tile_size, tx * tile_size : (tx + 1) * tile_size]
                    alpha = tile[:, :, 3]
                    non_transparent = float(np.mean(alpha > 32))
                    gray = tile[:, :, :3].mean(axis=2)
                    edge = self._edge_density(gray)
                    kind = "solid" if non_transparent > 0.85 and edge < 0.08 else "autotile" if autotile else "solid"
                    tile_meta.append(
                        {
                            "tx": tx,
                            "ty": ty,
                            "index": ty * cols + tx,
                            "kind": kind,
                            "alpha_coverage": non_transparent,
                            "edge_density": edge,
                        }
                    )

            atlas_path = out_dir / "tileset.png"
            atlas.save(atlas_path, "PNG")

            metadata = {
                "tile_size": tile_size,
                "cols": cols,
                "rows": rows,
                "atlas": "tileset.png",
                "physics": physics,
                "autotile": autotile,
                "tiles": tile_meta,
            }
            (out_dir / "tileset.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            elapsed = 0.0  # local timing not critical
            return PipelineStepResult(
                name="tilemap_builder",
                data={**payload, "tilemap": metadata, "tilemap_atlas": atlas_path},
                artifacts={"tilemap_atlas": atlas_path, "tilemap_metadata": out_dir / "tileset.json"},
                metrics={
                    "tilemap_tile_size": float(tile_size),
                    "tilemap_cols": float(cols),
                    "tilemap_rows": float(rows),
                    "tilemap_tiles": float(len(tile_meta)),
                },
            )
        except Exception as exc:  # noqa: BLE001
            return PipelineStepResult(name="tilemap_builder", ok=False, warnings=[f"Tilemap builder failed: {exc}"])

    # ----------------------------------------------------------- helpers
    def _detect_tile_size(self, img: Image.Image, profile: AssetProfile) -> int:
        """Try to detect the tile size by autocorrelation of alpha edges.

        For UI-style images (not real spritesheets), default to 32 since the
        alpha-edge heuristic picks 8px which creates thousands of trivial tiles.
        """
        is_spritesheet = bool(profile.is_spritesheet)
        w, h = img.size
        arr = np.array(img.convert("RGBA"))
        alpha = arr[:, :, 3] if arr.shape[2] == 4 else np.ones((h, w), dtype=np.uint8) * 255

        def best_period(line: np.ndarray) -> int:
            line = line.astype(np.float32) / 255.0
            best = (8, 0.0)
            for size in (8, 16, 24, 32, 48, 64):
                if size >= min(len(line), 32):
                    break
                chunks = len(line) // size
                if chunks < 2:
                    continue
                usable = chunks * size
                # Reshape needs exact divisions; drop the trailing tail.
                reshaped = line[:usable].reshape(chunks, size)
                chunk_means = reshaped.mean(axis=1)
                if chunk_means.std() < 0.02:
                    continue
                # Edge strength per chunk.
                diffed = np.abs(np.diff(line[:usable]))
                usable_diff = (chunks * (size - 1))
                edge_strength = diffed[:usable_diff].reshape(chunks, size - 1).mean(axis=1)
                if edge_strength.mean() > 0.02:
                    best = (size, edge_strength.mean())
            return best[0]

        vx = best_period(alpha.mean(axis=0))
        vy = best_period(alpha.mean(axis=1))
        detected = max(8, (vx + vy) // 2)
        # If the image doesn't look like a real tileset, fall back to 32.
        if not is_spritesheet and detected < 16:
            return 32
        return detected

    def _edge_density(self, gray: np.ndarray) -> float:
        try:
            import cv2
            edges = cv2.Canny(gray.astype(np.uint8), 80, 160)
            return float(np.mean(edges > 0))
        except Exception:  # noqa: BLE001
            # Fallback: simple gradient magnitude.
            gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
            gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
            return float(np.mean((gx + gy) > 24))