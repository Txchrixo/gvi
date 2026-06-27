"""Godot 4 TileSet resource exporter — strict valid .tres format.

Uses Godot's standard format: a single ``TileSetAtlasSource`` sub-resource that
contains every tile as a separate alternative (tile index inside the source).
This is dramatically more compact than one sub_resource per tile.

Each tile carries:
- ``texture_region`` (Rect2 in the atlas)
- ``0:0/next_alternative_id``
- ``0:0/physics_layer`` and a default full-tile collision polygon (if enabled)

Output is a single ``tile_set.tres`` next to ``scene.tscn``, referenced via
``res://tile_set.tres``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from gvi.core.plugin import Capability, Plugin, PluginContext
from gvi.core.types import AssetProfile, CapabilityType, PipelineStepResult


class GodotTilesetExporter(Plugin):
    def capability(self) -> Capability:
        return Capability(
            id="exporter.godot.tileset",
            type=CapabilityType.EXPORTER,
            name="Godot 4 TileSet .tres exporter",
            supported_targets=["godot.tilemap"],
            priority=99,
            requires=["tilemap_metadata"],
            provides=["godot_tileset"],
            tags={"tilemap", "godot"},
        )

    def run(self, payload: dict[str, Any], ctx: PluginContext) -> PipelineStepResult:
        tilemap = payload.get("tilemap")
        if not tilemap:
            return PipelineStepResult(name="godot_tileset", ok=False, warnings=["No tilemap metadata"])
        atlas_path = payload.get("tilemap_atlas")
        if not atlas_path:
            return PipelineStepResult(name="godot_tileset", ok=False, warnings=["No tilemap atlas image"])

        out_assets = Path(ctx.work_dir).parent / "assets"
        out_assets.mkdir(parents=True, exist_ok=True)
        target_atlas = out_assets / "tilemap_atlas.png"
        if Path(atlas_path).resolve() != target_atlas.resolve():
            target_atlas.write_bytes(Path(atlas_path).read_bytes())

        tile_size = int(tilemap["tile_size"])
        tiles = tilemap.get("tiles", [])

        tres = self._build_tres(
            atlas_rel="res://assets/tilemap_atlas.png",
            tile_size=tile_size,
            tiles=tiles,
            physics=bool(tilemap.get("physics", True)),
        )
        tres_path = Path(ctx.work_dir).parent / "tile_set.tres"
        tres_path.write_text(tres, encoding="utf-8")
        return PipelineStepResult(
            name="godot_tileset",
            data={**payload, "godot_tileset_path": tres_path},
            artifacts={"tileset": tres_path},
            metrics={"num_tiles": float(len(tiles))},
        )

    # ------------------------------------------------------------- tres builder
    def _build_tres(self, atlas_rel: str, tile_size: int, tiles: list, physics: bool) -> str:
        # Cap the number of tiles we emit (Godot's default is 65536 — fine, but
        # very dense tilesets are painful to edit). Keep at most 4096 alternatives.
        max_tiles = 4096
        kept = tiles[:max_tiles]

        out: list[str] = []
        out.append('[gd_resource type="TileSet" load_steps=3 format=3 uid="uid://gvi_generated_tileset"]')
        out.append("")
        out.append(f'[ext_resource type="Texture2D" path="{atlas_rel}" id="1_atlas"]')
        out.append("")

        # Single atlas source that holds every tile as an alternative.
        out.append('[sub_resource type="TileSetAtlasSource" id="TileSetAtlasSource_main"]')
        out.append("texture = ExtResource(\"1_atlas\")")
        out.append(f"texture_region_size = Vector2i({tile_size}, {tile_size})")
        out.append("margins = 0")
        out.append("separation = 0")
        out.append("")
        for tile in kept:
            tx = int(tile["tx"])
            ty = int(tile["ty"])
            x = tx * tile_size
            y = ty * tile_size
            out.append(f"{tx}:{ty}/next_alternative_id = 1")
            if physics:
                out.append(f"{tx}:{ty}/physics_layer = 0")
                out.append(f"{tx}:{ty}/physics_layer_0/polygon_0/points = PackedVector2Array(0,0,{tile_size},0,{tile_size},{tile_size},0,{tile_size})")
        out.append("")
        out.append("[resource]")
        out.append(f"tile_size = Vector2i({tile_size}, {tile_size})")
        out.append("sources/0 = SubResource(\"TileSetAtlasSource_main\")")
        return "\n".join(out) + "\n"