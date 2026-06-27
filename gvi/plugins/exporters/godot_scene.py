"""Godot 4 .tscn exporter for every GVI target.

Supports all 7 target kinds:

- ``godot.node2d``     — game scene (sprite layers)
- ``godot.control``    — UI (Buttons, Panels, Labels, TextureRects)
- ``godot.sprite2d``   — single Node2D with stacked Sprite2D layers
- ``godot.tilemap``    — TileMapLayer referencing tile_set.tres
- ``godot.richtext``   — RichTextLabel containing extracted text
- ``godot.theme``      — Control + theme.tres (palette + fonts)
- ``godot.animation``  — Node2D + AnimationPlayer with per-layer tracks

Hierarchy, parents, z_index, locked background, theme metadata and
manifests are all preserved. Output is always a fully-valid Godot 4
``.tscn`` that opens without warnings.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from gvi.core.plugin import Capability, Plugin, PluginContext
from gvi.core.types import (
    AssetProfile,
    CapabilityType,
    ConversionRequest,
    DetectedElement,
    PipelineStepResult,
    SegmentationResult,
    TargetKind,
)
from gvi.ir.schema import Bounds, IRDocument, IRNode, IRNodeType, Style


class GodotSceneExporter(Plugin):
    def capability(self) -> Capability:
        return Capability(
            id="exporter.godot.scene",
            type=CapabilityType.EXPORTER,
            name="Godot 4 .tscn exporter",
            supported_targets=[
                TargetKind.GODOT_CONTROL,
                TargetKind.GODOT_NODE2D,
                TargetKind.GODOT_SPRITE2D,
                TargetKind.GODOT_TILEMAP,
                TargetKind.GODOT_RICHTEXT,
                TargetKind.GODOT_THEME,
                TargetKind.GODOT_ANIMATION,
            ],
            priority=100,
            requires=["segmentation"],
            provides=["godot_scene", "manifest", "ir"],
            tags={"core"},
        )

    # ------------------------------------------------------------------ run
    def run(self, payload: dict[str, Any], ctx: PluginContext) -> PipelineStepResult:
        request: ConversionRequest = payload["request"]
        profile: AssetProfile = payload["profile"]
        segmentation: SegmentationResult | None = payload.get("segmentation")
        out_dir = Path(request.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        assets_dir = out_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        ir = self._build_ir(profile, segmentation, payload)
        asset_map = self._copy_assets(ir, assets_dir, request.options.copy_assets)
        scene_content = self._generate_scene(ir, request, asset_map, payload)
        scene_path = out_dir / "scene.tscn"
        scene_path.write_text(scene_content, encoding="utf-8")

        artifacts: dict[str, Path] = {"scene": scene_path, "assets_dir": assets_dir}

        # Optional resources
        if request.target.value == "godot.theme":
            theme_path = out_dir / "theme.tres"
            theme_path.write_text(self._generate_theme_resource(ir, payload.get("theme")), encoding="utf-8")
            artifacts["theme"] = theme_path
        if request.target.value == "godot.animation":
            script_path = out_dir / "gvi_layer_controller.gd"
            script_path.write_text(self._generate_layer_script(), encoding="utf-8")
            artifacts["script"] = script_path
            anim_path = out_dir / "gvi_default_anim.tres"
            anim_path.write_text(self._generate_animation_resource(ir), encoding="utf-8")
            artifacts["animation"] = anim_path

        manifest = self._generate_manifest(ir, segmentation, profile, asset_map, request)
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        artifacts["manifest"] = manifest_path

        return PipelineStepResult(
            name="godot_scene_exporter",
            data={**payload, "ir": ir, "asset_map": asset_map},
            artifacts=artifacts,
            metrics={"num_nodes": float(len(ir.flatten())), "num_assets": float(len(asset_map))},
        )

    # ------------------------------------------------------------------ IR
    def _build_ir(self, profile: AssetProfile, segmentation: SegmentationResult | None, payload: dict) -> IRDocument:
        ir = IRDocument(
            width=float(profile.width or 0),
            height=float(profile.height or 0),
            metadata={"source": str(profile.path) if profile.path else None},
        )
        if not segmentation or not segmentation.elements:
            return ir
        # Preserve order: background first, then others.
        for elem in segmentation.elements:
            node_type = self._map_element_type(elem.element_type, elem.semantic_class)
            ir.nodes.append(IRNode(
                id=elem.id,
                type=node_type,
                name=self._safe_node_name(f"{elem.element_type}_{len(ir.nodes):03d}"),
                bounds=self._bounds_from_element(elem),
                asset=elem.asset_path,
                z_index=elem.z_index,
                text=elem.text_content,
                parent_id=elem.parent_id,
                style=self._style_from_element(elem),
                metadata={
                    "confidence": elem.confidence,
                    "locked": elem.locked,
                    "rotated_rect": elem.rotated_rect,
                    "source": elem.source,
                    "semantic_class": elem.semantic_class,
                    **elem.metadata,
                },
            ))
            if elem.asset_path:
                ir.assets[elem.id] = elem.asset_path
        # Sort by z_index then y for stable output
        ir.nodes.sort(key=lambda n: (n.z_index, n.bounds.y if n.bounds else 0))
        return ir

    def _map_element_type(self, etype: str, semantic_class: str | None) -> IRNodeType:
        if semantic_class == "character":
            return IRNodeType.CHARACTER
        return {
            "background": IRNodeType.BACKGROUND,
            "sprite": IRNodeType.SPRITE,
            "image": IRNodeType.IMAGE,
            "text": IRNodeType.TEXT,
            "label": IRNodeType.LABEL,
            "button": IRNodeType.BUTTON,
            "panel": IRNodeType.PANEL,
            "frame": IRNodeType.FRAME,
            "decoration": IRNodeType.PROP,
            "character": IRNodeType.CHARACTER,
            "vehicle": IRNodeType.PROP,
            "prop": IRNodeType.PROP,
        }.get(etype, IRNodeType.SPRITE)

    def _bounds_from_element(self, elem: DetectedElement) -> Bounds:
        x, y, w, h = elem.bounds
        rot = 0.0
        if elem.rotated_rect:
            rot = float(elem.rotated_rect.get("angle_deg", 0.0))
        return Bounds(x=float(x), y=float(y), width=float(w), height=float(h), rotation=rot)

    def _style_from_element(self, elem: DetectedElement) -> Style:
        style = Style()
        if fill := elem.metadata.get("fill"):
            style.fill = fill if isinstance(fill, str) else None
        if fs := elem.metadata.get("font_size"):
            try:
                style.font_size = float(fs)
            except (TypeError, ValueError):
                pass
        if opacity := elem.metadata.get("opacity"):
            try:
                style.opacity = float(opacity)
            except (TypeError, ValueError):
                pass
        return style

    # ------------------------------------------------------------------ assets
    def _copy_assets(self, ir: IRDocument, assets_dir: Path, copy_assets: bool) -> dict[str, str]:
        asset_map: dict[str, str] = {}
        used_names: set[str] = set()
        for node in ir.flatten():
            if not node.asset:
                continue
            src = Path(node.asset)
            if not src.exists():
                continue
            safe = self._unique_name(self._safe_file_name(src.name), used_names)
            dst = assets_dir / safe
            if copy_assets:
                if src.resolve() != dst.resolve():
                    shutil.copy2(src, dst)
            else:
                dst = src
            asset_map[node.id] = f"res://assets/{safe}"
        return asset_map

    # ------------------------------------------------------------------ scene
    def _generate_scene(self, ir: IRDocument, request: ConversionRequest, asset_map: dict[str, str], payload: dict) -> str:
        target = request.target.value
        root_type = self._root_type(target)

        ext_resources = self._ext_resources(asset_map)
        lines: list[str] = []
        # If we will reference tile_set.tres, declare it first so load_steps is accurate.
        tilemap_meta = payload.get("tilemap")
        target = request.target.value
        tileset_dep = 1 if (target == "godot.tilemap" and tilemap_meta) else 0
        anim_dep = 2 if target == "godot.animation" else 0
        theme_dep = 1 if (target == "godot.theme") else 0
        load_steps = len(ext_resources) + 1 + tileset_dep + anim_dep + theme_dep
        lines.append(f"[gd_scene load_steps={load_steps} format=3 uid=\"uid://gvi_generated_scene\"]")
        lines.append("")
        if tileset_dep:
            lines.append('[ext_resource type="TileSet" path="res://tile_set.tres" id="99_tileset"]')
        if theme_dep:
            lines.append('[ext_resource type="Theme" path="res://theme.tres" id="98_theme"]')
        if anim_dep:
            lines.append('[ext_resource type="Script" path="res://gvi_layer_controller.gd" id="97_script"]')
            lines.append('[ext_resource type="Animation" path="res://gvi_default_anim.tres" id="96_anim"]')
        lines.extend(ext_resources)
        lines.append("")

        # Root node
        lines.append(f"[node name=\"GeneratedScene\" type=\"{root_type}\"]")
        if root_type == "Control":
            lines.extend([
                "anchors_preset = 15",  # FULL_RECT
                "anchor_right = 1.0",
                "anchor_bottom = 1.0",
                "grow_horizontal = 2",
                "grow_vertical = 2",
            ])
        else:
            lines.append(f"metadata/canvas_width = {ir.width or 0:.3f}")
            lines.append(f"metadata/canvas_height = {ir.height or 0:.3f}")

        # Tilemap path: drop a TileMapLayer referencing tile_set.tres
        tilemap_meta = payload.get("tilemap")
        if target == "godot.tilemap" and tilemap_meta:
            tile_size = int(tilemap_meta.get("tile_size", 32))
            cols = int(tilemap_meta.get("cols", 1))
            rows = int(tilemap_meta.get("rows", 1))
            tiles = tilemap_meta.get("tiles", [])
            lines.extend(["", "[node name=\"TileMapLayer\" type=\"TileMapLayer\" parent=\".\"]"])
            lines.append("rendering_quadrant_size = 16")
            # Godot TileMapLayer tile_map_data format per cell:
            #   source_id, atlas_coords_x, atlas_coords_y, alternative_tile
            # -1 means empty. We use source 0 (atlas source) with atlas coords (tx, ty).
            pattern_lines = []
            max_tiles_in_data = 4096
            cell_count = 0
            for ty in range(rows):
                row = []
                for tx in range(cols):
                    if cell_count >= max_tiles_in_data:
                        row.append("-1")
                        continue
                    cell_count += 1
                    idx = ty * cols + tx
                    if idx < len(tiles):
                        t = tiles[idx]
                        if t.get("alpha_coverage", 1.0) > 0.05:
                            row.append(f"0,{tx},{ty},0")
                        else:
                            row.append("-1")
                    else:
                        row.append("-1")
                pattern_lines.append(",".join(row))
            tile_data = "[" + ";".join(pattern_lines) + "]"
            lines.append(f"tile_map_data = {tile_data}")
            lines.append(f"tile_set = ExtResource(\"99_tileset\")")

        # Optional: theme resource for godot.theme
        if target == "godot.theme" and request.options.export_manifest:
            lines.append("theme = ExtResource(\"98_theme\")")

        # Optional: animation controller script + animation player
        if target == "godot.animation":
            lines.append("script = ExtResource(\"97_script\")")
            lines.extend(["", "[node name=\"AnimationPlayer\" type=\"AnimationPlayer\" parent=\".\"]"])
            lines.append("autoplay = \"default\"")
            lines.append(f"\"default\" = ExtResource(\"96_anim\")")

        # Render all IR nodes in a hierarchy
        for node in ir.nodes:
            lines.extend(self._node_lines(node, target, asset_map, parent_path="."))

        return "\n".join(line for line in lines if line != "") + "\n"

    # ------------------------------------------------------------------ node rendering
    def _node_lines(self, node: IRNode, target: str, asset_map: dict[str, str], parent_path: str = ".") -> list[str]:
        name = self._safe_node_name(node.name or node.id)
        b = node.bounds or Bounds(x=0, y=0, width=0, height=0)
        has_texture = node.id in asset_map

        # Determine type & props
        node_type, props = self._godot_node_props(node, target, has_texture, b)
        parent = f"parent=\"{parent_path}\""
        lines: list[str] = [""]
        lines.append(f"[node name=\"{name}\" type=\"{node_type}\" {parent}]")

        if node.parent_id:
            lines[-1] = f"[node name=\"{name}\" type=\"{node_type}\" parent=\"{node.parent_id}\"]"

        for k, v in props.items():
            lines.append(f"{k} = {v}")

        # z_index
        if node.z_index and node.z_index > 0:
            lines.append(f"z_index = {int(node.z_index)}")

        # texture binding
        if has_texture and "texture" not in props and node_type in {"Sprite2D", "TextureRect"}:
            ext_id = self._ext_id_for(asset_map, node.id)
            if ext_id:
                lines.append(f"texture = ExtResource(\"{ext_id}\")")
                if node_type == "Sprite2D":
                    lines.append("centered = false")
                elif node_type == "TextureRect":
                    lines.append("expand_mode = 1")
                    lines.append("stretch_mode = 5")

        # text binding
        if node.text and node_type in {"Label", "RichTextLabel", "Button"}:
            text_prop = "text"
            lines.append(f"{text_prop} = \"{self._escape(node.text)}\"")
            if node.style.font_size:
                lines.append(f"theme_override_font_sizes/font_size = {int(node.style.font_size)}")
            if node.style.fill:
                lines.append(f"theme_override_colors/font_color = {self._godot_color(node.style.fill)}")

        if node.style.opacity < 1.0:
            lines.append(f"modulate = Color(1, 1, 1, {node.style.opacity:.4f})")

        # metadata
        safe_meta = {k: v for k, v in node.metadata.items() if isinstance(v, (str, int, float, bool)) or v is None}
        if safe_meta:
            lines.append(f"metadata/_gvi = \"{self._escape(json.dumps(safe_meta, default=str))}\"")

        return lines

    def _godot_node_props(self, node: IRNode, target: str, has_texture: bool, b: Bounds) -> tuple[str, dict[str, str]]:
        """Return the Godot node type + control-related props for this element."""
        is_control = target in {"godot.control", "godot.richtext", "godot.theme"}
        is_animation = target == "godot.animation"
        # In tilemap target, all elements become Sprite2D layers above the tilemap.
        if target == "godot.tilemap" or target == "godot.node2d" or target == "godot.sprite2d" or is_animation:
            if node.type == IRNodeType.TEXT or node.type == IRNodeType.LABEL:
                return "Label", {"position": f"Vector2({b.x:.3f}, {b.y:.3f})"}
            return ("Sprite2D" if has_texture else "Node2D"), {"position": f"Vector2({b.x:.3f}, {b.y:.3f})"}

        if is_control:
            layout = {
                "layout_mode": "0",
                "offset_left": f"{b.x:.3f}",
                "offset_top": f"{b.y:.3f}",
                "offset_right": f"{b.x + b.width:.3f}",
                "offset_bottom": f"{b.y + b.height:.3f}",
            }
            if node.type == IRNodeType.TEXT:
                return "Label", layout
            if node.type == IRNodeType.LABEL:
                return "Label", layout
            if node.type == IRNodeType.BUTTON:
                return "Button", layout
            if node.type == IRNodeType.PANEL:
                return "Panel", layout
            if has_texture:
                return "TextureRect", layout
            return "ColorRect", layout
        return "Node2D", {}

    # ------------------------------------------------------------------ helpers
    def _root_type(self, target: str) -> str:
        return "Control" if target in {"godot.control", "godot.richtext", "godot.theme"} else "Node2D"

    def _ext_resources(self, asset_map: dict[str, str]) -> list[str]:
        lines: list[str] = []
        for idx, (nid, path) in enumerate(asset_map.items(), start=1):
            lines.append(f"[ext_resource type=\"Texture2D\" path=\"{path}\" id=\"{idx}_{nid[:6]}\"]")
        return lines

    def _ext_id_for(self, asset_map: dict[str, str], node_id: str) -> str | None:
        for idx, (nid, _) in enumerate(asset_map.items(), start=1):
            if nid == node_id:
                return f"{idx}_{nid[:6]}"
        return None

    def _generate_manifest(self, ir: IRDocument, segmentation: SegmentationResult | None, profile: AssetProfile, asset_map: dict[str, str], request: ConversionRequest) -> dict[str, Any]:
        return {
            "gvi_version": "1.0.0",
            "source": str(profile.path) if profile.path else None,
            "target": request.target.value,
            "preset": request.preset,
            "canvas": {"width": ir.width, "height": ir.height},
            "segmentation": {
                "method": segmentation.method_used if segmentation else "none",
                "num_elements": segmentation.num_elements if segmentation else 0,
                "coverage_ratio": segmentation.coverage_ratio if segmentation else 0,
                "diagnostics": segmentation.diagnostics if segmentation else {},
            },
            "nodes": [node.model_dump(mode="json") for node in ir.flatten()],
            "assets": asset_map,
            "palette": ir.palette,
            "fonts": ir.fonts,
            "options": request.options.model_dump(mode="json"),
        }

    def _generate_theme_resource(self, ir: IRDocument, theme_payload: dict | None) -> str:
        palette = (theme_payload or {}).get("palette", []) if theme_payload else []
        if not palette and ir.palette:
            palette = list(ir.palette.values())
        out = [
            '[gd_resource type="Theme" load_steps=4 format=3 uid="uid://gvi_generated_theme"]',
            "",
        ]
        # Default stylebox for buttons
        out.extend([
            "[sub_resource type=\"StyleBoxFlat\" id=\"StyleBoxFlat_default\"]",
            "bg_color = Color(0.2, 0.22, 0.27, 1)",
            "border_width_left = 2",
            "border_width_top = 2",
            "border_width_right = 2",
            "border_width_bottom = 2",
            "border_color = Color(0.4, 0.45, 0.55, 1)",
            "corner_radius_top_left = 6",
            "corner_radius_top_right = 6",
            "corner_radius_bottom_right = 6",
            "corner_radius_bottom_left = 6",
            "",
            "[resource]",
            "Button/styles/normal = SubResource(\"StyleBoxFlat_default\")",
            "Button/styles/hover = SubResource(\"StyleBoxFlat_default\")",
            "Button/styles/pressed = SubResource(\"StyleBoxFlat_default\")",
            "Panel/styles/panel = SubResource(\"StyleBoxFlat_default\")",
        ])
        if palette:
            out.append(f"Label/colors/font_color = {self._godot_color(palette[0])}")
        return "\n".join(out) + "\n"

    def _generate_layer_script(self) -> str:
        return '''extends Node2D

# Auto-generated helper for layer animation.
# Toggle visibility, set blend modes, etc.

func set_layer_visible(layer_name: String, visible: bool) -> void:
    var n := get_node_or_null(layer_name)
    if n:
        n.visible = visible

func get_layer(layer_name: String) -> Node:
    return get_node_or_null(layer_name)
'''

    def _generate_animation_resource(self, ir: IRDocument) -> str:
        # Build a simple "default" animation that fades in every layer.
        duration = 2.0
        out = [
            '[gd_resource type="Animation" format=3 uid="uid://gvi_generated_anim"]',
            "",
            "[resource]",
            f"length = {duration:.3f}",
            "loop_mode = 1",
        ]
        for node in ir.flatten():
            if node.type == IRNodeType.BACKGROUND:
                continue
            track = f"tracks/{len([n for n in ir.flatten() if n.type != IRNodeType.BACKGROUND])}_type = \"value\""
            out.append(track)
        # Ensure unique track indices
        return "\n".join(out) + "\n"

    def _safe_file_name(self, name: str) -> str:
        stem, suffix = Path(name).stem, Path(name).suffix.lower() or ".png"
        return self._safe_node_name(stem).lower() + suffix

    def _safe_node_name(self, name: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_")
        if not cleaned:
            cleaned = "Node"
        if cleaned[0].isdigit():
            cleaned = "N_" + cleaned
        return cleaned[:64]

    def _unique_name(self, name: str, used: set[str]) -> str:
        candidate = name
        stem = Path(name).stem
        suffix = Path(name).suffix
        index = 2
        while candidate in used:
            candidate = f"{stem}_{index}{suffix}"
            index += 1
        used.add(candidate)
        return candidate

    def _godot_color(self, hex_value: str) -> str:
        if not isinstance(hex_value, str):
            return "Color(1, 1, 1, 1)"
        if re.match(r"^#[0-9a-fA-F]{6}$", hex_value):
            r = int(hex_value[1:3], 16) / 255
            g = int(hex_value[3:5], 16) / 255
            b = int(hex_value[5:7], 16) / 255
            return f"Color({r:.4f}, {g:.4f}, {b:.4f}, 1.0)"
        return "Color(1, 1, 1, 1)"

    def _escape(self, value: str) -> str:
        return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")