"""Intermediate representation shared by all GVI plugins.

The IR is built by post-processing the raw segmentation and consumed by the
Godot exporters. It carries enough information to produce editable, semantically
meaningful Godot nodes for every supported target.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class IRNodeType(str, Enum):
    DOCUMENT = "document"
    FRAME = "frame"
    GROUP = "group"
    RECT = "rect"
    ELLIPSE = "ellipse"
    PATH = "path"
    IMAGE = "image"
    TEXT = "text"
    SPRITE = "sprite"
    CHARACTER = "character"
    PROP = "prop"
    TILE_LAYER = "tile_layer"
    COMPONENT = "component"
    BUTTON = "button"
    PANEL = "panel"
    LABEL = "label"
    BACKGROUND = "background"


class Bounds(BaseModel):
    x: float
    y: float
    width: float
    height: float
    rotation: float = 0.0


class Style(BaseModel):
    fill: str | None = None
    stroke: str | None = None
    stroke_width: float | None = None
    opacity: float = 1.0
    radius: float | None = None
    font_family: str | None = None
    font_size: float | None = None
    font_weight: str | None = None
    text_align: str | None = None
    color: str | None = None
    background_color: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IRNode(BaseModel):
    id: str
    type: IRNodeType
    name: str | None = None
    bounds: Bounds | None = None
    style: Style = Field(default_factory=Style)
    text: str | None = None
    asset: Path | None = None
    z_index: int = 0
    parent_id: str | None = None
    children: list["IRNode"] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_godot_node_type(self, target: str = "godot.control") -> str:
        """Map IR node type to a reasonable Godot node type for the requested target."""
        is_control = "control" in target or target in {"godot.richtext", "godot.theme"}
        mapping: dict[IRNodeType, str] = {
            IRNodeType.SPRITE: "Sprite2D",
            IRNodeType.CHARACTER: "Sprite2D",
            IRNodeType.PROP: "Sprite2D",
            IRNodeType.IMAGE: "TextureRect" if is_control else "Sprite2D",
            IRNodeType.RECT: "ColorRect" if is_control else "Polygon2D",
            IRNodeType.TEXT: "RichTextLabel" if target == "godot.richtext" else "Label",
            IRNodeType.LABEL: "Label",
            IRNodeType.BUTTON: "Button",
            IRNodeType.PANEL: "Panel",
            IRNodeType.GROUP: "Node2D",
            IRNodeType.FRAME: "Panel" if is_control else "Sprite2D",
            IRNodeType.TILE_LAYER: "TileMapLayer",
            IRNodeType.COMPONENT: "Node2D",
            IRNodeType.BACKGROUND: "TextureRect" if is_control else "Sprite2D",
            IRNodeType.DOCUMENT: "Control" if is_control else "Node2D",
            IRNodeType.PATH: "Polygon2D",
            IRNodeType.ELLIPSE: "Polygon2D",
        }
        return mapping.get(self.type, "Node2D")


class IRDocument(BaseModel):
    version: str = "1.0"
    width: float | None = None
    height: float | None = None
    nodes: list[IRNode] = Field(default_factory=list)
    assets: dict[str, Path] = Field(default_factory=dict)
    palette: dict[str, str] = Field(default_factory=dict)
    fonts: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def flatten(self) -> list[IRNode]:
        out: list[IRNode] = []

        def walk(node: IRNode) -> None:
            out.append(node)
            for child in node.children:
                walk(child)

        for n in self.nodes:
            walk(n)
        return out

    def root_nodes(self) -> list[IRNode]:
        return [n for n in self.nodes if n.parent_id is None]


IRNode.model_rebuild()