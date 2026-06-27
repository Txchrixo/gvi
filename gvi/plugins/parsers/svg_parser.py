"""Advanced SVG parser.

Extracts:
- viewBox dimensions and full tree
- named groups (``<g>`` with ``id`` / ``inkscape:label``) → "groups"
- rect, circle, ellipse, line, polygon, polyline, path, image, text, use, symbol
- inline ``style`` parsing for fill / stroke / opacity
- text elements with their actual string content
- rasterized fidelity background via cairosvg when available
- nested groups become hierarchy in the IR
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from gvi.core.plugin import Capability, Plugin, PluginContext
from gvi.core.types import AssetKind, AssetProfile, CapabilityType, DetectedElement, PipelineStepResult, SegmentationResult


_SVG_NS = "{http://www.w3.org/2000/svg}"
_INK_NS = "{http://www.inkscape.org/namespaces/inkscape}"


class SVGParserPlugin(Plugin):
    def capability(self) -> Capability:
        return Capability(
            id="parser.svg",
            type=CapabilityType.PARSER,
            name="Advanced SVG structure parser",
            supported_mime_types=["image/svg+xml"],
            supported_extensions=[".svg", ".svgz"],
            priority=95,
            provides=["segmentation", "raster_profile", "ir_hints"],
            tags={"vector", "structured"},
        )

    def run(self, payload: dict[str, Any], ctx: PluginContext) -> PipelineStepResult:
        profile: AssetProfile = payload["profile"]
        if not profile.path:
            return PipelineStepResult(name="svg_parser", ok=False, errors=["Missing SVG path"])

        out_dir = Path(ctx.work_dir) / "assets"
        out_dir.mkdir(parents=True, exist_ok=True)
        warnings: list[str] = []

        width, height = int(profile.width or 1024), int(profile.height or 768)
        bg_path = self._rasterize_svg(profile.path, out_dir / f"{profile.path.stem}_source.png", width, height, warnings)
        elements: list[DetectedElement] = []
        if bg_path:
            elements.append(DetectedElement(
                id="background_source",
                element_type="background",
                bounds=(0, 0, width, height),
                asset_path=bg_path,
                z_index=-1000,
                locked=True,
                source="svg_rasterizer",
                metadata={"source": "svg_rasterized"},
            ))

        try:
            tree = ET.parse(profile.path)
            root = tree.getroot()
        except Exception as exc:
            warnings.append(f"SVG parse failed: {exc}")
            root = None

        idx_counter = {"v": 0}
        if root is not None:
            self._walk(root, idx_counter, out_dir, width, height, elements, parent_group_id=None)

        raster_profile = profile.model_copy(deep=True)
        if bg_path:
            raster_profile.path = bg_path
            raster_profile.kind = AssetKind.RASTER
            raster_profile.extension = ".png"
            raster_profile.mime_type = "image/png"
            raster_profile.has_alpha = True
        raster_profile.width = width
        raster_profile.height = height
        raster_profile.metadata.update({
            "source_svg": str(profile.path),
            "svg_elements_extracted": max(0, len(elements) - 1),
        })

        segmentation = SegmentationResult(
            elements=elements,
            num_elements=len(elements),
            coverage_ratio=1.0 if bg_path else 0.0,
            method_used="svg_parser",
            diagnostics={"structured_elements": max(0, len(elements) - 1)},
        ).sort_and_index()

        return PipelineStepResult(
            name="svg_parser",
            data={**payload, "profile": raster_profile, "segmentation": segmentation, "parsed_asset": bg_path},
            artifacts={"parsed_asset": bg_path} if bg_path else {},
            metrics={"svg_elements": float(max(0, len(elements) - 1))},
            warnings=warnings,
        )

    # -------------------------------------------------------- helpers
    def _rasterize_svg(self, src: Path, dst: Path, width: int, height: int, warnings: list[str]) -> Path | None:
        try:
            import cairosvg
            cairosvg.svg2png(url=str(src), write_to=str(dst), output_width=width, output_height=height)
            return dst
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"cairosvg unavailable ({exc}); continuing with structured nodes only.")
            return None

    def _walk(self, node, idx_counter, out_dir, canvas_w, canvas_h, elements, parent_group_id):
        tag = self._strip_ns(node.tag)
        if tag == "g":
            gid = node.attrib.get("id") or node.attrib.get(_INK_NS + "label") or f"g_{idx_counter['v']}"
            idx_counter["v"] += 1
            for child in node:
                self._walk(child, idx_counter, out_dir, canvas_w, canvas_h, elements, parent_group_id=gid)
            return
        if tag in {"defs", "metadata", "title", "desc", "style", "filter", "clipPath", "mask"}:
            return

        attrs = node.attrib
        style = self._parse_style(attrs.get("style", ""))
        fill_raw = attrs.get("fill") or style.get("fill", "#000000")
        stroke_raw = attrs.get("stroke") or style.get("stroke")
        opacity = float(attrs.get("opacity", style.get("opacity", 1.0)))

        if tag == "rect":
            self._emit_rect(node, idx_counter, out_dir, canvas_w, canvas_h, elements, fill_raw, opacity, parent_group_id)
        elif tag == "circle":
            self._emit_circle(node, idx_counter, out_dir, canvas_w, canvas_h, elements, fill_raw, opacity, parent_group_id)
        elif tag == "ellipse":
            self._emit_ellipse(node, idx_counter, out_dir, canvas_w, canvas_h, elements, fill_raw, opacity, parent_group_id)
        elif tag == "line":
            self._emit_line(node, idx_counter, out_dir, canvas_w, canvas_h, elements, stroke_raw, opacity, parent_group_id)
        elif tag in {"polygon", "polyline"}:
            self._emit_polygon(node, idx_counter, out_dir, canvas_w, canvas_h, elements, fill_raw, opacity, parent_group_id)
        elif tag == "path":
            self._emit_path(node, idx_counter, out_dir, canvas_w, canvas_h, elements, fill_raw, opacity, parent_group_id)
        elif tag == "image":
            self._emit_image(node, idx_counter, out_dir, canvas_w, canvas_h, elements, parent_group_id)
        elif tag == "text":
            self._emit_text(node, idx_counter, out_dir, canvas_w, canvas_h, elements, fill_raw, opacity, parent_group_id)
        elif tag in {"use", "symbol"}:
            self._emit_use(node, idx_counter, out_dir, canvas_w, canvas_h, elements, parent_group_id)

        # walk children for non-group elements
        if tag not in {"g"}:
            for child in node:
                self._walk(child, idx_counter, out_dir, canvas_w, canvas_h, elements, parent_group_id=parent_group_id)

    # ------------------------------------------------ emit helpers
    def _emit_rect(self, node, idx_counter, out_dir, cw, ch, elements, fill, opacity, parent_id):
        idx = idx_counter["v"]; idx_counter["v"] += 1
        a = node.attrib
        x, y = self._num(a.get("x")), self._num(a.get("y"))
        w, h = self._num(a.get("width")), self._num(a.get("height"))
        if w <= 0 or h <= 0:
            return
        etype = "panel" if w > cw * 0.25 or h > ch * 0.15 else "button"
        color = self._normalize_color(fill)
        asset = out_dir / f"svg_rect_{idx:03d}.png"
        self._placeholder(asset, int(max(1, w)), int(max(1, h)), color, "rect", opacity)
        elements.append(DetectedElement(
            id=f"svg_rect_{idx:03d}", element_type=etype, bounds=(int(x), int(y), int(w), int(h)),
            asset_path=asset, z_index=idx, confidence=0.9, parent_id=parent_id, source="svg",
            metadata={"source": f"svg_rect", "fill": color, "opacity": opacity},
        ))

    def _emit_circle(self, node, idx_counter, out_dir, cw, ch, elements, fill, opacity, parent_id):
        idx = idx_counter["v"]; idx_counter["v"] += 1
        a = node.attrib
        r = self._num(a.get("r"))
        cx, cy = self._num(a.get("cx")), self._num(a.get("cy"))
        if r <= 0:
            return
        color = self._normalize_color(fill)
        asset = out_dir / f"svg_circle_{idx:03d}.png"
        self._placeholder(asset, int(r * 2), int(r * 2), color, "ellipse", opacity)
        elements.append(DetectedElement(
            id=f"svg_circle_{idx:03d}", element_type="sprite", bounds=(int(cx - r), int(cy - r), int(r * 2), int(r * 2)),
            asset_path=asset, z_index=idx, confidence=0.9, parent_id=parent_id, source="svg",
            metadata={"source": "svg_circle", "fill": color, "opacity": opacity},
        ))

    def _emit_ellipse(self, node, idx_counter, out_dir, cw, ch, elements, fill, opacity, parent_id):
        idx = idx_counter["v"]; idx_counter["v"] += 1
        a = node.attrib
        rx, ry = self._num(a.get("rx")), self._num(a.get("ry"))
        cx, cy = self._num(a.get("cx")), self._num(a.get("cy"))
        if rx <= 0 or ry <= 0:
            return
        color = self._normalize_color(fill)
        asset = out_dir / f"svg_ellipse_{idx:03d}.png"
        self._placeholder(asset, int(rx * 2), int(ry * 2), color, "ellipse", opacity)
        elements.append(DetectedElement(
            id=f"svg_ellipse_{idx:03d}", element_type="sprite", bounds=(int(cx - rx), int(cy - ry), int(rx * 2), int(ry * 2)),
            asset_path=asset, z_index=idx, confidence=0.9, parent_id=parent_id, source="svg",
            metadata={"source": "svg_ellipse", "fill": color, "opacity": opacity},
        ))

    def _emit_line(self, node, idx_counter, out_dir, cw, ch, elements, stroke, opacity, parent_id):
        idx = idx_counter["v"]; idx_counter["v"] += 1
        a = node.attrib
        x1, y1, x2, y2 = (self._num(a.get("x1")), self._num(a.get("y1")), self._num(a.get("x2")), self._num(a.get("y2")))
        color = self._normalize_color(stroke or "#000000")
        bx, by = min(x1, x2), min(y1, y2)
        bw, bh = max(1, abs(x2 - x1)), max(1, abs(y2 - y1))
        asset = out_dir / f"svg_line_{idx:03d}.png"
        self._placeholder(asset, max(2, int(bw)), max(2, int(bh)), color, "line", opacity)
        elements.append(DetectedElement(
            id=f"svg_line_{idx:03d}", element_type="decoration", bounds=(int(bx), int(by), int(bw), int(bh)),
            asset_path=asset, z_index=idx, confidence=0.7, parent_id=parent_id, source="svg",
            metadata={"source": "svg_line", "fill": color, "opacity": opacity},
        ))

    def _emit_polygon(self, node, idx_counter, out_dir, cw, ch, elements, fill, opacity, parent_id):
        idx = idx_counter["v"]; idx_counter["v"] += 1
        pts = self._parse_points(node.attrib.get("points", ""))
        if not pts:
            return
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        bx, by = min(xs), min(ys)
        bw, bh = max(xs) - bx, max(ys) - by
        if bw <= 0 or bh <= 0:
            return
        color = self._normalize_color(fill)
        asset = out_dir / f"svg_poly_{idx:03d}.png"
        self._placeholder(asset, int(max(1, bw)), int(max(1, bh)), color, "rect", opacity)
        elements.append(DetectedElement(
            id=f"svg_poly_{idx:03d}", element_type="sprite", bounds=(int(bx), int(by), int(bw), int(bh)),
            asset_path=asset, z_index=idx, confidence=0.7, parent_id=parent_id, source="svg",
            metadata={"source": "svg_polygon", "fill": color, "opacity": opacity, "points": pts},
        ))

    def _emit_path(self, node, idx_counter, out_dir, cw, ch, elements, fill, opacity, parent_id):
        idx = idx_counter["v"]; idx_counter["v"] += 1
        d = node.attrib.get("d", "")
        bbox = self._path_bbox(d)
        if not bbox:
            return
        bx, by, bw, bh = bbox
        color = self._normalize_color(fill or "#000000")
        asset = out_dir / f"svg_path_{idx:03d}.png"
        self._placeholder(asset, int(max(1, bw)), int(max(1, bh)), color, "rect", opacity)
        elements.append(DetectedElement(
            id=f"svg_path_{idx:03d}", element_type="sprite", bounds=(int(bx), int(by), int(max(1, bw)), int(max(1, bh))),
            asset_path=asset, z_index=idx, confidence=0.6, parent_id=parent_id, source="svg",
            metadata={"source": "svg_path", "fill": color, "opacity": opacity},
        ))

    def _emit_image(self, node, idx_counter, out_dir, cw, ch, elements, parent_id):
        idx = idx_counter["v"]; idx_counter["v"] += 1
        a = node.attrib
        x, y = self._num(a.get("x")), self._num(a.get("y"))
        w, h = self._num(a.get("width")), self._num(a.get("height"))
        if w <= 0 or h <= 0:
            return
        href = a.get("{http://www.w3.org/1999/xlink}href") or a.get("href") or ""
        elements.append(DetectedElement(
            id=f"svg_image_{idx:03d}", element_type="image", bounds=(int(x), int(y), int(w), int(h)),
            z_index=idx, confidence=0.8, parent_id=parent_id, source="svg",
            metadata={"source": "svg_image", "href": href},
        ))

    def _emit_text(self, node, idx_counter, out_dir, cw, ch, elements, fill, opacity, parent_id):
        idx = idx_counter["v"]; idx_counter["v"] += 1
        a = node.attrib
        x, y = self._num(a.get("x")), self._num(a.get("y"))
        font_size = self._num(a.get("font-size"), 16)
        text_content = "".join(node.itertext()).strip()
        if not text_content:
            return
        color = self._normalize_color(fill or "#000000")
        w = max(12, len(text_content) * font_size * 0.6)
        h = font_size * 1.4
        asset = out_dir / f"svg_text_{idx:03d}.png"
        self._placeholder(asset, int(w), int(h), color, "rect", opacity)
        elements.append(DetectedElement(
            id=f"svg_text_{idx:03d}", element_type="text", bounds=(int(x), int(max(0, y - h)), int(w), int(h)),
            asset_path=asset, text_content=text_content, z_index=idx, confidence=0.95,
            parent_id=parent_id, source="svg",
            metadata={"source": "svg_text", "fill": color, "font_size": font_size, "opacity": opacity},
        ))

    def _emit_use(self, node, idx_counter, out_dir, cw, ch, elements, parent_id):
        idx = idx_counter["v"]; idx_counter["v"] += 1
        a = node.attrib
        x, y = self._num(a.get("x")), self._num(a.get("y"))
        w, h = self._num(a.get("width"), 32), self._num(a.get("height"), 32)
        href = a.get("{http://www.w3.org/1999/xlink}href") or a.get("href") or ""
        elements.append(DetectedElement(
            id=f"svg_use_{idx:03d}", element_type="sprite", bounds=(int(x), int(y), int(w), int(h)),
            z_index=idx, confidence=0.6, parent_id=parent_id, source="svg",
            metadata={"source": "svg_use", "href": href},
        ))

    # ------------------------------------------------ utility helpers
    def _placeholder(self, path: Path, w: int, h: int, fill_rgba, shape: str, opacity: float) -> None:
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        rgba = fill_rgba + (int(255 * opacity),)
        if shape == "ellipse":
            draw.ellipse((0, 0, w - 1, h - 1), fill=rgba)
        elif shape == "line":
            draw.line((0, 0, w - 1, h - 1), fill=rgba, width=max(1, min(w, h) // 4))
        else:
            draw.rectangle((0, 0, w - 1, h - 1), fill=rgba)
        img.save(path)

    def _parse_style(self, raw: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for chunk in raw.split(";"):
            if ":" in chunk:
                k, v = chunk.split(":", 1)
                out[k.strip()] = v.strip()
        return out

    def _parse_points(self, raw: str) -> list[tuple[float, float]]:
        nums = re.findall(r"-?\d+(?:\.\d+)?", raw)
        out: list[tuple[float, float]] = []
        for i in range(0, len(nums) - 1, 2):
            out.append((float(nums[i]), float(nums[i + 1])))
        return out

    def _path_bbox(self, d: str) -> tuple[float, float, float, float] | None:
        nums = re.findall(r"-?\d+(?:\.\d+)?", d)
        if not nums:
            return None
        coords = [float(n) for n in nums]
        xs = coords[0::2]
        ys = coords[1::2]
        if not xs or not ys:
            return None
        return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))

    def _normalize_color(self, raw: str) -> tuple[int, int, int]:
        raw = (raw or "#000000").strip()
        if raw.startswith("#") and len(raw) == 7:
            try:
                return tuple(int(raw[i:i+2], 16) for i in (1, 3, 5))  # type: ignore[return-value]
            except ValueError:
                return (0, 0, 0)
        named = {"black": (0, 0, 0), "white": (255, 255, 255), "red": (255, 0, 0), "green": (0, 128, 0), "blue": (0, 0, 255), "gray": (128, 128, 128), "grey": (128, 128, 128)}
        return named.get(raw.lower(), (0, 0, 0))

    def _num(self, value, default=0.0):
        if value is None:
            return default
        m = re.search(r"-?\d+(?:\.\d+)?", str(value))
        return float(m.group(0)) if m else default

    def _strip_ns(self, tag: str) -> str:
        return tag.rsplit("}", 1)[-1]