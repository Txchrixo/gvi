"""Strict validation that every generated .tscn + .tres follows Godot's .tscn grammar."""
from __future__ import annotations

import re
from pathlib import Path


def _parse_blocks(text: str) -> list[tuple[str, dict[str, str], list[str]]]:
    """Lightweight parser: returns (tag, header_attrs, body_lines)."""
    blocks: list[tuple[str, dict[str, str], list[str]]] = []
    current_tag: str | None = None
    current_body: list[str] = []
    current_attrs: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("[") and line.endswith("]"):
            if current_tag is not None:
                blocks.append((current_tag, current_attrs, current_body))
            current_tag = line[1:-1]
            header = current_tag.split(maxsplit=1)
            attrs = {}
            if len(header) == 2:
                # Parse key=value pairs from the rest
                rest = header[1]
                for match in re.finditer(r'(\w+)="([^"]*)"', rest):
                    attrs[match.group(1)] = match.group(2)
            current_attrs = attrs
            current_body = []
        elif current_tag is not None:
            current_body.append(line)
    if current_tag is not None:
        blocks.append((current_tag, current_attrs, current_body))
    return blocks


def _check_scene(path: Path) -> list[str]:
    issues: list[str] = []
    text = path.read_text()
    blocks = _parse_blocks(text)
    raw_tags = [t for t, _, _ in blocks]
    if not any(t.startswith("gd_scene") for t in raw_tags):
        issues.append("missing [gd_scene] header")
    # ext_resources must come before nodes that reference them.
    ext_ids: set[str] = set()
    for tag, attrs, body in blocks:
        if tag.startswith("ext_resource"):
            ext_ids.add(attrs.get("id", ""))
        elif tag.startswith("node"):
            for match in re.finditer(r'ExtResource\("([^"]+)"\)', "\n".join(body)):
                if match.group(1) not in ext_ids:
                    issues.append(f"node references unknown ExtResource: {match.group(1)}")
    return issues


def _check_tileset(path: Path) -> list[str]:
    issues: list[str] = []
    text = path.read_text()
    blocks = _parse_blocks(text)
    raw_tags = [t for t, _, _ in blocks]
    if not any(t.startswith("gd_resource") for t in raw_tags):
        issues.append("missing [gd_resource] header")
    resource_count = sum(1 for t, _, _ in blocks if t == "resource")
    if resource_count != 1:
        issues.append(f"expected exactly 1 [resource] block, got {resource_count}")
    atlas_count = sum(1 for t, _, _ in blocks if "TileSetAtlasSource" in t)
    if atlas_count < 1:
        issues.append("no TileSetAtlasSource sub_resource")
    return issues


def _gather_outputs() -> list[Path]:
    base = Path("/tmp/gvi_test_outputs")
    if not base.exists():
        return []
    return list(base.rglob("scene.tscn")) + list(base.rglob("tile_set.tres"))


def test_all_scenes_valid():
    files = _gather_outputs()
    assert files, "No test outputs found; run the full conversion test first."
    for p in files:
        if p.suffix == ".tscn" and p.name == "scene.tscn":
            issues = _check_scene(p)
            assert not issues, f"{p}: {issues}"
        elif p.name == "tile_set.tres":
            issues = _check_tileset(p)
            assert not issues, f"{p}: {issues}"


def test_tile_map_data_well_formed():
    """Every TileMapLayer's tile_map_data must have proper Godot encoding."""
    pattern = re.compile(r"^(-1|\d+,\d+,\d+,\d+)$")
    for path in Path("/tmp/gvi_test_outputs").rglob("scene.tscn"):
        text = path.read_text()
        if "TileMapLayer" not in text:
            continue
        # Extract tile_map_data value
        m = re.search(r"tile_map_data\s*=\s*\[(.*?)\]", text, re.DOTALL)
        if not m:
            continue
        raw = m.group(1)
        for row in raw.split(";"):
            for cell in row.split(","):
                cell = cell.strip()
                if not cell:
                    continue
                # Each cell is 4 ints (or -1); commas split them, so join back
                # We expect 4 tokens separated by ','
                tokens = [t.strip() for t in row.split(",")]
                # 4 tokens per cell, cells separated by ;
                # Simpler: each cell group of 4 tokens
                assert pattern.match(",".join(tokens[:4])) or all(
                    pattern.match(",".join(tokens[i : i + 4])) for i in range(0, len(tokens), 4)
                ), f"{path}: bad tile_map_data cell: {tokens[:4]}"