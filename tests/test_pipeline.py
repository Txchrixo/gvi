"""End-to-end smoke test: convert each test image to all 7 targets."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from gvi.core.orchestrator import Orchestrator
from gvi.core.types import ConversionOptions, ConversionRequest, TargetKind

TEST_IMAGES = Path(__file__).resolve().parent.parent / "test_images"
OUTPUT_DIR = Path("/tmp/gvi_test_outputs")


@pytest.fixture(scope="module", autouse=True)
def clean_outputs():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    yield


def _images() -> list[Path]:
    return sorted(TEST_IMAGES.glob("*.png"))


@pytest.mark.parametrize("image", _images(), ids=lambda p: p.name)
@pytest.mark.parametrize(
    "target",
    [
        TargetKind.GODOT_NODE2D,
        TargetKind.GODOT_CONTROL,
        TargetKind.GODOT_SPRITE2D,
        TargetKind.GODOT_TILEMAP,
        TargetKind.GODOT_RICHTEXT,
        TargetKind.GODOT_THEME,
        TargetKind.GODOT_ANIMATION,
    ],
)
def test_convert_all_targets(image: Path, target: TargetKind):
    out = OUTPUT_DIR / f"{image.stem}_{target.value.replace('.', '_')}"
    opts = ConversionOptions(
        tilemap_tile_size=32,
        fail_fast=True,
        include_background=True,
        ocr=target.value in {"godot.control", "godot.richtext", "godot.theme"},
    )
    request = ConversionRequest(
        input_path=image,
        target=target,
        output_dir=out,
        preset="balanced",
        options=opts,
    )
    result = Orchestrator().convert(request)
    scene = result.artifacts.get("scene")
    assert scene is not None, f"Scene not produced for {target}"
    assert scene.exists()
    content = scene.read_text()
    assert "[gd_scene" in content
    assert "uid=" in content
    if target == TargetKind.GODOT_TILEMAP:
        tileset = result.artifacts.get("tileset")
        assert tileset is not None and tileset.exists()
        tres = tileset.read_text()
        assert "TileSet" in tres
        assert "tile_size" in tres
        assert "TileSetAtlasSource" in tres
    assert not result.errors, f"Errors: {result.errors}"


def test_inspect_works():
    profile = Orchestrator().inspect(TEST_IMAGES / "ui_mockup.png")
    assert profile.width is not None and profile.height is not None
    assert profile.kind.value == "raster"


def test_tileset_is_well_formed():
    """Tileset .tres file must contain exactly one [resource] block."""
    out = OUTPUT_DIR / "wall_of_frames_godot_tilemap"
    if out.exists():
        shutil.rmtree(out)
    opts = ConversionOptions(tilemap_tile_size=32, fail_fast=True)
    req = ConversionRequest(
        input_path=TEST_IMAGES / "wall_of_frames.png",
        target=TargetKind.GODOT_TILEMAP,
        output_dir=out,
        preset="tilemap",
        options=opts,
    )
    result = Orchestrator().convert(req)
    tres = result.artifacts.get("tileset")
    assert tres is not None
    text = tres.read_text()
    # Exactly one [resource] block at the end (not many).
    assert text.count("[resource]") == 1
    # And it must have an atlas source.
    assert "TileSetAtlasSource" in text
    # Must be load_steps > 2 (1 atlas sub_resource + ext_resource).
    assert "load_steps=3" in text


def test_control_scene_has_text_labels():
    out = OUTPUT_DIR / "ui_mockup_godot_control"
    if out.exists():
        shutil.rmtree(out)
    opts = ConversionOptions(fail_fast=True, ocr=True)
    req = ConversionRequest(
        input_path=TEST_IMAGES / "ui_mockup.png",
        target=TargetKind.GODOT_CONTROL,
        output_dir=out,
        preset="balanced",
        options=opts,
    )
    result = Orchestrator().convert(req)
    scene = result.artifacts["scene"].read_text()
    assert "TextureRect" in scene
    assert "Control" in scene


def test_animation_target_emits_animation_player():
    out = OUTPUT_DIR / "ui_mockup_godot_animation"
    if out.exists():
        shutil.rmtree(out)
    opts = ConversionOptions(fail_fast=True)
    req = ConversionRequest(
        input_path=TEST_IMAGES / "ui_mockup.png",
        target=TargetKind.GODOT_ANIMATION,
        output_dir=out,
        preset="balanced",
        options=opts,
    )
    result = Orchestrator().convert(req)
    scene = result.artifacts["scene"].read_text()
    assert "AnimationPlayer" in scene
    assert (out / "gvi_layer_controller.gd").exists()
    assert (out / "gvi_default_anim.tres").exists()


def test_theme_target_emits_theme_resource():
    out = OUTPUT_DIR / "ui_mockup_godot_theme"
    if out.exists():
        shutil.rmtree(out)
    opts = ConversionOptions(fail_fast=True)
    req = ConversionRequest(
        input_path=TEST_IMAGES / "ui_mockup.png",
        target=TargetKind.GODOT_THEME,
        output_dir=out,
        preset="balanced",
        options=opts,
    )
    result = Orchestrator().convert(req)
    scene = result.artifacts["scene"].read_text()
    theme = result.artifacts["theme"]
    assert theme is not None
    text = theme.read_text()
    assert "Theme" in text
    assert "StyleBoxFlat" in text