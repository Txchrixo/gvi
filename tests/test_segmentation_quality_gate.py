"""Regression gate against the v1.0.0 over-segmentation bug, and unit tests
for the new geometric primitives (homography rectification, GrabCut).

Unlike tests/test_pipeline.py (which exercises the full pydantic-based
orchestrator), everything here imports only
`gvi.plugins.segmenters._opencv_core`, which has zero dependency on
pydantic/gvi.core. That is deliberate: it means these tests can run (and
were actually run, during development -- see docs/CHANGELOG_v1.1.md for the
real numbers) even in an environment where `pip install -e .` hasn't been
done yet, which makes them a much faster, more reliable CI smoke layer.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from gvi.plugins.segmenters._opencv_core import (
    classify_scene,
    grabcut_refine,
    rectify_if_rotated,
    run_segmentation,
)

TEST_IMAGES = Path(__file__).resolve().parent.parent / "test_images"

# (name, filename, max_allowed_elements, min_expected_elements)
QUALITY_GATE_CASES = [
    ("wall_of_frames", "wall_of_frames.png", 15, 4),
    ("ui_mockup", "ui_mockup.png", 15, 2),
    ("sprite_scene", "sprite_scene.png", 15, 1),
    ("map_render", "map_render.png", 15, 1),
    ("robot_photo", "robot_photo.png", 20, 1),
    ("monster_can", "monster_can.png", 15, 1),
    ("trashpick_logo", "trashpick_logo.png", 15, 1),
]


@pytest.mark.parametrize("name,filename,ceiling,floor", QUALITY_GATE_CASES, ids=[c[0] for c in QUALITY_GATE_CASES])
def test_no_fragment_storm_and_no_zero_detection(tmp_path, name, filename, ceiling, floor):
    """The test that would have caught both the v1.0.0 regressions
    automatically: 0 elements on a flat synthetic scene (the original bug),
    and 100+ elements on any real photo/render (the v1.0.0 "fix")."""
    img = cv2.imread(str(TEST_IMAGES / filename), cv2.IMREAD_UNCHANGED)
    assert img is not None, f"could not load {filename}"
    res = run_segmentation(img, tmp_path / name, preset="balanced")
    n = res["num_elements"]
    assert n <= ceiling, f"{name}: {n} elements > ceiling {ceiling} -- fragment-storm regression"
    assert n >= floor, f"{name}: {n} elements < floor {floor} -- zero-detection regression"


def _rotated_rect_image(angle_deg=18.0, rect_w=300, rect_h=180, canvas=500):
    img = np.full((canvas, canvas, 3), (40, 40, 40), dtype=np.uint8)
    rect = ((canvas / 2, canvas / 2), (rect_w, rect_h), angle_deg)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillConvexPoly(img, box, (200, 130, 40))
    return img, box


def test_homography_rectifies_tilted_quad():
    img, box = _rotated_rect_image(angle_deg=18.0)
    contour = box.reshape(-1, 1, 2)
    out = rectify_if_rotated(img, None, contour, min_angle_deg=4.0)
    assert out is not None
    rgba = out["image"]
    opaque_ratio = float(np.mean(rgba[:, :, 3] > 200))
    assert opaque_ratio > 0.90


def test_homography_skips_near_upright_quad():
    img, box = _rotated_rect_image(angle_deg=1.0)
    contour = box.reshape(-1, 1, 2)
    assert rectify_if_rotated(img, None, contour, min_angle_deg=4.0) is None


def test_homography_skips_non_rectangular_blob():
    canvas = 400
    img = np.full((canvas, canvas, 3), (40, 40, 40), dtype=np.uint8)
    pts = []
    for i in range(24):
        a = 2 * math.pi * i / 24
        r = 80 + 40 * math.sin(a * 5)
        pts.append((int(200 + r * math.cos(a)), int(200 + r * math.sin(a))))
    pts = np.array(pts, dtype=np.int32)
    cv2.fillPoly(img, [pts], (90, 160, 90))
    assert rectify_if_rotated(img, None, pts.reshape(-1, 1, 2), min_angle_deg=4.0) is None


def test_grabcut_trims_loose_bbox():
    bg = np.random.randint(60, 90, (300, 300, 3), dtype=np.uint8)
    img = bg.copy()
    cv2.circle(img, (150, 150), 70, (210, 90, 60), -1)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    mask = grabcut_refine(img, (60, 60, 180, 180))
    assert mask is not None
    fg_ratio = float(np.mean(mask > 0))
    assert 0.25 < fg_ratio < 0.75, f"GrabCut did not meaningfully trim the box (ratio={fg_ratio:.3f})"


def test_grabcut_bails_on_tiny_box():
    img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    assert grabcut_refine(img, (10, 10, 5, 5)) is None


def test_scene_classifier_separates_flat_from_textured():
    flat = np.full((200, 200, 3), (220, 220, 220), dtype=np.uint8)
    cv2.rectangle(flat, (20, 20), (90, 90), (60, 140, 220), -1)
    textured = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    assert classify_scene(flat, None)["complexity"] == "flat"
    assert classify_scene(textured, None)["complexity"] == "textured"


def _write_fake_godot(bin_dir: Path, mode: str) -> Path:
    """A fake `godot4` executable so the subprocess/parsing logic in
    _godot_headless.py can be tested beyond just its "binary not found"
    fallback -- without needing a real Godot install in CI."""
    script = bin_dir / "godot4"
    if mode == "error":
        body = (
            "#!/bin/bash\n"
            "echo 'Godot Engine v4.3.stable.official - fake binary'\n"
            "echo \"ERROR: Parse Error: Could not resolve external resource\"\n"
            "echo 'SCRIPT ERROR: Parse Error: Expected end of statement'\n"
            "exit 1\n"
        )
    else:
        body = (
            "#!/bin/bash\n"
            "echo 'Godot Engine v4.3.stable.official - fake binary'\n"
            "echo 'Project imported successfully.'\n"
            "exit 0\n"
        )
    script.write_text(body)
    script.chmod(0o755)
    return script


def test_godot_headless_accepts_a_clean_import(tmp_path, monkeypatch):
    from gvi.plugins.validators._godot_headless import validate_project_headless, scaffold_minimal_project

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_godot(bin_dir, "clean")
    monkeypatch.setenv("PATH", f"{bin_dir}:{__import__('os').environ['PATH']}")

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "scene.tscn").write_text('[gd_scene load_steps=1 format=3]\n[node name="X" type="Node2D"]\n')
    scaffold_minimal_project(project_dir)

    res = validate_project_headless(project_dir)
    assert res.available is True
    assert res.ok is True
    assert res.errors == []


def test_godot_headless_reports_a_real_import_error(tmp_path, monkeypatch):
    from gvi.plugins.validators._godot_headless import validate_project_headless, scaffold_minimal_project

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_godot(bin_dir, "error")
    monkeypatch.setenv("PATH", f"{bin_dir}:{__import__('os').environ['PATH']}")

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "scene.tscn").write_text('[gd_scene load_steps=1 format=3]\n[node name="X" type="Node2D"]\n')
    scaffold_minimal_project(project_dir)

    res = validate_project_headless(project_dir)
    assert res.available is True
    assert res.ok is False
    assert any("Parse Error" in e for e in res.errors)


def test_godot_headless_falls_back_honestly_when_no_binary(tmp_path, monkeypatch):
    from gvi.plugins.validators._godot_headless import validate_project_headless

    monkeypatch.setenv("PATH", str(tmp_path))  # empty dir, nothing on PATH
    monkeypatch.delenv("GODOT_BIN", raising=False)
    res = validate_project_headless(tmp_path)
    assert res.available is False
    assert res.ok is None
    assert res.skipped_reason is not None
