"""Real Godot-engine validation, with an honest, explicit fallback.

The v1.0.0 validator only ran a hand-rolled regex parser against the .tscn
grammar -- it could tell you the file LOOKED like a scene file, never that
Godot itself would actually accept it. This module tries to shell out to a
real Godot 4 binary in headless mode to import/parse the generated project
and report genuine engine-level errors.

Design choices:
  - Never raises if Godot isn't installed -- returns a result that says so
    explicitly (`available: False`), so callers/CI can decide whether that's
    acceptable instead of silently believing a fake "valid" result.
  - Looks for `godot4`, `godot`, then `GODOT_BIN` env var, in that order.
  - Uses `--headless --import --quit-after <n>` which is the documented way
    to force Godot to (re)import a project and exit without opening a
    window; it surfaces import errors on stderr/stdout and via exit code.
  - Has a hard timeout so a hung import can't stall CI.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class HeadlessValidationResult:
    available: bool
    ok: bool | None = None  # None when `available` is False
    binary_used: str | None = None
    returncode: int | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    errors: list[str] = field(default_factory=list)
    skipped_reason: str | None = None


def find_godot_binary() -> str | None:
    import os

    for candidate in (os.environ.get("GODOT_BIN"), "godot4", "godot", "godot-headless"):
        if not candidate:
            continue
        path = shutil.which(candidate)
        if path:
            return path
    return None


def validate_project_headless(project_dir: Path, timeout_s: int = 60) -> HeadlessValidationResult:
    """Run `godot --headless --import` against a minimal project containing
    the generated scene, and report whether Godot itself accepted it.

    `project_dir` must already contain a `project.godot` file (see
    `scaffold_minimal_project` below) plus the scene/assets to check.
    """
    binary = find_godot_binary()
    if binary is None:
        return HeadlessValidationResult(
            available=False,
            skipped_reason=(
                "No Godot 4 binary found on PATH (tried $GODOT_BIN, godot4, godot, "
                "godot-headless). Falling back to the structural .tscn grammar "
                "checker only -- this is NOT the same guarantee as Godot actually "
                "accepting the file. Install Godot 4 and make it available on PATH "
                "(or set GODOT_BIN) to get a real engine-level check."
            ),
        )

    try:
        proc = subprocess.run(
            [binary, "--headless", "--path", str(project_dir), "--import", "--quit-after", "1"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return HeadlessValidationResult(
            available=True,
            ok=False,
            binary_used=binary,
            errors=[f"Godot headless import timed out after {timeout_s}s"],
        )
    except OSError as exc:
        return HeadlessValidationResult(
            available=True,
            ok=False,
            binary_used=binary,
            errors=[f"Failed to execute Godot binary: {exc}"],
        )

    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    error_lines = [
        line for line in combined.splitlines()
        if "ERROR" in line or "SCRIPT ERROR" in line or "Parse Error" in line
    ]
    ok = proc.returncode == 0 and not error_lines
    return HeadlessValidationResult(
        available=True,
        ok=ok,
        binary_used=binary,
        returncode=proc.returncode,
        stdout_tail="\n".join((proc.stdout or "").splitlines()[-20:]),
        stderr_tail="\n".join((proc.stderr or "").splitlines()[-20:]),
        errors=error_lines,
    )


def scaffold_minimal_project(scene_dir: Path) -> Path:
    """Write the smallest possible project.godot next to a generated
    scene.tscn so `--import` has something to chew on. Idempotent.
    """
    project_file = scene_dir / "project.godot"
    if not project_file.exists():
        project_file.write_text(
            '; Minimal scaffold project written by GVI for headless validation only.\n'
            'config_version=5\n\n[application]\n\nconfig/name="gvi_validation"\n'
            'run/main_scene="res://scene.tscn"\n',
            encoding="utf-8",
        )
    return project_file
