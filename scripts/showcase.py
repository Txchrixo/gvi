"""Generate a full Godot showcase project — every image × every target in one folder.

Run: python scripts/showcase.py
Output: /workspace/gvi_v1/outputs/godot_showcase/
"""
from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gvi.core.orchestrator import Orchestrator  # noqa: E402
from gvi.core.types import ConversionOptions, ConversionRequest, TargetKind  # noqa: E402

ROOT = Path("/workspace/gvi_v1")
OUTPUT = ROOT / "outputs" / "godot_showcase"
TEST_IMAGES = ROOT / "test_images"


def main() -> int:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    failures: list[str] = []

    for img in sorted(TEST_IMAGES.glob("*.png")):
        for target in TargetKind:
            out = OUTPUT / img.stem / target.value.replace(".", "_")
            opts = ConversionOptions(
                tilemap_tile_size=32,
                fail_fast=True,
                ocr=target.value in {"godot.control", "godot.richtext", "godot.theme"},
            )
            req = ConversionRequest(
                input_path=img,
                target=target,
                output_dir=out,
                preset="balanced",
                options=opts,
            )
            try:
                r = Orchestrator().convert(req)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{img.name} → {target.value}: {exc}")
                continue
            manifest.append({
                "source": img.name,
                "target": target.value,
                "output_dir": str(out.relative_to(ROOT)),
                "ok": r.ok,
                "scene": str(r.artifacts.get("scene").relative_to(ROOT)) if r.artifacts.get("scene") else None,
                "tileset": str(r.artifacts.get("tileset").relative_to(ROOT)) if r.artifacts.get("tileset") else None,
                "theme": str(r.artifacts.get("theme").relative_to(ROOT)) if r.artifacts.get("theme") else None,
                "animation": str(r.artifacts.get("animation").relative_to(ROOT)) if r.artifacts.get("animation") else None,
                "manifest": str(r.artifacts.get("manifest").relative_to(ROOT)) if r.artifacts.get("manifest") else None,
                "num_elements": int(r.metrics.get("num_elements", 0)),
            })
            status = "OK" if r.ok else "FAIL"
            print(f"{status:4} {img.name:25} -> {target.value:20} ({out.relative_to(OUTPUT)})")

    # Write the showcase manifest.
    (OUTPUT / "showcase_manifest.json").write_text(
        __import__("json").dumps({"entries": manifest, "failures": failures}, indent=2),
        encoding="utf-8",
    )

    # Bundle a single zip you can hand to a teammate.
    zip_path = OUTPUT / "gvi_showcase.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in OUTPUT.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(OUTPUT)))

    passed = sum(1 for m in manifest if m["ok"])
    print(f"\n{passed}/{len(manifest)} conversions OK")
    print(f"Showcase: {OUTPUT}")
    print(f"Bundled: {zip_path} ({zip_path.stat().st_size // 1024} KB)")
    if failures:
        print(f"\nFailures:")
        for f in failures:
            print("  -", f)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())