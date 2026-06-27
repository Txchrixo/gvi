"""Benchmark GVI on the sample test images across all targets."""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gvi.core.orchestrator import Orchestrator  # noqa: E402
from gvi.core.types import ConversionOptions, ConversionRequest, TargetKind  # noqa: E402

TEST_IMAGES = Path("/workspace/gvi_v1/test_images")
OUTPUT_DIR = Path("/tmp/gvi_benchmark")


def main() -> int:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for img in sorted(TEST_IMAGES.glob("*.png")):
        for target in TargetKind:
            out = OUTPUT_DIR / f"{img.stem}_{target.value.replace('.', '_')}"
            opts = ConversionOptions(
                tilemap_tile_size=32,
                fail_fast=True,
                ocr=target.value in {"godot.control", "godot.richtext", "godot.theme"},
                sam2_enabled=False,  # too slow for benchmark; enable manually
                semantic_detection=True,
            )
            req = ConversionRequest(input_path=img, target=target, output_dir=out, preset="balanced", options=opts)
            t0 = time.perf_counter()
            r = Orchestrator().convert(req)
            elapsed = (time.perf_counter() - t0) * 1000
            scene = r.artifacts.get("scene")
            scene_size = scene.stat().st_size if scene else 0
            ts = r.artifacts.get("tileset")
            ts_size = ts.stat().st_size if ts else 0
            row = {
                "image": img.name,
                "target": target.value,
                "ok": r.ok,
                "elapsed_ms": round(elapsed, 1),
                "scene_bytes": scene_size,
                "tileset_bytes": ts_size,
                "num_elements": r.metrics.get("num_elements", 0),
                "num_assets": r.metrics.get("num_assets", 0),
                "warnings": len(r.warnings),
                "errors": len(r.errors),
            }
            rows.append(row)
            status = "OK" if r.ok else "FAIL"
            print(f"{status:4} {img.name:25} {target.value:20} {elapsed:6.0f} ms scene={scene_size:6}B ts={ts_size:7}B")
    out_json = OUTPUT_DIR / "benchmark.json"
    out_json.write_text(json.dumps(rows, indent=2))
    passed = sum(1 for r in rows if r["ok"])
    print(f"\n{passed}/{len(rows)} benchmark runs OK — results in {out_json}")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())