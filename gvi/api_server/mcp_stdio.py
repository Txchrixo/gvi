"""Stdio MCP-style tool server for GVI.

Reads one JSON request per line and returns one JSON response per line.
Compatible with simple MCP-style agent integrations.
"""
from __future__ import annotations

import json
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from gvi.core.orchestrator import Orchestrator
from gvi.core.types import ConversionOptions, ConversionRequest, TargetKind


def _ok(id, result):
    return {"id": id, "ok": True, "result": result}


def _err(id, error):
    return {"id": id, "ok": False, "error": str(error)}


def _handle(request: dict[str, Any]) -> dict[str, Any]:
    rid = request.get("id")
    method = request.get("method")
    params = request.get("params", {}) or {}
    try:
        if method == "tools/list":
            return _ok(rid, {"tools": [
                {"name": "gvi.inspect", "description": "Inspect an image / vector / PSD / PDF."},
                {"name": "gvi.convert", "description": "Convert any image to a Godot scene."},
            ]})
        if method == "tools/call":
            tool = params.get("name")
            args = params.get("arguments", {})
            if tool == "gvi.inspect":
                path = Path(args["path"]).expanduser().resolve()
                profile = Orchestrator().inspect(path)
                return _ok(rid, profile.model_dump(mode="json"))
            if tool == "gvi.convert":
                path = Path(args["path"]).expanduser().resolve()
                target = args.get("target", "godot.node2d")
                preset = args.get("preset", "balanced")
                out_dir = Path(args.get("out") or tempfile.mkdtemp(prefix="gvi-mcp-")) / uuid.uuid4().hex
                options = ConversionOptions(**(args.get("options") or {}))
                request_obj = ConversionRequest(input_path=path, target=TargetKind(target), output_dir=out_dir, preset=preset, options=options)
                result = Orchestrator().convert(request_obj)
                return _ok(rid, {
                    "ok": result.ok,
                    "pipeline_id": result.pipeline_id,
                    "artifacts": {k: str(v) for k, v in result.artifacts.items()},
                    "metrics": result.metrics,
                    "warnings": result.warnings,
                    "errors": result.errors,
                    "output_dir": str(out_dir),
                })
            return _err(rid, f"Unknown tool: {tool}")
        return _err(rid, f"Unknown method: {method}")
    except Exception as exc:  # noqa: BLE001
        return _err(rid, exc)


def main() -> None:
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            sys.stdout.write(json.dumps({"ok": False, "error": f"bad json: {exc}"}) + "\n")
            sys.stdout.flush()
            continue
        resp = _handle(req)
        sys.stdout.write(json.dumps(resp, default=str) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()