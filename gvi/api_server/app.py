"""FastAPI server exposing GVI as a REST API."""
from __future__ import annotations

import io
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from gvi.core.orchestrator import Orchestrator
from gvi.core.types import ConversionOptions, ConversionRequest, TargetKind

app = FastAPI(title="GVI API", version="1.0.0")


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "version": "1.0.0"}


@app.post("/inspect")
async def inspect(file: UploadFile = File(...)) -> JSONResponse:
    """Inspect an uploaded asset."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    try:
        profile = Orchestrator().inspect(tmp_path)
        return JSONResponse(content=profile.model_dump(mode="json"))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


@app.post("/convert")
async def convert(
    file: UploadFile = File(...),
    target: str = Form("godot.node2d"),
    preset: str = Form("balanced"),
    sam2: bool = Form(False),
    sam2_model: str = Form("sam2_hiera_small"),
    semantic: bool = Form(True),
    ocr: bool = Form(True),
    tile_size: int = Form(32),
    include_background: bool = Form(True),
    max_elements: int = Form(500),
) -> JSONResponse:
    """Convert an uploaded asset to a Godot scene."""
    try:
        target_kind = TargetKind(target)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid target: {target}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / (file.filename or "input.bin")
        with tmp_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        out_dir = Path(tmp_dir) / "out"
        opts = ConversionOptions(
            include_background=include_background,
            max_elements=max_elements,
            sam2_enabled=sam2,
            sam2_model=sam2_model,  # type: ignore[arg-type]
            semantic_detection=semantic,
            include_text_elements=ocr,
            tilemap_tile_size=tile_size,
        )
        request = ConversionRequest(
            input_path=tmp_path,
            target=target_kind,
            output_dir=out_dir,
            preset=preset,  # type: ignore[arg-type]
            options=opts,
        )
        try:
            result = Orchestrator().convert(request)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc))
        # Bundle the resulting files into a single zip
        import zipfile
        zip_path = Path(tmp_dir) / "result.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in out_dir.rglob("*"):
                if p.is_file():
                    zf.write(p, arcname=str(p.relative_to(out_dir)))
        with zip_path.open("rb") as f:
            zip_bytes = f.read()
        return JSONResponse(
            content={
                "ok": result.ok,
                "pipeline_id": result.pipeline_id,
                "warnings": result.warnings,
                "errors": result.errors,
                "artifacts": {k: str(v) for k, v in result.artifacts.items()},
                "metrics": result.metrics,
                "result_zip_b64": __import__("base64").b64encode(zip_bytes).decode(),
            }
        )