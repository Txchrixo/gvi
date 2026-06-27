"""Top-level orchestrator — runs the planned pipeline and produces a Godot scene."""
from __future__ import annotations

import json
import time
from pathlib import Path

from gvi.core.plugin import PluginContext
from gvi.core.planner import PipelinePlanner
from gvi.core.registry import PluginRegistry
from gvi.core.types import (
    AssetProfile,
    ConversionRequest,
    ConversionResult,
    SegmentationResult,
)


class Orchestrator:
    """Executes a planned pipeline on the input asset."""

    def __init__(self, registry: PluginRegistry | None = None) -> None:
        self.registry = registry or PluginRegistry.with_builtins()
        self.planner = PipelinePlanner(self.registry)

    # ------------------------------------------------------------- inspect ---
    def inspect(self, input_path: Path) -> AssetProfile:
        probe = self.registry.get("detector.asset_probe")
        work_dir = Path("/tmp") / f"gvi-inspect-{int(time.time() * 1000)}"
        work_dir.mkdir(parents=True, exist_ok=True)
        ctx = PluginContext(work_dir=str(work_dir), cache_dir=str(Path.home() / ".cache" / "gvi"))
        result = probe.run(input_path, ctx)
        if result.errors:
            raise RuntimeError("; ".join(result.errors))
        profile = result.data
        if not isinstance(profile, AssetProfile):
            raise RuntimeError("Asset probe did not return AssetProfile")

        # Run heuristics if raster
        if profile.kind.value in {"raster", "spritesheet", "tilemap_source"}:
            for aid in ("analyzer.raster_heuristics", "analyzer.spritesheet"):
                if self.registry.has(aid):
                    try:
                        ar = self.registry.get(aid).run(profile, ctx)
                        if isinstance(ar.data, AssetProfile):
                            profile = ar.data
                        profile.notes.extend(ar.warnings)
                    except Exception as exc:  # noqa: BLE001
                        profile.notes.append(f"{aid} failed: {exc}")
        return profile

    # ------------------------------------------------------------- convert ---
    def convert(self, request: ConversionRequest) -> ConversionResult:
        request.output_dir.mkdir(parents=True, exist_ok=True)
        profile = self.inspect(request.input_path)
        plan = self.planner.plan(profile, request)

        work_dir = Path(request.output_dir) / ".work"
        work_dir.mkdir(parents=True, exist_ok=True)
        ctx = PluginContext(
            work_dir=str(work_dir),
            cache_dir=str(Path.home() / ".cache" / "gvi"),
            options={
                "target": request.target.value,
                "preset": request.preset,
                **request.user_hints,
                **request.options.model_dump(mode="json"),
            },
        )

        payload: dict = {"request": request, "profile": profile, "ir": None, "segmentation": None, "plan": plan}
        warnings: list[str] = []
        errors: list[str] = []
        metrics: dict[str, float] = {}
        artifacts: dict[str, Path] = {}
        total_start = time.perf_counter()

        # Persist the planned pipeline for transparency.
        plan_path = request.output_dir / "pipeline_plan.json"
        plan_path.write_text(
            json.dumps({"id": plan.id, "steps": list(plan.steps), "reason": plan.reason}, indent=2),
            encoding="utf-8",
        )
        artifacts["pipeline_plan"] = plan_path

        for step_id in plan.steps:
            if not self.registry.has(step_id):
                warnings.append(f"Step '{step_id}' skipped: plugin not registered")
                continue
            plugin = self.registry.get(step_id)
            step_start = time.perf_counter()
            try:
                step_result = plugin.run(payload, ctx)
            except Exception as exc:  # noqa: BLE001
                step_result = None
                errors.append(f"Step '{step_id}' crashed: {exc}")
                metrics[f"{step_id}_failed"] = 1.0
                if request.options.fail_fast:
                    break
                continue
            finally:
                metrics[f"{step_id}_time_ms"] = (time.perf_counter() - step_start) * 1000

            if step_result is None:
                continue

            warnings.extend(step_result.warnings)
            errors.extend(step_result.errors)
            metrics.update(step_result.metrics)
            artifacts.update(step_result.artifacts)

            if step_result.data is not None:
                if isinstance(step_result.data, dict):
                    payload.update(step_result.data)
                else:
                    payload["data"] = step_result.data
                    if isinstance(step_result.data, SegmentationResult):
                        payload["segmentation"] = step_result.data

            if step_result.errors and request.options.fail_fast:
                break

        metrics["total_time_ms"] = (time.perf_counter() - total_start) * 1000

        return ConversionResult(
            request=request,
            profile=payload.get("profile", profile),
            pipeline_id=plan.id,
            output_dir=request.output_dir,
            artifacts=artifacts,
            metrics=metrics,
            warnings=warnings,
            errors=errors,
        )