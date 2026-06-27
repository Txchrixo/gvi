"""Hierarchy builder — groups small elements inside large panels/frames.

When an image contains a big rectangle that fully contains several smaller
sprites/text, those children become children of the parent in the IR. This
makes the resulting Godot scene tree much more meaningful and editable.
"""
from __future__ import annotations

from typing import Any

from gvi.core.plugin import Capability, Plugin, PluginContext
from gvi.core.types import CapabilityType, PipelineStepResult


class HierarchyBuilder(Plugin):
    def capability(self) -> Capability:
        return Capability(
            id="postprocess.hierarchy",
            type=CapabilityType.POSTPROCESSOR,
            name="Parent/child hierarchy builder",
            priority=70,
            provides=["ir", "hierarchy"],
        )

    def run(self, payload: dict[str, Any], ctx: PluginContext) -> PipelineStepResult:
        seg = payload.get("segmentation")
        if seg is None:
            return PipelineStepResult(name="hierarchy", ok=False, warnings=["No segmentation available for hierarchy"])
        elements = list(seg.elements)
        background = [e for e in elements if e.element_type == "background"]
        containers = [e for e in elements if e.element_type in {"panel", "frame"}]
        others = [e for e in elements if e.element_type not in {"background", "panel", "frame"}]

        # Sort containers largest-first so children are attached to the deepest enclosing parent.
        containers.sort(key=lambda e: e.bounds[2] * e.bounds[3], reverse=True)

        # Greedy containment assignment.
        for elem in others:
            parent_id = None
            best_area = None
            for cont in containers:
                if self._contains(cont.bounds, elem.bounds):
                    area = cont.bounds[2] * cont.bounds[3]
                    if best_area is None or area < best_area:
                        parent_id = cont.id
                        best_area = area
            elem.parent_id = parent_id

        seg.elements = background + containers + others
        seg.sort_and_index()

        return PipelineStepResult(
            name="hierarchy",
            data={**payload, "segmentation": seg},
            metrics={"num_root": float(len([e for e in seg.elements if not e.parent_id]))},
            artifacts={},
        )

    def _contains(self, outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
        ox, oy, ow, oh = outer
        ix, iy, iw, ih = inner
        return ox <= ix and oy <= iy and ox + ow >= ix + iw and oy + oh >= iy + ih