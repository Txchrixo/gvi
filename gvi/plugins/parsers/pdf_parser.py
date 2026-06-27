"""PDF parser — rasterizes the requested page and extracts text spans."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from gvi.core.plugin import Capability, Plugin, PluginContext
from gvi.core.types import AssetKind, AssetProfile, CapabilityType, DetectedElement, PipelineStepResult, SegmentationResult


class PDFParserPlugin(Plugin):
    def capability(self) -> Capability:
        return Capability(
            id="parser.pdf",
            type=CapabilityType.PARSER,
            name="PDF page rasterizer + text spans",
            supported_extensions=[".pdf"],
            priority=85,
            provides=["segmentation", "raster_profile", "text_elements"],
            tags={"document", "pdf"},
        )

    def run(self, payload: dict[str, Any], ctx: PluginContext) -> PipelineStepResult:
        profile: AssetProfile = payload["profile"]
        if not profile.path:
            return PipelineStepResult(name="pdf_parser", ok=False, errors=["Missing PDF path"])

        out_dir = Path(ctx.work_dir) / "assets"
        out_dir.mkdir(parents=True, exist_ok=True)
        warnings: list[str] = []

        try:
            import fitz  # PyMuPDF
        except ImportError:
            return PipelineStepResult(
                name="pdf_parser", ok=False,
                errors=["PyMuPDF is required for PDF import. Install with: python -m pip install 'gvi[pdf]'"],
            )
        warnings.append(
            "PyMuPDF (the 'pdf' extra) is AGPL-3.0/commercial dual-licensed, "
            "not MIT like GVI's core dependencies -- see NOTICE_LICENSING.md "
            "before publishing a fork or running this as a public service "
            "with PDF import enabled."
        )

        opts = ctx.options
        page_index = int(opts.get("pdf_page", 0))
        dpi = int(opts.get("pdf_dpi", 160))

        try:
            doc = fitz.open(profile.path)
            if page_index >= doc.page_count:
                page_index = 0
            page = doc[page_index]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, alpha=True)
            raster_path = out_dir / f"{profile.path.stem}_page{page_index}.png"
            pix.save(str(raster_path))
            page_w = pix.width
            page_h = pix.height

            # Text spans -> DetectedElement of type text
            text_elements: list[DetectedElement] = []
            text_dict = page.get_text("dict")
            for block in text_dict.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text:
                            continue
                        bbox = span.get("bbox", [0, 0, 0, 0])
                        # Convert from PDF coordinates to pixel coordinates.
                        sx = page_w / page.rect.width
                        sy = page_h / page.rect.height
                        x = int(bbox[0] * sx)
                        y = int(bbox[1] * sy)
                        w = int((bbox[2] - bbox[0]) * sx)
                        h = int((bbox[3] - bbox[1]) * sy)
                        text_elements.append(DetectedElement(
                            id=f"pdf_text_{len(text_elements):04d}",
                            element_type="text",
                            bounds=(x, y, w, h),
                            text_content=text,
                            confidence=0.95,
                            z_index=100 + len(text_elements),
                            source="pdf",
                            metadata={"pdf_font": span.get("font", ""), "pdf_size": float(span.get("size", 0)), "pdf_color": int(span.get("color", 0))},
                        ))

            elements: list[DetectedElement] = [DetectedElement(
                id="background_source",
                element_type="background",
                bounds=(0, 0, page_w, page_h),
                asset_path=raster_path,
                z_index=-1000,
                locked=True,
                source="pdf_raster",
                metadata={"source": "pdf_page_rasterized", "page": page_index},
            )]
            elements.extend(text_elements)

            raster_profile = profile.model_copy(deep=True)
            raster_profile.path = raster_path
            raster_profile.kind = AssetKind.RASTER
            raster_profile.extension = ".png"
            raster_profile.mime_type = "image/png"
            raster_profile.width = page_w
            raster_profile.height = page_h
            raster_profile.has_alpha = True

            segmentation = SegmentationResult(
                elements=elements,
                num_elements=len(elements),
                coverage_ratio=1.0,
                method_used="pdf_parser",
                diagnostics={"page": page_index, "num_text_spans": len(text_elements)},
            ).sort_and_index()

            doc.close()
            return PipelineStepResult(
                name="pdf_parser",
                data={**payload, "profile": raster_profile, "segmentation": segmentation, "parsed_asset": raster_path},
                artifacts={"parsed_asset": raster_path},
                metrics={"pdf_text_spans": float(len(text_elements))},
                warnings=warnings,
            )

        except Exception as exc:  # noqa: BLE001
            return PipelineStepResult(name="pdf_parser", ok=False, errors=[f"PDF processing failed: {exc}"])