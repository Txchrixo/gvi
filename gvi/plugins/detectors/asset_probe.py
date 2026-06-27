"""Asset probe — detects format, dimensions, alpha, etc.

Reliable detection is the foundation of every pipeline.
"""
from __future__ import annotations

import mimetypes
import struct
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from gvi.core.plugin import Capability, Plugin, PluginContext
from gvi.core.types import AssetKind, AssetProfile, CapabilityType, PipelineStepResult


class AssetProbePlugin(Plugin):
    def capability(self) -> Capability:
        return Capability(
            id="detector.asset_probe",
            type=CapabilityType.DETECTOR,
            name="Asset type and metadata probe",
            priority=100,
            provides=["asset_profile"],
            tags={"core"},
        )

    def run(self, payload, ctx: PluginContext) -> PipelineStepResult:
        path = Path(payload).expanduser().resolve()
        if not path.exists():
            return PipelineStepResult(name="asset_probe", ok=False, errors=[f"File not found: {path}"])

        ext = path.suffix.lower()
        mime = mimetypes.guess_type(str(path))[0]
        profile = AssetProfile(
            path=path,
            extension=ext,
            mime_type=mime,
            metadata={"size_bytes": path.stat().st_size},
        )
        warnings: list[str] = []

        if ext in {".svg", ".svgz"}:
            profile.kind = AssetKind.VECTOR
            profile.mime_type = "image/svg+xml"
            profile.has_vector_data = True
            self._probe_svg_dimensions(path, profile)
        elif ext == ".pdf":
            profile.kind = AssetKind.DOCUMENT
            profile.mime_type = "application/pdf"
            self._probe_pdf(path, profile, warnings)
        elif ext in {".psd", ".psb"}:
            profile.kind = AssetKind.PSD
            profile.has_layers = True
            profile.mime_type = "image/vnd.adobe.photoshop"
            self._probe_psd(path, profile, warnings)
        elif ext in {".fig", ".figma"}:
            profile.kind = AssetKind.FIGMA
            profile.notes.append("Figma binary/API import is planned; export SVG/PNG/PDF for now.")
        else:
            try:
                with Image.open(path) as img:
                    profile.kind = self._classify_raster(img)
                    profile.mime_type = Image.MIME.get(img.format, mime or "image/unknown")
                    profile.width, profile.height = img.size
                    profile.has_alpha = img.mode in {"RGBA", "LA"} or "transparency" in img.info
                    profile.channels = len(img.getbands())
                    dpi = img.info.get("dpi")
                    if dpi:
                        profile.dpi = tuple(int(v) for v in dpi[:2])
                    profile.metadata.update(
                        {"pil_format": img.format, "mode": img.mode, "frames": getattr(img, "n_frames", 1)}
                    )
            except UnidentifiedImageError:
                profile.kind = AssetKind.UNKNOWN
                profile.notes.append("Unknown file type; add a parser plugin or export to PNG/SVG/PDF.")

        return PipelineStepResult(name="asset_probe", data=profile, warnings=warnings)

    # --------------------------------------------------------------- helpers
    def _classify_raster(self, img: Image.Image) -> AssetKind:
        w, h = img.size
        if img.format == "GIF" and getattr(img, "n_frames", 1) > 1:
            return AssetKind.SPRITESHEET
        if w <= 1024 and h <= 1024 and img.mode == "P":
            return AssetKind.SPRITESHEET
        return AssetKind.RASTER

    def _probe_svg_dimensions(self, path: Path, profile: AssetProfile) -> None:
        try:
            import re
            text = path.read_text(encoding="utf-8", errors="ignore")[:8192]
            w = re.search(r'\bwidth=["\']([0-9.]+)', text)
            h = re.search(r'\bheight=["\']([0-9.]+)', text)
            vb = re.search(r'\bviewBox=["\']\s*[-0-9.]+\s+[-0-9.]+\s+([0-9.]+)\s+([0-9.]+)', text)
            if w and h:
                profile.width = int(float(w.group(1)))
                profile.height = int(float(h.group(1)))
            elif vb:
                profile.width = int(float(vb.group(1)))
                profile.height = int(float(vb.group(2)))
            profile.has_text = "<text" in text
            profile.has_vector_data = True
            profile.has_layers = "<g " in text or "<g>" in text
        except Exception as exc:  # noqa: BLE001
            profile.notes.append(f"SVG probe failed: {exc}")

    def _probe_pdf(self, path: Path, profile: AssetProfile, warnings: list[str]) -> None:
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(path)
            profile.page_count = doc.page_count
            if doc.page_count:
                rect = doc[0].rect
                profile.width = int(rect.width)
                profile.height = int(rect.height)
                text = doc[0].get_text("text")
                profile.has_text = bool(text.strip())
            doc.close()
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"PDF deep probe unavailable ({exc}); install PyMuPDF for better PDF support.")

    def _probe_psd(self, path: Path, profile: AssetProfile, warnings: list[str]) -> None:
        try:
            from psd_tools import PSDImage
            psd = PSDImage.open(path)
            profile.width, profile.height = psd.size
            descendants = list(psd.descendants())
            profile.metadata["num_layers"] = len(descendants)
            profile.has_text = any(getattr(layer, "kind", "") == "type" for layer in descendants)
        except Exception:
            try:
                with path.open("rb") as f:
                    header = f.read(26)
                if header[:4] == b"8BPS" and len(header) >= 26:
                    channels, height, width = struct.unpack(">HII", header[12:22])
                    profile.width = int(width)
                    profile.height = int(height)
                    profile.channels = int(channels)
                warnings.append("psd-tools not installed; layer extraction will fall back to flattened PSD when possible.")
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"PSD probe failed: {exc}")