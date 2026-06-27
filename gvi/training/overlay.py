"""Visual QA overlays for predictions and annotations."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from gvi.training.annotations import AnnotationFile


PALETTE = [
    "#00e5ff", "#ffea00", "#ff5252", "#69f0ae", "#b388ff", "#ff9100",
    "#f50057", "#64ffda", "#c6ff00", "#40c4ff", "#ffd180", "#ea80fc",
]


def draw_overlay(annotation: AnnotationFile, output_path: Path) -> Path:
    image = Image.open(annotation.image_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for idx, obj in enumerate(annotation.objects):
        color = PALETTE[idx % len(PALETTE)]
        x, y, w, h = obj.bbox_xywh
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
        if obj.polygon and obj.polygon.is_valid:
            draw.line(obj.polygon.points + [obj.polygon.points[0]], fill=color, width=2)
        label = f"{obj.class_name} {obj.confidence:.2f}"
        draw.rectangle([x, max(0, y - 14), x + len(label) * 7 + 6, y], fill=(0, 0, 0, 180))
        draw.text((x + 3, max(0, y - 13)), label, fill=color, font=font)
    composed = Image.alpha_composite(image, overlay)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    composed.convert("RGB").save(output_path)
    return output_path
