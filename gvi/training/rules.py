"""Rule-based semantic scoring for game/UI elements.

The trained model should not be the only decision-maker. These rules adjust the
model confidence using shape, geometry and Godot-target context.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gvi.training.annotations import AnnotationObject


GODOT_CANDIDATES = {
    "platform": ["StaticBody2D", "CollisionPolygon2D", "Sprite2D"],
    "wall": ["StaticBody2D", "CollisionPolygon2D", "Sprite2D"],
    "floor": ["StaticBody2D", "CollisionShape2D", "Sprite2D"],
    "ladder": ["Area2D", "CollisionShape2D", "Sprite2D"],
    "spike": ["Area2D", "CollisionPolygon2D", "Sprite2D"],
    "hazard": ["Area2D", "CollisionPolygon2D", "Sprite2D"],
    "door": ["Area2D", "CollisionShape2D", "Sprite2D"],
    "exit": ["Area2D", "Marker2D", "Sprite2D"],
    "enemy": ["Marker2D", "CharacterBody2D", "Sprite2D"],
    "player_spawn": ["Marker2D"],
    "pickup": ["Area2D", "Sprite2D"],
    "coin": ["Area2D", "Sprite2D"],
    "button": ["Area2D", "Button", "Sprite2D"],
    "lever": ["Area2D", "Sprite2D"],
    "water": ["Area2D", "Polygon2D", "Sprite2D"],
    "background": ["Sprite2D", "TextureRect", "ParallaxBackground"],
    "foreground_decoration": ["Sprite2D"],
    "light": ["PointLight2D", "Sprite2D"],
    "text": ["Label", "RichTextLabel"],
    "ui_button": ["Button", "TextureButton"],
    "ui_panel": ["Panel", "NinePatchRect", "TextureRect"],
    "ui_text": ["Label", "RichTextLabel"],
    "icon": ["TextureRect", "Sprite2D"],
}


@dataclass(slots=True)
class RuleDecision:
    final_class: str
    rule_score: float
    context_score: float
    final_confidence: float
    godot_candidates: list[str]
    needs_review: bool
    reason: str | None = None


def score_object(obj: AnnotationObject, image_width: int, image_height: int, target_mode: str = "platformer-level") -> RuleDecision:
    x, y, w, h = obj.bbox_xywh
    aspect = w / max(h, 1)
    area_ratio = (w * h) / max(image_width * image_height, 1)
    rule = 0.75
    reason: str | None = None

    cls = obj.class_name
    if cls in {"platform", "floor", "wall"}:
        if aspect >= 1.4 or area_ratio > 0.02:
            rule += 0.15
        else:
            rule -= 0.15
            reason = "Platform/wall candidate has unusual geometry."
    elif cls == "ladder":
        if h > w and aspect < 0.8:
            rule += 0.15
        else:
            rule -= 0.25
            reason = "Ladder candidate is not vertical enough."
    elif cls in {"spike", "hazard"}:
        if area_ratio < 0.05:
            rule += 0.10
        else:
            rule -= 0.10
            reason = "Hazard candidate is too large; may be decoration."
    elif cls in {"enemy", "pickup", "coin", "light"}:
        if area_ratio < 0.08:
            rule += 0.05
        else:
            rule -= 0.15
            reason = "Entity candidate is too large; review semantics."
    elif cls.startswith("ui_") or cls in {"text", "icon"}:
        if "ui" in target_mode or "control" in target_mode:
            rule += 0.15
        else:
            rule -= 0.05

    context = 0.8
    if target_mode in {"platformer-level", "tilemap"} and cls in {"platform", "wall", "floor", "ladder", "spike", "hazard", "door", "enemy", "pickup"}:
        context += 0.12
    if target_mode in {"ui-control", "godot.control"} and (cls.startswith("ui_") or cls in {"text", "icon"}):
        context += 0.12

    rule = max(0.0, min(1.0, rule))
    context = max(0.0, min(1.0, context))
    final = 0.55 * obj.confidence + 0.25 * rule + 0.20 * context
    needs_review = final < 0.72 or bool(obj.needs_review) or (reason is not None and rule < 0.65)
    if needs_review and reason is None:
        reason = "Low final confidence after model + rule + context scoring."
    return RuleDecision(
        final_class=cls,
        rule_score=rule,
        context_score=context,
        final_confidence=max(0.0, min(1.0, final)),
        godot_candidates=GODOT_CANDIDATES.get(cls, ["Sprite2D"]),
        needs_review=needs_review,
        reason=reason,
    )
