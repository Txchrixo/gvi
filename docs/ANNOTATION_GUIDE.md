# Annotation Guide

## Core platformer classes

Start with these classes:

```text
platform, wall, floor, ladder, spike, hazard, door, exit, player_spawn, enemy, pickup, coin, button, lever, water, background, foreground_decoration, light
```

For a first model, focus on:

```text
platform, ladder, spike, door, enemy, pickup, background, wall
```

## Rules

### platform
Annotate only the visible solid gameplay surface. Use masks/polygons for irregular shapes. Do not include background grid lines unless they are part of the platform sprite.

### ladder
Annotate the full climbable ladder, including side rails and rungs.

### spike/hazard
Annotate the actual dangerous shape, not empty space around it.

### enemy
Annotate the visible character/entity body. If unsure whether it is enemy/player/NPC, use `enemy` but mark review in GVI JSON or correct later.

### pickup/coin/light
Small floating objects should be tight masks. If the object is decorative, mark `foreground_decoration` instead.

### background
Use for non-interactive regions. Avoid overusing background in YOLO training because it can dominate the dataset.

## Quality rules

- Tight masks beat loose boxes.
- Ambiguous label beats wrong label.
- Keep class definitions consistent.
- Review all objects below 0.72 confidence.
- Keep validation images from different sources than training images.
