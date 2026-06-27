# Dataset Card — GVI platformer

## Goal
Train a student segmentation model that recognizes 2D game / UI elements and
converts them into reliable Godot scene IR objects.

## Classes
- `platform`: Solid traversable gameplay surface.
- `wall`: Solid vertical or background boundary.
- `floor`: Main ground/floor collision surface.
- `ladder`: Climbable vertical ladder.
- `spike`: Pointed hazard, often triangular.
- `hazard`: Generic dangerous gameplay area.
- `door`: Door or level transition object.
- `exit`: Goal or level exit marker.
- `player_spawn`: Initial player spawn marker.
- `enemy`: Enemy or character-like gameplay entity.
- `pickup`: Collectible or interactive pickup.
- `coin`: Coin-like collectible.
- `button`: In-level button or trigger.
- `lever`: In-level lever or switch.
- `water`: Water/lava/liquid-like region.
- `background`: Non-interactive background layer.
- `foreground_decoration`: Non-interactive foreground decoration.
- `light`: Light orb or lamp marker.

## Sources
Document every source you collect. Respect game licenses and avoid shipping
third-party copyrighted assets in public model demos unless you have permission.
Use synthetic examples and your own assets for public releases.

## Annotation rules
- Use tight masks for interactive elements.
- Use full visible object masks for enemies, pickups and doors.
- Use collision-relevant masks for platforms, walls and hazards.
- Put ambiguous elements in `needs_review` instead of guessing.

## Quality gates
- Minimum 50 images for prototype.
- 200+ corrected images before a demo model.
- Track mAP, per-class recall, false positives and GVI fidelity error.
