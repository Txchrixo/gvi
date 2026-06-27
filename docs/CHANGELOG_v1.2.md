# Changelog v1.2.0 — Training Ready

Added:

- `gvi training` CLI subcommands.
- `gvi dataset` CLI subcommands.
- Platformer/UI/TileMap taxonomy configs.
- GVI JSON annotation schema.
- YOLO segmentation label export.
- Heuristic auto-label backend.
- YOLO teacher backend.
- Grounded-SAM endpoint adapter.
- Active-learning review generation.
- Visual overlay generation.
- Synthetic platformer dataset generator.
- Training docs, annotation guide, teacher backend guide and AI-engineer portfolio guide.

Validation run:

- `python -m pytest tests/test_training_system.py -q` passed.
- `python -m compileall -q gvi` passed.
- CLI smoke tests for `dataset init` and `training train --dry-run` passed.
