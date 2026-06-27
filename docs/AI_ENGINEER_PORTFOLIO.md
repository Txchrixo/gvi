# How to present this as AI Engineer / Machine Learning Engineer work

This project is not only a Godot utility. It is an applied computer-vision and ML system.

## Strong technical bullets

- Designed an intermediate representation for converting visual assets into editable game-engine scene graphs.
- Built a teacher-student training loop using zero-shot/weak labeling and YOLO segmentation fine-tuning.
- Implemented active learning with confidence-based review queues and visual QA overlays.
- Combined model confidence, geometric rules and target-engine context to improve semantic classification.
- Built a CLI-ready dataset management system with taxonomies, dataset cards and reproducible train/evaluate commands.
- Prepared adapter architecture for Grounding DINO/SAM2, YOLO, OCR, PSD/SVG/PDF parsers and Godot exporters.

## Demo path

1. Show raw image.
2. Run auto-label.
3. Show overlay.
4. Correct 2 labels.
5. Train or dry-run train.
6. Predict with student model.
7. Export to Godot scene.
8. Show `.tscn` hierarchy.

## Metrics to track

- mAP50 / mAP50-95 for segmentation.
- Per-class precision/recall.
- Review rate: % objects needing human correction.
- Godot fidelity score: visual reconstruction similarity.
- Scene usability: % platforms with valid collisions.
