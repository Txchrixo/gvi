# GVI v1.2.1 — Audit & hardening of the training layer

This patch is a verification pass on the v1.2 "training-ready" claim. The
training modules were not just read — they were **executed end-to-end** in an
offline sandbox (via the test-only pydantic/rich shims in
`offline_test_shims/`, see that folder's README), and the issues found were
fixed.

## Verified working by actually running it
- `dataset init` → real YOLO folder tree + valid `data.yaml` (18 classes).
- `training synthetic` → real images + valid YOLO-segmentation label files
  (class_id + normalized polygon). Lets you train with zero collected images.
- `dataset stats`, `dataset ingest`.
- `autolabel --backend heuristic` → GVI JSON annotations with the exact schema
  from the spec (class / bbox / polygon / confidence / godot_candidates /
  needs_review).
- GVI→YOLO label export (43 labels exported in the test run).
- `review` (active-learning review.json).
- Business-rule engine `score_object`: a horizontal "ladder" is correctly
  demoted to needs_review with reason "Ladder candidate is not vertical
  enough"; a vertical one passes. Godot node candidates correct.
- Overlay rendering.
- `train --dry-run` → emits the real `yolo segment train ...` command.
- All 4 packaged tests in `tests/test_training_system.py`.

## Bugs fixed in this patch
1. **CLI `--semantic` default was still `True`** (flagged back in v1.1, never
   fixed). It overrode the `semantic_detection=False` default in
   `ConversionOptions`, so every `gvi convert` silently re-enabled the
   COCO-pretrained YOLO path the user never asked for. Now `False`.
2. **Path-vs-str fragility on every public training entry point.** `init_dataset`,
   `generate_synthetic_platformer`, `autolabel_directory`, `build_review_file`,
   `evaluate_dataset`, `export_gvi_annotations_to_yolo`, and `TrainConfig` all
   called `.resolve()` / used `/` on their path argument directly. That works
   when typer hands them a `Path`, but crashed (`'str' object has no attribute
   'resolve'`, `unsupported operand type(s) for /: 'str' and 'str'`) for any
   programmatic caller, notebook, or test passing a string. All now coerce
   with `Path(...)` first. The full pipeline was then re-run end-to-end with
   **all paths as plain strings** to confirm.

## Not verified (be skeptical until you run it yourself)
- The real `yolo segment train` run, the `yolo` and `grounded-sam` autolabel
  backends, real `predict`, and Colab training — all need GPU/network/
  ultralytics not present in the audit sandbox. They fail cleanly with an
  explicit "Install training deps first" message when the dependency is
  missing, rather than crashing obscurely.
- The pydantic shim used to run all of the above is deliberately lenient (no
  strict type validation). Run `pytest tests/` after a real
  `pip install -e ".[dev,training]"` to get pydantic's real guarantees.
