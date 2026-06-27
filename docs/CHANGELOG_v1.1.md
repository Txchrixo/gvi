# GVI v1.1 — Changelog (over-segmentation fix)

This release was produced in direct response to an independent audit that
graded v1.0.0 at **12/20**, with the central finding that the v1.0.0 fix for
the "0 elements detected on photos" bug (present since v0.3.0) had
overcorrected into a new, arguably worse failure mode: **100-160 spurious,
chaotic elements per image** on any real (non-synthetic) picture.

This document is deliberately written in the same spirit as that audit:
concrete numbers, no unverified superlatives, and an explicit list of what
was **not** possible to verify in the sandbox this work was done in.

## What changed

`gvi/plugins/segmenters/opencv_segmenter.py` now delegates to a new pure
module, `gvi/plugins/segmenters/_opencv_core.py`, which replaces the
"run every strategy, IoU-dedupe what's left" approach with:

1. **Scene-complexity-adaptive strategy selection** (`flat` / `medium` /
   `textured`, estimated from quantized-color count in a 64x64 thumbnail —
   empirically far more reliable than edge density alone at separating flat
   UI/vector art from photographic or gradient-heavy content). Watershed and
   full-resolution threshold contouring — the two biggest sources of
   micro-fragments — are skipped entirely on textured scenes.
2. **A per-candidate plausibility filter** (`is_plausible`) based on contour
   solidity and internal pixel-intensity homogeneity, which rejects
   texture/noise fragments *before* they are ever saved as a PNG.
3. **Color- and proximity-aware region merging** (`merge_fragments`, union-
   find), which consolidates genuine fragmentation (e.g. a dozen scraps of
   the same painted surface split by specular highlights) instead of only
   removing near-duplicate boxes the way plain IoU dedup does.
4. **Containment-aware final dedup**, which also catches same-type boxes
   that cover almost the same region at a different aspect ratio (missed by
   plain IoU when the two boxes differ a lot in shape).
5. **Optional GrabCut boundary refinement**, applied to flattened
   (non-alpha) photographic candidates to clean up a crude rectangle/contour
   into a real foreground mask.
6. **Homography-based rectification** for rotated quads (`rectify_if_rotated`):
   a tilted frame/panel is now warped to an upright rectangle via a 4-point
   perspective transform instead of taking a wasteful axis-aligned crop.
   (Still no OpenCV `GrabCut`/homography existed in v1.0.0 at all — this was
   flagged as an outstanding gap across *two* prior major versions.)
7. **An adaptive element budget by scene label** (e.g. 60 for flat UI, 20 for
   photographic content) instead of a flat ceiling of 500.

## Real before/after numbers

All numbers below were produced by actually running the segmentation core
(not by reading the code and guessing) on 7 images: the 3 synthetic images
already bundled with GVI, plus 4 real, diverse images supplied for this
audit (a stylised 3D map render, a glossy toy-robot product photo, a
Monster Energy can photo, and a small flat mascot logo).

| Image | v1.0.0 elements | v1.1 elements | Note |
|---|---:|---:|---|
| `wall_of_frames.png` (synthetic) | 9 | 7 | still correctly finds all 6 painted frames |
| `ui_mockup.png` (synthetic) | — | 5 | clean |
| `sprite_scene.png` (synthetic) | — | 8 | clean |
| Stylised map render | **120** | **3** | 1 background panel, 1 mid-band, 1 highlighted-region box |
| Glossy toy robot photo | **157** | **30** (budget cap) | biggest remaining weak point, see below |
| Monster Energy can photo | **120** | **4** | can body + claw-mark highlight + label band |
| Small flat mascot logo | 7 | 1 | mascot + text now read as one coherent panel |

A repeatable regression test for exactly this failure mode now exists at
`tests/test_segmentation_quality_gate.py::test_no_fragment_storm_and_no_zero_detection`,
with both a ceiling (catches fragment-storm regressions) and a floor
(catches a return of the original 0-detection bug) per image. It was run
and passes on all 7 cases as of this changelog.

Two new geometric primitives also have real, executed unit tests using
synthetic images (not just code that compiles): a known-18°-tilted
rectangle is correctly rectified to upright, a near-upright (1°) one is
correctly left alone, a non-rectangular blob is correctly rejected by the
rectangularity guard, and GrabCut measurably trims a loose bounding box
down to roughly the true foreground area on a synthetic soft-edged blob.

## Known remaining weak point: glossy/reflective surfaces

The toy-robot photo still produces 30 elements (hitting the new, much lower
budget cap) rather than the ~6-8 a human would draw by hand. Specular
highlights and paint texture on glossy plastic split what should be one
part (an arm, a leg) into several disjoint, slightly-different-colored
blobs; the color+proximity merge introduced in this release closes most but
not all of that gap. Closing it further with classical CV alone has
diminishing returns — this is exactly the kind of case the original
research notes pointed at zero-shot ML segmentation (SAM 2, or ideally
Grounding DINO for open-vocabulary prompting) for, not more contour-merging
heuristics.

## What could NOT be verified in this sandbox (be skeptical of anything
## claiming otherwise without re-testing)

This work was done in an offline sandbox with **no network access**, no
`torch`/`ultralytics`/`easyocr`/`pydantic`/`typer`/`rich`/`fastapi`
installed, and no Godot binary present. Concretely, this means:

- **The full pydantic-based orchestrator pipeline was never executed
  end-to-end in this environment.** All verification above ran the
  segmentation core directly against raw images. The plugin adapter
  (`opencv_segmenter.py`) that wraps the core's output into
  `DetectedElement`/`SegmentationResult` pydantic models was updated and
  passes a static syntax check, but **was not run**, since `pydantic` isn't
  installed here. Run `pytest tests/` yourself after `pip install -e .` to
  confirm the wiring is correct end-to-end.
- **YOLO, SAM 2, and EasyOCR integrations were not touched and were not
  tested**, online or offline. Whatever was true of their correctness
  before this changelog remains unverified now. In particular, the prior
  audit's concern that COCO-pretrained `yolov8n-seg` has no classes
  relevant to game/UI/product assets stands and was not addressed here.
- **The new Godot-headless validation hook
  (`gvi/plugins/validators/_godot_headless.py`) could only be verified on
  its "Godot not installed" fallback path**, which behaves correctly and
  reports `available: False` with an explicit reason instead of silently
  pretending validation passed. The actual "Godot accepts/rejects this
  scene" path requires a real Godot 4 binary and was never exercised here.
  Do not treat this feature as proven until it has been run against a real
  Godot install.
- **AGPL licensing risk from `ultralytics` (used for YOLO and the SAM 2
  extra) was not resolved**, only documented (see `NOTICE_LICENSING.md`).
  No code changes were made to remove or replace that dependency.

## Bonus: wider end-to-end test coverage for free

The 4 real images used to verify this fix were copied into `test_images/`
(`map_render.png`, `robot_photo.png`, `monster_can.png`,
`trashpick_logo.png`). `tests/test_pipeline.py` globs everything in that
folder, so the existing 3-images × 7-targets = 21 end-to-end smoke test
matrix automatically becomes 7 × 7 = 49 once dependencies are installed and
that suite is run — more real-world coverage, no test code changes needed.
This was not run in the audit sandbox (no pydantic), so treat "49 passing"
as a prediction to confirm, not a verified fact.

---

# v1.1.1 patch — closing the gaps the v1.1.0 self-review explicitly flagged

The v1.1.0 changelog above ended with a list of "what could NOT be verified
in this sandbox." This patch went back and closed every item on that list
that was actually possible to close offline, and is honest about the one
category that genuinely isn't (YOLO/SAM2/EasyOCR model execution, which
needs network + GPU + model weights this sandbox does not have).

## 1. The real orchestrator pipeline now actually runs end-to-end here

Built a small, explicitly-labeled **test-only** pydantic-compatible shim
(`offline_test_shims/pydantic/`, not shipped in the package, not a
replacement for real pydantic) implementing just enough of `BaseModel` /
`Field` / `field_validator` / `model_dump` for `gvi.core.types` and
`gvi.core.plugin` to import and run unmodified. This let the real
`Orchestrator().convert(...)` run against real images for the first time
in this audit, and it found two real bugs that static syntax-checking could
never have caught:

- `IRNode.model_rebuild()` is called at import time for the self-referential
  `children: list["IRNode"]` field; the shim didn't have that method,
  so the entire package failed to import. Fixed by adding a no-op
  `model_rebuild()` (the shim does no forward-ref resolution to begin with,
  so there's nothing to rebuild).
- The Godot-headless validation hook added in v1.1.0 was only wired into
  `SceneValidator.validate_scene_file()` (the standalone `gvi validate` CLI
  path). The **live conversion pipeline** calls `SceneValidator.run()`,
  which has its own `validate_segmentation()` code path that never called
  the headless hook at all -- so in practice, every real `gvi convert` was
  silently skipping the new validation feature entirely. Fixed by invoking
  it from `run()` too, against the `scene.tscn` the exporter step just
  wrote.

All 7 test images (3 synthetic + 4 real) now convert end-to-end with zero
errors through the real `Orchestrator`, confirmed by actually running it,
not by reading the code. `validation.json` for each now contains a populated
`godot_headless` field with an honest "not available" reason instead of a
silent `null`.

**Still not verified:** this shim is deliberately lenient (no real type
coercion/validation). Run `pytest tests/` after `pip install pydantic` to
get the real guarantees back; treat the shim run as "the wiring isn't
obviously broken," not "pydantic's validation behavior is confirmed."

## 2. YOLO semantic detection: disabled by default, not silently broken

Didn't (and can't, offline) fix the "COCO classes are irrelevant to game
assets" problem at the model level. Instead, `semantic_detection` now
defaults to `False` in `ConversionOptions`, with the reasoning written
directly in the field's docstring/comment. Verified by re-running the
orchestrator and confirming `segmenter.yolo` no longer appears in
`pipeline_plan.json` by default, and the "ultralytics missing" warning is
gone for users who never asked for semantic detection in the first place.
It remains available as an explicit opt-in for inputs that genuinely
contain real-world COCO-class objects.

## 3. Godot-headless validation: the actual subprocess/parsing logic is now tested

Still no real Godot binary in this sandbox. But a fake `godot4` executable
(a two-line bash script, generated on the fly in the test, not shipped as a
binary) now exercises the parts of `_godot_headless.py` that were
previously completely untested: a clean exit (`ok: True`), a simulated
parse-error exit (`ok: False`, with the actual error lines captured), and
the original "binary not found" fallback. Three real, passing tests now
cover this instead of one.

**Still not verified:** none of this proves real Godot would actually
accept any specific generated `.tscn`. It proves the code that *would* ask
Godot, and parse its answer, behaves correctly when given Godot-shaped
output. Run it against a real Godot 4 install before trusting "ok: True"
for an actual scene.

## 4. The robot photo: 30 elements down to 9, via two real fixes (not just retuning)

Two actual bugs were found and fixed by debugging this specific case, not
by turning knobs blindly:

- **Specular-highlight suppression was added** (`suppress_specular_highlights`)
  but had ~zero effect on the robot photo specifically -- its glossy look
  turned out to come from continuous gold-paint shading (max V channel
  value 221 across the whole foreground), not blown-out white highlights
  (which the original heuristic, V>232, was tuned for). The highlight
  heuristic still helps other glossy-highlight cases; it just wasn't the
  right tool for this one.
- **The actual fix** was increasing the morphological closing kernel used
  on each k-means color cluster before contour extraction (`close_kernel`,
  new parameter on `_segment_color_regions`), specifically for `medium`-
  complexity scenes. A smoothly shaded curved surface crosses a k-means
  color-bin boundary many times across its area, and each crossing was
  previously becoming its own tiny contour. Bridging those gaps with a
  21x21 closing kernel (vs. the default 3x3) collapsed 55 raw color
  fragments on the robot down to ~13 before any merging even happens.
- **A real bug was found in the merge step while debugging the Monster
  can regression this caused**: `merge_fragments` compared mean color over
  each candidate's full bounding *box*, not its actual contour mask. A
  sparse/elongated shape (e.g. the can's "M" claw-mark logo) has a bbox
  that's mostly background, so its bbox-average color came out close to
  the can body's color purely by dilution -- causing the merge step to
  wrongly fuse a distinct logo mark into the can body. Fixed with
  `_mean_color_masked`, which only averages pixels inside the actual
  contour. Also added a size-ratio gate to the final containment-dedupe
  check, so a small nested detail (the logo) isn't swallowed by a much
  bigger same-type parent (the can body) the way the v1.1.0 "fix" for the
  map's redundant panels was accidentally doing here.

Net result, measured by actually re-running the full 7-image suite after
each change: robot photo 157 (v1.0.0) -> 30 (v1.1.0) -> **9** (this patch);
Monster can 120 -> 4 -> back up to **8** with the M-mark logo correctly
preserved as its own element again (it had collapsed to 1 mid-patch before
the masked-color fix).

## 5. PyMuPDF AGPL notice is now a runtime warning, not just a markdown file

`parser.pdf` now emits an explicit warning (surfaced through the normal
`PipelineStepResult.warnings` -> `ConversionResult.warnings` path every
caller already reads) every time it actually runs, pointing at
`NOTICE_LICENSING.md`. Not just documentation someone has to go looking
for.

## Updated regression gate

`robot_photo`'s ceiling in `tests/test_segmentation_quality_gate.py` was
tightened from 40 (set generously around the old 30-element result) to 20,
now that the real number is 9 -- so a future regression back toward 30+
would fail loudly again instead of hiding under a ceiling sized for the bug
it was supposed to catch.
