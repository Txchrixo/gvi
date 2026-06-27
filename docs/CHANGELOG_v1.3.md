# GVI v1.3 — Hugging Face teacher backend (Grounding DINO + SAM 2)

Closes the long-standing "grounded-sam" gap. Earlier versions only had a
placeholder backend that POSTed to an external HTTP server; there was no way to
run a real zero-shot teacher locally. v1.3 adds `gvi/training/hf_teacher.py`, a
`HuggingFaceTeacherBackend` that runs **Grounding DINO + SAM 2** locally via
`transformers`.

## What's new
- `gvi/training/hf_teacher.py` — new teacher backend.
  - Grounding DINO: taxonomy text prompts -> bounding boxes (open-vocabulary,
    so it finds game/UI classes COCO-pretrained models never learned).
  - SAM 2: each box -> precise mask -> polygon (the YOLO-seg labels need this).
  - Degrades gracefully: no SAM 2 -> box used as 4-point polygon; missing
    transformers/torch -> clear actionable RuntimeError; one bad image -> does
    not crash the batch.
  - Every label passes through `score_object` for Godot node candidates +
    needs_review, identical to the yolo/heuristic backends.
- Registered in `get_backend()` under aliases: `huggingface`, `hf`,
  `grounding-dino`, `dino`. Heavy deps imported lazily (verified: importing the
  whole `gvi.training` package pulls in neither transformers nor torch).
- New `[hf]` extra (`transformers`, `torch`, `torchvision`, `timm`,
  `accelerate`); `[training]` now includes it so the teacher works out of the box.
- CLI: `gvi training autolabel --backend huggingface ...` (help text updated).
- Docs: `docs/HF_TEACHER.md`, updated `docs/TEACHER_BACKENDS.md`, and
  `docs/CLAUDE_CODE_HANDOFF_HF.md` (a ready-to-paste prompt that gives Claude
  Code the full context to install, run real inference, and finish verification
  on a machine with GPU/network).
- 4 new packaged tests in `tests/test_training_system.py` (8 total, all green
  offline).

## Verified offline (no transformers/GPU; injected fake DINO/SAM2 engines)
- Backend registration + all 4 aliases.
- Lazy imports (no heavy deps loaded at import time).
- Clean RuntimeError when transformers/torch absent; no corrupted labels written
  on failure.
- Prompt construction from the taxonomy; phrase->canonical-class mapping
  (incl. unknown-phrase fallback).
- SAM2 mask -> polygon conversion (rectangle -> 4 corners; empty -> []; circle
  -> simplified polygon).
- Full `label_image` wiring with a fake DINO: detector output -> AnnotationObject
  -> score_object (low-confidence box correctly flagged needs_review; godot
  candidates set) -> valid YOLO-seg line. And the DINO+SAM2 path producing a
  real polygon and `source="huggingface:dino+sam2"`.
- Full test suite still green; all other backends (heuristic/yolo/grounded-sam)
  unaffected.

## NOT verified (needs real weights + GPU + network — do this on your machine)
- Actual Grounding DINO and SAM 2 inference quality on real images.
- transformers API drift: the exact auto-class names
  (`AutoModelForZeroShotObjectDetection`), the SAM2 access path
  (`Sam2Model`/`Sam2Processor` vs the standalone `sam2` package), and the
  `post_process_grounded_object_detection` keyword names can vary by transformers
  version. `docs/CLAUDE_CODE_HANDOFF_HF.md` lists exactly what to adapt if a
  TypeError/ImportError shows up at runtime.

Run `gvi training autolabel --backend huggingface` on a few real screenshots and
inspect `dataset/overlays/*.jpg` to judge real recall. The code paths around the
models are proven; the model calls themselves are the last mile to confirm.

## Note on scope
Deliberately did NOT add Hugging Face `datasets` or Hub integration. At the
current dataset scale (hundreds of images in folders) they add dependencies
without solving a real problem. Add them when the need is real, not for a CV
line. Ultralytics remains the student trainer; the HF backend is the teacher.
The two are complementary, not competitors — and because the dataset stays in
standard YOLO/COCO format, switching the student trainer later costs a label
conversion, not a rewrite.
