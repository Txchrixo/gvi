# Hugging Face teacher backend (Grounding DINO + SAM 2)

This backend closes the long-standing "grounded-sam" gap. Earlier versions only
had a placeholder that called an external HTTP server. This one runs the teacher
models **locally** through Hugging Face `transformers`, which is the workflow the
training docs recommend: a powerful zero-shot teacher pre-labels your images, you
correct the labels, then a small fast YOLO student learns your domain.

## What it does

```
your image  ──►  Grounding DINO  ──►  boxes from text prompts
                  (open-vocab)        ("ladder", "spike", "orange enemy"...)
                       │
                       ▼
                    SAM 2  ──►  precise masks  ──►  polygons (YOLO-seg labels)
                       │
                       ▼
            GVI AnnotationObject  (class / bbox / polygon / confidence /
                                   godot_candidates / needs_review)
```

- **Grounding DINO** turns the taxonomy's text prompts into bounding boxes. This
  is what lets the teacher find arbitrary game/UI classes that COCO-pretrained
  models (like plain YOLO) have never seen.
- **SAM 2** refines each box into a precise mask, which becomes the polygon that
  YOLO-segmentation training actually needs. If SAM 2 isn't available, the box
  is used as a 4-point polygon so training can still begin.
- Every produced label is run through the same `score_object` business rules as
  the other backends (sets Godot node candidates, flags `needs_review`, etc.).

## Install

```bash
python -m pip install -e ".[hf]"      # just the HF teacher
# or
python -m pip install -e ".[training]" # YOLO student + HF teacher together
```

`hf` pulls in `transformers`, `torch`, `torchvision`, `timm`, `accelerate`.
First run downloads the model weights (a few GB) from the Hugging Face Hub.

A GPU is strongly recommended for the teacher. On CPU it works but is slow —
fine for tens of images to bootstrap, painful for hundreds. No GPU? Run the
autolabel step on Google Colab/Kaggle, commit the resulting labels, and train
the student wherever you like.

## Use

```bash
gvi training autolabel ./dataset/raw \
    --dataset ./dataset \
    --backend huggingface \
    --classes platform --classes ladder --classes spike \
    --classes door --classes enemy --classes pickup \
    --conf 0.30
```

Aliases for `--backend`: `huggingface`, `hf`, `grounding-dino`, `dino`.

Then review and train as usual:

```bash
gvi training review ./dataset --cvat
gvi training train ./dataset --model yolo11n-seg.pt --epochs 80
```

## Tuning

| Option | Meaning | Default |
|---|---|---|
| `--conf` | Grounding DINO box threshold (higher = fewer, surer boxes) | 0.30 |
| `--model` | override the DINO checkpoint | `IDEA-Research/grounding-dino-base` |

The constructor (`HuggingFaceTeacherBackend`) also exposes `text_threshold`,
`sam2_model`, `use_sam2`, `device`, and `review_below` for programmatic use.

## Prompts come from your taxonomy

The quality of zero-shot detection depends heavily on the prompt phrases. They
live in `gvi/training/configs/platformer_classes.yaml` under each class's
`prompts:` list. Improving those phrases ("vertical climbable ladder", "spiked
floor trap") improves teacher recall directly — edit the taxonomy, not the code.

## Graceful degradation (by design)

- transformers/torch missing → clear `RuntimeError` telling you what to install.
- SAM 2 unavailable but Grounding DINO works → boxes still produced, polygons
  fall back to the rectangle. Training can proceed.
- A single image failing → does not crash the whole run.

## What was verified, and what wasn't

Verified offline (no transformers/GPU needed): backend registration + lazy
imports, clean failure when deps are missing, prompt construction from the
taxonomy, phrase→class mapping, mask→polygon conversion, and the full
`label_image` wiring (parsing detector output → `AnnotationObject` →
`score_object` → YOLO label line) using injected fake DINO/SAM2 engines.

NOT verified here (needs the real models + GPU + network, so confirm on your
machine): the actual Grounding DINO and SAM 2 inference quality on real images.
Run the `gvi training autolabel --backend huggingface` command on a handful of
your own screenshots and inspect `dataset/overlays/*.jpg` to judge real recall.
