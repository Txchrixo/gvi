# GVI Training System — Ready to Train

This version adds a complete Machine Learning loop to GVI:

```text
Raw visual input
↓
Teacher auto-labeling: heuristic / YOLO / Grounded-SAM endpoint
↓
Human correction: CVAT / Roboflow / Label Studio
↓
Student training: YOLO segmentation
↓
Active learning: low-confidence review
↓
GVI inference → Scene IR → Godot .tscn
```

## 1. Install

Base dev install:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e ".[dev,training]"
```

CPU is enough to run dataset commands and small tests. For real YOLO training, a GPU is recommended.

## 2. Create your first dataset

```bash
gvi dataset init --type platformer ./dataset
gvi training synthetic --dataset ./dataset --count 80 --split train
gvi training synthetic --dataset ./dataset --count 20 --split val
gvi dataset stats ./dataset
```

This creates a tiny synthetic platformer dataset so the whole training path is testable immediately.

## 3. Add real screenshots/images

Put your own images here:

```text
dataset/raw/
```

or ingest a folder:

```bash
gvi dataset ingest ./my_raw_images --dataset ./dataset
```

Accepted raster formats: PNG, JPG/JPEG, WebP, BMP. PSD/SVG/PDF can be rendered/extracted by the existing GVI pipeline first, then their exported rasters can enter the training dataset.

## 4. Auto-label

Fast offline bootstrap:

```bash
gvi training autolabel ./dataset/raw   --dataset ./dataset   --backend heuristic   --classes platform --classes ladder --classes spike --classes door --classes enemy --classes pickup
```

YOLO teacher, once you already have a model:

```bash
gvi training autolabel ./dataset/raw   --dataset ./dataset   --backend yolo   --model ./dataset/models/best.pt
```

Grounded-SAM/SAM2 teacher is integrated as an endpoint adapter to avoid vendoring huge model repos. Run a local teacher service, then:

```bash
gvi training autolabel ./dataset/raw   --dataset ./dataset   --backend grounded-sam   --teacher-endpoint http://127.0.0.1:7860/label
```

Expected endpoint response:

```json
{
  "objects": [
    {
      "class_name": "platform",
      "bbox_xywh": [10, 20, 140, 30],
      "polygon": [[10,20],[150,20],[150,50],[10,50]],
      "confidence": 0.86
    }
  ]
}
```

## 5. Human correction

Use one of these tools:

- CVAT: best open source option for segmentation review.
- Roboflow: easiest hosted workflow.
- Label Studio: flexible annotation/review flows.

GVI creates:

```text
dataset/overlays/*_overlay.jpg
dataset/review/review.json
```

Run:

```bash
gvi training review ./dataset --cvat
```

Then open `dataset/review/CVAT_QUICKSTART.md`.

## 6. Train YOLO segmentation

Dry run first:

```bash
gvi training train ./dataset --model yolo11n-seg.pt --epochs 80 --imgsz 640 --batch 8 --dry-run
```

Then train:

```bash
gvi training train ./dataset --model yolo11n-seg.pt --epochs 80 --imgsz 640 --batch 8
```

Recommended progression:

```text
50 images   → smoke/prototype
200 images  → first useful model
500 images  → solid demo
1000+ images → serious domain model
3000+ images → robust multi-style model
```

## 7. Evaluate and predict

```bash
gvi training evaluate ./dataset

gvi training predict ./test_images/level.png   --model ./dataset/runs/gvi_platformer_yolo/weights/best.pt   --out ./dataset/predictions
```

## 8. Best-practice active learning loop

```text
1. Collect 100 new images.
2. Auto-label with teacher/student.
3. Review low-confidence objects only.
4. Export corrected labels.
5. Retrain.
6. Compare metrics and Godot reconstruction fidelity.
```

The point is not to make the model perfect at once. The point is to build a repeatable ML loop with dataset versioning, review files, overlays, training configs and metrics.

## 9. What makes this AI-engineer ready

This project now demonstrates:

- dataset design and taxonomy management
- weak supervision / teacher-student labeling
- segmentation training with YOLO
- active learning
- model inference integration
- rule-based post-processing
- domain-specific semantic mapping to Godot nodes
- reproducible CLI workflows
- overlays, review files and dataset cards
