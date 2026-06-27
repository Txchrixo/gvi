# Teacher Backends

GVI supports four teacher styles.

## 1. heuristic

Works offline, no GPU, no heavy models. It uses contours and geometry to create first labels. It is useful only for bootstrapping and testing the workflow.

```bash
gvi training autolabel ./dataset/raw --dataset ./dataset --backend heuristic
```

## 2. yolo

Uses an existing YOLO segmentation model to produce GVI JSON + YOLO labels.

```bash
gvi training autolabel ./dataset/raw --dataset ./dataset --backend yolo --model ./dataset/models/best.pt
```

## 3. huggingface  (recommended high-quality teacher)

Grounding DINO + SAM 2, run locally via `transformers`. This is the best
zero-shot teacher: it detects arbitrary classes from your taxonomy's text
prompts and produces precise masks. See **docs/HF_TEACHER.md** for full details.

```bash
python -m pip install -e ".[hf]"
gvi training autolabel ./dataset/raw --dataset ./dataset --backend huggingface \
    --classes platform --classes ladder --classes spike --conf 0.30
```

Aliases: `huggingface`, `hf`, `grounding-dino`, `dino`.

## 4. grounded-sam endpoint

Kept for users who already run a Grounded-SAM2 server and prefer to call it
over HTTP instead of running models in-process. For most people, backend #3
(`huggingface`) is simpler — it needs no separate server.

Expected API:

```http
POST /label
multipart image=<file>
form prompts=["platform", "ladder", "spike"]
```

Response:

```json
{
  "objects": [
    {"class_name": "ladder", "bbox_xywh": [1,2,30,90], "polygon": [[1,2],[31,2],[31,92],[1,92]], "confidence": 0.88}
  ]
}
```
