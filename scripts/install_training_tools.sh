#!/usr/bin/env bash
set -euo pipefail
python -m pip install -U pip
python -m pip install -e ".[dev,training]"
cat <<'EOF'

Training tools installed.

Optional external tools:
- CVAT with Docker for segmentation review
- Roboflow or Label Studio for hosted/manual labeling
- NVIDIA CUDA drivers if you want fast local YOLO training

Next:
  gvi dataset init --type platformer ./dataset
  gvi training synthetic --dataset ./dataset --count 80 --split train
  gvi training train ./dataset --dry-run
EOF
