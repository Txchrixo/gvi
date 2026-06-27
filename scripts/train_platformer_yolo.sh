#!/usr/bin/env bash
set -euo pipefail
DATASET=${1:-./dataset}
MODEL=${2:-yolo11n-seg.pt}
EPOCHS=${3:-80}
IMGSZ=${4:-640}
BATCH=${5:-8}

gvi training train "$DATASET"   --model "$MODEL"   --epochs "$EPOCHS"   --imgsz "$IMGSZ"   --batch "$BATCH"
