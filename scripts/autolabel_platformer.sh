#!/usr/bin/env bash
set -euo pipefail
RAW=${1:-./dataset/raw}
DATASET=${2:-./dataset}
BACKEND=${3:-heuristic}

gvi training autolabel "$RAW"   --dataset "$DATASET"   --backend "$BACKEND"   --classes platform   --classes ladder   --classes spike   --classes door   --classes enemy   --classes pickup   --classes wall   --classes background
