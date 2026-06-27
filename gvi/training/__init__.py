"""Training and active-learning utilities for GVI.

This package intentionally keeps heavyweight ML dependencies optional. The CLI
can initialize datasets, build configs, create review files and generate simple
heuristic auto-labels without GPU packages. YOLO/teacher-model backends are
loaded lazily only when a user asks for them.
"""
from __future__ import annotations

__all__ = [
    "AnnotationObject",
    "AnnotationFile",
    "Taxonomy",
    "DatasetConfig",
]

from gvi.training.annotations import AnnotationFile, AnnotationObject
from gvi.training.taxonomy import DatasetConfig, Taxonomy
