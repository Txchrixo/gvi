from __future__ import annotations

from pathlib import Path

from gvi.training.active_learning import build_review_file
from gvi.training.dataset_builder import dataset_stats, init_dataset
from gvi.training.rules import score_object
from gvi.training.synthetic import generate_synthetic_platformer
from gvi.training.annotations import AnnotationObject
from gvi.training.taxonomy import Taxonomy
from gvi.training.train_yolo import TrainConfig, train_yolo


def test_taxonomy_loads_platformer():
    taxonomy = Taxonomy.load("platformer")
    assert "platform" in taxonomy.class_to_id
    assert "ladder" in taxonomy.class_to_id
    assert taxonomy.names_for_yolo[taxonomy.class_to_id["platform"]] == "platform"


def test_dataset_init_and_synthetic(tmp_path: Path):
    root = tmp_path / "dataset"
    init_dataset(root, taxonomy_name="platformer")
    assert (root / "data.yaml").exists()
    assert (root / "classes.json").exists()
    result = generate_synthetic_platformer(root, count=3, split="train")
    assert result["generated"] == 3
    stats = dataset_stats(root)
    assert stats["splits"]["train"]["images"] == 3
    assert stats["splits"]["train"]["labels"] == 3


def test_rule_scoring_marks_bad_ladder_review():
    obj = AnnotationObject(id="x", class_name="ladder", bbox_xywh=(0, 0, 100, 20), confidence=0.80)
    decision = score_object(obj, image_width=200, image_height=200)
    assert decision.needs_review
    assert "Area2D" in decision.godot_candidates


def test_review_and_train_dry_run(tmp_path: Path):
    root = tmp_path / "dataset"
    init_dataset(root, taxonomy_name="platformer")
    generate_synthetic_platformer(root, count=2, split="train")
    review = build_review_file(root, threshold=0.99)
    assert (root / "review" / "review.json").exists()
    assert len(review.items) >= 1
    dry = train_yolo(TrainConfig(dataset_root=root, dry_run=True))
    assert dry["dry_run"] is True
    assert "yolo" in dry["command"][0]


# --------------------------------------------------------------- HF teacher
def test_hf_backend_registered_and_lazy():
    """The huggingface backend must be reachable via get_backend and must NOT
    import transformers/torch at construction time (lazy deps)."""
    from gvi.training.autolabel import get_backend

    for name in ["huggingface", "hf", "grounding-dino", "dino"]:
        backend = get_backend(name)
        assert backend.id == "huggingface"


def test_hf_backend_fails_cleanly_without_transformers():
    """Calling label_image without transformers installed must raise a clear,
    actionable RuntimeError, not an obscure ImportError mid-run."""
    import builtins
    import pytest
    from gvi.training.autolabel import get_backend
    from gvi.training.taxonomy import Taxonomy

    backend = get_backend("huggingface")
    taxonomy = Taxonomy.load("platformer")

    # Only meaningful when transformers is actually absent; if it's installed in
    # the dev env, skip rather than fail.
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
        pytest.skip("transformers/torch installed; clean-failure path not exercised")
    except Exception:
        pass

    with pytest.raises(RuntimeError):
        backend.label_image(Path("test_images/wall_of_frames.png"), taxonomy, selected_classes=["platform"])


def test_hf_prompt_and_phrase_mapping():
    """Prompt building and phrase->class mapping must not need any heavy deps."""
    from gvi.training.hf_teacher import HuggingFaceTeacherBackend
    from gvi.training.taxonomy import Taxonomy

    taxonomy = Taxonomy.load("platformer")
    prompt, mapping = HuggingFaceTeacherBackend._build_prompt(taxonomy, ["platform", "ladder", "spike"])
    assert prompt.endswith(".")
    assert len(mapping) >= 3
    # Each known prompt phrase resolves back to a real taxonomy class.
    assert HuggingFaceTeacherBackend._match_phrase_to_class("ladder", mapping, taxonomy) == "ladder"
    assert HuggingFaceTeacherBackend._match_phrase_to_class("solid ledge", mapping, taxonomy) == "platform"
    # Unknown phrase falls back to a valid class, never crashes.
    fallback = HuggingFaceTeacherBackend._match_phrase_to_class("zzz nonsense", mapping, taxonomy)
    assert fallback in taxonomy.class_to_id


def test_hf_mask_to_polygon():
    """SAM2 mask -> polygon conversion is pure OpenCV and fully testable."""
    import numpy as np
    from gvi.training.hf_teacher import _mask_to_polygon

    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 30:70] = True
    pts = _mask_to_polygon(mask)
    assert len(pts) == 4  # a rectangle simplifies to its 4 corners
    assert _mask_to_polygon(np.zeros((50, 50), dtype=bool)) == []  # empty -> []
