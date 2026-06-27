"""GVI 1.0.0 CLI — convert any image into any Godot component."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from gvi.core.orchestrator import Orchestrator
from gvi.core.types import ConversionOptions, ConversionRequest, TargetKind

app = typer.Typer(help="Godot Vision Orchestrator CLI", no_args_is_help=True, add_completion=False)
console = Console()


def _print_json(data: Any) -> None:
    console.print_json(json.dumps(data, indent=2, default=str))


# ---------------------------------------------------------------- inspect
@app.command()
def inspect(
    input_path: Path = typer.Argument(..., help="Path to input asset"),
    json_out: bool = typer.Option(False, "--json", help="Print JSON only"),
) -> None:
    """Inspect an asset and print its detected profile."""
    profile = Orchestrator().inspect(input_path)
    if json_out:
        _print_json(profile.model_dump(mode="json"))
        return
    table = Table(title="Asset Profile", show_header=True)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    rows = {
        "Path": str(profile.path),
        "Kind": profile.kind.value,
        "Format": profile.extension or "unknown",
        "MIME": profile.mime_type or "unknown",
        "Dimensions": f"{profile.width or '?'} x {profile.height or '?'}",
        "Has Alpha": str(profile.has_alpha),
        "Has Text": str(profile.has_text),
        "Has UI Layout": str(profile.has_ui_layout),
        "Has Pixel Art": str(profile.has_pixel_art),
        "Is Spritesheet": str(profile.is_spritesheet),
        "Semantic Hint": str(profile.semantic_hint),
        "Scene Kind": str(profile.scene_kind),
        "Quality Score": f"{profile.quality_score:.3f}" if profile.quality_score is not None else "N/A",
    }
    for k, v in rows.items():
        table.add_row(k, v)
    for key, value in profile.metadata.items():
        if isinstance(value, (str, int, float, bool)):
            table.add_row(f"meta.{key}", str(value))
    for note in profile.notes:
        table.add_row("note", note)
    console.print(table)


# ---------------------------------------------------------------- convert
@app.command()
def convert(
    input_path: Path = typer.Argument(..., help="Path to input asset"),
    target: str = typer.Option("godot.node2d", "--target", help="Target: godot.node2d/control/sprite2d/tilemap/richtext/theme/animation"),
    out: Path = typer.Option(Path("outputs/godot"), "--out", help="Output directory"),
    preset: str = typer.Option("balanced", "--preset", help="Pipeline preset: fast/balanced/fidelity/semantic/tilemap"),
    include_background: bool = typer.Option(True, "--include-background/--no-background", help="Add locked source-preservation layer"),
    background_mode: str = typer.Option("source", "--background-mode", help="source/solid/transparent/none"),
    max_elements: int = typer.Option(500, "--max-elements", help="Maximum editable elements"),
    min_area_ratio: float = typer.Option(0.0001, "--min-area-ratio", help="Minimum contour area / image area"),
    debug_overlay: bool = typer.Option(False, "--debug-overlay", help="Write segmentation overlay image"),
    sam2: bool = typer.Option(False, "--sam2", help="Use SAM 2 (downloads weights on first run)"),
    sam2_model: str = typer.Option("sam2_hiera_small", "--sam2-model", help="sam2_hiera_tiny/small/base_plus/large"),
    semantic_detection: bool = typer.Option(False, "--semantic/--no-semantic", help="Use YOLO for semantic class labels (COCO-pretrained; off by default, see docs)"),
    ocr: bool = typer.Option(True, "--ocr/--no-ocr", help="Run EasyOCR for editable text labels"),
    tilemap_tile_size: int = typer.Option(32, "--tile-size", help="Tile size for godot.tilemap"),
    fail_fast: bool = typer.Option(False, "--fail-fast"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Convert an input asset to any Godot 4 component."""
    try:
        target_kind = TargetKind(target)
    except ValueError:
        console.print(f"[red]Invalid target: {target}. Use one of: {[t.value for t in TargetKind]}[/red]")
        raise typer.Exit(1)
    try:
        options = ConversionOptions(
            include_background=include_background,
            background_mode=background_mode,  # type: ignore[arg-type]
            max_elements=max_elements,
            min_area_ratio=min_area_ratio,
            make_debug_overlay=debug_overlay,
            sam2_enabled=sam2,
            sam2_model=sam2_model,  # type: ignore[arg-type]
            semantic_detection=semantic_detection,
            include_text_elements=ocr,
            tilemap_tile_size=tilemap_tile_size,
            fail_fast=fail_fast,
        )
    except Exception as exc:
        console.print(f"[red]Invalid options: {exc}[/red]")
        raise typer.Exit(1)

    request = ConversionRequest(
        input_path=input_path,
        target=target_kind,
        output_dir=out,
        preset=preset,  # type: ignore[arg-type]
        options=options,
    )
    with console.status("[bold green]Converting asset..."):
        result = Orchestrator().convert(request)
    if json_out:
        data = result.model_dump(mode="json")
        data["ok"] = result.ok
        _print_json(data)
        return

    table = Table(title="Conversion Result", show_header=True)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("OK", str(result.ok))
    table.add_row("Pipeline", result.pipeline_id)
    table.add_row("Output Dir", str(result.output_dir))
    table.add_row("Warnings", str(len(result.warnings)))
    table.add_row("Errors", str(len(result.errors)))
    for key, path in result.artifacts.items():
        table.add_row(f"artifact.{key}", str(path))
    for key, value in result.metrics.items():
        table.add_row(f"metric.{key}", f"{value:.4f}")
    for warning in result.warnings[:10]:
        table.add_row("warning", f"[yellow]{warning}[/yellow]")
    for error in result.errors[:10]:
        table.add_row("error", f"[red]{error}[/red]")
    console.print(table)
    scene_path = result.artifacts.get("scene")
    if scene_path:
        console.print(f"\n[bold green]Scene ready:[/bold green] {scene_path}")
    tileset_path = result.artifacts.get("tileset")
    if tileset_path:
        console.print(f"[bold green]TileSet ready:[/bold green] {tileset_path}")


# ---------------------------------------------------------------- validate
@app.command()
def validate(
    scene_path: Path = typer.Argument(..., help="Path to .tscn file"),
    source_image: Path | None = typer.Option(None, "--source", help="Source image for fidelity comparison"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Validate a generated Godot scene using manifest.json and optional source image."""
    from gvi.plugins.validators.scene_validator import SceneValidator
    result = SceneValidator().validate_scene_file(scene_path, source_image)
    if json_out:
        _print_json(result.model_dump(mode="json"))
        return
    table = Table(title="Validation Result", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Valid", str(result.is_valid))
    table.add_row("SSIM", f"{result.ssim_score:.4f}" if result.ssim_score is not None else "N/A")
    table.add_row("PSNR", f"{result.psnr_score:.2f}" if result.psnr_score is not None else "N/A")
    table.add_row("IoU", f"{result.iou_score:.4f}" if result.iou_score is not None else "N/A")
    table.add_row("Warnings", str(len(result.warnings)))
    table.add_row("Errors", str(len(result.errors)))
    for w in result.warnings[:8]:
        table.add_row("warning", f"[yellow]{w}[/yellow]")
    for e in result.errors[:8]:
        table.add_row("error", f"[red]{e}[/red]")
    console.print(table)


# ---------------------------------------------------------------- serve
@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8080, "--port"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Run the FastAPI server (requires `gvi[api]`)."""
    try:
        import uvicorn
    except ImportError:
        console.print("[red]FastAPI server deps missing. Install with: python -m pip install 'gvi[api]'[/red]")
        raise typer.Exit(1)
    console.print(f"[green]Starting GVI API server on {host}:{port}[/green]")
    uvicorn.run("gvi.api_server.app:app", host=host, port=port, reload=reload)


# ---------------------------------------------------------------- mcp
@app.command("mcp")
def mcp_server() -> None:
    """Run the stdio agent/MCP-compatible tool server."""
    from gvi.api_server.mcp_stdio import main
    main()



# ---------------------------------------------------------------- training / dataset
training_app = typer.Typer(help="Training, auto-labeling and active-learning commands", no_args_is_help=True)
dataset_app = typer.Typer(help="Dataset management commands", no_args_is_help=True)
app.add_typer(dataset_app, name="dataset")
app.add_typer(training_app, name="training")


@dataset_app.command("init")
def dataset_init(
    dataset_root: Path = typer.Argument(..., help="Dataset output directory"),
    type: str = typer.Option("platformer", "--type", help="Taxonomy: platformer/ui/tilemap or path to YAML"),
    force: bool = typer.Option(False, "--force", help="Overwrite README/DATASET_CARD templates"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Create a training dataset scaffold ready for YOLO segmentation."""
    from gvi.training.dataset_builder import init_dataset

    artifacts = init_dataset(dataset_root, taxonomy_name=type, force=force)
    if json_out:
        _print_json({k: str(v) for k, v in artifacts.items()})
        return
    table = Table(title="GVI Dataset Initialized", show_header=True)
    table.add_column("Artifact", style="cyan")
    table.add_column("Path", style="green")
    for k, v in artifacts.items():
        table.add_row(k, str(v))
    console.print(table)


@dataset_app.command("ingest")
def dataset_ingest(
    source: Path = typer.Argument(..., help="Image file or folder to copy"),
    dataset: Path = typer.Option(Path("dataset"), "--dataset", help="Dataset root"),
    split: bool = typer.Option(False, "--split", help="Copy directly into train/val/test instead of raw/"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Copy raw images into a GVI dataset."""
    from gvi.training.dataset_builder import ingest_raw_images

    copied = ingest_raw_images(dataset, source, split=split)
    if json_out:
        _print_json({"copied": [str(p) for p in copied], "count": len(copied)})
    else:
        console.print(f"[green]Copied {len(copied)} image(s) into {dataset}[/green]")


@dataset_app.command("stats")
def dataset_stats_cmd(
    dataset: Path = typer.Argument(..., help="Dataset root"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Show dataset health and class counts."""
    from gvi.training.dataset_builder import dataset_stats

    stats = dataset_stats(dataset)
    if json_out:
        _print_json(stats)
        return
    table = Table(title="Dataset Stats", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Root", str(stats.get("dataset_root")))
    table.add_row("Raw images", str(stats.get("raw_images")))
    for split, values in (stats.get("splits") or {}).items():
        table.add_row(f"{split}.images", str(values.get("images")))
        table.add_row(f"{split}.labels", str(values.get("labels")))
    for cls, count in (stats.get("classes") or {}).items():
        table.add_row(f"class.{cls}", str(count))
    console.print(table)


@training_app.command("autolabel")
def training_autolabel(
    source: Path = typer.Argument(..., help="Raw image file/folder"),
    dataset: Path = typer.Option(Path("dataset"), "--dataset", help="Dataset root"),
    taxonomy: str = typer.Option("platformer", "--taxonomy", help="Taxonomy name or YAML path"),
    backend: str = typer.Option("heuristic", "--backend", help="heuristic/yolo/huggingface/grounded-sam"),
    classes: list[str] = typer.Option(None, "--classes", help="Classes to request from teacher"),
    model: str | None = typer.Option(None, "--model", help="YOLO teacher model path/name"),
    teacher_endpoint: str | None = typer.Option(None, "--teacher-endpoint", help="Local Grounded-SAM server endpoint"),
    conf: float = typer.Option(0.25, "--conf", help="Teacher confidence threshold"),
    split: str = typer.Option("train", "--split", help="train/val/test"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Auto-label raw images using a teacher backend and export YOLO segmentation labels."""
    from gvi.training.autolabel import autolabel_directory

    result = autolabel_directory(source, dataset, taxonomy_name=taxonomy, backend=backend, classes=classes, model=model, teacher_endpoint=teacher_endpoint, conf=conf, split=split)
    if json_out:
        _print_json(result)
    else:
        console.print(f"[green]Auto-labelled {result['images']} image(s) with backend={result['backend']}[/green]")
        console.print(f"Dataset: {result['dataset_root']}")


@training_app.command("synthetic")
def training_synthetic(
    dataset: Path = typer.Option(Path("dataset"), "--dataset", help="Dataset root"),
    count: int = typer.Option(50, "--count", help="Number of synthetic images"),
    split: str = typer.Option("train", "--split", help="train/val/test"),
    seed: int = typer.Option(42, "--seed"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Generate a tiny synthetic platformer dataset to test the training path."""
    from gvi.training.synthetic import generate_synthetic_platformer

    result = generate_synthetic_platformer(dataset, count=count, split=split, seed=seed)
    if json_out:
        _print_json(result)
    else:
        console.print(f"[green]Generated {result['generated']} synthetic sample(s)[/green]")


@training_app.command("review")
def training_review(
    dataset: Path = typer.Argument(..., help="Dataset root"),
    threshold: float = typer.Option(0.72, "--threshold", help="Confidence threshold"),
    max_items: int = typer.Option(200, "--max-items"),
    cvat: bool = typer.Option(False, "--cvat", help="Write CVAT quickstart file"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Create review.json for low-confidence/ambiguous labels."""
    from gvi.training.active_learning import build_review_file
    from gvi.training.review import write_cvat_quickstart

    review = build_review_file(dataset, threshold=threshold, max_items=max_items)
    cvat_path = write_cvat_quickstart(dataset) if cvat else None
    if json_out:
        _print_json({"review": review.model_dump(mode="json"), "cvat": str(cvat_path) if cvat_path else None})
        return
    console.print(f"[green]Review file written:[/green] {dataset / 'review' / 'review.json'}")
    console.print(f"Items to review: {len(review.items)}")
    if cvat_path:
        console.print(f"CVAT guide: {cvat_path}")


@training_app.command("train")
def training_train(
    dataset: Path = typer.Argument(..., help="Dataset root"),
    model: str = typer.Option("yolo11n-seg.pt", "--model", help="YOLO segmentation model"),
    epochs: int = typer.Option(80, "--epochs"),
    imgsz: int = typer.Option(640, "--imgsz"),
    batch: int = typer.Option(8, "--batch"),
    device: str | None = typer.Option(None, "--device", help="cpu, 0, 0,1 ..."),
    workers: int = typer.Option(4, "--workers"),
    name: str = typer.Option("gvi_platformer_yolo", "--name"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print command without running"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Train a YOLO segmentation student model."""
    from gvi.training.train_yolo import TrainConfig, train_yolo

    result = train_yolo(TrainConfig(dataset_root=dataset, model=model, epochs=epochs, imgsz=imgsz, batch=batch, device=device, workers=workers, name=name, dry_run=dry_run))
    if json_out:
        _print_json(result)
    else:
        console.print("[green]Training command:[/green]")
        console.print(" ".join(result["command"]))
        console.print(f"OK: {result.get('ok', result.get('dry_run'))}")
        if result.get("expected_best"):
            console.print(f"Expected best.pt: {result['expected_best']}")


@training_app.command("evaluate")
def training_evaluate(
    dataset: Path = typer.Argument(..., help="Dataset root"),
    model: Path | None = typer.Option(None, "--model", help="Optional YOLO model to validate"),
    imgsz: int = typer.Option(640, "--imgsz"),
    split: str = typer.Option("val", "--split"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Evaluate dataset readiness and optionally validate a YOLO model."""
    from gvi.training.evaluate import evaluate_dataset, evaluate_yolo

    result = evaluate_dataset(dataset)
    if model:
        result["yolo"] = evaluate_yolo(dataset, model, imgsz=imgsz, split=split, dry_run=dry_run)
    if json_out:
        _print_json(result)
    else:
        console.print_json(json.dumps(result, indent=2, default=str))


@training_app.command("predict")
def training_predict(
    image: Path = typer.Argument(..., help="Image to predict"),
    model: Path = typer.Option(..., "--model", help="Trained YOLO .pt model"),
    out: Path = typer.Option(Path("predictions"), "--out"),
    taxonomy: str = typer.Option("platformer", "--taxonomy"),
    conf: float = typer.Option(0.25, "--conf"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Run a trained model and write predictions.json + overlay."""
    from gvi.training.predict import predict_image

    result = predict_image(image, model, out, taxonomy_name=taxonomy, conf=conf)
    if json_out:
        _print_json(result)
    else:
        console.print(f"[green]Predictions:[/green] {result['predictions']}")
        console.print(f"[green]Overlay:[/green] {result['overlay']}")


# ---------------------------------------------------------------- version
@app.command()
def version() -> None:
    import gvi
    console.print(f"[bold]GVI[/bold] (Godot Vision Orchestrator) v{gvi.__version__}")


if __name__ == "__main__":
    app()