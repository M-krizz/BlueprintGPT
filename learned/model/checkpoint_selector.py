"""
checkpoint_selector.py - Select a production checkpoint without relying on
broken zero-score cache entries or ad-hoc newest-file fallbacks.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


PRODUCTION_CHECKPOINT_PREFERENCE = (
    "improved_fullsize.pt",
    "improved_v1.pt",
    "improved_v2.pt",
    "kaggle_test.pt",
)

DEFAULT_VALIDATION_SPECS = [
    {
        "rooms": [
            {"type": "Bedroom", "count": 2},
            {"type": "Kitchen", "count": 1},
            {"type": "Bathroom", "count": 1},
            {"type": "LivingRoom", "count": 1},
        ],
        "plot_area_sqm": 100.0,
        "boundary_polygon": [(0, 0), (12, 0), (12, 10), (0, 10)],
        "entrance_point": (6.0, 0.0),
    },
    {
        "rooms": [
            {"type": "Bedroom", "count": 2},
            {"type": "Kitchen", "count": 1},
            {"type": "DiningRoom", "count": 1},
            {"type": "Store", "count": 1},
        ],
        "plot_area_sqm": 125.0,
        "boundary_polygon": [(0, 0), (15, 0), (15, 10), (0, 10)],
        "entrance_point": (7.5, 0.0),
    },
]


@dataclass
class CheckpointMetrics:
    """Metrics for evaluating checkpoint quality."""

    checkpoint_path: str
    epoch: int
    val_loss: float

    compliance_rate: float
    realism_score: float
    overlap_rate: float
    repair_severity: float

    min_dim_violations: int
    generation_success_rate: float

    composite_score: float


def _is_finite_number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _checkpoint_preference_bonus(checkpoint_path: Path) -> float:
    name = checkpoint_path.name
    if name in PRODUCTION_CHECKPOINT_PREFERENCE:
        rank = PRODUCTION_CHECKPOINT_PREFERENCE.index(name)
        return (len(PRODUCTION_CHECKPOINT_PREFERENCE) - rank) * 0.01
    return 0.0


def _is_ephemeral_checkpoint(checkpoint_path: Path) -> bool:
    stem = checkpoint_path.stem.lower()
    return (
        stem.startswith(("test_", "quick_", "debug_", "scratch_"))
        or "quick" in stem
        or stem in {"test", "quick", "debug", "scratch"}
    )


def _candidate_checkpoints(checkpoint_dir: Path, *, allow_ephemeral: bool = False) -> List[Path]:
    checkpoint_files = sorted(checkpoint_dir.glob("*.pt"))
    if allow_ephemeral:
        return checkpoint_files

    production = [path for path in checkpoint_files if not _is_ephemeral_checkpoint(path)]
    return production or checkpoint_files


def _checkpoint_metadata(checkpoint_path: Path) -> Dict[str, float]:
    metadata = {"epoch": 0, "loss": float("inf"), "val_loss": float("inf")}

    if not TORCH_AVAILABLE:
        return metadata

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        return metadata

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss")
    val_loss = checkpoint.get("val_loss")

    metadata["epoch"] = int(epoch or 0)
    if _is_finite_number(loss):
        metadata["loss"] = float(loss)
    if _is_finite_number(val_loss):
        metadata["val_loss"] = float(val_loss)
    elif _is_finite_number(loss):
        metadata["val_loss"] = float(loss)

    return metadata


def _heuristic_metrics(checkpoint_path: Path) -> CheckpointMetrics:
    metadata = _checkpoint_metadata(checkpoint_path)
    score = 0.0

    if _is_finite_number(metadata["val_loss"]):
        score += 1.0 / (1.0 + float(metadata["val_loss"]))
    else:
        score += 0.05

    score += min(int(metadata["epoch"]), 100) / 1000.0
    score += _checkpoint_preference_bonus(checkpoint_path)

    return CheckpointMetrics(
        checkpoint_path=str(checkpoint_path),
        epoch=int(metadata["epoch"]),
        val_loss=float(metadata["val_loss"]),
        compliance_rate=0.0,
        realism_score=0.0,
        overlap_rate=1.0,
        repair_severity=1.0,
        min_dim_violations=999,
        generation_success_rate=0.0,
        composite_score=round(score, 4),
    )


def _looks_like_real_evaluation(metrics: CheckpointMetrics) -> bool:
    return any(
        float(value) > 0.0
        for value in (
            metrics.compliance_rate,
            metrics.realism_score,
            metrics.generation_success_rate,
        )
    )


def _load_cached_metrics(cache_path: Path, checkpoint_dir: Path) -> Dict[Path, CheckpointMetrics]:
    if not cache_path.exists():
        return {}

    try:
        cache_data = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    cached: Dict[Path, CheckpointMetrics] = {}
    if not isinstance(cache_data, list):
        return cached

    for item in cache_data:
        if not isinstance(item, dict):
            continue

        try:
            metrics = CheckpointMetrics(**item)
        except TypeError:
            continue

        checkpoint_name = Path(metrics.checkpoint_path).name
        checkpoint_path = checkpoint_dir / checkpoint_name

        if not checkpoint_path.exists():
            continue
        if not _looks_like_real_evaluation(metrics):
            continue

        cached[checkpoint_path] = metrics

    return cached


def evaluate_checkpoint(
    checkpoint_path: Path,
    validation_specs: List[Dict],
    num_samples: int = 20,
) -> CheckpointMetrics:
    """Evaluate a checkpoint against a validation set and return quality metrics."""
    heuristic = _heuristic_metrics(checkpoint_path)
    if not TORCH_AVAILABLE:
        return heuristic

    try:
        from learned.integration.learned_to_building_adapter import adapt_generated_layout_to_building
        from learned.integration.realism_score import compute_realism_score
        from learned.integration.repair_gate import validate_and_repair_generated_layout
        from learned.model.sample import constrained_sample_layout, load_model

        model, tokenizer = load_model(str(checkpoint_path))
    except Exception:
        return heuristic

    total_attempts = 0
    successful_generations = 0
    compliant_layouts = 0
    total_realism = 0.0
    total_overlaps = 0
    total_repair_severity = 0.0
    total_min_dim_violations = 0

    for spec in validation_specs:
        boundary_polygon = spec.get("boundary_polygon", [(0, 0), (15, 0), (15, 12), (0, 12)])
        entrance_point = spec.get("entrance_point")
        plot_area_sqm = float(spec.get("plot_area_sqm", 100.0))

        for _ in range(num_samples):
            total_attempts += 1

            try:
                rooms = constrained_sample_layout(
                    model,
                    tokenizer,
                    spec,
                    plot_area_sqm=plot_area_sqm,
                    temperature=0.8,
                    max_new_tokens=150,
                )
            except Exception:
                continue

            if not rooms:
                continue

            successful_generations += 1

            try:
                building = adapt_generated_layout_to_building(
                    rooms,
                    boundary_poly=boundary_polygon,
                    entrance=entrance_point,
                    spec=spec,
                    regulation_data="ontology/regulation_data.json",
                )
            except Exception:
                continue

            realism = compute_realism_score(
                building,
                plot_area_sqm=plot_area_sqm,
                regulation_file="ontology/regulation_data.json",
            )
            total_realism += realism.overall
            total_min_dim_violations += realism.min_dim_violations

            if any("overlap" in str(item).lower() for item in realism.details.get("min_dim_violations_list", [])):
                total_overlaps += 1

            try:
                _, _, status, _, repair_report = validate_and_repair_generated_layout(
                    building,
                    boundary_polygon=boundary_polygon,
                    entrance_point=entrance_point,
                    regulation_file="ontology/regulation_data.json",
                    spec=spec,
                    run_ontology=False,
                )

                if status == "COMPLIANT":
                    compliant_layouts += 1

                total_repair_severity += repair_report.severity_score
            except Exception:
                total_repair_severity += 1.0

    if total_attempts == 0 or successful_generations == 0:
        return heuristic

    generation_success_rate = successful_generations / total_attempts
    compliance_rate = compliant_layouts / successful_generations
    avg_realism = total_realism / successful_generations
    overlap_rate = total_overlaps / successful_generations
    avg_repair_severity = total_repair_severity / successful_generations

    composite_score = (
        0.40 * compliance_rate
        + 0.25 * avg_realism
        + 0.20 * generation_success_rate
        + 0.15 * (1.0 - avg_repair_severity)
    )

    return CheckpointMetrics(
        checkpoint_path=str(checkpoint_path),
        epoch=heuristic.epoch,
        val_loss=heuristic.val_loss,
        compliance_rate=round(compliance_rate, 4),
        realism_score=round(avg_realism, 4),
        overlap_rate=round(overlap_rate, 4),
        repair_severity=round(avg_repair_severity, 4),
        min_dim_violations=total_min_dim_violations,
        generation_success_rate=round(generation_success_rate, 4),
        composite_score=round(composite_score, 4),
    )


def select_best_checkpoint(
    checkpoint_dir: Path,
    validation_specs: Optional[List[Dict]] = None,
    cache_file: Optional[str] = "checkpoint_metrics.json",
    num_samples: int = 20,
    *,
    evaluate_missing: bool = False,
    allow_ephemeral: bool = False,
) -> Path:
    """Select the best checkpoint using valid cache entries or training metadata."""
    checkpoint_files = _candidate_checkpoints(checkpoint_dir, allow_ephemeral=allow_ephemeral)

    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

    if len(checkpoint_files) == 1:
        return checkpoint_files[0]

    validation_specs = validation_specs or DEFAULT_VALIDATION_SPECS
    cache_path = checkpoint_dir / cache_file if cache_file else None
    cached_metrics = _load_cached_metrics(cache_path, checkpoint_dir) if cache_path else {}

    checkpoint_metrics: List[CheckpointMetrics] = []
    for checkpoint_path in checkpoint_files:
        if checkpoint_path in cached_metrics:
            metrics = cached_metrics[checkpoint_path]
            print(f"Using cached metrics for {checkpoint_path.name}")
        elif evaluate_missing:
            print(f"Evaluating {checkpoint_path.name}...")
            metrics = evaluate_checkpoint(checkpoint_path, validation_specs, num_samples=num_samples)
        else:
            metrics = _heuristic_metrics(checkpoint_path)
        checkpoint_metrics.append(metrics)

    if cache_path and evaluate_missing:
        try:
            cache_path.write_text(
                json.dumps([metric.__dict__ for metric in checkpoint_metrics], indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            print(f"Warning: Could not save metrics cache: {exc}")

    checkpoint_metrics.sort(
        key=lambda metric: (
            metric.composite_score,
            _looks_like_real_evaluation(metric),
            metric.epoch,
            -_candidate_checkpoints(
                checkpoint_dir,
                allow_ephemeral=True,
            ).index(Path(metric.checkpoint_path)),
        ),
        reverse=True,
    )

    print("\nCheckpoint Ranking:")
    print("-" * 80)
    for index, metric in enumerate(checkpoint_metrics[:5], start=1):
        name = Path(metric.checkpoint_path).name
        print(
            f"{index:2}. {name:20} | Score: {metric.composite_score:.3f} | "
            f"Compliance: {metric.compliance_rate:.2f} | Realism: {metric.realism_score:.2f} | "
            f"Repair: {metric.repair_severity:.2f}"
        )

    best_checkpoint = Path(checkpoint_metrics[0].checkpoint_path)
    print(f"\nSelected: {best_checkpoint.name}")
    return best_checkpoint


def get_model_checkpoint_path() -> str:
    """Get the best available model checkpoint path."""
    env_checkpoint = os.getenv("LAYOUT_MODEL_CHECKPOINT")
    if env_checkpoint and Path(env_checkpoint).exists():
        print(f"Using manual checkpoint: {env_checkpoint}")
        return env_checkpoint

    checkpoint_dir = Path("learned/model/checkpoints")
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    best_checkpoint = select_best_checkpoint(
        checkpoint_dir,
        evaluate_missing=False,
        allow_ephemeral=False,
    )
    print(f"Auto-selected best checkpoint: {best_checkpoint}")
    return str(best_checkpoint)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Select best model checkpoint")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="learned/model/checkpoints",
        help="Directory containing checkpoint files",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to generate per validation spec",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not read or write cached evaluation metrics",
    )
    parser.add_argument(
        "--allow-ephemeral",
        action="store_true",
        help="Include quick/test checkpoints in the ranking.",
    )

    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    cache_file = None if args.no_cache else "checkpoint_metrics.json"

    best_checkpoint = select_best_checkpoint(
        checkpoint_dir,
        cache_file=cache_file,
        num_samples=args.num_samples,
        evaluate_missing=True,
        allow_ephemeral=args.allow_ephemeral,
    )

    print(f"\nBest checkpoint: {best_checkpoint}")
    print(f"Set environment: export LAYOUT_MODEL_CHECKPOINT='{best_checkpoint}'")
