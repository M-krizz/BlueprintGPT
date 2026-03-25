"""
train_kaggle.py - End-to-end training entrypoint for the Kaggle floor-plan dataset.

This script uses the repo's existing local Kaggle JSON export under
``learned/data/kaggle_json`` and wires it directly into the existing
LayoutTransformer training pipeline.

Usage
-----
    # Prepare train/val JSONL files only
    python -m learned.model.train_kaggle --prepare-only

    # Export topology JSONL pairs for later LLM/topology work
    python -m learned.model.train_kaggle --prepare-only \\
        --pairs-output training/kaggle_topology_pairs.jsonl

    # Train with the improved trainer
    python -m learned.model.train_kaggle \\
        --epochs 50 --batch-size 8 --accumulate 4 \\
        --save learned/model/checkpoints/kaggle_improved.pt

    # Quick smoke training on a small subset
    python -m learned.model.train_kaggle \\
        --limit 16 --epochs 1 --batch-size 2 --accumulate 1 --device cpu

    # Use a preset and override only the values you care about
    python -m learned.model.train_kaggle \\
        --preset kaggle_quick --device cuda --save learned/model/checkpoints/kaggle_quick.pt
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from learned.data.build_sequences import build_sequences_jsonl
from learned.data.tokenizer_layout import DEFAULT_NUM_BINS
from learned.model.train import train as train_baseline
from learned.model.train_improved import train_improved
from training.make_training_pairs import iter_pairs


DEFAULT_SOURCE_DIR = Path("learned/data/kaggle_json")
DEFAULT_TRAIN_JSONL = Path("learned/data/kaggle_train.jsonl")
DEFAULT_VAL_JSONL = Path("learned/data/kaggle_train_val.jsonl")
DEFAULT_SUMMARY_JSON = Path("learned/model/checkpoints/kaggle_dataset_summary.json")
DEFAULT_CHECKPOINT = Path("learned/model/checkpoints/kaggle_improved.pt")
DEFAULT_KAGGLE_DATASET = "mazharrehan/floorplan"
DEFAULT_TOPOLOGY_PAIRS = Path("training/kaggle_topology_pairs.jsonl")

KAGGLE_TRAINING_PRESETS = {
    "cpu_smoke": {
        "epochs": 1,
        "batch_size": 2,
        "accumulate_steps": 1,
        "lr": 1e-4,
        "weight_decay": 0.01,
        "warmup_steps": 10,
        "device": "cpu",
        "log_every": 5,
        "augment": False,
        "expansion_factor": 1,
        "aux_loss": False,
        "coverage_weight": 0.3,
        "overlap_weight": 0.4,
        "spread_weight": 0.1,
        "n_layers": 4,
        "n_heads": 4,
        "d_model": 128,
        "d_ff": 512,
        "dropout": 0.1,
    },
    "kaggle_quick": {
        "epochs": 20,
        "batch_size": 4,
        "accumulate_steps": 2,
        "lr": 1e-4,
        "weight_decay": 0.01,
        "warmup_steps": 100,
        "device": "cpu",
        "log_every": 20,
        "augment": True,
        "expansion_factor": 4,
        "aux_loss": True,
        "coverage_weight": 0.3,
        "overlap_weight": 0.4,
        "spread_weight": 0.1,
        "n_layers": 6,
        "n_heads": 4,
        "d_model": 192,
        "d_ff": 768,
        "dropout": 0.15,
    },
    "kaggle_full": {
        "epochs": 100,
        "batch_size": 8,
        "accumulate_steps": 4,
        "lr": 1e-4,
        "weight_decay": 0.01,
        "warmup_steps": 500,
        "device": "cpu",
        "log_every": 50,
        "augment": True,
        "expansion_factor": 8,
        "aux_loss": True,
        "coverage_weight": 0.3,
        "overlap_weight": 0.4,
        "spread_weight": 0.1,
        "n_layers": 8,
        "n_heads": 8,
        "d_model": 256,
        "d_ff": 1024,
        "dropout": 0.15,
    },
}


def _write_jsonl(records: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def _write_summary(summary: Dict[str, object], summary_output: Optional[str | Path]) -> None:
    if not summary_output:
        return
    summary_path = Path(summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _boundary_extent(boundary: list[list[float]] | None) -> tuple[float, float]:
    xs = [float(point[0]) for point in boundary or []]
    ys = [float(point[1]) for point in boundary or []]
    if not xs or not ys:
        return (0.0, 0.0)
    return (max(xs) - min(xs), max(ys) - min(ys))


def _collect_source_stats(source_dir: Path) -> Dict[str, object]:
    plot_types: Counter[str] = Counter()
    building_types: Counter[str] = Counter()
    room_types: Counter[str] = Counter()
    room_count_distribution: Counter[int] = Counter()
    boundary_widths: list[float] = []
    boundary_heights: list[float] = []

    source_files = sorted(source_dir.glob("*.json"))
    for source_path in source_files:
        with source_path.open("r", encoding="utf-8") as handle:
            sample = json.load(handle)
        plot_types[str(sample.get("plot_type") or "unknown")] += 1
        building_types[str(sample.get("building_type") or "unknown")] += 1
        rooms = sample.get("rooms", []) or []
        room_count_distribution[len(rooms)] += 1
        for room in rooms:
            room_type = str(room.get("type") or "unknown").strip() or "unknown"
            room_types[room_type] += 1
        width, height = _boundary_extent(sample.get("boundary"))
        if width > 0.0 and height > 0.0:
            boundary_widths.append(width)
            boundary_heights.append(height)

    avg_room_count = 0.0
    if room_count_distribution:
        total_rooms = sum(room_count * count for room_count, count in room_count_distribution.items())
        avg_room_count = total_rooms / max(sum(room_count_distribution.values()), 1)

    return {
        "source_file_count": len(source_files),
        "plot_types": dict(sorted(plot_types.items())),
        "building_types": dict(sorted(building_types.items())),
        "room_types": dict(sorted(room_types.items())),
        "room_count_distribution": {str(key): value for key, value in sorted(room_count_distribution.items())},
        "average_room_count": round(avg_room_count, 3),
        "average_boundary_width": round(sum(boundary_widths) / max(len(boundary_widths), 1), 3) if boundary_widths else 0.0,
        "average_boundary_height": round(sum(boundary_heights) / max(len(boundary_heights), 1), 3) if boundary_heights else 0.0,
    }


def _write_topology_pairs(source_dir: Path, pairs_output: str | Path) -> Dict[str, object]:
    output_path = Path(pairs_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pair_count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for pair in iter_pairs(source_dir):
            handle.write(json.dumps(pair, ensure_ascii=True) + "\n")
            pair_count += 1
    return {
        "pairs_output": str(output_path),
        "pair_count": pair_count,
    }


def _resolve_source_dir(
    source_dir: str | Path,
    *,
    use_kagglehub: bool = False,
    kaggle_dataset: str = DEFAULT_KAGGLE_DATASET,
) -> Path:
    source_dir = Path(source_dir)
    if source_dir.exists():
        return source_dir
    if not use_kagglehub:
        raise FileNotFoundError(
            f"Kaggle source directory not found: {source_dir}. "
            "Expected local floor-plan JSON files under learned/data/kaggle_json."
        )
    try:
        import kagglehub
    except ImportError as exc:
        raise FileNotFoundError(
            f"Kaggle source directory not found: {source_dir}, and kagglehub is unavailable."
        ) from exc

    downloaded = Path(kagglehub.dataset_download(kaggle_dataset))
    if source_dir.name and (downloaded / source_dir.name).exists():
        candidate = downloaded / source_dir.name
        if list(candidate.glob("*.json")):
            return candidate
    if list(downloaded.glob("*.json")):
        return downloaded
    for candidate in downloaded.rglob("*.json"):
        return candidate.parent
    raise FileNotFoundError(
        f"Downloaded Kaggle dataset from {kaggle_dataset}, but no floor-plan JSON files were found under {downloaded}."
    )


def prepare_kaggle_jsonl(
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    *,
    train_output: str | Path = DEFAULT_TRAIN_JSONL,
    val_output: str | Path = DEFAULT_VAL_JSONL,
    summary_output: Optional[str | Path] = DEFAULT_SUMMARY_JSON,
    split: float = 0.9,
    num_bins: int = DEFAULT_NUM_BINS,
    max_len: int = 256,
    seed: int = 42,
    limit: Optional[int] = None,
    use_kagglehub: bool = False,
    kaggle_dataset: str = DEFAULT_KAGGLE_DATASET,
    pairs_output: Optional[str | Path] = None,
) -> Dict[str, object]:
    source_dir = _resolve_source_dir(
        source_dir,
        use_kagglehub=use_kagglehub,
        kaggle_dataset=kaggle_dataset,
    )
    train_output = Path(train_output)
    val_output = Path(val_output)

    records = build_sequences_jsonl(source_dir, num_bins=num_bins, max_len=max_len)
    rng = random.Random(seed)
    rng.shuffle(records)

    if limit is not None:
        records = records[: max(1, int(limit))]
    if len(records) < 2:
        raise RuntimeError("Need at least 2 Kaggle records to create train/val splits.")

    split = max(0.1, min(float(split), 0.99))
    split_idx = max(1, min(len(records) - 1, int(round(len(records) * split))))
    train_records = records[:split_idx]
    val_records = records[split_idx:]

    _write_jsonl(train_records, train_output)
    _write_jsonl(val_records, val_output)

    summary: Dict[str, object] = {
        "source_dir": str(source_dir),
        "train_output": str(train_output),
        "val_output": str(val_output),
        "record_count": len(records),
        "train_record_count": len(train_records),
        "val_record_count": len(val_records),
        "split": split,
        "num_bins": num_bins,
        "max_len": max_len,
        "seed": seed,
        "limit": limit,
        "sample_plan_ids": [record.get("plan_id") for record in records[:5]],
        "source_stats": _collect_source_stats(source_dir),
    }

    if pairs_output:
        summary["topology_pairs"] = _write_topology_pairs(source_dir, pairs_output)

    _write_summary(summary, summary_output)

    return summary


def train_on_kaggle(
    *,
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    train_output: str | Path = DEFAULT_TRAIN_JSONL,
    val_output: str | Path = DEFAULT_VAL_JSONL,
    summary_output: Optional[str | Path] = DEFAULT_SUMMARY_JSON,
    split: float = 0.9,
    num_bins: int = DEFAULT_NUM_BINS,
    max_len: int = 256,
    seed: int = 42,
    limit: Optional[int] = None,
    use_kagglehub: bool = False,
    kaggle_dataset: str = DEFAULT_KAGGLE_DATASET,
    prepare_only: bool = False,
    trainer: str = "improved",
    epochs: int = 100,
    batch_size: int = 8,
    accumulate_steps: int = 4,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 500,
    save_path: str | Path = DEFAULT_CHECKPOINT,
    device: str = "cpu",
    log_every: int = 50,
    augment: bool = True,
    expansion_factor: int = 8,
    aux_loss: bool = True,
    coverage_weight: float = 0.3,
    overlap_weight: float = 0.4,
    spread_weight: float = 0.1,
    n_layers: int = 8,
    n_heads: int = 8,
    d_model: int = 256,
    d_ff: int = 1024,
    dropout: float = 0.15,
    preset_name: Optional[str] = None,
    pairs_output: Optional[str | Path] = None,
) -> Dict[str, object]:
    summary = prepare_kaggle_jsonl(
        source_dir,
        train_output=train_output,
        val_output=val_output,
        summary_output=summary_output,
        split=split,
        num_bins=num_bins,
        max_len=max_len,
        seed=seed,
        limit=limit,
        use_kagglehub=use_kagglehub,
        kaggle_dataset=kaggle_dataset,
        pairs_output=pairs_output,
    )

    if prepare_only:
        return summary

    if trainer == "baseline":
        train_baseline(
            str(train_output),
            val_path=str(val_output),
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            num_bins=num_bins,
            save_path=str(save_path),
            device=device,
            log_every=log_every,
        )
    else:
        train_improved(
            str(train_output),
            val_path=str(val_output),
            epochs=epochs,
            batch_size=batch_size,
            accumulate_steps=accumulate_steps,
            lr=lr,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            num_bins=num_bins,
            save_path=str(save_path),
            device=device,
            log_every=log_every,
            augment=augment,
            expansion_factor=expansion_factor,
            aux_loss=aux_loss,
            coverage_weight=coverage_weight,
            overlap_weight=overlap_weight,
            spread_weight=spread_weight,
            n_layers=n_layers,
            n_heads=n_heads,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )

    summary = dict(summary)
    summary["trainer"] = trainer
    summary["save_path"] = str(save_path)
    summary["training_config"] = {
        "epochs": epochs,
        "batch_size": batch_size,
        "accumulate_steps": accumulate_steps,
        "lr": lr,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "device": device,
        "log_every": log_every,
        "trainer": trainer,
        "preset": preset_name,
        "augment": augment,
        "expansion_factor": expansion_factor,
        "aux_loss": aux_loss,
        "coverage_weight": coverage_weight,
        "overlap_weight": overlap_weight,
        "spread_weight": spread_weight,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "d_model": d_model,
        "d_ff": d_ff,
        "dropout": dropout,
    }
    _write_summary(summary, summary_output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and train the LayoutTransformer on the Kaggle floor-plan dataset.")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR), help="Directory of Kaggle floor-plan JSON files")
    parser.add_argument("--train-output", default=str(DEFAULT_TRAIN_JSONL))
    parser.add_argument("--val-output", default=str(DEFAULT_VAL_JSONL))
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY_JSON))
    parser.add_argument("--use-kagglehub", action="store_true", help="Download or resolve the Kaggle dataset via kagglehub when the source directory is missing")
    parser.add_argument("--kaggle-dataset", default=DEFAULT_KAGGLE_DATASET, help="Kaggle dataset handle used with --use-kagglehub")
    parser.add_argument("--split", type=float, default=0.9)
    parser.add_argument("--num-bins", type=int, default=DEFAULT_NUM_BINS)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick smoke runs")
    parser.add_argument("--pairs-output", default=None, help="Optional JSONL output for topology-training pairs derived from the same Kaggle source")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--trainer", choices=["improved", "baseline"], default="improved")
    parser.add_argument("--preset", choices=sorted(KAGGLE_TRAINING_PRESETS), default=None, help="Apply a named training preset before explicit overrides")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", "--batch", dest="batch_size", type=int, default=None)
    parser.add_argument("--accumulate", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--augment", action="store_true", default=None, help="Enable improved-trainer augmentation")
    parser.add_argument("--no-augment", action="store_false", dest="augment")
    parser.add_argument("--expansion", type=int, default=None, help="Augmentation expansion factor for the improved trainer")
    parser.add_argument("--aux-loss", action="store_true", default=None, help="Enable improved-trainer auxiliary losses")
    parser.add_argument("--no-aux-loss", action="store_false", dest="aux_loss")
    parser.add_argument("--coverage-weight", type=float, default=None)
    parser.add_argument("--overlap-weight", type=float, default=None)
    parser.add_argument("--spread-weight", type=float, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--save", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    resolved = dict(KAGGLE_TRAINING_PRESETS.get(args.preset or "", {}))
    fallback_defaults = {
        "epochs": 100,
        "batch_size": 8,
        "accumulate_steps": 4,
        "lr": 1e-4,
        "weight_decay": 0.01,
        "warmup_steps": 500,
        "log_every": 50,
        "augment": True,
        "expansion_factor": 8,
        "aux_loss": True,
        "coverage_weight": 0.3,
        "overlap_weight": 0.4,
        "spread_weight": 0.1,
        "n_layers": 8,
        "n_heads": 8,
        "d_model": 256,
        "d_ff": 1024,
        "dropout": 0.15,
    }

    cli_overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "accumulate_steps": args.accumulate,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "log_every": args.log_every,
        "augment": args.augment,
        "expansion_factor": args.expansion,
        "aux_loss": args.aux_loss,
        "coverage_weight": args.coverage_weight,
        "overlap_weight": args.overlap_weight,
        "spread_weight": args.spread_weight,
        "n_layers": args.layers,
        "n_heads": args.heads,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "dropout": args.dropout,
    }
    for key, fallback in fallback_defaults.items():
        if cli_overrides[key] is not None:
            resolved[key] = cli_overrides[key]
        elif key not in resolved:
            resolved[key] = fallback

    summary = train_on_kaggle(
        source_dir=args.source_dir,
        train_output=args.train_output,
        val_output=args.val_output,
        summary_output=args.summary_output,
        split=args.split,
        num_bins=args.num_bins,
        max_len=args.max_len,
        seed=args.seed,
        limit=args.limit,
        use_kagglehub=args.use_kagglehub,
        kaggle_dataset=args.kaggle_dataset,
        pairs_output=args.pairs_output,
        prepare_only=args.prepare_only,
        trainer=args.trainer,
        epochs=resolved["epochs"],
        batch_size=resolved["batch_size"],
        accumulate_steps=resolved["accumulate_steps"],
        lr=resolved["lr"],
        weight_decay=resolved["weight_decay"],
        warmup_steps=resolved["warmup_steps"],
        save_path=args.save,
        device=args.device,
        log_every=resolved["log_every"],
        augment=resolved["augment"],
        expansion_factor=resolved["expansion_factor"],
        aux_loss=resolved["aux_loss"],
        coverage_weight=resolved["coverage_weight"],
        overlap_weight=resolved["overlap_weight"],
        spread_weight=resolved["spread_weight"],
        n_layers=resolved["n_layers"],
        n_heads=resolved["n_heads"],
        d_model=resolved["d_model"],
        d_ff=resolved["d_ff"],
        dropout=resolved["dropout"],
        preset_name=args.preset,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
