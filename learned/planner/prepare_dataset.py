"""Build and split the planner dataset for training."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from learned.planner.data import build_planner_dataset



def append_jsonl_records(output_path: str, extra_paths: list[str]) -> int:
    output = Path(output_path)
    appended = 0
    with output.open("a", encoding="utf-8") as out_handle:
        for extra_path in extra_paths:
            extra_file = Path(extra_path)
            if not extra_file.exists():
                continue
            for line in extra_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                out_handle.write(line + "\n")
                appended += 1
    return appended



def _program_signature(record: dict) -> str:
    counts: dict[str, int] = {}
    for room in record.get("rooms", []):
        room_type = str(room.get("type") or "").strip()
        if not room_type:
            continue
        counts[room_type] = counts.get(room_type, 0) + 1
    if not counts:
        return "empty"
    return "|".join(f"{room_type}:{counts[room_type]}" for room_type in sorted(counts))



def _load_record_lines(input_path: str) -> list[tuple[str, dict]]:
    loaded: list[tuple[str, dict]] = []
    for line in Path(input_path).read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        loaded.append((raw, json.loads(raw)))
    return loaded



def _random_split(records: list[str], *, val_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    split_index = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * (1.0 - val_ratio)))))
    return shuffled[:split_index], shuffled[split_index:]



def _program_stratified_split(records: list[tuple[str, dict]], *, val_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    grouped: dict[str, list[str]] = defaultdict(list)
    for raw, record in records:
        grouped[_program_signature(record)].append(raw)

    desired_val = max(1, min(len(records) - 1, int(round(len(records) * val_ratio))))
    val_by_group: dict[str, int] = {}
    remainder_rank: list[tuple[float, str]] = []

    for signature, group_records in grouped.items():
        rng.shuffle(group_records)
        if len(group_records) <= 1:
            val_by_group[signature] = 0
            continue
        exact = len(group_records) * val_ratio
        base = min(len(group_records) - 1, int(math.floor(exact)))
        val_by_group[signature] = max(0, base)
        remainder_rank.append((exact - base, signature))

    current_val = sum(val_by_group.values())
    if current_val == 0:
        for _, signature in sorted(remainder_rank, reverse=True):
            if len(grouped[signature]) > 1:
                val_by_group[signature] = 1
                current_val += 1
                break

    for _, signature in sorted(remainder_rank, reverse=True):
        if current_val >= desired_val:
            break
        capacity = len(grouped[signature]) - 1
        if val_by_group[signature] >= capacity:
            continue
        val_by_group[signature] += 1
        current_val += 1

    train_records: list[str] = []
    val_records: list[str] = []
    for signature in sorted(grouped):
        group_records = grouped[signature]
        val_count = min(val_by_group.get(signature, 0), len(group_records) - 1)
        val_records.extend(group_records[:val_count])
        train_records.extend(group_records[val_count:])

    if not train_records or not val_records:
        return _random_split([raw for raw, _ in records], val_ratio=val_ratio, seed=seed)

    return train_records, val_records



def _record_stats(records: list[dict]) -> dict:
    program_counts: Counter[str] = Counter()
    teacher_records = 0
    room_type_counts: Counter[str] = Counter()
    for record in records:
        program_counts[_program_signature(record)] += 1
        if record.get("teacher_metadata"):
            teacher_records += 1
        for room in record.get("rooms", []):
            room_type = str(room.get("type") or "").strip()
            if room_type:
                room_type_counts[room_type] += 1
    return {
        "record_count": len(records),
        "teacher_record_count": teacher_records,
        "base_record_count": max(len(records) - teacher_records, 0),
        "unique_program_count": len(program_counts),
        "top_programs": [
            {"program": program, "count": count}
            for program, count in sorted(program_counts.items(), key=lambda item: (-item[1], item[0]))[:10]
        ],
        "room_type_counts": {key: value for key, value in sorted(room_type_counts.items())},
    }



def build_split_summary(train_path: str, val_path: str) -> dict:
    train_records = [record for _, record in _load_record_lines(train_path)]
    val_records = [record for _, record in _load_record_lines(val_path)]

    train_programs = {_program_signature(record) for record in train_records}
    val_programs = {_program_signature(record) for record in val_records}
    overlap = sorted(train_programs & val_programs)
    val_only = sorted(val_programs - train_programs)
    train_only = sorted(train_programs - val_programs)

    return {
        "train": _record_stats(train_records),
        "val": _record_stats(val_records),
        "program_overlap_count": len(overlap),
        "program_overlap_examples": overlap[:10],
        "validation_only_program_count": len(val_only),
        "validation_only_program_examples": val_only[:10],
        "train_only_program_count": len(train_only),
        "train_only_program_examples": train_only[:10],
    }



def split_records(
    input_path: str,
    train_output_path: str,
    val_output_path: str,
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
    split_mode: str = "program",
) -> tuple[int, int]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")

    loaded_records = _load_record_lines(input_path)
    if len(loaded_records) < 2:
        raise ValueError("Need at least two planner records to create train/val splits.")

    if split_mode == "program":
        train_records, val_records = _program_stratified_split(loaded_records, val_ratio=val_ratio, seed=seed)
    elif split_mode == "random":
        train_records, val_records = _random_split([raw for raw, _ in loaded_records], val_ratio=val_ratio, seed=seed)
    else:
        raise ValueError("split_mode must be 'program' or 'random'.")

    train_output = Path(train_output_path)
    val_output = Path(val_output_path)
    train_output.parent.mkdir(parents=True, exist_ok=True)
    val_output.parent.mkdir(parents=True, exist_ok=True)
    train_output.write_text("\n".join(train_records) + "\n", encoding="utf-8")
    val_output.write_text("\n".join(val_records) + "\n", encoding="utf-8")
    return len(train_records), len(val_records)



def main():
    parser = argparse.ArgumentParser(description="Build and split planner JSONL data")
    parser.add_argument(
        "--input-pattern",
        default="learned/data/kaggle_json/*.json",
        help="Glob pattern for input plan JSON files",
    )
    parser.add_argument(
        "--output",
        default="learned/planner/data/planner_records.jsonl",
        help="Output JSONL path for the combined planner records",
    )
    parser.add_argument(
        "--train-output",
        default="learned/planner/data/planner_train.jsonl",
        help="Output JSONL path for the train split",
    )
    parser.add_argument(
        "--val-output",
        default="learned/planner/data/planner_val.jsonl",
        help="Output JSONL path for the validation split",
    )
    parser.add_argument(
        "--summary-output",
        default="learned/planner/data/planner_split_summary.json",
        help="Optional JSON path for train/val split summary",
    )
    parser.add_argument(
        "--append-jsonl",
        nargs="*",
        default=[],
        help="Optional planner-record JSONL files to append before the train/val split",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation fraction used when writing train/val splits",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for shuffling before split",
    )
    parser.add_argument(
        "--split-mode",
        choices=("program", "random"),
        default="program",
        help="Split strategy for train/val generation",
    )
    parser.add_argument(
        "--adjacency-tolerance-ratio",
        type=float,
        default=0.015,
        help="Boundary-relative tolerance used when deriving adjacency labels",
    )
    args = parser.parse_args()

    written = build_planner_dataset(
        input_pattern=args.input_pattern,
        output_path=args.output,
        adjacency_tolerance_ratio=args.adjacency_tolerance_ratio,
    )
    appended = append_jsonl_records(args.output, args.append_jsonl)
    train_count, val_count = split_records(
        args.output,
        args.train_output,
        args.val_output,
        val_ratio=args.val_ratio,
        seed=args.seed,
        split_mode=args.split_mode,
    )
    summary = build_split_summary(args.train_output, args.val_output)
    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[planner] wrote {written} records to {args.output}")
    if appended:
        print(f"[planner] appended {appended} extra planner records")
    print(f"[planner] split dataset into {train_count} train and {val_count} validation records using {args.split_mode} mode")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
