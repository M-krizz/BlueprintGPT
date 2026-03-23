"""CLI for planner dataset generation."""

from __future__ import annotations

import argparse

from learned.planner.data import build_planner_dataset


def main():
    parser = argparse.ArgumentParser(description="Build planner JSONL dataset from plan JSON files")
    parser.add_argument(
        "--input-pattern",
        default="learned/data/kaggle_json/*.json",
        help="Glob pattern for input plan JSON files",
    )
    parser.add_argument(
        "--output",
        default="learned/planner/data/planner_records.jsonl",
        help="Output JSONL path",
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
    print(f"[planner] wrote {written} records to {args.output}")


if __name__ == "__main__":
    main()
