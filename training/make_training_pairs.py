from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from nl_interface.program_planner import (
    build_room_program,
    build_semantic_spec,
    build_zoning_plan,
    infer_layout_type_from_counts,
)


DEFAULT_INPUT_DIR = Path("learned/data/kaggle_json")
DEFAULT_OUTPUT_FILE = Path("training/kaggle_topology_pairs.jsonl")


def _boundary_extent(boundary: List[List[float]]) -> Tuple[float, float]:
    xs = [float(point[0]) for point in boundary or []]
    ys = [float(point[1]) for point in boundary or []]
    if not xs or not ys:
        return (0.0, 0.0)
    return (max(xs) - min(xs), max(ys) - min(ys))


def _room_counts(data: Dict[str, Any]) -> Dict[str, int]:
    counts = Counter()
    for room in data.get("rooms", []) or []:
        room_type = str(room.get("type") or "").strip()
        if room_type:
            counts[room_type] += 1
    return dict(counts)


def _prompt_from_counts(plot_type: str, room_counts: Dict[str, int]) -> str:
    ordered = ", ".join(f"{count} {room_type}" for room_type, count in sorted(room_counts.items()))
    layout_type = infer_layout_type_from_counts(room_counts)
    if layout_type:
        return f"Design a {layout_type} residential floor plan on a {plot_type} plot with {ordered}."
    return f"Design a residential floor plan on a {plot_type} plot with {ordered}."


def build_training_pair(source_path: Path) -> Dict[str, Any]:
    with source_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    room_counts = _room_counts(data)
    boundary_width, boundary_height = _boundary_extent(data.get("boundary") or [])
    layout_type = infer_layout_type_from_counts(room_counts)
    plot_type = str(data.get("plot_type") or "custom")
    spec = {
        "building_type": data.get("building_type") or "Residential",
        "layout_type": layout_type,
        "plot_type": plot_type,
        "rooms": [{"type": room_type, "count": count} for room_type, count in sorted(room_counts.items())],
        "preferences": {"adjacency": [], "privacy": {}},
        "style_hints": [],
    }
    semantic_spec = build_semantic_spec(spec, resolution=None, user_prompt=None)
    room_program = build_room_program(semantic_spec)
    zoning_plan = build_zoning_plan(room_program, semantic_spec, resolution=None, source_spec=spec)
    return {
        "source_file": str(source_path.as_posix()),
        "prompt": _prompt_from_counts(plot_type, room_counts),
        "semantic_spec_target": semantic_spec.to_dict(),
        "room_program_target": room_program.to_dict(),
        "topology_target": zoning_plan.to_dict(),
        "source_layout_summary": {
            "plot_type": plot_type,
            "layout_type": layout_type,
            "room_counts": room_counts,
            "boundary_extent_source_units": {
                "width": round(boundary_width, 3),
                "height": round(boundary_height, 3),
            },
        },
    }


def iter_pairs(input_dir: Path) -> Iterable[Dict[str, Any]]:
    for source_path in sorted(input_dir.glob("*.json")):
        yield build_training_pair(source_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create JSONL training pairs from Kaggle layout JSON files.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    args = parser.parse_args()

    pairs = list(iter_pairs(args.input_dir))
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=True) + "\n")
    print(f"Wrote {len(pairs)} training pairs to {args.output_file}")


if __name__ == "__main__":
    main()
