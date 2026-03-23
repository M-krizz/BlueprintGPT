"""Profile planner JSONL datasets for training-data quality checks."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List


def load_records(path: str) -> List[Dict]:
    records: List[Dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _sorted_dict(counter: Counter) -> Dict[str, int]:
    return {key: value for key, value in sorted(counter.items(), key=lambda item: (-item[1], item[0]))}


def _program_signature(record: Dict) -> str:
    room_counts: Counter = Counter()
    for room in record.get("rooms", []):
        room_type = str(room.get("type") or "").strip()
        if room_type:
            room_counts[room_type] += 1
    if not room_counts:
        return "empty"
    return "|".join(f"{room_type}:{room_counts[room_type]}" for room_type in sorted(room_counts))


def profile_records(records: List[Dict], *, top_programs: int = 10) -> Dict:
    plot_type_counts: Counter = Counter()
    room_type_counts: Counter = Counter()
    room_count_histogram: Counter = Counter()
    program_counts: Counter = Counter()
    teacher_style_counts: Counter = Counter()

    teacher_record_count = 0
    total_rooms = 0

    for record in records:
        plot_type = str(record.get("plot_type") or "Unknown")
        plot_type_counts[plot_type] += 1

        rooms = record.get("rooms", [])
        total_rooms += len(rooms)
        room_count_histogram[str(len(rooms))] += 1
        program_counts[_program_signature(record)] += 1

        teacher_metadata = record.get("teacher_metadata") or {}
        if teacher_metadata:
            teacher_record_count += 1
            style = str(teacher_metadata.get("style") or "").strip()
            if style:
                teacher_style_counts[style] += 1

        for room in rooms:
            room_type = str(room.get("type") or "Unknown")
            room_type_counts[room_type] += 1

    avg_room_count = round(total_rooms / len(records), 4) if records else 0.0
    top_program_list = [
        {"program": program, "count": count}
        for program, count in sorted(program_counts.items(), key=lambda item: (-item[1], item[0]))[:top_programs]
    ]

    return {
        "record_count": len(records),
        "teacher_record_count": teacher_record_count,
        "base_record_count": max(len(records) - teacher_record_count, 0),
        "avg_room_count": avg_room_count,
        "plot_type_counts": _sorted_dict(plot_type_counts),
        "room_type_counts": _sorted_dict(room_type_counts),
        "room_count_histogram": _sorted_dict(room_count_histogram),
        "teacher_style_counts": _sorted_dict(teacher_style_counts),
        "top_programs": top_program_list,
    }


def main():
    parser = argparse.ArgumentParser(description="Profile planner JSONL records")
    parser.add_argument("--input", required=True, help="Planner JSONL file to profile")
    parser.add_argument("--top-programs", type=int, default=10, help="Number of room-program signatures to include")
    parser.add_argument("--output-json", default="", help="Optional JSON file for the profile summary")
    args = parser.parse_args()

    summary = profile_records(load_records(args.input), top_programs=args.top_programs)
    payload = json.dumps(summary, indent=2)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")

    print(payload)


if __name__ == "__main__":
    main()
