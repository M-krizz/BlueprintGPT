"""Data preparation utilities for the planner model."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from shapely.geometry import Polygon
from torch.utils.data import Dataset


PLANNER_ROOM_TYPES: List[str] = [
    "Bedroom",
    "Bathroom",
    "Kitchen",
    "LivingRoom",
    "DiningRoom",
    "Garage",
    "Store",
    "Study",
    "Balcony",
    "WC",
]
PLANNER_ROOM_TYPE_TO_ID = {room_type: index + 1 for index, room_type in enumerate(PLANNER_ROOM_TYPES)}
PLANNER_ROOM_ASPECT_RATIOS = {
    "LivingRoom": 1.4,
    "DiningRoom": 1.2,
    "Kitchen": 1.3,
    "Bedroom": 1.25,
    "Bathroom": 0.8,
    "WC": 0.7,
    "Garage": 1.5,
    "Store": 1.0,
    "Study": 1.1,
    "Balcony": 2.0,
}
PLANNER_ROOM_ID_TO_ASPECT = {
    index + 1: float(PLANNER_ROOM_ASPECT_RATIOS.get(room_type, 1.2))
    for index, room_type in enumerate(PLANNER_ROOM_TYPES)
}

PLANNER_PLOT_TYPES: List[str] = ["5Marla", "10Marla", "20Marla", "Custom", "Residential"]
PLANNER_PLOT_TYPE_TO_ID = {plot_type: index + 1 for index, plot_type in enumerate(PLANNER_PLOT_TYPES)}

PLANNER_SKIP_TYPES = {
    "Corridor",
    "Passage",
    "Stairs",
    "Staircase",
    "Lobby",
    "Lounge",
    "Lawn",
    "OpenSpace",
    "SideGarden",
    "DressingArea",
    "PrayerRoom",
    "ServantQuarter",
    "Backyard",
    "Laundry",
    "Unknown",
}

PLANNER_CANONICAL_MAP = {
    "DrawingRoom": "LivingRoom",
    "Hall": "LivingRoom",
    "Lounge": "LivingRoom",
    "Lobby": "LivingRoom",
    "Dining": "DiningRoom",
    "Storage": "Store",
    "Toilet": "WC",
}


def canonicalize_planner_room_type(raw_type: str) -> Optional[str]:
    room_type = PLANNER_CANONICAL_MAP.get(str(raw_type or "").strip(), str(raw_type or "").strip())
    if not room_type or room_type in PLANNER_SKIP_TYPES:
        return None
    return room_type if room_type in PLANNER_ROOM_TYPE_TO_ID else None


def _polygon_area(points: List[List[float]]) -> float:
    polygon = Polygon(points)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    return float(polygon.area)


def _polygon_centroid(points: List[List[float]], bounds: Tuple[float, float, float, float]) -> Tuple[float, float]:
    polygon = Polygon(points)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    cx, cy = polygon.centroid.x, polygon.centroid.y
    min_x, min_y, max_x, max_y = bounds
    width = max(max_x - min_x, 1e-6)
    height = max(max_y - min_y, 1e-6)
    return (
        max(0.0, min(1.0, (cx - min_x) / width)),
        max(0.0, min(1.0, (cy - min_y) / height)),
    )


def _touch_or_near(poly_a: List[List[float]], poly_b: List[List[float]], tolerance: float) -> bool:
    polygon_a = Polygon(poly_a)
    polygon_b = Polygon(poly_b)
    if not polygon_a.is_valid:
        polygon_a = polygon_a.buffer(0)
    if not polygon_b.is_valid:
        polygon_b = polygon_b.buffer(0)
    if polygon_a.touches(polygon_b):
        return True
    return polygon_a.distance(polygon_b) <= tolerance


def build_planner_record(plan: Dict, adjacency_tolerance_ratio: float = 0.015) -> Optional[Dict]:
    """Convert one Kaggle-style plan JSON into a planner-training record."""
    boundary = plan.get("boundary") or plan.get("boundary_polygon")
    rooms = plan.get("rooms", [])
    if not boundary or len(boundary) < 3 or not rooms:
        return None

    boundary_polygon = Polygon(boundary)
    if not boundary_polygon.is_valid:
        boundary_polygon = boundary_polygon.buffer(0)
    if boundary_polygon.is_empty or boundary_polygon.area <= 0:
        return None

    bounds = boundary_polygon.bounds
    min_x, min_y, max_x, max_y = bounds
    boundary_area = float(boundary_polygon.area)
    tolerance = max(max_x - min_x, max_y - min_y) * adjacency_tolerance_ratio

    type_counts: Counter = Counter()
    planner_rooms: List[Dict] = []

    for room in rooms:
        room_type = canonicalize_planner_room_type(room.get("type") or room.get("room_type"))
        polygon = room.get("polygon")
        if room_type is None or not polygon or len(polygon) < 3:
            continue

        type_counts[room_type] += 1
        room_name = f"{room_type}_{type_counts[room_type]}"
        planner_rooms.append(
            {
                "name": room_name,
                "type": room_type,
                "area_ratio": round(_polygon_area(polygon) / max(boundary_area, 1e-6), 6),
                "centroid": [round(v, 6) for v in _polygon_centroid(polygon, bounds)],
                "polygon": polygon,
            }
        )

    if not planner_rooms:
        return None

    adjacency: List[List[str]] = []
    for left_index, left_room in enumerate(planner_rooms):
        for right_room in planner_rooms[left_index + 1:]:
            if _touch_or_near(left_room["polygon"], right_room["polygon"], tolerance):
                adjacency.append([left_room["name"], right_room["name"]])

    return {
        "plan_id": plan.get("plan_id") or plan.get("file_name"),
        "building_type": plan.get("building_type", "Residential"),
        "plot_type": plan.get("plot_type", "Custom"),
        "boundary_bbox": [round(min_x, 3), round(min_y, 3), round(max_x, 3), round(max_y, 3)],
        "rooms": [
            {
                "name": room["name"],
                "type": room["type"],
                "area_ratio": room["area_ratio"],
                "centroid": room["centroid"],
            }
            for room in planner_rooms
        ],
        "adjacency": adjacency,
        "contact_pairs": adjacency,
    }


def iter_plan_files(input_pattern: str) -> Iterable[Path]:
    yield from sorted(Path().glob(input_pattern))


def build_planner_dataset(
    input_pattern: str,
    output_path: str,
    adjacency_tolerance_ratio: float = 0.015,
) -> int:
    """Build a planner-training JSONL dataset from Kaggle-style plan JSON files."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with output.open("w", encoding="utf-8") as handle:
        for path in iter_plan_files(input_pattern):
            try:
                plan = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue

            record = build_planner_record(plan, adjacency_tolerance_ratio=adjacency_tolerance_ratio)
            if record is None:
                continue

            handle.write(json.dumps(record) + "\n")
            written += 1

    return written


class PlannerJsonlDataset(Dataset):
    """Fixed-shape dataset for planner-model training."""

    def __init__(self, path: str, max_rooms: int = 20):
        self.max_rooms = max_rooms
        self.records = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        room_type_ids = torch.zeros(self.max_rooms, dtype=torch.long)
        centroid_targets = torch.zeros(self.max_rooms, 2, dtype=torch.float32)
        area_targets = torch.zeros(self.max_rooms, dtype=torch.float32)
        adjacency_targets = torch.zeros(self.max_rooms, self.max_rooms, dtype=torch.float32)
        contact_targets = torch.zeros(self.max_rooms, self.max_rooms, dtype=torch.float32)
        room_mask = torch.zeros(self.max_rooms, dtype=torch.bool)

        room_names: Dict[str, int] = {}
        for room_index, room in enumerate(record.get("rooms", [])[: self.max_rooms]):
            room_type_ids[room_index] = PLANNER_ROOM_TYPE_TO_ID.get(room["type"], 0)
            centroid_targets[room_index] = torch.tensor(room.get("centroid", [0.5, 0.5]), dtype=torch.float32)
            area_targets[room_index] = float(room.get("area_ratio", 0.0))
            room_mask[room_index] = True
            room_names[room["name"]] = room_index

        for left_name, right_name in record.get("adjacency", []):
            left_index = room_names.get(left_name)
            right_index = room_names.get(right_name)
            if left_index is None or right_index is None:
                continue
            adjacency_targets[left_index, right_index] = 1.0
            adjacency_targets[right_index, left_index] = 1.0

        for left_name, right_name in record.get("contact_pairs", record.get("adjacency", [])):
            left_index = room_names.get(left_name)
            right_index = room_names.get(right_name)
            if left_index is None or right_index is None:
                continue
            contact_targets[left_index, right_index] = 1.0
            contact_targets[right_index, left_index] = 1.0

        plot_type_id = PLANNER_PLOT_TYPE_TO_ID.get(record.get("plot_type", "Custom"), PLANNER_PLOT_TYPE_TO_ID["Custom"])

        return {
            "room_type_ids": room_type_ids,
            "plot_type_id": torch.tensor(plot_type_id, dtype=torch.long),
            "centroid_targets": centroid_targets,
            "area_targets": area_targets,
            "adjacency_targets": adjacency_targets,
            "contact_targets": contact_targets,
            "room_mask": room_mask,
        }
