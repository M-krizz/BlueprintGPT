"""
build_coco.py – Convert per-image annotation JSONs (from extract_annotations.py)
into a single COCO-format JSON file consumable by ``dataset_parser.parse_coco_json``.

Also emits a parallel directory of ``FloorPlanSample``-compatible JSON files
that ``dataset_parser.parse_json_directory`` can read directly.

Usage
-----
    python -m learned.data.build_coco \\
        --annotations learned/data/annotations/ \\
        --output-coco learned/data/kaggle_coco.json \\
        --output-json-dir learned/data/kaggle_json/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Room-type mapping:  dataset names  →  model canonical names
# ---------------------------------------------------------------------------
# The Kaggle dataset uses room-type names that don't all match the
# LayoutTokenizer vocabulary.  We normalise here.
ROOM_TYPE_MAP: Dict[str, str] = {
    "Bedroom": "Bedroom",
    "Bathroom": "Bathroom",
    "Kitchen": "Kitchen",
    "DrawingRoom": "DrawingRoom",
    "Garage": "Garage",
    "Lounge": "Lounge",
    "Lobby": "Lobby",
    "Passage": "Passage",
    "Stairs": "Stairs",
    "Lawn": "Lawn",
    "OpenSpace": "OpenSpace",
    "Staircase": "Staircase",
    "SideGarden": "SideGarden",
    "Dining": "Dining",
    "DressingArea": "DressingArea",
    "Store": "Store",
    "PrayerRoom": "PrayerRoom",
    "ServantQuarter": "ServantQuarter",
    "Backyard": "Backyard",
    "Laundry": "Laundry",
}

# Reverse map for category IDs
ALL_ROOM_TYPES = sorted(set(ROOM_TYPE_MAP.values()))


def _bbox_xywh(rooms_entry: dict) -> Tuple[float, float, float, float]:
    """Return COCO-style [x, y, w, h] from the extract_annotations bbox."""
    return tuple(rooms_entry["bbox"])  # already [x, y, w, h]


# ---------------------------------------------------------------------------
# Build COCO JSON
# ---------------------------------------------------------------------------

def build_coco(annotation_dir: Path) -> dict:
    """Read all annotation JSONs from *annotation_dir* and build COCO dict."""
    categories = [
        {"id": i + 1, "name": rtype}
        for i, rtype in enumerate(ALL_ROOM_TYPES)
    ]
    cat_name2id: dict[str, int] = {c["name"]: c["id"] for c in categories}

    images: list[dict] = []
    annotations: list[dict] = []
    ann_id = 0

    for idx, fpath in enumerate(sorted(annotation_dir.glob("*.json")), start=1):
        with open(fpath, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        img_id = idx
        images.append({
            "id": img_id,
            "file_name": data["file_name"],
            "width": data["image_width"],
            "height": data["image_height"],
            "plot_type": data.get("plot_type", "Unknown"),
            "floor": data.get("floor", "GF"),
        })

        for room in data.get("rooms", []):
            rtype_raw = room["room_type"]
            rtype = ROOM_TYPE_MAP.get(rtype_raw, rtype_raw)
            cat_id = cat_name2id.get(rtype)
            if cat_id is None:
                continue  # unknown type – skip
            ann_id += 1
            x, y, w, h = _bbox_xywh(room)
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [x, y, w, h],
                "area": room.get("area_px", w * h),
                "segmentation": [room["polygon"]] if "polygon" in room else [],
                "iscrowd": 0,
            })

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


# ---------------------------------------------------------------------------
# Build per-plan JSON (FloorPlanSample-compatible)
# ---------------------------------------------------------------------------

def build_sample_jsons(annotation_dir: Path, output_dir: Path) -> int:
    """Convert per-image annotations into FloorPlanSample-style JSONs.

    Returns the number of files written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for fpath in sorted(annotation_dir.glob("*.json")):
        with open(fpath, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        w = data["image_width"]
        h = data["image_height"]

        # Boundary = full image rectangle
        boundary = [[0, 0], [w, 0], [w, h], [0, h]]

        rooms: list[dict] = []
        for room in data.get("rooms", []):
            rtype_raw = room["room_type"]
            rtype = ROOM_TYPE_MAP.get(rtype_raw, rtype_raw)
            bx, by, bw, bh = room["bbox"]
            rooms.append({
                "type": rtype,
                "bbox": [bx, by, bx + bw, by + bh],  # [x_min, y_min, x_max, y_max]
                "polygon": room.get("polygon", []),
            })

        sample = {
            "building_type": "Residential",
            "plot_type": data.get("plot_type", "Unknown"),
            "boundary": boundary,
            "rooms": rooms,
        }

        out_file = output_dir / (fpath.stem + ".json")
        with open(out_file, "w", encoding="utf-8") as fh:
            json.dump(sample, fh, indent=2)
        count += 1

    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build COCO / FloorPlanSample JSONs from extracted annotations"
    )
    parser.add_argument("--annotations", required=True,
                        help="Directory of per-image annotation JSONs")
    parser.add_argument("--output-coco", default="learned/data/kaggle_coco.json",
                        help="Output path for COCO JSON")
    parser.add_argument("--output-json-dir", default="learned/data/kaggle_json",
                        help="Output directory for FloorPlanSample JSONs")
    args = parser.parse_args()

    ann_dir = Path(args.annotations)

    # COCO
    coco = build_coco(ann_dir)
    coco_path = Path(args.output_coco)
    coco_path.parent.mkdir(parents=True, exist_ok=True)
    with open(coco_path, "w", encoding="utf-8") as fh:
        json.dump(coco, fh, indent=2)
    n_imgs = len(coco["images"])
    n_anns = len(coco["annotations"])
    print(f"COCO  → {coco_path}  ({n_imgs} images, {n_anns} annotations)")

    # Per-plan JSONs
    n = build_sample_jsons(ann_dir, Path(args.output_json_dir))
    print(f"JSON  → {args.output_json_dir}/  ({n} files)")


if __name__ == "__main__":
    main()
