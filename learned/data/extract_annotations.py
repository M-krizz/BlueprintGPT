"""
extract_annotations.py – Extract per-image room annotations from floor plan PNGs.

Uses colour-segmentation (nearest-colour matching in RGB space) with OpenCV
connected-component labelling to produce per-instance bounding boxes and
polygon contours.

If ``--metadata-dir`` is supplied the Kaggle FloorPlan_Metadata JSON files
are read to enrich each annotation with physical-area data.

Output JSON format (one file per image, consumed by ``build_coco.py``)
----------------------------------------------------------------------
{
  "file_name":    "10Marla_GF_FP_001_V01.png",
  "plan_id":      "10Marla_GF_FP_001_V01",
  "plot_type":    "10Marla",
  "floor":        "GF",
  "image_width":  512,
  "image_height": 927,
  "rooms": [
    {
      "room_type":    "Bedroom",
      "instance_id":  1,
      "bbox":         [x, y, w, h],
      "area_px":      12345,
      "polygon":      [[x1,y1], [x2,y2], ...]
    }
  ]
}

Usage
-----
    # Single image:
    python -m learned.data.extract_annotations \\
        --image path/to/plan.png \\
        --color-map learned/data/color_map.json \\
        --output learned/data/annotations

    # Batch (entire directory):
    python -m learned.data.extract_annotations \\
        --image-dir path/to/dataset/ \\
        --color-map learned/data/color_map.json \\
        --output learned/data/annotations \\
        --metadata-dir path/to/FloorPlan_Metadata/data_JSON
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_COLOR_MAP = Path(__file__).with_name("color_map.json")
DEFAULT_TOLERANCE = 40   # max Euclidean RGB distance to match a colour
MIN_AREA_PX      = 150   # discard connected components smaller than this


# ── Helper: parse plan ID from filename ──────────────────────────────────────

_FNAME_RE = re.compile(
    r"(?P<plot>[^_]+Marla)_(?P<floor>GF|FF|SF|TF|BF)_FP_(?P<num>\d+)_(?P<ver>V\d+)",
    re.IGNORECASE,
)


def _parse_filename(name: str) -> Tuple[str, str, str]:
    """Return (plan_id, plot_type, floor) extracted from *name*."""
    stem = Path(name).stem
    m = _FNAME_RE.search(stem)
    if m:
        plot  = m.group("plot")
        floor = m.group("floor").upper()
        no    = m.group("num")
        ver   = m.group("ver").upper()
        plan_id = f"{plot}_GF_FP_{no}_{ver}"
        return plan_id, plot, floor
    return stem, "Unknown", "GF"


# ── Core: colour-based segmentation ──────────────────────────────────────────

def _load_color_map(path: str | Path) -> Dict[str, np.ndarray]:
    """Load color map JSON → {room_type: np.array([R,G,B], dtype=uint8)}."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return {k: np.array(v, dtype=np.uint8) for k, v in raw.items()}


def _classify_image(
    bgr: np.ndarray,
    color_map: Dict[str, np.ndarray],
    tolerance: int,
) -> np.ndarray:
    """Return int32 label image where label i corresponds to room type i in color_map.
    Pixels that don't match any colour within *tolerance* get label -1.
    """
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).reshape(-1, 3).astype(np.float32)

    types = list(color_map.keys())
    centroids = np.array([color_map[t] for t in types], dtype=np.float32)  # [K, 3]

    # Chunk-wise nearest-centroid to stay under ~200 MB
    CHUNK = 60_000
    label_flat = np.full(h * w, -1, dtype=np.int32)
    for start in range(0, len(rgb), CHUNK):
        chunk = rgb[start : start + CHUNK]                            # [C, 3]
        diffs = chunk[:, None, :] - centroids[None, :, :]             # [C, K, 3]
        dists = np.sqrt((diffs ** 2).sum(axis=-1))                    # [C, K]
        best  = np.argmin(dists, axis=-1)                             # [C]
        min_d = dists[np.arange(len(chunk)), best]
        label_flat[start : start + CHUNK] = np.where(min_d <= tolerance, best, -1)

    return label_flat.reshape(h, w)


def _extract_rooms(
    label_img: np.ndarray,
    type_list: List[str],
    min_area: int = MIN_AREA_PX,
) -> List[dict]:
    """Run connected-component analysis per room type and return annotation dicts."""
    rooms: list[dict] = []
    instance_counter: Dict[str, int] = {}

    for type_idx, rtype in enumerate(type_list):
        mask = (label_img == type_idx).astype(np.uint8)
        if mask.sum() == 0:
            continue

        # Morphological close to bridge thin gaps (1-pixel walls)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            closed, connectivity=4
        )

        for comp_id in range(1, n_labels):  # 0 is background
            area_px = int(stats[comp_id, cv2.CC_STAT_AREA])
            if area_px < min_area:
                continue

            x = int(stats[comp_id, cv2.CC_STAT_LEFT])
            y = int(stats[comp_id, cv2.CC_STAT_TOP])
            cw = int(stats[comp_id, cv2.CC_STAT_WIDTH])
            ch = int(stats[comp_id, cv2.CC_STAT_HEIGHT])

            # Polygon: contour of this component
            comp_mask = (labels == comp_id).astype(np.uint8)
            contours, _ = cv2.findContours(
                comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                # Keep largest contour and simplify
                cnt = max(contours, key=cv2.contourArea)
                epsilon = 0.01 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                polygon = approx.reshape(-1, 2).tolist()
            else:
                polygon = [[x, y], [x + cw, y], [x + cw, y + ch], [x, y + ch]]

            instance_counter[rtype] = instance_counter.get(rtype, 0) + 1
            rooms.append({
                "room_type":   rtype,
                "instance_id": instance_counter[rtype],
                "bbox":        [x, y, cw, ch],
                "area_px":     area_px,
                "polygon":     polygon,
            })

    return rooms


# ── Metadata enrichment ───────────────────────────────────────────────────────

def _load_metadata(meta_dir: Path, plan_id: str) -> Optional[dict]:
    """Try to find and load a matching metadata JSON for *plan_id*."""
    for ext in (".json",):
        p = meta_dir / (plan_id + ext)
        if p.exists():
            with open(p, encoding="utf-8") as fh:
                return json.load(fh)
    return None


def _enrich_with_metadata(annotation: dict, meta: dict) -> None:
    """Add sqft information to each room entry from metadata if available."""
    sqft_map: Dict[str, list] = meta.get("room_instance_areas_sqft", {})
    counters: Dict[str, int] = {}
    for room in annotation["rooms"]:
        rtype = room["room_type"]
        idx   = counters.get(rtype, 0)
        counters[rtype] = idx + 1
        if rtype in sqft_map and idx < len(sqft_map[rtype]):
            room["area_sqft"] = sqft_map[rtype][idx]


# ── Top-level: process one image ─────────────────────────────────────────────

def process_image(
    image_path: str | Path,
    color_map: Dict[str, np.ndarray],
    output_dir: Path,
    metadata_dir: Optional[Path] = None,
    tolerance: int = DEFAULT_TOLERANCE,
    min_area: int = MIN_AREA_PX,
    overwrite: bool = False,
) -> Optional[Path]:
    """Extract annotations from one floor plan image and write JSON.

    Returns the output path, or *None* if the image was skipped.
    """
    image_path = Path(image_path)
    plan_id, plot_type, floor = _parse_filename(image_path.name)

    out_path = output_dir / (plan_id + ".json")
    if out_path.exists() and not overwrite:
        return None  # already done

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        print(f"  [WARN] Cannot read {image_path.name} – skipping")
        return None

    h, w = bgr.shape[:2]

    type_list  = list(color_map.keys())
    label_img  = _classify_image(bgr, color_map, tolerance)
    rooms      = _extract_rooms(label_img, type_list, min_area)

    annotation = {
        "file_name":    image_path.name,
        "plan_id":      plan_id,
        "plot_type":    plot_type,
        "floor":        floor,
        "image_width":  w,
        "image_height": h,
        "rooms":        rooms,
    }

    if metadata_dir is not None:
        meta = _load_metadata(metadata_dir, plan_id)
        if meta:
            _enrich_with_metadata(annotation, meta)

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(annotation, fh, indent=2)

    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Extract room annotations from floor-plan PNG images"
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image",     help="Single PNG to process")
    src.add_argument("--image-dir", help="Directory of PNG images")

    ap.add_argument(
        "--color-map",
        default=str(DEFAULT_COLOR_MAP),
        help="JSON file mapping room types to [R,G,B] colours "
             f"(default: {DEFAULT_COLOR_MAP})",
    )
    ap.add_argument(
        "--output",
        default="learned/data/annotations",
        help="Output directory for annotation JSONs",
    )
    ap.add_argument(
        "--metadata-dir",
        default=None,
        help="Optional directory of Kaggle FloorPlan_Metadata JSONs for sqft enrichment",
    )
    ap.add_argument(
        "--tolerance",
        type=int,
        default=DEFAULT_TOLERANCE,
        help=f"Max Euclidean RGB distance for colour matching (default {DEFAULT_TOLERANCE})",
    )
    ap.add_argument(
        "--min-area",
        type=int,
        default=MIN_AREA_PX,
        help=f"Minimum connected-component area in pixels (default {MIN_AREA_PX})",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-process images even if output JSON already exists",
    )
    args = ap.parse_args()

    output_dir   = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    color_map    = _load_color_map(args.color_map)
    metadata_dir = Path(args.metadata_dir) if args.metadata_dir else None

    if args.image:
        images = [Path(args.image)]
    else:
        img_dir = Path(args.image_dir)
        images  = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.PNG"))

    done = skipped = 0
    for img_path in images:
        result = process_image(
            img_path,
            color_map,
            output_dir,
            metadata_dir=metadata_dir,
            tolerance=args.tolerance,
            min_area=args.min_area,
            overwrite=args.overwrite,
        )
        if result is not None:
            done += 1
            if done % 20 == 0:
                print(f"  processed {done}/{len(images)} …")
        else:
            skipped += 1

    print(
        f"Done.  {done} annotation(s) written to {output_dir}  "
        f"({skipped} skipped/already present)"
    )


if __name__ == "__main__":
    main()
