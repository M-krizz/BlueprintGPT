"""
build_sequences.py – Build training-ready token sequences from FloorPlanSample
JSON files (produced by ``build_coco.build_sample_jsons``).

Usage
-----
    # Tensor output (.pt):
    python -m learned.data.build_sequences \\
        --json-dir learned/data/kaggle_json/ \\
        --output learned/data/sequences.pt

    # JSONL output with 80/20 train-val split:
    python -m learned.data.build_sequences \\
        --source learned/data/kaggle_json/ \\
        --output learned/data/kaggle_train.jsonl \\
        --bins 256 --split 0.8

CLI aliases
-----------
    --source   = --json-dir
    --bins     = --num-bins
    --split    = fraction of data kept for training (rest -> *_val.jsonl)

Output formats
--------------
*.pt    - torch.save dict: {"sequences": LongTensor[N, max_len], "lengths": LongTensor[N]}
*.jsonl - one JSON object per line:
          {"plan_id": str, "tokens": List[int], "num_rooms": int,
           "building_type": str}
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

import torch

from learned.data.tokenizer_layout import (
    LayoutTokenizer,
    RoomBox,
    DEFAULT_NUM_BINS,
)


# =============================================================================
#  Parsing
# =============================================================================

def _rooms_from_sample(sample: dict, img_w: float, img_h: float) -> List[RoomBox]:
    """Convert a FloorPlanSample rooms list into normalised RoomBoxes."""
    rooms: list[RoomBox] = []
    scale_x = 1.0 / max(img_w, 1)
    scale_y = 1.0 / max(img_h, 1)

    for r in sample.get("rooms", []):
        rtype = r.get("type", "Unknown")
        bx = r.get("bbox")
        if bx is None or len(bx) != 4:
            continue
        x_min, y_min, x_max, y_max = bx
        rooms.append(RoomBox(
            room_type=rtype,
            x_min=x_min * scale_x,
            y_min=y_min * scale_y,
            x_max=x_max * scale_x,
            y_max=y_max * scale_y,
        ))
    return rooms


# =============================================================================
#  Core builders
# =============================================================================

def build_sequences(
    json_dir: str | Path,
    *,
    num_bins: int = DEFAULT_NUM_BINS,
    max_len: int = 256,
) -> dict:
    """Tokenize every plan JSON in json_dir and return padded tensor dict.

    Returns dict with:
        sequences : LongTensor [N, max_len]
        lengths   : LongTensor [N]
    """
    tokenizer = LayoutTokenizer(num_bins=num_bins)
    sequences: list[list[int]] = []

    for fpath in sorted(Path(json_dir).glob("*.json")):
        with open(fpath, "r", encoding="utf-8") as fh:
            sample = json.load(fh)

        boundary = sample.get("boundary", [[0, 0], [100, 0], [100, 100], [0, 100]])
        img_w = max(p[0] for p in boundary) - min(p[0] for p in boundary) or 1
        img_h = max(p[1] for p in boundary) - min(p[1] for p in boundary) or 1

        rooms = _rooms_from_sample(sample, img_w, img_h)
        if not rooms:
            continue

        building_type = sample.get("building_type", "Residential")
        seq = tokenizer.encode_sample(rooms, building_type=building_type)

        if len(seq) > max_len:
            continue  # skip extremely large plans

        sequences.append(seq)

    if not sequences:
        raise RuntimeError(f"No valid sequences found in {json_dir}")

    # Pad to uniform length
    padded = torch.zeros(len(sequences), max_len, dtype=torch.long)
    lengths = torch.zeros(len(sequences), dtype=torch.long)
    for i, seq in enumerate(sequences):
        L = min(len(seq), max_len)
        padded[i, :L] = torch.tensor(seq[:L], dtype=torch.long)
        lengths[i] = L

    return {"sequences": padded, "lengths": lengths}


def build_sequences_jsonl(
    json_dir: str | Path,
    *,
    num_bins: int = DEFAULT_NUM_BINS,
    max_len: int = 256,
) -> List[dict]:
    """Tokenize every plan JSON in json_dir and return list of JSONL records.

    Each record: {"plan_id": str, "tokens": List[int], "num_rooms": int,
                  "building_type": str}
    """
    tokenizer = LayoutTokenizer(num_bins=num_bins)
    records: list[dict] = []

    for fpath in sorted(Path(json_dir).glob("*.json")):
        with open(fpath, "r", encoding="utf-8") as fh:
            sample = json.load(fh)

        boundary = sample.get("boundary", [[0, 0], [100, 0], [100, 100], [0, 100]])
        img_w = max(p[0] for p in boundary) - min(p[0] for p in boundary) or 1
        img_h = max(p[1] for p in boundary) - min(p[1] for p in boundary) or 1

        rooms = _rooms_from_sample(sample, img_w, img_h)
        if not rooms:
            continue

        building_type = sample.get("building_type", "Residential")
        seq = tokenizer.encode_sample(rooms, building_type=building_type)

        if len(seq) > max_len:
            continue

        records.append({
            "plan_id":       fpath.stem,
            "tokens":        seq,
            "num_rooms":     len(rooms),
            "building_type": building_type,
        })

    if not records:
        raise RuntimeError(f"No valid sequences found in {json_dir}")

    return records


# =============================================================================
#  CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Build token sequences from plan JSONs")

    # Source directory: primary flag + --source alias
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--json-dir", dest="json_dir",
                     help="Directory of FloorPlanSample JSONs")
    src.add_argument("--source", dest="json_dir",
                     help="Alias for --json-dir")

    ap.add_argument("--output", default="learned/data/sequences.pt",
                    help="Output file (.pt for tensors, .jsonl for text lines)")
    ap.add_argument("--max-len", type=int, default=256)

    # Bins: primary + alias
    ap.add_argument("--num-bins", "--bins", dest="num_bins",
                    type=int, default=DEFAULT_NUM_BINS,
                    help="Coordinate discretisation bins (default %(default)s)")

    ap.add_argument("--split", type=float, default=1.0,
                    help="Train fraction for JSONL output (e.g. 0.8 produces "
                         "80%% train + 20%% *_val.jsonl). Ignored for .pt output.")

    args = ap.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.suffix.lower() == ".jsonl":
        # --- JSONL path -------------------------------------------------------
        records = build_sequences_jsonl(
            args.json_dir, num_bins=args.num_bins, max_len=args.max_len
        )
        random.shuffle(records)

        if args.split < 1.0:
            split_idx  = max(1, int(len(records) * args.split))
            train_recs = records[:split_idx]
            val_recs   = records[split_idx:]
            val_path   = out.with_stem(out.stem + "_val")
        else:
            train_recs = records
            val_recs   = []

        with open(out, "w", encoding="utf-8") as fh:
            for rec in train_recs:
                fh.write(json.dumps(rec) + "\n")
        print(f"Saved {len(train_recs)} train sequences -> {out}")

        if val_recs:
            with open(val_path, "w", encoding="utf-8") as fh:
                for rec in val_recs:
                    fh.write(json.dumps(rec) + "\n")
            print(f"Saved {len(val_recs)} val sequences   -> {val_path}")

    else:
        # --- PT (tensor) path -------------------------------------------------
        data = build_sequences(
            args.json_dir, num_bins=args.num_bins, max_len=args.max_len
        )
        torch.save(data, out)
        print(
            f"Saved {data['sequences'].shape[0]} sequences -> {out}  "
            f"(max_len={args.max_len}, shape={tuple(data['sequences'].shape)})"
        )


if __name__ == "__main__":
    main()
