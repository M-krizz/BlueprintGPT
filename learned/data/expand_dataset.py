"""
expand_dataset.py - Expand training data using deterministic augmentations.

Takes the existing training JSONL and creates an expanded version with
all 8 transformation variants (original + flips + rotations).

Usage:
    python -m learned.data.expand_dataset \\
        --input learned/data/kaggle_train.jsonl \\
        --output learned/data/kaggle_train_expanded.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from learned.data.tokenizer_layout import LayoutTokenizer, RoomBox
from learned.data.augmentation import deterministic_augment, jitter_coordinates


def expand_dataset(
    input_path: str,
    output_path: str,
    add_jitter_variants: bool = True,
    num_jitter_variants: int = 2,
    verbose: bool = True
) -> int:
    """Expand a JSONL dataset with deterministic augmentations.

    Args:
        input_path: Path to input .jsonl file
        output_path: Path to output expanded .jsonl file
        add_jitter_variants: Also add jittered versions
        num_jitter_variants: Number of jitter variants per augmentation

    Returns:
        Number of samples written
    """
    tokenizer = LayoutTokenizer()
    records = []

    # Load input
    with open(input_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if verbose:
        print(f"[EXPAND] Loaded {len(records)} records from {input_path}")

    expanded = []

    for i, rec in enumerate(records):
        tokens = rec["tokens"]
        plan_id = rec.get("plan_id", f"plan_{i}")
        building_type = rec.get("building_type", "Residential")
        num_rooms = rec.get("num_rooms", 0)

        # Decode tokens to rooms
        rooms = tokenizer.decode_rooms(tokens)
        if not rooms:
            # Can't augment, keep original
            expanded.append(rec)
            continue

        # Get building type from tokens
        decoded_btype = tokenizer.decode_building_type(tokens)

        # Generate all 8 deterministic variants
        variants = deterministic_augment(rooms)

        for aug_rooms, transform_name in variants:
            # Re-encode
            aug_tokens = tokenizer.encode_sample(aug_rooms, decoded_btype)

            new_rec = {
                "plan_id": f"{plan_id}_{transform_name}",
                "tokens": aug_tokens,
                "num_rooms": len(aug_rooms),
                "building_type": building_type,
                "transform": transform_name,
            }
            expanded.append(new_rec)

            # Add jittered variants
            if add_jitter_variants:
                for j in range(num_jitter_variants):
                    jittered = jitter_coordinates(aug_rooms, sigma=0.01, clamp=0.025)
                    jit_tokens = tokenizer.encode_sample(jittered, decoded_btype)
                    jit_rec = {
                        "plan_id": f"{plan_id}_{transform_name}_jitter{j}",
                        "tokens": jit_tokens,
                        "num_rooms": len(jittered),
                        "building_type": building_type,
                        "transform": f"{transform_name}_jitter{j}",
                    }
                    expanded.append(jit_rec)

        if verbose and (i + 1) % 50 == 0:
            print(f"[EXPAND] Processed {i + 1}/{len(records)} records...")

    # Write output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for rec in expanded:
            fh.write(json.dumps(rec) + "\n")

    if verbose:
        base_variants = 8  # deterministic transforms
        jitter_mult = 1 + num_jitter_variants if add_jitter_variants else 1
        expected = len(records) * base_variants * jitter_mult
        print(f"[EXPAND] Wrote {len(expanded)} records to {output_path}")
        print(f"[EXPAND] Expansion factor: {len(expanded) / len(records):.1f}x")

    return len(expanded)


def main():
    ap = argparse.ArgumentParser(description="Expand dataset with augmentations")
    ap.add_argument("--input", "-i", required=True, help="Input .jsonl file")
    ap.add_argument("--output", "-o", required=True, help="Output .jsonl file")
    ap.add_argument("--no-jitter", action="store_true", help="Skip jitter variants")
    ap.add_argument("--jitter-variants", type=int, default=2,
                    help="Number of jitter variants per augmentation")
    args = ap.parse_args()

    expand_dataset(
        args.input,
        args.output,
        add_jitter_variants=not args.no_jitter,
        num_jitter_variants=args.jitter_variants,
    )


if __name__ == "__main__":
    main()
