"""
analyze_training_bounds.py — Analyze boundary dimensions in training data.

Computes statistics to calibrate MinDimProcessor normalization parameters.
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple


def parse_tokens_to_coords(tokens: List[int], B: int = 256) -> List[Tuple[float, float, float, float]]:
    """Parse token sequence to extract room coordinates.

    Format: [BOS, num_rooms, room_types..., SEP, coords..., EOS]
    Coords are: [room_id, x1, y1, x2, y2, SEP, ...]

    Returns list of (x1, y1, x2, y2) tuples in normalized [0, 1] space.
    """
    BOS, EOS, SEP = 1, 2, 3

    try:
        # Find first SEP (after room types)
        sep_idx = tokens.index(SEP)
        num_rooms = tokens[1]

        # Parse coordinates
        coords = []
        i = sep_idx + 1
        while i < len(tokens) - 1:  # Skip EOS
            if tokens[i] == EOS:
                break
            if tokens[i] == SEP:
                i += 1
                continue

            # Expect: room_id, x1, y1, x2, y2
            if i + 4 < len(tokens):
                room_id = tokens[i]
                x1, y1, x2, y2 = tokens[i+1], tokens[i+2], tokens[i+3], tokens[i+4]

                # Normalize to [0, 1]
                x1_norm = x1 / (B - 1)
                y1_norm = y1 / (B - 1)
                x2_norm = x2 / (B - 1)
                y2_norm = y2 / (B - 1)

                coords.append((x1_norm, y1_norm, x2_norm, y2_norm))
                i += 5
            else:
                break

        return coords
    except (ValueError, IndexError):
        return []


def compute_boundary_size(coords: List[Tuple[float, float, float, float]]) -> Tuple[float, float]:
    """Compute bounding box of all rooms (implied boundary)."""
    if not coords:
        return 0.0, 0.0

    all_x = []
    all_y = []
    for x1, y1, x2, y2 in coords:
        all_x.extend([x1, x2])
        all_y.extend([y1, y2])

    width = max(all_x) - min(all_x)
    height = max(all_y) - min(all_y)

    return width, height


def analyze_training_data(jsonl_path: str) -> dict:
    """Analyze training data to extract boundary statistics."""
    path = Path(jsonl_path)

    widths = []
    heights = []
    diagonals = []
    max_dims = []

    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line)
            tokens = data['tokens']

            coords = parse_tokens_to_coords(tokens)
            if coords:
                w, h = compute_boundary_size(coords)
                widths.append(w)
                heights.append(h)
                diagonals.append((w**2 + h**2)**0.5)
                max_dims.append(max(w, h))

    # Convert to numpy for stats
    widths = np.array(widths)
    heights = np.array(heights)
    diagonals = np.array(diagonals)
    max_dims = np.array(max_dims)

    return {
        'count': len(widths),
        'width': {
            'mean': float(np.mean(widths)),
            'std': float(np.std(widths)),
            'median': float(np.median(widths)),
            'p25': float(np.percentile(widths, 25)),
            'p75': float(np.percentile(widths, 75)),
            'p90': float(np.percentile(widths, 90)),
            'min': float(np.min(widths)),
            'max': float(np.max(widths)),
        },
        'height': {
            'mean': float(np.mean(heights)),
            'std': float(np.std(heights)),
            'median': float(np.median(heights)),
            'p25': float(np.percentile(heights, 25)),
            'p75': float(np.percentile(heights, 75)),
            'p90': float(np.percentile(heights, 90)),
            'min': float(np.min(heights)),
            'max': float(np.max(heights)),
        },
        'diagonal': {
            'mean': float(np.mean(diagonals)),
            'std': float(np.std(diagonals)),
            'median': float(np.median(diagonals)),
        },
        'max_dim': {
            'mean': float(np.mean(max_dims)),
            'median': float(np.median(max_dims)),
        }
    }


if __name__ == '__main__':
    train_file = 'learned/data/kaggle_train_expanded.jsonl'
    val_file = 'learned/data/kaggle_val_expanded.jsonl'

    print("Analyzing training data...")
    train_stats = analyze_training_data(train_file)

    print("\n" + "="*60)
    print("TRAINING DATA BOUNDARY STATISTICS")
    print("="*60)
    print(f"\nSamples analyzed: {train_stats['count']}")

    print("\nWidth (normalized [0, 1]):")
    print(f"  Mean:   {train_stats['width']['mean']:.4f}")
    print(f"  Median: {train_stats['width']['median']:.4f}")
    print(f"  Std:    {train_stats['width']['std']:.4f}")
    print(f"  Range:  [{train_stats['width']['min']:.4f}, {train_stats['width']['max']:.4f}]")
    print(f"  P25-P75: [{train_stats['width']['p25']:.4f}, {train_stats['width']['p75']:.4f}]")
    print(f"  P90:    {train_stats['width']['p90']:.4f}")

    print("\nHeight (normalized [0, 1]):")
    print(f"  Mean:   {train_stats['height']['mean']:.4f}")
    print(f"  Median: {train_stats['height']['median']:.4f}")
    print(f"  Std:    {train_stats['height']['std']:.4f}")
    print(f"  Range:  [{train_stats['height']['min']:.4f}, {train_stats['height']['max']:.4f}]")
    print(f"  P25-P75: [{train_stats['height']['p25']:.4f}, {train_stats['height']['p75']:.4f}]")
    print(f"  P90:    {train_stats['height']['p90']:.4f}")

    print("\nDiagonal (normalized [0, 1]):")
    print(f"  Mean:   {train_stats['diagonal']['mean']:.4f}")
    print(f"  Median: {train_stats['diagonal']['median']:.4f}")

    print("\nMax dimension (normalized [0, 1]):")
    print(f"  Mean:   {train_stats['max_dim']['mean']:.4f}")
    print(f"  Median: {train_stats['max_dim']['median']:.4f}")

    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS FOR MinDimProcessor")
    print("="*60)

    # Current heuristic assumes boundary_m = 15.0
    # Room width in meters: room_width_m = (x2 - x1) * boundary_m
    # Room width normalized: room_width_norm = (x2 - x1)
    # So: room_width_m = room_width_norm * boundary_m

    # Typical boundary sizes (assuming 10-15m physical boundaries)
    for boundary_m in [10.0, 12.0, 15.0, 18.0, 20.0]:
        avg_width_m = train_stats['width']['mean'] * boundary_m
        avg_height_m = train_stats['height']['mean'] * boundary_m
        print(f"\nIf boundary = {boundary_m}m:")
        print(f"  Avg layout width:  {avg_width_m:.1f}m")
        print(f"  Avg layout height: {avg_height_m:.1f}m")

    # Validation data
    if Path(val_file).exists():
        print("\n\nAnalyzing validation data...")
        val_stats = analyze_training_data(val_file)
        print(f"\nValidation samples: {val_stats['count']}")
        print(f"Width mean: {val_stats['width']['mean']:.4f} (train: {train_stats['width']['mean']:.4f})")
        print(f"Height mean: {val_stats['height']['mean']:.4f} (train: {train_stats['height']['mean']:.4f})")

    # Save to JSON
    output = {
        'train': train_stats,
    }
    if Path(val_file).exists():
        output['val'] = val_stats

    with open('learned/data/boundary_stats.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\n✓ Saved statistics to: learned/data/boundary_stats.json")
