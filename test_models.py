"""
Test and compare original vs improved model layouts.

Usage:
    python test_models.py
"""
import os
import sys
from learned.model.sample import load_model, constrained_sample_layout
from learned.data.tokenizer_layout import LayoutTokenizer

def test_model(checkpoint_path, name):
    """Generate a sample layout and report statistics."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Checkpoint: {checkpoint_path}")
    print('='*60)

    # Load model
    model, tokenizer = load_model(checkpoint_path, device='cpu')

    # Test spec: 3BHK (3 bedrooms, 2 bathrooms, 1 kitchen, 1 living room)
    spec = {
        "rooms": [
            {"type": "Bedroom", "count": 3},
            {"type": "Bathroom", "count": 2},
            {"type": "Kitchen", "count": 1},
            {"type": "LivingRoom", "count": 1},
        ]
    }

    # Generate 5 samples
    results = []
    for i in range(5):
        rooms = constrained_sample_layout(
            model, tokenizer,
            spec=spec,
            building_type="Residential",
            temperature=0.85,
            device='cpu'
        )

        if not rooms:
            print(f"  Sample {i+1}: FAILED (empty)")
            continue

        # Calculate stats
        x_coords = [r.x_min for r in rooms] + [r.x_max for r in rooms]
        y_coords = [r.y_min for r in rooms] + [r.y_max for r in rooms]

        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        coverage = x_range * y_range

        # Check overlap
        overlaps = 0
        for i, r1 in enumerate(rooms):
            for r2 in rooms[i+1:]:
                # Check intersection
                ix1 = max(r1.x_min, r2.x_min)
                iy1 = max(r1.y_min, r2.y_min)
                ix2 = min(r1.x_max, r2.x_max)
                iy2 = min(r1.y_max, r2.y_max)
                if ix1 < ix2 and iy1 < iy2:
                    overlaps += 1

        results.append({
            'rooms': len(rooms),
            'coverage': coverage,
            'overlaps': overlaps,
            'x_range': x_range,
            'y_range': y_range,
        })

        print(f"  Sample {i+1}: {len(rooms)} rooms, coverage={coverage:.2f}, overlaps={overlaps}, spread=({x_range:.2f}, {y_range:.2f})")

    # Summary
    if results:
        avg_coverage = sum(r['coverage'] for r in results) / len(results)
        avg_overlaps = sum(r['overlaps'] for r in results) / len(results)
        avg_x = sum(r['x_range'] for r in results) / len(results)
        avg_y = sum(r['y_range'] for r in results) / len(results)

        print(f"\n  AVERAGE: coverage={avg_coverage:.2f}, overlaps={avg_overlaps:.1f}, spread=({avg_x:.2f}, {avg_y:.2f})")
        return {
            'name': name,
            'avg_coverage': avg_coverage,
            'avg_overlaps': avg_overlaps,
            'avg_x_range': avg_x,
            'avg_y_range': avg_y,
        }

    return None

if __name__ == "__main__":
    print("Model Comparison Test")
    print("=" * 60)

    # Test original model
    original = test_model("learned/model/checkpoints/kaggle_test.pt", "Original Model")

    # Test improved model
    improved = test_model("learned/model/checkpoints/improved_v1.pt", "Improved Model")

    # Comparison
    if original and improved:
        print(f"\n{'='*60}")
        print("COMPARISON")
        print('='*60)
        print(f"Coverage:    {improved['avg_coverage']:.2f} vs {original['avg_coverage']:.2f} ({improved['avg_coverage']/original['avg_coverage']:+.1%})")
        print(f"Overlaps:    {improved['avg_overlaps']:.1f} vs {original['avg_overlaps']:.1f} ({(improved['avg_overlaps']-original['avg_overlaps']):.1f})")
        print(f"X-Spread:    {improved['avg_x_range']:.2f} vs {original['avg_x_range']:.2f} ({improved['avg_x_range']/original['avg_x_range']:+.1%})")
        print(f"Y-Spread:    {improved['avg_y_range']:.2f} vs {original['avg_y_range']:.2f} ({improved['avg_y_range']/original['avg_y_range']:+.1%})")
        print()
        print("Better coverage & spread = rooms fill the plot better")
        print("Fewer overlaps = cleaner layouts")
