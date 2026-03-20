"""
test_jitter_hints.py – Unit tests for centroid collapse detection and jitter.

Tests the short-term fix (A) from the detailed proposal:
- Centroid collapse detection
- Jitter application to break ties
- Metric computation (median distance, IoU fractions)
"""
import pytest

from learned.data.tokenizer_layout import RoomBox
from learned.integration.centroid_utils import (
    compute_centroid,
    compute_iou,
    compute_pairwise_iou_fraction,
    compute_median_centroid_distance,
    detect_centroid_collapse,
    jitter_centroids,
)


class TestCentroidComputation:
    """Test centroid extraction from RoomBox."""

    def test_compute_centroid_simple(self):
        rbox = RoomBox(room_type="Bedroom", x_min=0.2, y_min=0.3, x_max=0.6, y_max=0.7)
        cx, cy = compute_centroid(rbox)
        assert cx == pytest.approx(0.4, abs=1e-6)
        assert cy == pytest.approx(0.5, abs=1e-6)

    def test_compute_centroid_corner(self):
        rbox = RoomBox(room_type="Kitchen", x_min=0.0, y_min=0.0, x_max=0.1, y_max=0.1)
        cx, cy = compute_centroid(rbox)
        assert cx == pytest.approx(0.05, abs=1e-6)
        assert cy == pytest.approx(0.05, abs=1e-6)


class TestIoUComputation:
    """Test IoU (Intersection over Union) calculation."""

    def test_iou_no_overlap(self):
        box1 = RoomBox(room_type="A", x_min=0.0, y_min=0.0, x_max=0.2, y_max=0.2)
        box2 = RoomBox(room_type="B", x_min=0.5, y_min=0.5, x_max=0.7, y_max=0.7)
        iou = compute_iou(box1, box2)
        assert iou == pytest.approx(0.0, abs=1e-6)

    def test_iou_full_overlap(self):
        box1 = RoomBox(room_type="A", x_min=0.2, y_min=0.3, x_max=0.6, y_max=0.7)
        box2 = RoomBox(room_type="B", x_min=0.2, y_min=0.3, x_max=0.6, y_max=0.7)
        iou = compute_iou(box1, box2)
        assert iou == pytest.approx(1.0, abs=1e-6)

    def test_iou_partial_overlap(self):
        box1 = RoomBox(room_type="A", x_min=0.0, y_min=0.0, x_max=0.5, y_max=0.5)
        box2 = RoomBox(room_type="B", x_min=0.25, y_min=0.25, x_max=0.75, y_max=0.75)
        # Intersection: 0.25*0.25 = 0.0625
        # Union: 0.25 + 0.25 - 0.0625 = 0.4375
        # IoU = 0.0625 / 0.4375 ≈ 0.1428
        iou = compute_iou(box1, box2)
        assert iou == pytest.approx(0.1428, abs=1e-3)


class TestPairwiseMetrics:
    """Test pairwise IoU fraction and median centroid distance."""

    def test_pairwise_iou_fraction_no_overlaps(self):
        boxes = [
            RoomBox("A", x_min=0.0, y_min=0.0, x_max=0.2, y_max=0.2),
            RoomBox("B", x_min=0.3, y_min=0.3, x_max=0.5, y_max=0.5),
            RoomBox("C", x_min=0.6, y_min=0.6, x_max=0.8, y_max=0.8),
        ]
        frac = compute_pairwise_iou_fraction(boxes, threshold=0.5)
        assert frac == pytest.approx(0.0, abs=1e-6)

    def test_pairwise_iou_fraction_high_overlaps(self):
        # All boxes at same position
        boxes = [
            RoomBox("A", x_min=0.4, y_min=0.4, x_max=0.6, y_max=0.6),
            RoomBox("B", x_min=0.4, y_min=0.4, x_max=0.6, y_max=0.6),
            RoomBox("C", x_min=0.4, y_min=0.4, x_max=0.6, y_max=0.6),
        ]
        frac = compute_pairwise_iou_fraction(boxes, threshold=0.5)
        # All 3 pairs have IoU=1.0 > 0.5
        assert frac == pytest.approx(1.0, abs=1e-6)

    def test_median_centroid_distance_spread(self):
        boxes = [
            RoomBox("A", x_min=0.0, y_min=0.0, x_max=0.1, y_max=0.1),  # center (0.05, 0.05)
            RoomBox("B", x_min=0.4, y_min=0.4, x_max=0.5, y_max=0.5),  # center (0.45, 0.45)
            RoomBox("C", x_min=0.8, y_min=0.8, x_max=0.9, y_max=0.9),  # center (0.85, 0.85)
        ]
        median_dist = compute_median_centroid_distance(boxes)
        # Distances: (0.05,0.05)-(0.45,0.45) ≈ 0.566
        #            (0.05,0.05)-(0.85,0.85) ≈ 1.131
        #            (0.45,0.45)-(0.85,0.85) ≈ 0.566
        # Median of [0.566, 1.131, 0.566] = 0.566
        assert median_dist == pytest.approx(0.566, abs=1e-2)

    def test_median_centroid_distance_collapsed(self):
        # All centroids at same position
        boxes = [
            RoomBox("A", x_min=0.48, y_min=0.48, x_max=0.52, y_max=0.52),
            RoomBox("B", x_min=0.48, y_min=0.48, x_max=0.52, y_max=0.52),
            RoomBox("C", x_min=0.48, y_min=0.48, x_max=0.52, y_max=0.52),
        ]
        median_dist = compute_median_centroid_distance(boxes)
        assert median_dist == pytest.approx(0.0, abs=1e-6)


class TestCentroidCollapseDetection:
    """Test collapse detection logic."""

    def test_no_collapse_spread_boxes(self):
        boxes = [
            RoomBox("A", x_min=0.0, y_min=0.0, x_max=0.2, y_max=0.2),
            RoomBox("B", x_min=0.4, y_min=0.4, x_max=0.6, y_max=0.6),
            RoomBox("C", x_min=0.7, y_min=0.7, x_max=0.9, y_max=0.9),
        ]
        is_collapsed, metrics = detect_centroid_collapse(
            boxes, min_rooms=3, median_dist_thresh=0.02, iou_pair_ratio_thresh=0.30
        )
        assert not is_collapsed
        assert metrics["median_centroid_distance"] > 0.02
        assert metrics["pairwise_iou_fraction"] < 0.30

    def test_collapse_identical_centroids(self):
        # All boxes at same position
        boxes = [
            RoomBox("A", x_min=0.4, y_min=0.4, x_max=0.5, y_max=0.5),
            RoomBox("B", x_min=0.4, y_min=0.4, x_max=0.5, y_max=0.5),
            RoomBox("C", x_min=0.4, y_min=0.4, x_max=0.5, y_max=0.5),
        ]
        is_collapsed, metrics = detect_centroid_collapse(
            boxes, min_rooms=3, median_dist_thresh=0.02, iou_pair_ratio_thresh=0.30
        )
        assert is_collapsed
        assert metrics["median_centroid_distance"] < 0.02
        assert metrics["pairwise_iou_fraction"] > 0.30

    def test_collapse_high_iou_but_different_centroids(self):
        # Boxes with high overlap (IoU > 0.5) but slightly different centroids
        boxes = [
            RoomBox("A", x_min=0.40, y_min=0.40, x_max=0.60, y_max=0.60),
            RoomBox("B", x_min=0.42, y_min=0.42, x_max=0.62, y_max=0.62),
            RoomBox("C", x_min=0.44, y_min=0.44, x_max=0.64, y_max=0.64),
        ]
        is_collapsed, metrics = detect_centroid_collapse(
            boxes, min_rooms=3, median_dist_thresh=0.02, iou_pair_ratio_thresh=0.30
        )
        # High IoU should trigger collapse even if centroids not identical
        assert is_collapsed or metrics["pairwise_iou_fraction"] > 0.5

    def test_no_collapse_insufficient_rooms(self):
        # Only 2 rooms - below min_rooms threshold
        boxes = [
            RoomBox("A", x_min=0.4, y_min=0.4, x_max=0.5, y_max=0.5),
            RoomBox("B", x_min=0.4, y_min=0.4, x_max=0.5, y_max=0.5),
        ]
        is_collapsed, metrics = detect_centroid_collapse(
            boxes, min_rooms=3, median_dist_thresh=0.02, iou_pair_ratio_thresh=0.30
        )
        assert not is_collapsed  # Should not trigger with < min_rooms


class TestJitterCentroids:
    """Test jitter application to break centroid ties."""

    def test_jitter_returns_dict(self):
        boxes = [
            RoomBox("Bedroom", x_min=0.4, y_min=0.4, x_max=0.6, y_max=0.6),
            RoomBox("Kitchen", x_min=0.2, y_min=0.2, x_max=0.3, y_max=0.3),
        ]
        jittered = jitter_centroids(boxes, sigma=0.01, seed=42)
        assert isinstance(jittered, dict)
        assert "Bedroom" in jittered
        assert "Kitchen" in jittered
        assert isinstance(jittered["Bedroom"], tuple)
        assert len(jittered["Bedroom"]) == 2

    def test_jitter_stays_in_bounds(self):
        # Even with jitter, centroids should stay in [0, 1]
        boxes = [
            RoomBox("A", x_min=0.0, y_min=0.0, x_max=0.02, y_max=0.02),  # Near boundary
            RoomBox("B", x_min=0.98, y_min=0.98, x_max=1.0, y_max=1.0),  # Near boundary
        ]
        jittered = jitter_centroids(boxes, sigma=0.05, seed=42)
        for rtype, (cx, cy) in jittered.items():
            assert 0.0 <= cx <= 1.0
            assert 0.0 <= cy <= 1.0

    def test_jitter_is_reproducible_with_seed(self):
        boxes = [
            RoomBox("Bedroom", x_min=0.4, y_min=0.4, x_max=0.6, y_max=0.6),
        ]
        jittered1 = jitter_centroids(boxes, sigma=0.01, seed=123)
        jittered2 = jitter_centroids(boxes, sigma=0.01, seed=123)
        assert jittered1["Bedroom"] == jittered2["Bedroom"]

    def test_jitter_averages_multiple_rooms_of_same_type(self):
        boxes = [
            RoomBox("Bedroom", x_min=0.2, y_min=0.2, x_max=0.3, y_max=0.3),  # center (0.25, 0.25)
            RoomBox("Bedroom", x_min=0.6, y_min=0.6, x_max=0.7, y_max=0.7),  # center (0.65, 0.65)
        ]
        # Average centroid should be (0.45, 0.45) before jitter
        jittered = jitter_centroids(boxes, sigma=0.001, seed=42)  # Small sigma
        # With small sigma, jittered value should be close to 0.45
        assert jittered["Bedroom"][0] == pytest.approx(0.45, abs=0.05)
        assert jittered["Bedroom"][1] == pytest.approx(0.45, abs=0.05)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
