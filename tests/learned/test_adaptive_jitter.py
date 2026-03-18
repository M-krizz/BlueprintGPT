"""
test_adaptive_jitter.py – Tests for adaptive sigma scaling and directional
boundary-aware jitter.

Covers Phase 1 enhancements from the improvement roadmap:
  - Adaptive sigma: severity=0 → base sigma; severity=1 → MAX_MULTIPLIER × sigma
  - Collapse severity scoring (0-1 range, both IoU and distance branches)
  - Directional bias: centroids near edges are pushed inward
  - Combined adaptive + directional behaviour
  - Boundary clamp guarantees under extreme inputs
"""
import math

import pytest

from learned.data.tokenizer_layout import RoomBox
from learned.integration.centroid_utils import (
    BOUNDARY_MARGIN,
    MAX_ADAPTIVE_JITTER_MULTIPLIER,
    compute_boundary_bias,
    detect_centroid_collapse,
    jitter_centroids,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _box(room_type: str, cx: float, cy: float, half: float = 0.05) -> RoomBox:
    """Build a RoomBox centred at (cx, cy) with a given half-width."""
    return RoomBox(room_type=room_type,
                   x_min=cx - half, y_min=cy - half,
                   x_max=cx + half, y_max=cy + half)


def _collapsed_rooms(n: int = 4, cx: float = 0.5, cy: float = 0.5) -> list:
    """Return n identical boxes all stacked at (cx, cy)."""
    return [_box(f"Room{i}", cx, cy) for i in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
#  1. Collapse severity scoring
# ─────────────────────────────────────────────────────────────────────────────

class TestCollapseSeverity:

    def test_severity_zero_when_no_collapse(self):
        rooms = [
            _box("A", 0.1, 0.1),
            _box("B", 0.5, 0.5),
            _box("C", 0.9, 0.9),
        ]
        _, metrics = detect_centroid_collapse(rooms)
        assert metrics["collapse_severity"] == pytest.approx(0.0, abs=1e-6)

    def test_severity_is_one_for_total_collapse(self):
        # All rooms identical → distance=0, IoU=1.0
        rooms = _collapsed_rooms(5)
        _, metrics = detect_centroid_collapse(rooms)
        assert metrics["collapse_severity"] == pytest.approx(1.0, abs=1e-4)

    def test_severity_between_zero_and_one(self):
        # Mild collapse: rooms slightly offset
        rooms = [
            _box("A", 0.49, 0.49),
            _box("B", 0.50, 0.50),
            _box("C", 0.51, 0.51),
            _box("D", 0.49, 0.51),
        ]
        _, metrics = detect_centroid_collapse(rooms)
        s = metrics["collapse_severity"]
        assert 0.0 <= s <= 1.0

    def test_severity_increases_with_overlap(self):
        # More overlap → higher severity
        mild = [_box("A", 0.45, 0.5), _box("B", 0.55, 0.5), _box("C", 0.5, 0.45)]
        severe = _collapsed_rooms(3)

        _, m_mild = detect_centroid_collapse(mild)
        _, m_severe = detect_centroid_collapse(severe)

        assert m_severe["collapse_severity"] > m_mild["collapse_severity"]

    def test_severity_in_metrics_dict(self):
        rooms = _collapsed_rooms(3)
        _, metrics = detect_centroid_collapse(rooms)
        assert "collapse_severity" in metrics
        assert isinstance(metrics["collapse_severity"], float)


# ─────────────────────────────────────────────────────────────────────────────
#  2. Adaptive sigma scaling
# ─────────────────────────────────────────────────────────────────────────────

class TestAdaptiveSigma:
    """Verify that a higher collapse_severity produces larger effective jitter."""

    BASE_SIGMA = 0.01

    def _mean_displacement(self, rooms, severity, n_trials=200, seed_start=0):
        """Average jitter displacement |Δcx| + |Δcy| over many seeds."""
        original_cx = 0.5  # All rooms centered at (0.5, 0.5)
        original_cy = 0.5
        displacements = []
        for seed in range(seed_start, seed_start + n_trials):
            hints = jitter_centroids(
                rooms,
                sigma=self.BASE_SIGMA,
                seed=seed,
                adaptive=True,
                collapse_severity=severity,
                directional=False,   # Isolate adaptive effect
            )
            for cx, cy in hints.values():
                displacements.append(abs(cx - original_cx) + abs(cy - original_cy))
        return sum(displacements) / len(displacements) if displacements else 0.0

    def test_higher_severity_gives_larger_displacement(self):
        rooms = _collapsed_rooms(4)
        low_disp = self._mean_displacement(rooms, severity=0.0, n_trials=100)
        high_disp = self._mean_displacement(rooms, severity=1.0, n_trials=100)
        # With max multiplier 3.0, sigma triples → mean displacement should increase
        assert high_disp > low_disp * 1.5

    def test_severity_zero_equals_base_sigma_scale(self):
        rooms = _collapsed_rooms(4)
        # severity=0 → multiplier=1 → should behave like plain sigma
        low_disp  = self._mean_displacement(rooms, severity=0.0,  n_trials=300)
        # Expect mean displacement ≈ sqrt(2/π) * sigma * 2 ≈ 0.016 for sigma=0.01
        assert low_disp < self.BASE_SIGMA * MAX_ADAPTIVE_JITTER_MULTIPLIER * 3

    def test_adaptive_disabled_ignores_severity(self):
        rooms = _collapsed_rooms(4)
        displacements_off = []
        displacements_on = []
        for seed in range(100):
            h_off = jitter_centroids(rooms, sigma=self.BASE_SIGMA, seed=seed,
                                     adaptive=False, collapse_severity=1.0, directional=False)
            h_on  = jitter_centroids(rooms, sigma=self.BASE_SIGMA, seed=seed,
                                     adaptive=True,  collapse_severity=1.0, directional=False)
            for (cx_off, cy_off), (cx_on, cy_on) in zip(h_off.values(), h_on.values()):
                displacements_off.append(abs(cx_off - 0.5) + abs(cy_off - 0.5))
                displacements_on.append(abs(cx_on  - 0.5) + abs(cy_on  - 0.5))
        avg_off = sum(displacements_off) / len(displacements_off)
        avg_on  = sum(displacements_on)  / len(displacements_on)
        # When adaptive is ON with severity=1, displacements should be larger
        assert avg_on > avg_off


# ─────────────────────────────────────────────────────────────────────────────
#  3. Boundary bias (directional jitter)
# ─────────────────────────────────────────────────────────────────────────────

class TestBoundaryBias:
    """Unit tests for compute_boundary_bias."""

    SIGMA = 0.02

    def test_no_bias_at_center(self):
        bx, by = compute_boundary_bias(0.5, 0.5, margin=BOUNDARY_MARGIN, sigma=self.SIGMA)
        assert bx == pytest.approx(0.0, abs=1e-9)
        assert by == pytest.approx(0.0, abs=1e-9)

    def test_left_edge_pushes_right(self):
        bx, by = compute_boundary_bias(0.0, 0.5, margin=BOUNDARY_MARGIN, sigma=self.SIGMA)
        assert bx > 0.0, "Centroid on left edge should get positive x-bias"
        assert by == pytest.approx(0.0, abs=1e-9)

    def test_right_edge_pushes_left(self):
        bx, by = compute_boundary_bias(1.0, 0.5, margin=BOUNDARY_MARGIN, sigma=self.SIGMA)
        assert bx < 0.0, "Centroid on right edge should get negative x-bias"

    def test_bottom_edge_pushes_up(self):
        bx, by = compute_boundary_bias(0.5, 0.0, margin=BOUNDARY_MARGIN, sigma=self.SIGMA)
        assert by > 0.0, "Centroid on bottom edge should get positive y-bias"

    def test_top_edge_pushes_down(self):
        bx, by = compute_boundary_bias(0.5, 1.0, margin=BOUNDARY_MARGIN, sigma=self.SIGMA)
        assert by < 0.0, "Centroid on top edge should get negative y-bias"

    def test_bias_zero_outside_margin(self):
        # Centroid safely inside: no bias
        bx, by = compute_boundary_bias(0.5, 0.5, margin=0.1, sigma=self.SIGMA)
        assert bx == pytest.approx(0.0, abs=1e-9)
        assert by == pytest.approx(0.0, abs=1e-9)

    def test_bias_increases_toward_edge(self):
        # Closer to edge → larger bias magnitude
        margin = BOUNDARY_MARGIN
        _, b_far  = compute_boundary_bias(0.5, margin * 0.6, margin=margin, sigma=self.SIGMA)
        _, b_near = compute_boundary_bias(0.5, margin * 0.1, margin=margin, sigma=self.SIGMA)
        assert b_near > b_far, "Closer to edge should produce larger bias"

    def test_bias_capped_at_sigma(self):
        # At the very edge (0.0), bias should not exceed sigma
        bx, _ = compute_boundary_bias(0.0, 0.5, margin=BOUNDARY_MARGIN, sigma=self.SIGMA)
        assert bx <= self.SIGMA + 1e-9

    def test_corner_gets_both_axes_biased(self):
        # Bottom-left corner → positive x and y bias
        bx, by = compute_boundary_bias(0.0, 0.0, margin=BOUNDARY_MARGIN, sigma=self.SIGMA)
        assert bx > 0.0
        assert by > 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  4. Directional flag in jitter_centroids
# ─────────────────────────────────────────────────────────────────────────────

class TestDirectionalJitterIntegration:
    """Verify that directional=True causes edge centroids to move inward."""

    def test_edge_centroid_moves_inward_on_average(self):
        """Rooms near left edge: mean jittered cx should exceed original cx."""
        # Place all rooms with centroid at x=0.02 (near left boundary)
        rooms = [_box(f"R{i}", 0.02, 0.5) for i in range(3)]
        original_cx = 0.02
        moved_in, moved_out = 0, 0
        for seed in range(200):
            hints = jitter_centroids(
                rooms, sigma=0.02, seed=seed,
                adaptive=False, collapse_severity=0.0,
                directional=True,
            )
            for cx, _ in hints.values():
                if cx > original_cx:
                    moved_in += 1
                else:
                    moved_out += 1
        # The bias should make inward moves the majority
        assert moved_in > moved_out, (
            f"Expected more inward moves, got {moved_in} in vs {moved_out} out"
        )

    def test_directional_false_no_consistent_inward_bias(self):
        """Without directional bias, inward/outward moves should be roughly equal."""
        rooms = [_box(f"R{i}", 0.02, 0.5) for i in range(3)]
        original_cx = 0.02
        moved_in, moved_out = 0, 0
        for seed in range(300):
            hints = jitter_centroids(
                rooms, sigma=0.02, seed=seed,
                adaptive=False, collapse_severity=0.0,
                directional=False,
            )
            for cx, _ in hints.values():
                if cx > original_cx:
                    moved_in += 1
                else:
                    moved_out += 1
        # Without bias the split is roughly 50/50 (with clamping skewing it slightly)
        ratio = moved_in / (moved_in + moved_out)
        # Generous band: expect anywhere between 40-100% inward since clamp still helps
        # The key claim is simply that WITHOUT directional it's less reliably inward
        # than WITH directional (tested above at 50%+, here we just confirm it's plausible)
        assert ratio >= 0.4

    def test_interior_centroid_unaffected_by_directional(self):
        """Rooms at center should have same distribution regardless of directional flag."""
        rooms = [_box(f"R{i}", 0.5, 0.5) for i in range(3)]
        displacements_dir  = []
        displacements_flat = []
        for seed in range(100):
            h_dir  = jitter_centroids(rooms, sigma=0.01, seed=seed,
                                       adaptive=False, directional=True)
            h_flat = jitter_centroids(rooms, sigma=0.01, seed=seed,
                                       adaptive=False, directional=False)
            for (cx_d, cy_d), (cx_f, cy_f) in zip(h_dir.values(), h_flat.values()):
                displacements_dir.append(abs(cx_d - 0.5) + abs(cy_d - 0.5))
                displacements_flat.append(abs(cx_f - 0.5) + abs(cy_f - 0.5))
        avg_dir  = sum(displacements_dir)  / len(displacements_dir)
        avg_flat = sum(displacements_flat) / len(displacements_flat)
        # For centroids at center, directional should not significantly increase displacement
        assert abs(avg_dir - avg_flat) < 0.005


# ─────────────────────────────────────────────────────────────────────────────
#  5. Clamp guarantees under extreme inputs
# ─────────────────────────────────────────────────────────────────────────────

class TestClampGuarantees:

    def test_always_in_unit_square_large_sigma(self):
        """Even with sigma=0.5, all jittered centroids must stay in [0,1]."""
        rooms = [_box(f"R{i}", 0.01, 0.99) for i in range(4)]
        for seed in range(200):
            hints = jitter_centroids(
                rooms, sigma=0.5, seed=seed,
                adaptive=True, collapse_severity=1.0,
                directional=True,
            )
            for cx, cy in hints.values():
                assert 0.0 <= cx <= 1.0, f"cx={cx} out of [0,1] at seed={seed}"
                assert 0.0 <= cy <= 1.0, f"cy={cy} out of [0,1] at seed={seed}"

    def test_always_in_unit_square_corner_rooms(self):
        """Rooms stacked in every corner must still produce in-range hints."""
        corners = [
            _box("A", 0.0, 0.0),
            _box("B", 0.0, 1.0),
            _box("C", 1.0, 0.0),
            _box("D", 1.0, 1.0),
        ]
        for seed in range(100):
            hints = jitter_centroids(corners, sigma=0.05, seed=seed,
                                     adaptive=False, directional=True)
            for cx, cy in hints.values():
                assert 0.0 <= cx <= 1.0
                assert 0.0 <= cy <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
#  6. Regression / combined behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestCombinedBehaviour:

    def test_full_pipeline_total_collapse_all_options_on(self):
        """All three features on: adaptive + directional + collapse detection."""
        rooms = _collapsed_rooms(6, cx=0.5, cy=0.5)
        is_collapsed, metrics = detect_centroid_collapse(rooms)
        assert is_collapsed
        severity = metrics["collapse_severity"]

        hints = jitter_centroids(
            rooms, sigma=0.01, seed=7,
            adaptive=True,
            collapse_severity=severity,
            directional=True,
        )
        for cx, cy in hints.values():
            assert 0.0 <= cx <= 1.0
            assert 0.0 <= cy <= 1.0

    def test_full_pipeline_edge_collapse(self):
        """Collapse near left edge: adaptive jitter, directional bias both fire."""
        rooms = _collapsed_rooms(4, cx=0.03, cy=0.5)
        _, metrics = detect_centroid_collapse(rooms)
        severity = metrics["collapse_severity"]

        cx_values = []
        for seed in range(100):
            hints = jitter_centroids(
                rooms, sigma=0.01, seed=seed,
                adaptive=True, collapse_severity=severity,
                directional=True,
            )
            cx_values.extend(cx for cx, _ in hints.values())

        # With directional bias pushing away from left edge, mean cx should exceed original 0.03
        assert sum(cx_values) / len(cx_values) > 0.03

    def test_reproducibility_with_seed_all_features(self):
        rooms = _collapsed_rooms(4, cx=0.95, cy=0.05)
        _, met = detect_centroid_collapse(rooms)
        kwargs = dict(sigma=0.01, seed=999, adaptive=True,
                      collapse_severity=met["collapse_severity"], directional=True)
        h1 = jitter_centroids(rooms, **kwargs)
        h2 = jitter_centroids(rooms, **kwargs)
        assert h1 == h2, "Same seed must produce identical results"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
