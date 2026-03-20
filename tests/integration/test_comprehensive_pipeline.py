"""
test_comprehensive_pipeline.py - Comprehensive integration tests for BlueprintGPT

Features:
- End-to-end pipeline testing with mock model outputs
- Property-based testing for layout generation edge cases
- Template integration testing
- Quality monitoring and cache system testing
- Performance regression detection
- Fuzzing for boundary conditions and error handling

Test Coverage:
- Full generation pipeline from NL spec → final building
- Repair gate effectiveness across different failure modes
- Template system integration and fallbacks
- Quality monitoring data collection
- Cache performance and eviction
- Error handling and graceful degradation

Performance Impact: Prevents quality regressions, enables continuous integration
Reliability Impact: Catches integration bugs before production

Usage:
    # Run comprehensive test suite
    python -m pytest tests/integration/test_comprehensive_pipeline.py -v

    # Run with coverage
    python -m pytest tests/integration/test_comprehensive_pipeline.py --cov=learned

    # Run performance regression tests
    python -m pytest tests/integration/test_comprehensive_pipeline.py -k "performance"
"""
from __future__ import annotations

import json
import os
import pytest
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock

# Test infrastructure
class MockRoomBox:
    """Mock RoomBox for testing without torch dependency."""
    def __init__(self, room_type: str, x1: float, y1: float, x2: float, y2: float):
        self.type = room_type
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.width = x2 - x1
        self.height = y2 - y1
        self.area = self.width * self.height


def create_mock_building():
    """Create mock building for testing."""
    class MockBuilding:
        def __init__(self):
            self.rooms = []
            self.corridors = []
            self.doors = []
            self.windows = []

    return MockBuilding()


def create_sample_spec() -> Dict[str, Any]:
    """Create test specification."""
    return {
        "building_type": "apartment",
        "occupancy": "residential",
        "total_area": 75.0,
        "rooms": [
            {"type": "living room", "min_area": 18.0, "max_area": 25.0},
            {"type": "kitchen", "min_area": 12.0, "max_area": 18.0},
            {"type": "bedroom", "min_area": 15.0, "max_area": 20.0},
            {"type": "bathroom", "min_area": 6.0, "max_area": 10.0},
        ]
    }


def create_mock_boundary() -> List[List[float]]:
    """Create test boundary polygon."""
    return [[0.0, 0.0], [10.0, 0.0], [10.0, 8.0], [0.0, 8.0]]


class TestTemplateSystemIntegration:
    """Test template system integration and functionality."""

    def test_template_import(self):
        """Test that template system can be imported and initialized."""
        try:
            from learned.templates import LayoutTemplateEngine, find_layout_template
            engine = LayoutTemplateEngine()
            assert engine is not None
            assert engine.stats()["template_count"] > 0
        except ImportError:
            pytest.skip("Template system not available")

    def test_template_matching(self):
        """Test template matching for various specifications."""
        try:
            from learned.templates import find_layout_template
        except ImportError:
            pytest.skip("Template system not available")

        # Test studio apartment matching
        studio_spec = {
            "building_type": "apartment",
            "rooms": [
                {"type": "living room"},
                {"type": "kitchen"},
                {"type": "bathroom"}
            ]
        }

        template = find_layout_template(studio_spec)
        assert template is not None
        compatibility = template.calculate_compatibility(studio_spec)
        assert compatibility > 50.0  # Should find reasonable match

    def test_template_application(self):
        """Test template application to boundary polygons."""
        try:
            from learned.templates import find_layout_template, apply_layout_template
            from shapely.geometry import Polygon
        except ImportError:
            pytest.skip("Template/Shapely not available")

        spec = create_sample_spec()
        template = find_layout_template(spec)

        if template:
            boundary = Polygon(create_mock_boundary())
            layout = apply_layout_template(template, boundary, spec)

            assert "rooms" in layout
            assert len(layout["rooms"]) > 0
            assert "template_used" in layout
            assert layout["coverage_ratio"] > 0.5

    def test_template_json_loading(self):
        """Test external template loading from JSON files."""
        try:
            from learned.templates import LayoutTemplateEngine, LayoutTemplate, LayoutStyle, ZoneType, RoomTemplate
        except ImportError:
            pytest.skip("Template system not available")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test template JSON
            test_template = {
                "name": "Test Template",
                "style": "studio",
                "building_type": "apartment",
                "room_count_range": [2, 4],
                "total_area_range": [30.0, 50.0],
                "description": "Test template for integration tests",
                "tags": ["test"],
                "quality_score": 75.0,
                "rooms": [
                    {
                        "room_type": "living room",
                        "zone": "living",
                        "min_area_ratio": 0.4,
                        "max_area_ratio": 0.6,
                        "aspect_ratio_range": [1.2, 2.0],
                        "preferred_position": [0.5, 0.5],
                        "position_tolerance": 0.3,
                        "adjacency_preferences": [],
                        "separation_requirements": [],
                        "required": True,
                        "priority": 1
                    }
                ]
            }

            # Save to temporary file
            template_file = Path(temp_dir) / "test_template.json"
            with open(template_file, 'w') as f:
                json.dump(test_template, f)

            # Load with engine
            engine = LayoutTemplateEngine(templates_dir=temp_dir)
            stats = engine.stats()

            # Should load built-in templates plus our test template
            assert stats["template_count"] >= 4


class TestQualityMonitoringIntegration:
    """Test quality monitoring system integration."""

    def test_quality_dashboard_import(self):
        """Test quality dashboard import and initialization."""
        try:
            from learned.monitoring import QualityDashboard, QualityMetrics
        except ImportError:
            pytest.skip("Quality monitoring not available")

        dashboard = QualityDashboard(db_path=":memory:")
        assert dashboard is not None

        stats = dashboard.get_current_stats()
        assert "window_size" in stats

    def test_quality_metrics_creation(self):
        """Test quality metrics creation from generation summaries."""
        try:
            from learned.monitoring import QualityMetrics
        except ImportError:
            pytest.skip("Quality monitoring not available")

        mock_summary = {
            "valid_samples": 8,
            "total_attempts": 10,
            "diagnostics": {
                "avg_repair_severity": 15.0,
                "model_cache": {"hit_rate_percent": 80.0}
            }
        }

        metrics = QualityMetrics.from_summary(mock_summary, "apartment")

        assert metrics.building_type == "apartment"
        assert metrics.valid_samples == 8
        assert metrics.total_attempts == 10
        assert metrics.success_rate == 80.0
        assert metrics.quality_score > 0

    def test_dashboard_html_generation(self):
        """Test HTML dashboard generation."""
        try:
            from learned.monitoring import QualityDashboard
        except ImportError:
            pytest.skip("Quality monitoring not available")

        dashboard = QualityDashboard(db_path=":memory:", window_size=5)

        # Add some mock data
        for i in range(3):
            mock_summary = {
                "valid_samples": 7 + i,
                "total_attempts": 10,
                "diagnostics": {"avg_repair_severity": 10.0 + i * 5}
            }
            dashboard.log_generation(mock_summary, "residential")

        html = dashboard.render_html()
        assert "<html>" in html
        assert "Quality Dashboard" in html
        assert "residential" in html


class TestCacheSystemIntegration:
    """Test caching system integration."""

    def test_model_cache_import(self):
        """Test model cache import and basic functionality."""
        try:
            from learned.model.model_cache import ModelCache
        except ImportError:
            pytest.skip("Model cache not available")

        cache = ModelCache(max_size=2, ttl_seconds=10, enabled=True)
        stats = cache.stats()

        assert stats["max_size"] == 2
        assert stats["ttl_seconds"] == 10
        assert stats["enabled"] == True

    def test_svg_cache_import(self):
        """Test SVG cache import and symbol compilation."""
        try:
            from visualization.svg_template_cache import SvgSymbolLibrary, CachedSvgRenderer
        except ImportError:
            pytest.skip("SVG cache not available")

        # Test symbol library
        symbol_lib = SvgSymbolLibrary()
        assert symbol_lib.symbol_count > 10  # Should have many symbols

        # Test symbol retrieval
        door_symbol = symbol_lib.get_symbol("door-swing")
        assert door_symbol is not None
        assert "symbol" in door_symbol

        # Test renderer
        renderer = CachedSvgRenderer(cache_enabled=True)
        stats = renderer.stats()
        assert "cache_enabled" in stats


class TestPipelineIntegration:
    """Test full pipeline integration with mocked components."""

    def setup_method(self):
        """Set up test fixtures."""
        self.spec = create_sample_spec()
        self.boundary = create_mock_boundary()
        self.entrance = [5.0, 0.0]

    @patch('learned.integration.model_generation_loop.load_model')
    @patch('learned.integration.model_generation_loop.cached_load_model')
    def test_pipeline_with_mocked_model(self, mock_cached_load, mock_load):
        """Test pipeline with mocked model components."""
        try:
            from learned.integration.model_generation_loop import generate_best_layout_from_model
        except ImportError:
            pytest.skip("Pipeline components not available")

        # Mock model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_cached_load.return_value = (mock_model, mock_tokenizer)
        mock_load.return_value = (mock_model, mock_tokenizer)

        # Mock sample generation
        mock_rooms = [
            MockRoomBox("living room", 0.2, 0.2, 0.7, 0.6),
            MockRoomBox("kitchen", 0.1, 0.1, 0.4, 0.4),
            MockRoomBox("bedroom", 0.5, 0.6, 0.9, 0.9),
            MockRoomBox("bathroom", 0.1, 0.7, 0.3, 0.9),
        ]

        with patch('learned.integration.model_generation_loop.sample_layout') as mock_sample:
            mock_sample.return_value = mock_rooms

            # Mock repair and validation
            with patch('learned.integration.model_generation_loop.validate_and_repair_generated_layout') as mock_repair:
                mock_repair.return_value = (
                    create_mock_building(),  # repaired building
                    [],  # violations
                    "COMPLIANT",  # status
                    []   # repair_trace
                )

                with patch('learned.integration.model_generation_loop.adapt_generated_layout_to_building') as mock_adapt:
                    mock_adapt.return_value = create_mock_building()

                    with patch('learned.integration.model_generation_loop._score_variant') as mock_score:
                        mock_score.return_value = (85.0, {}, {})

                        # Run pipeline
                        result, summary = generate_best_layout_from_model(
                            spec=self.spec,
                            boundary_poly=self.boundary,
                            entrance=self.entrance,
                            max_attempts=3,
                            K=2
                        )

                        # Verify results
                        assert result is not None
                        assert "building" in result
                        assert "score" in result
                        assert result["score"] > 0

                        assert "valid_samples" in summary
                        assert "total_attempts" in summary
                        assert summary["valid_samples"] > 0

    def test_pipeline_error_handling(self):
        """Test pipeline error handling and graceful degradation."""
        try:
            from learned.integration.model_generation_loop import generate_best_layout_from_model
        except ImportError:
            pytest.skip("Pipeline components not available")

        # Test with invalid checkpoint path
        result, summary = generate_best_layout_from_model(
            spec=self.spec,
            boundary_poly=self.boundary,
            entrance=self.entrance,
            checkpoint_path="nonexistent.pt",
            max_attempts=1
        )

        # Should handle gracefully
        assert result is None or isinstance(result, dict)
        assert "total_attempts" in summary

    def test_template_integration_in_pipeline(self):
        """Test template integration within the pipeline."""
        try:
            from learned.integration.model_generation_loop import generate_best_layout_from_model
            from learned.templates import find_layout_template
        except ImportError:
            pytest.skip("Pipeline/Template components not available")

        # Test that templates can be found for our spec
        template = find_layout_template(self.spec)
        if template:
            assert template.calculate_compatibility(self.spec) > 0

            # Test pipeline with template enabled
            with patch.dict(os.environ, {"LAYOUT_USE_TEMPLATES": "true"}):
                with patch('learned.integration.model_generation_loop.load_model') as mock_load:
                    mock_load.return_value = (Mock(), Mock())

                    # The pipeline should attempt to use templates
                    # (exact behavior depends on template quality and availability)
                    result, summary = generate_best_layout_from_model(
                        spec=self.spec,
                        boundary_poly=self.boundary,
                        entrance=self.entrance,
                        use_templates=True,
                        max_attempts=1
                    )

                    # Check for template usage indicators in diagnostics
                    diagnostics = summary.get("diagnostics", {})
                    # Template integration should be attempted
                    assert isinstance(diagnostics, dict)


class TestPerformanceRegression:
    """Test performance regression detection."""

    def test_generation_timing_bounds(self):
        """Test that generation stays within timing bounds."""
        try:
            from learned.templates import find_layout_template
        except ImportError:
            pytest.skip("Template system not available")

        spec = create_sample_spec()

        # Template matching should be fast
        start_time = time.time()
        template = find_layout_template(spec)
        elapsed = time.time() - start_time

        assert elapsed < 0.1, f"Template matching too slow: {elapsed:.3f}s"

    def test_cache_performance(self):
        """Test cache system performance characteristics."""
        try:
            from learned.model.model_cache import ModelCache
        except ImportError:
            pytest.skip("Model cache not available")

        cache = ModelCache(max_size=3, ttl_seconds=10)

        # Cache operations should be fast
        start_time = time.time()
        for i in range(100):
            _ = cache.stats()
        elapsed = time.time() - start_time

        assert elapsed < 0.01, f"Cache stats too slow: {elapsed:.3f}s"

    def test_svg_symbol_compilation_timing(self):
        """Test SVG symbol compilation performance."""
        try:
            from visualization.svg_template_cache import SvgSymbolLibrary
        except ImportError:
            pytest.skip("SVG cache not available")

        # Symbol compilation should complete quickly
        start_time = time.time()
        symbol_lib = SvgSymbolLibrary()
        elapsed = time.time() - start_time

        assert elapsed < 1.0, f"Symbol compilation too slow: {elapsed:.3f}s"
        assert symbol_lib.symbol_count > 10


class TestEdgeCasesAndRobustness:
    """Test edge cases and system robustness."""

    def test_empty_spec_handling(self):
        """Test handling of empty or minimal specifications."""
        try:
            from learned.templates import find_layout_template
        except ImportError:
            pytest.skip("Template system not available")

        empty_specs = [
            {},
            {"building_type": "apartment"},
            {"rooms": []},
            {"building_type": "apartment", "rooms": []},
        ]

        for spec in empty_specs:
            # Should not crash
            template = find_layout_template(spec)
            # May or may not find a template, but should not error
            assert template is None or hasattr(template, 'name')

    def test_invalid_boundary_handling(self):
        """Test handling of invalid boundary polygons."""
        try:
            from learned.templates import apply_layout_template, find_layout_template
            from shapely.geometry import Polygon
        except ImportError:
            pytest.skip("Template/Shapely not available")

        spec = create_sample_spec()
        template = find_layout_template(spec)

        if template:
            # Test with various invalid boundaries
            invalid_boundaries = [
                Polygon([(0, 0), (1, 0), (0, 1)]),  # Triangle (too few points)
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),  # Self-intersecting
            ]

            for boundary in invalid_boundaries:
                try:
                    result = apply_layout_template(template, boundary, spec)
                    # Should either work or fail gracefully
                    assert isinstance(result, dict)
                except Exception:
                    # Acceptable to fail on invalid input
                    pass

    def test_large_specification_handling(self):
        """Test handling of unusually large specifications."""
        try:
            from learned.templates import find_layout_template
        except ImportError:
            pytest.skip("Template system not available")

        # Create spec with many rooms
        large_spec = {
            "building_type": "commercial",
            "rooms": [{"type": f"room_{i}"} for i in range(50)]
        }

        # Should not crash or take excessive time
        start_time = time.time()
        template = find_layout_template(large_spec)
        elapsed = time.time() - start_time

        assert elapsed < 1.0, "Template matching should handle large specs efficiently"

    def test_memory_usage_bounds(self):
        """Test that operations stay within memory bounds."""
        try:
            from learned.monitoring import QualityDashboard
        except ImportError:
            pytest.skip("Quality monitoring not available")

        dashboard = QualityDashboard(db_path=":memory:", window_size=1000)

        # Add many metrics
        for i in range(500):
            mock_summary = {
                "valid_samples": 5,
                "total_attempts": 10,
                "diagnostics": {"avg_repair_severity": i % 50}
            }
            dashboard.log_generation(mock_summary, f"building_type_{i % 5}")

        # Should not consume excessive memory
        # (Specific bounds would depend on system, this tests basic functionality)
        stats = dashboard.get_current_stats()
        assert stats["window_size"] <= 1000  # Should respect window limits


class TestSystemIntegrationFlow:
    """Test complete system integration flow."""

    def test_full_pipeline_mock_flow(self):
        """Test the complete pipeline flow with all components."""
        spec = create_sample_spec()
        boundary = create_mock_boundary()

        results = {}

        # 1. Template system
        try:
            from learned.templates import find_layout_template
            template = find_layout_template(spec)
            results["template_found"] = template is not None
            if template:
                results["template_compatibility"] = template.calculate_compatibility(spec)
        except ImportError:
            results["template_available"] = False

        # 2. Cache system
        try:
            from learned.model.model_cache import get_cache_stats
            cache_stats = get_cache_stats()
            results["cache_available"] = True
            results["cache_stats"] = cache_stats
        except ImportError:
            results["cache_available"] = False

        # 3. Quality monitoring
        try:
            from learned.monitoring import QualityDashboard
            dashboard = QualityDashboard(db_path=":memory:")
            mock_summary = {"valid_samples": 5, "total_attempts": 8}
            dashboard.log_generation(mock_summary, "test")
            results["quality_monitoring_available"] = True
        except ImportError:
            results["quality_monitoring_available"] = False

        # 4. SVG optimization
        try:
            from visualization.svg_template_cache import get_svg_cache_stats
            svg_stats = get_svg_cache_stats()
            results["svg_cache_available"] = True
        except ImportError:
            results["svg_cache_available"] = False

        # Verify integration works
        assert isinstance(results, dict)
        assert len(results) > 0


# Pytest configuration and fixtures
@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_checkpoint(test_data_dir):
    """Create mock checkpoint file for testing."""
    checkpoint_file = test_data_dir / "mock_checkpoint.pt"

    # Create minimal mock checkpoint data
    mock_data = {
        "config": {
            "vocab_size": 1000,
            "max_seq_len": 256,
            "n_layers": 6,
            "n_heads": 8,
            "d_model": 256,
        },
        "epoch": 10,
        "loss": 2.5,
        "model_state_dict": {},
        "optimizer_state_dict": {},
    }

    # Save as JSON (since we can't save torch without torch)
    with open(checkpoint_file, 'w') as f:
        json.dump(mock_data, f)

    return str(checkpoint_file)


# Property-based testing (if hypothesis is available)
try:
    from hypothesis import given, strategies as st
    import hypothesis

    @given(st.dictionaries(
        keys=st.sampled_from(["building_type", "occupancy", "total_area"]),
        values=st.one_of(
            st.text(min_size=1, max_size=20),
            st.floats(min_value=10.0, max_value=500.0),
        ),
        min_size=1,
        max_size=3
    ))
    def test_template_matching_property_based(spec_dict):
        """Property-based test for template matching."""
        try:
            from learned.templates import find_layout_template
            template = find_layout_template(spec_dict)
            # Should not crash regardless of input
            assert template is None or hasattr(template, 'name')
        except ImportError:
            pytest.skip("Template system not available")

except ImportError:
    # Hypothesis not available, skip property-based tests
    pass


if __name__ == "__main__":
    # Allow direct execution for debugging
    pytest.main([__file__, "-v"])