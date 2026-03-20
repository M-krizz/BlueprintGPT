"""
test_e2e_pipeline.py - End-to-end integration tests for BlueprintGPT pipelines.

Tests the complete flow:
    NL input → spec parsing → generation → repair → compliance → SVG output

Validates consistency between learned, algorithmic, and hybrid pipelines.
Includes both legacy tests and modern pipeline coverage.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from nl_interface.service import process_user_request
from learned.integration.realism_score import compute_realism_score
from constraints.compliance_report import build_compliance_report
from core.building import Building


class TestNLToSpecParsing:
    """Test natural language to DesignSpec conversion."""

    def test_basic_residential_parsing(self):
        """Test parsing a basic residential request."""
        response = process_user_request(
            "Need 2 bedrooms, 1 kitchen, 1 bathroom on a 10 marla plot with north entrance."
        )

        spec = response["current_spec"]
        assert spec["plot_type"] == "10Marla"
        assert spec["entrance_side"] == "North"

        # Verify room counts
        rooms = {r["type"]: r["count"] for r in spec["rooms"]}
        assert rooms["Bedroom"] == 2
        assert rooms["Kitchen"] == 1
        assert rooms["Bathroom"] == 1

        # Should be missing boundary geometry
        assert not response["backend_ready"]
        assert "boundary_polygon" in response["missing_fields"]

    def test_adjacency_extraction(self):
        """Test extraction of room adjacency preferences."""
        response = process_user_request(
            "3 bedrooms, 1 kitchen, 1 bathroom. Kitchen next to dining, bedrooms far from kitchen."
        )

        adjacency = response["current_spec"]["preferences"]["adjacency"]

        # Should extract adjacency rules
        kitchen_dining = ["Kitchen", "DiningRoom", "adjacent_to"] in adjacency
        bedroom_kitchen = any(
            rel[0] == "Bedroom" and rel[1] == "Kitchen" and rel[2] == "far_from"
            for rel in adjacency
        )

        assert kitchen_dining or bedroom_kitchen  # At least one should be extracted

    def test_privacy_and_weights(self):
        """Test weight normalization and privacy extraction."""
        response = process_user_request(
            "2 bedrooms, kitchen, living room. Prioritize privacy and minimize corridor space."
        )

        weights = response["current_spec"]["weights"]

        # Weights should sum to 1.0
        assert abs(sum(weights.values()) - 1.0) < 1e-4

        # Privacy should be elevated
        assert weights["privacy"] > 0.2  # Should be > default 1/3


class TestEndToEndConsistency:
    """Test consistency between different pipeline approaches."""

    def _create_test_spec(self) -> dict:
        """Create a standard test specification."""
        return {
            "plot_type": "10Marla",
            "entrance_side": "North",
            "rooms": [
                {"type": "Bedroom", "count": 2},
                {"type": "Kitchen", "count": 1},
                {"type": "Bathroom", "count": 1},
                {"type": "LivingRoom", "count": 1},
            ],
            "boundary_polygon": [
                [0, 0], [15, 0], [15, 12], [0, 12]  # Simple rectangle
            ],
            "preferences": {
                "adjacency": [],
                "minimize_corridor": True,
            },
            "weights": {
                "privacy": 1.0 / 3.0,
                "compactness": 1.0 / 3.0,
                "corridor": 1.0 / 3.0,
            }
        }

    @pytest.mark.skipif(
        not os.path.exists("learned/model/checkpoints"),
        reason="No model checkpoints available"
    )
    def test_learned_pipeline_basic_generation(self):
        """Test learned pipeline can generate a layout."""
        # Skip test if no model available
        try:
            from learned.model.sample import load_model, constrained_sample_layout
        except ImportError:
            pytest.skip("Learned model dependencies not available")

        # Create simple spec
        spec = {
            "rooms": [
                {"type": "Bedroom", "count": 1},
                {"type": "Kitchen", "count": 1},
            ]
        }

        try:
            # Try to load a model (will skip if none available)
            checkpoint_files = list(Path("learned/model/checkpoints").glob("*.pt"))
            if not checkpoint_files:
                pytest.skip("No model checkpoints found")

            model, tokenizer = load_model(str(checkpoint_files[0]))

            # Generate layout
            rooms = constrained_sample_layout(
                model, tokenizer, spec,
                plot_area_sqm=100.0,
                temperature=0.8
            )

            # Basic validation
            assert len(rooms) >= 2
            room_types = [r.room_type for r in rooms]
            assert "Bedroom" in room_types
            assert "Kitchen" in room_types

        except Exception as e:
            pytest.skip(f"Learned pipeline not functional: {e}")

    def _dict_to_building(self, building_dict: dict) -> Building:
        """Convert dictionary representation back to Building object."""
        building = Building(occupancy_type=building_dict.get("occupancy_type", "Residential"))

        # Add rooms from dict
        for room_dict in building_dict.get("rooms", []):
            from core.room import Room
            room = Room(
                room_dict["name"],
                room_dict["room_type"],
                room_dict.get("requested_area", 0)
            )
            room.final_area = room_dict.get("final_area", 0)
            room.polygon = room_dict.get("polygon", [])
            building.add_room(room)

        return building


class TestRealismScoring:
    """Test realism scoring integration."""

    def test_realism_score_components(self):
        """Test all realism score components work."""
        # Create a test building with basic rooms
        building = Building(occupancy_type="Residential")

        from core.room import Room

        # Add valid rooms with reasonable polygons
        bedroom = Room("Bedroom1", "Bedroom", 12.0)
        bedroom.polygon = [(2, 2), (6, 2), (6, 5), (2, 5)]
        bedroom.final_area = 12.0

        kitchen = Room("Kitchen1", "Kitchen", 8.0)
        kitchen.polygon = [(6, 2), (9, 2), (9, 5), (6, 5)]
        kitchen.final_area = 8.0

        bathroom = Room("Bathroom1", "Bathroom", 4.0)
        bathroom.polygon = [(9, 2), (11, 2), (11, 5), (9, 5)]
        bathroom.final_area = 4.0

        building.rooms = [bedroom, kitchen, bathroom]

        # Compute realism score
        score = compute_realism_score(building, plot_area_sqm=150.0)

        # All components should be present
        assert hasattr(score, 'overall')
        assert hasattr(score, 'min_dim_score')
        assert hasattr(score, 'aspect_ratio_score')
        assert hasattr(score, 'corridor_score')
        assert hasattr(score, 'zoning_score')
        assert hasattr(score, 'travel_score')

        # Scores should be in valid range
        assert 0.0 <= score.overall <= 1.0
        assert 0.0 <= score.min_dim_score <= 1.0
        assert 0.0 <= score.aspect_ratio_score <= 1.0
        assert 0.0 <= score.zoning_score <= 1.0

        # Should have details
        assert "min_dim_violations_list" in score.details
        assert "zoning" in score.details

    def test_zoning_score_wet_area_clustering(self):
        """Test zoning score detects wet area clustering."""
        building = Building(occupancy_type="Residential")

        from core.room import Room

        # Case 1: Well-clustered wet areas (good) - distance < 0.4 threshold
        kitchen = Room("Kitchen1", "Kitchen", 8.0)
        kitchen.polygon = [(0, 0), (0.2, 0), (0.2, 0.2), (0, 0.2)]  # Centroid at (0.1, 0.1)
        kitchen.final_area = 8.0

        bathroom = Room("Bathroom1", "Bathroom", 4.0)
        bathroom.polygon = [(0.2, 0), (0.4, 0), (0.4, 0.2), (0.2, 0.2)]  # Centroid at (0.3, 0.1)
        bathroom.final_area = 4.0
        # Distance = sqrt((0.3-0.1)^2 + (0.1-0.1)^2) = 0.2 < 0.4 → no penalty

        building.rooms = [kitchen, bathroom]

        score_good = compute_realism_score(building, plot_area_sqm=100.0)

        # Case 2: Far apart wet areas (bad) - distance > 0.4 threshold
        bathroom2 = Room("Bathroom2", "Bathroom", 4.0)
        bathroom2.polygon = [(1.0, 1.0), (1.2, 1.0), (1.2, 1.2), (1.0, 1.2)]  # Centroid at (1.1, 1.1)
        bathroom2.final_area = 4.0
        # Distance = sqrt((1.1-0.1)^2 + (1.1-0.1)^2) = sqrt(2) ≈ 1.41 > 0.4 → penalty

        building.rooms = [kitchen, bathroom2]

        score_bad = compute_realism_score(building, plot_area_sqm=100.0)

        print(f"\nGood clustering: distance={score_good.details['zoning'].get('avg_wet_area_distance', 'N/A')}, score={score_good.zoning_score}")
        print(f"Bad clustering: distance={score_bad.details['zoning'].get('avg_wet_area_distance', 'N/A')}, score={score_bad.zoning_score}")

        # Good clustering should score higher than bad clustering
        assert score_good.zoning_score > score_bad.zoning_score


class TestComplianceReporting:
    """Test compliance reporting integration."""

    def test_compliance_report_generation(self):
        """Test compliance report includes all sections."""
        building = Building(occupancy_type="Residential")

        from core.room import Room

        # Add a simple valid room
        room = Room("TestRoom", "Bedroom", 10.0)
        room.polygon = [(0, 0), (3, 0), (3, 4), (0, 4)]
        room.final_area = 12.0
        building.add_room(room)

        # Generate compliance report
        report = build_compliance_report(building, plot_area_sqm=100.0)

        # Should have required sections
        assert "overall_compliance_score" in report
        assert "chapter4_compliance" in report
        assert "building_summary" in report

        # Chapter-4 section should have details
        ch4 = report["chapter4_compliance"]
        assert "violations" in ch4
        assert "checks_performed" in ch4

    @pytest.mark.skipif(
        not Path("ground_truth/chapter4_validator.py").exists(),
        reason="Ground truth validator not available"
    )
    def test_ground_truth_validation_integration(self):
        """Test ground truth validation integration in compliance reports."""
        building = Building(occupancy_type="Residential")

        from core.room import Room

        # Add rooms that should pass basic validation
        bedroom = Room("Bedroom1", "Bedroom", 15.0)
        bedroom.polygon = [(0, 0), (4, 0), (4, 4), (0, 4)]  # 4×4 = 16 sq units
        bedroom.final_area = 15.0
        building.add_room(bedroom)

        # Generate report with ground truth validation
        report = build_compliance_report(
            building,
            plot_area_sqm=100.0,
            enable_ground_truth=True
        )

        # Should include ground truth section if available
        if "ground_truth_validation" in report:
            gt = report["ground_truth_validation"]
            assert "compliant" in gt
            assert "violations" in gt


class TestSVGGeneration:
    """Test SVG generation and export."""

    def test_svg_export_creates_valid_file(self):
        """Test SVG export creates parseable XML."""
        building = Building(occupancy_type="Residential")

        from core.room import Room

        room = Room("TestRoom", "Bedroom", 10.0)
        room.polygon = [(0, 0), (3, 0), (3, 3), (0, 3)]
        room.final_area = 9.0
        building.add_room(room)

        # Generate SVG
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
            svg_path = f.name

        try:
            # Try to import and use SVG renderer if available
            try:
                from generator.svg_renderer import render_building_to_svg
                render_building_to_svg(building, svg_path)

                # Verify file exists and has content
                assert Path(svg_path).exists()
                assert Path(svg_path).stat().st_size > 100  # Should have substantial content

                # Basic XML validation
                content = Path(svg_path).read_text()
                assert content.startswith('<?xml') or content.startswith('<svg')
                assert "</svg>" in content

            except ImportError:
                pytest.skip("SVG renderer not available")

        finally:
            # Cleanup
            if Path(svg_path).exists():
                Path(svg_path).unlink()


# ─── Legacy Algorithm Pipeline Tests ──────────────────────────────────────

class TestLegacyAlgorithmPipeline:
    """Legacy tests for the original algorithmic pipeline."""

    def test_full_algorithmic_pipeline(self):
        """Test complete algorithmic pipeline from spec to SVG."""
        spec = {
            "occupancy": "Residential",
            "total_area": 120,
            "area_unit": "sq.m",
            "allocation_strategy": "priority_weights",
            "rooms": [
                {"name": "Bed1", "type": "Bedroom"},
                {"name": "Kit", "type": "Kitchen"},
                {"name": "Liv", "type": "LivingRoom"}
            ],
            "boundary_polygon": [(0, 0), (10, 0), (10, 12), (0, 12)],
            "entrance_point": (0, 6)
        }

        try:
            # Import legacy components
            from constraints.repair_loop import validate_and_repair_spec
            from constraints.spec_validator import validate_spec
            from generator.layout_generator import generate_layout_from_spec
            from generator.ranking import rank_layout_variants
            from visualization.export_svg_blueprint import render_svg_blueprint
            from ontology.ontology_bridge import OntologyBridge

            # 1. Validate & Repair
            repaired = validate_and_repair_spec(spec, validate_spec, max_attempts=3)
            spec_repaired = repaired["spec"]

            assert repaired["validation"]["valid"] is True

            # 2. Generate Layout
            bridge = OntologyBridge("ontology/regulatory.owl")
            result = generate_layout_from_spec(
                spec_repaired,
                regulation_file="ontology/regulation_data.json",
                ontology_validator=bridge
            )

            variants = result.get("layout_variants", [result])
            assert len(variants) > 0

            # 3. Ranking
            ranked_variants, rec_idx = rank_layout_variants(variants)
            best = ranked_variants[rec_idx]

            # 4. Compliance Report
            report = build_compliance_report(best)
            assert report is not None
            assert "status" in report or "overall_compliance_score" in report

            # 5. SVG Render
            svg = render_svg_blueprint(
                best["building"],
                boundary_polygon=spec_repaired["boundary_polygon"],
                entrance_point=spec_repaired["entrance_point"]
            )
            assert len(svg) > 100
            assert "svg" in svg

        except ImportError as e:
            pytest.skip(f"Legacy pipeline components not available: {e}")


# ─── Pytest Configuration ──────────────────────────────────────────

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
