from constraints.compliance_report import build_compliance_report
from learned.planner.geometry_synthesis import synthesize_room_geometry, synthesize_simple_doors
from nl_interface import runner



def test_synthesize_simple_doors_returns_segments_for_room_and_exit_connections():
    rooms = [
        {
            "name": "LivingRoom_1",
            "type": "LivingRoom",
            "polygon": [(0.0, 0.0), (3.0, 0.0), (3.0, 3.0), (0.0, 3.0)],
            "area": 9.0,
            "centroid": (1.5, 1.5),
        },
        {
            "name": "Bedroom_1",
            "type": "Bedroom",
            "polygon": [(3.0, 0.0), (6.0, 0.0), (6.0, 3.0), (3.0, 3.0)],
            "area": 9.0,
            "centroid": (4.5, 1.5),
        },
    ]

    doors = synthesize_simple_doors(
        rooms,
        boundary_polygon=[(0.0, 0.0), (6.0, 0.0), (6.0, 3.0), (0.0, 3.0)],
        entrance_point=(0.0, 1.5),
    )

    room_door = next(door for door in doors if door["door_type"] == "room_to_room")
    exit_door = next(door for door in doors if door["door_type"] == "exit")

    assert room_door["segment"] == ((3.0, 1.05), (3.0, 1.95))
    assert exit_door["segment"] == ((0.0, 1.0), (0.0, 2.0))



def test_synthesize_room_geometry_pulls_preferred_rooms_into_contact():
    planner_output = {
        "spatial_hints": {
            "LivingRoom_1": [0.2, 0.2],
            "Kitchen_1": [0.8, 0.8],
        },
        "area_ratios": {
            "LivingRoom_1": 0.18,
            "Kitchen_1": 0.18,
        },
        "room_order": ["LivingRoom_1", "Kitchen_1"],
        "room_zones": {
            "LivingRoom_1": "public",
            "Kitchen_1": "service",
        },
        "adjacency_preferences": [
            {"a": "LivingRoom_1", "b": "Kitchen_1", "score": 0.95},
        ],
    }
    spec = {
        "rooms": [
            {"name": "LivingRoom_1", "type": "LivingRoom", "area": 12.0},
            {"name": "Kitchen_1", "type": "Kitchen", "area": 10.0},
        ]
    }

    rooms = synthesize_room_geometry(
        planner_output,
        boundary_polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
        spec=spec,
    )

    doors = synthesize_simple_doors(rooms, boundary_polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])

    assert any(door["to_room"] == "Kitchen_1" for door in doors)



def test_synthesize_simple_doors_bridges_small_gap_between_rooms():
    rooms = [
        {
            "name": "LivingRoom_1",
            "type": "LivingRoom",
            "polygon": [(0.0, 0.0), (3.2, 0.0), (3.2, 2.8), (0.0, 2.8)],
            "area": 8.96,
            "centroid": (1.6, 1.4),
        },
        {
            "name": "Bedroom_1",
            "type": "Bedroom",
            "polygon": [(0.0, 2.9), (3.2, 2.9), (3.2, 5.7), (0.0, 5.7)],
            "area": 8.96,
            "centroid": (1.6, 4.3),
        },
    ]

    doors = synthesize_simple_doors(
        rooms,
        boundary_polygon=[(0.0, 0.0), (3.2, 0.0), (3.2, 5.7), (0.0, 5.7)],
    )

    bridge = next(door for door in doors if door["door_type"] == "room_to_room_bridge")
    assert bridge["segment"] == ((1.15, 2.85), (2.05, 2.85))



def test_build_compliance_report_marks_failed_planner_verification_non_compliant():
    result = {
        "source": "planner_direct",
        "metrics": {
            "fully_connected": True,
            "travel_distance_compliant": True,
            "required_exit_width": 1.0,
            "connectivity_to_exit": True,
            "max_travel_distance": 8.0,
            "max_allowed_travel_distance": 22.5,
            "corridor_width": 0.0,
            "circulation_walkable_area": 0.0,
        },
        "modifications": [],
        "allocation": None,
        "verification": {
            "passed": False,
            "issues": ["Overlap detected"],
            "metrics": {},
        },
        "input_spec": {},
        "bounding_box": {},
    }

    report = build_compliance_report(result)

    assert report["status"] == "NON_COMPLIANT"
    assert report["checks"]["planner_verification"] is False
    assert "Overlap detected" in report["violations"]



def test_run_planner_direct_backend_builds_real_connectivity_metrics(monkeypatch):
    monkeypatch.setattr(runner, "save_svg_blueprint", lambda *args, **kwargs: kwargs["output_path"])
    monkeypatch.setattr(runner, "save_compliance_report", lambda report, output_path: None)
    monkeypatch.setattr(runner, "build_evidence", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(runner, "explain", lambda *args, **kwargs: {"summary": "ok"})

    spec = {
        "occupancy": "Residential",
        "boundary_polygon": [(0.0, 0.0), (8.0, 0.0), (8.0, 4.0), (0.0, 4.0)],
        "entrance_point": (0.0, 2.0),
        "entrance_side": "West",
        "rooms": [
            {"name": "LivingRoom_1", "type": "LivingRoom", "area": 10.0},
            {"name": "Bedroom_1", "type": "Bedroom", "area": 10.0},
        ],
    }
    planner_guidance = {
        "source": "test",
        "spatial_hints": {
            "LivingRoom_1": [0.22, 0.5],
            "Bedroom_1": [0.78, 0.5],
        },
        "room_order": ["LivingRoom_1", "Bedroom_1"],
        "area_ratios": {
            "LivingRoom_1": 0.42,
            "Bedroom_1": 0.42,
        },
        "room_zones": {
            "LivingRoom_1": "public",
            "Bedroom_1": "private",
        },
        "adjacency_preferences": [
            {"a": "LivingRoom_1", "b": "Bedroom_1", "type": "prefer"},
        ],
    }

    result = runner.run_planner_direct_backend(
        spec,
        output_dir="outputs",
        output_prefix="test_planner_direct_runtime",
        planner_guidance=planner_guidance,
    )

    assert result["metrics"]["fully_connected"] is True
    assert result["metrics"]["connectivity_to_exit"] is True
    assert result["metrics"]["door_path_travel_distance"] < 999.0
    assert result["metrics"]["public_frontage_score"] >= 0.45
    assert result["metrics"]["bedroom_privacy_score"] >= 0.5
    assert "kitchen_living_score" in result["metrics"]
    assert "bathroom_access_score" in result["metrics"]
    assert "architectural_reasonableness" in result["metrics"]
    assert result["report_status"] == "COMPLIANT"



def test_run_planner_backend_rescues_with_plain_algorithmic_when_direct_fallback_is_non_compliant(monkeypatch):
    import learned.planner.inference as planner_inference

    planner_guidance = {
        "source": "test",
        "spatial_hints": {},
        "room_order": [],
        "area_ratios": {},
        "room_zones": {},
        "adjacency_preferences": [],
    }
    monkeypatch.setattr(planner_inference, "predict_planner_guidance", lambda *args, **kwargs: planner_guidance)

    def fake_algorithmic(spec, **kwargs):
        if spec.get("planner_guidance"):
            raise ValueError("guided packing failed")
        return {
            "status": "completed",
            "backend_target": "algorithmic",
            "report_status": "COMPLIANT",
            "violations": [],
            "artifact_paths": {"svg": "algo.svg", "report": "algo.json"},
            "metrics": {"fully_connected": True},
        }

    monkeypatch.setattr(runner, "run_algorithmic_backend", fake_algorithmic)
    monkeypatch.setattr(runner, "run_planner_direct_backend", lambda *args, **kwargs: {
        "status": "completed",
        "backend_target": "planner_direct",
        "report_status": "NON_COMPLIANT",
        "violations": ["Layout graph is not fully connected"],
        "artifact_paths": {"svg": "direct.svg", "report": "direct.json"},
    })

    result = runner.run_planner_backend(
        {
            "occupancy": "Residential",
            "boundary_polygon": [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
            "entrance_point": (0.0, 2.0),
            "rooms": [{"name": "LivingRoom_1", "type": "LivingRoom"}],
        },
        output_dir="outputs",
        output_prefix="test_planner_rescue",
    )

    assert result["winning_source"] == "algorithmic_rescue"
    assert result["planner_direct_attempt"]["report_status"] == "NON_COMPLIANT"
    assert result["planner_fallback_reason"] == "guided packing failed"


def test_run_algorithmic_backend_rescues_with_planner_direct_when_all_variants_fail(monkeypatch):
    monkeypatch.setattr(runner, "validate_and_repair_spec", lambda spec, validator, max_attempts=3: {
        "spec": dict(spec),
        "validation": {},
        "repair_attempts": 0,
    })
    monkeypatch.setattr(runner, "validate_spec", lambda spec: {"valid": True})
    monkeypatch.setattr(runner, "generate_layout_from_spec", lambda *args, **kwargs: {
        "layout_variants": [
            {
                "source": "algorithmic",
                "strategy_name": "balanced",
                "building": object(),
                "metrics": {
                    "fully_connected": False,
                    "max_travel_distance": 10.0,
                    "max_allowed_travel_distance": 20.0,
                    "adjacency_satisfaction": 0.6,
                    "alignment_score": 0.7,
                    "corridor_width": 1.2,
                    "circulation_walkable_area": 8.0,
                    "total_area": 100.0,
                    "max_room_area_error": 0.1,
                },
            }
        ]
    })
    monkeypatch.setattr(runner, "run_planner_direct_backend", lambda *args, **kwargs: {
        "status": "completed",
        "backend_target": "planner_direct",
        "report_status": "COMPLIANT",
        "violations": [],
        "artifact_paths": {"svg": "direct.svg", "report": "direct.json"},
        "metrics": {"fully_connected": True},
    })

    result = runner.run_algorithmic_backend(
        {
            "occupancy": "Residential",
            "boundary_polygon": [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
            "entrance_point": (0.0, 2.0),
            "rooms": [{"name": "LivingRoom_1", "type": "LivingRoom"}],
        },
        output_dir="outputs",
        output_prefix="test_algorithmic_rescue",
    )

    assert result["winning_source"] == "planner_direct_rescue"
    assert result["algorithmic_attempt"]["reasons"] == ["Not fully connected"]


def test_run_algorithmic_backend_retries_with_expanded_boundary_on_area_drift(monkeypatch):
    monkeypatch.setattr(runner, "validate_and_repair_spec", lambda spec, validator, max_attempts=3: {
        "spec": dict(spec),
        "validation": {},
        "repair_attempts": 0,
    })
    monkeypatch.setattr(runner, "validate_spec", lambda spec: {"valid": True})
    monkeypatch.setattr(runner, "rank_layout_variants", lambda variants: (variants, {}))
    monkeypatch.setattr(runner, "save_svg_blueprint", lambda *args, **kwargs: kwargs["output_path"])
    monkeypatch.setattr(runner, "save_compliance_report", lambda report, output_path: None)
    monkeypatch.setattr(runner, "build_compliance_report", lambda chosen: {"status": "COMPLIANT", "violations": []})
    monkeypatch.setattr(runner, "build_evidence", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(runner, "explain", lambda *args, **kwargs: {"summary": "ok"})

    boundary_widths = []

    def fake_generate(spec, **kwargs):
        xs = [point[0] for point in spec["boundary_polygon"]]
        width = max(xs) - min(xs)
        boundary_widths.append(round(width, 1))
        if len(boundary_widths) == 1:
            return {
                "layout_variants": [
                    {
                        "source": "algorithmic",
                        "strategy_name": "balanced",
                        "building": object(),
                        "metrics": {
                            "fully_connected": True,
                            "max_travel_distance": 10.0,
                            "max_allowed_travel_distance": 20.0,
                            "adjacency_satisfaction": 0.6,
                            "alignment_score": 0.7,
                            "corridor_width": 1.2,
                            "circulation_walkable_area": 8.0,
                            "total_area": 100.0,
                            "max_room_area_error": 1.1,
                        },
                    }
                ]
            }
        return {
            "layout_variants": [
                {
                    "source": "algorithmic",
                    "strategy_name": "balanced",
                    "building": object(),
                    "metrics": {
                        "fully_connected": True,
                        "max_travel_distance": 9.0,
                        "max_allowed_travel_distance": 20.0,
                        "adjacency_satisfaction": 0.62,
                        "alignment_score": 0.72,
                        "corridor_width": 1.2,
                        "circulation_walkable_area": 9.0,
                        "total_area": 120.0,
                        "max_room_area_error": 0.12,
                    },
                }
            ]
        }

    monkeypatch.setattr(runner, "generate_layout_from_spec", fake_generate)

    result = runner.run_algorithmic_backend(
        {
            "occupancy": "Residential",
            "boundary_polygon": [(0.0, 0.0), (12.4, 0.0), (12.4, 8.9), (0.0, 8.9)],
            "entrance_point": (6.2, 8.9),
            "rooms": [{"name": f"Room_{idx}", "type": "Bedroom"} for idx in range(7)],
        },
        output_dir="outputs",
        output_prefix="test_algorithmic_boundary_recovery",
    )

    assert len(boundary_widths) == 2
    assert boundary_widths[1] > boundary_widths[0]
    assert result["winning_source"] == "algorithmic_boundary_recovery"
    assert result["auto_boundary_adjustment"]["new_boundary_size"][0] > result["auto_boundary_adjustment"]["prior_boundary_size"][0]


def test_run_algorithmic_backend_retries_with_expanded_boundary_on_compact_connectivity_failure(monkeypatch):
    monkeypatch.setattr(runner, "validate_and_repair_spec", lambda spec, validator, max_attempts=3: {
        "spec": dict(spec),
        "validation": {},
        "repair_attempts": 0,
    })
    monkeypatch.setattr(runner, "validate_spec", lambda spec: {"valid": True})
    monkeypatch.setattr(runner, "rank_layout_variants", lambda variants: (variants, {}))
    monkeypatch.setattr(runner, "save_svg_blueprint", lambda *args, **kwargs: kwargs["output_path"])
    monkeypatch.setattr(runner, "save_compliance_report", lambda report, output_path: None)
    monkeypatch.setattr(runner, "build_compliance_report", lambda chosen: {"status": "COMPLIANT", "violations": []})
    monkeypatch.setattr(runner, "build_evidence", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(runner, "explain", lambda *args, **kwargs: {"summary": "ok"})

    boundary_widths = []

    def fake_generate(spec, **kwargs):
        xs = [point[0] for point in spec["boundary_polygon"]]
        width = max(xs) - min(xs)
        boundary_widths.append(round(width, 1))
        if len(boundary_widths) == 1:
            return {
                "layout_variants": [
                    {
                        "source": "algorithmic",
                        "strategy_name": "compact-balanced",
                        "building": object(),
                        "metrics": {
                            "fully_connected": False,
                            "max_travel_distance": 10.0,
                            "max_allowed_travel_distance": 20.0,
                            "adjacency_satisfaction": 0.6,
                            "alignment_score": 0.7,
                            "corridor_width": 1.2,
                            "circulation_walkable_area": 6.0,
                            "total_area": 55.0,
                            "max_room_area_error": 0.12,
                        },
                    }
                ]
            }
        return {
            "layout_variants": [
                {
                    "source": "algorithmic",
                    "strategy_name": "compact-balanced",
                    "building": object(),
                    "metrics": {
                        "fully_connected": True,
                        "max_travel_distance": 9.0,
                        "max_allowed_travel_distance": 20.0,
                        "adjacency_satisfaction": 0.62,
                        "alignment_score": 0.72,
                        "corridor_width": 1.2,
                        "circulation_walkable_area": 7.0,
                        "total_area": 60.0,
                        "max_room_area_error": 0.12,
                    },
                }
            ]
        }

    monkeypatch.setattr(runner, "generate_layout_from_spec", fake_generate)

    result = runner.run_algorithmic_backend(
        {
            "occupancy": "Residential",
            "total_area": 55.0,
            "boundary_polygon": [(0.0, 0.0), (8.1, 0.0), (8.1, 6.8), (0.0, 6.8)],
            "entrance_point": (4.05, 6.8),
            "rooms": [
                {"name": "LivingRoom_1", "type": "LivingRoom"},
                {"name": "Kitchen_1", "type": "Kitchen"},
                {"name": "Bedroom_1", "type": "Bedroom"},
                {"name": "Bedroom_2", "type": "Bedroom"},
                {"name": "Bathroom_1", "type": "Bathroom"},
                {"name": "Bathroom_2", "type": "Bathroom"},
            ],
        },
        output_dir="outputs",
        output_prefix="test_algorithmic_compact_connectivity_recovery",
    )

    assert len(boundary_widths) == 2
    assert boundary_widths[1] > boundary_widths[0]
    assert result["winning_source"] == "algorithmic_boundary_recovery"


def test_run_planner_direct_backend_marks_overlapping_layout_non_compliant(monkeypatch):
    monkeypatch.setattr(runner, "save_svg_blueprint", lambda *args, **kwargs: kwargs["output_path"])
    monkeypatch.setattr(runner, "save_compliance_report", lambda report, output_path: None)
    monkeypatch.setattr(runner, "build_evidence", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(runner, "explain", lambda *args, **kwargs: {"summary": "ok"})

    overlapping_rooms = [
        {
            "name": "LivingRoom_1",
            "type": "LivingRoom",
            "polygon": [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
            "area": 16.0,
            "centroid": (2.0, 2.0),
        },
        {
            "name": "Kitchen_1",
            "type": "Kitchen",
            "polygon": [(2.0, 1.0), (6.0, 1.0), (6.0, 5.0), (2.0, 5.0)],
            "area": 16.0,
            "centroid": (4.0, 3.0),
        },
    ]

    monkeypatch.setattr(
        "learned.planner.geometry_synthesis.synthesize_room_geometry",
        lambda *args, **kwargs: overlapping_rooms,
    )

    result = runner.run_planner_direct_backend(
        {
            "occupancy": "Residential",
            "boundary_polygon": [(0.0, 0.0), (8.0, 0.0), (8.0, 8.0), (0.0, 8.0)],
            "entrance_point": (4.0, 8.0),
            "entrance_side": "North",
            "rooms": [
                {"name": "LivingRoom_1", "type": "LivingRoom", "area": 12.0},
                {"name": "Kitchen_1", "type": "Kitchen", "area": 8.0},
            ],
        },
        output_dir="outputs",
        output_prefix="test_planner_direct_overlap_fail",
        planner_guidance={
            "source": "test",
            "spatial_hints": {"LivingRoom_1": [0.4, 0.3], "Kitchen_1": [0.6, 0.4]},
            "room_order": ["LivingRoom_1", "Kitchen_1"],
            "area_ratios": {"LivingRoom_1": 0.2, "Kitchen_1": 0.15},
            "room_zones": {"LivingRoom_1": "public", "Kitchen_1": "service"},
            "adjacency_preferences": [{"a": "LivingRoom_1", "b": "Kitchen_1", "score": 1.0}],
        },
    )

    assert result["report_status"] == "NON_COMPLIANT"
    assert result["quality_gate_passed"] is False
    assert any("Overlap detected" in issue for issue in result["verification"]["issues"])





def test_run_planner_backend_rebuilds_clean_rescue_spec_from_original_request(monkeypatch):
    import learned.planner.inference as planner_inference

    planner_guidance = {
        "source": "test",
        "spatial_hints": {},
        "room_order": [],
        "area_ratios": {},
        "room_zones": {},
        "adjacency_preferences": [],
    }

    def fake_predict(spec, *args, **kwargs):
        spec["poison"] = True
        return planner_guidance

    monkeypatch.setattr(planner_inference, "predict_planner_guidance", fake_predict)

    def fake_algorithmic(spec, **kwargs):
        if spec.get("planner_guidance"):
            raise ValueError("guided packing failed")
        if spec.get("poison"):
            raise ValueError("poisoned rescue spec")
        return {
            "status": "completed",
            "backend_target": "algorithmic",
            "report_status": "COMPLIANT",
            "violations": [],
            "artifact_paths": {"svg": "algo.svg", "report": "algo.json"},
            "metrics": {"fully_connected": True},
        }

    monkeypatch.setattr(runner, "run_algorithmic_backend", fake_algorithmic)
    monkeypatch.setattr(runner, "run_planner_direct_backend", lambda *args, **kwargs: {
        "status": "completed",
        "backend_target": "planner_direct",
        "report_status": "NON_COMPLIANT",
        "violations": ["Layout graph is not fully connected"],
        "artifact_paths": {"svg": "direct.svg", "report": "direct.json"},
    })

    result = runner.run_planner_backend(
        {
            "occupancy": "Residential",
            "boundary_polygon": [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
            "entrance_point": (0.0, 2.0),
            "rooms": [{"name": "LivingRoom_1", "type": "LivingRoom"}],
        },
        output_dir="outputs",
        output_prefix="test_planner_rescue_clean_spec",
    )

    assert result["winning_source"] == "algorithmic_rescue"
    assert result["report_status"] == "COMPLIANT"


def test_run_planner_backend_uses_compact_algorithmic_baseline_when_available(monkeypatch):
    import learned.planner.inference as planner_inference

    planner_guidance = {
        "source": "test",
        "spatial_hints": {},
        "room_order": [],
        "area_ratios": {},
        "room_zones": {},
        "adjacency_preferences": [],
    }
    monkeypatch.setattr(planner_inference, "predict_planner_guidance", lambda *args, **kwargs: planner_guidance)

    def fake_algorithmic(spec, **kwargs):
        if spec.get("planner_guidance"):
            raise AssertionError("guided algorithmic path should be skipped for compact baseline")
        return {
            "status": "completed",
            "backend_target": "algorithmic",
            "report_status": "COMPLIANT",
            "violations": [],
            "artifact_paths": {"svg": "algo.svg", "report": "algo.json"},
            "metrics": {"fully_connected": True},
        }

    monkeypatch.setattr(runner, "run_algorithmic_backend", fake_algorithmic)
    monkeypatch.setattr(runner, "run_planner_direct_backend", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("planner_direct should not run")))

    result = runner.run_planner_backend(
        {
            "occupancy": "Residential",
            "total_area": 55.0,
            "boundary_polygon": [(0.0, 0.0), (8.0, 0.0), (8.0, 7.0), (0.0, 7.0)],
            "entrance_point": (4.0, 0.0),
            "rooms": [
                {"name": "LivingRoom_1", "type": "LivingRoom"},
                {"name": "Kitchen_1", "type": "Kitchen"},
                {"name": "Bedroom_1", "type": "Bedroom"},
                {"name": "Bedroom_2", "type": "Bedroom"},
                {"name": "Bathroom_1", "type": "Bathroom"},
                {"name": "Bathroom_2", "type": "Bathroom"},
            ],
        },
        output_dir="outputs",
        output_prefix="test_planner_compact_baseline",
    )

    assert result["winning_source"] == "algorithmic_compact_baseline"
    assert result["planner_summary"]["compact_baseline_only"] is True
    assert result["report_status"] == "COMPLIANT"


def test_run_planner_backend_preserves_zoning_guidance_for_compact_baseline(monkeypatch):
    seen = {}

    def fake_algorithmic(spec, **kwargs):
        seen["planner_guidance_source"] = (spec.get("planner_guidance") or {}).get("source")
        seen["learned_spatial_hints"] = dict(spec.get("learned_spatial_hints") or {})
        return {
            "status": "completed",
            "backend_target": "algorithmic",
            "report_status": "COMPLIANT",
            "violations": [],
            "artifact_paths": {"svg": "algo.svg", "report": "algo.json"},
            "metrics": {"fully_connected": True},
        }

    monkeypatch.setattr(runner, "run_algorithmic_backend", fake_algorithmic)
    monkeypatch.setattr(runner, "run_planner_direct_backend", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("planner_direct should not run")))

    zoning_guidance = {
        "source": "zoning-plan",
        "spatial_hints": {"LivingRoom_1": [0.5, 0.2]},
        "room_order": ["LivingRoom_1", "Kitchen_1", "Bedroom_1", "Bedroom_2", "Bathroom_1", "Bathroom_2"],
        "adjacency_preferences": [{"a": "Kitchen_1", "b": "LivingRoom_1", "type": "prefer"}],
    }

    result = runner.run_planner_backend(
        {
            "occupancy": "Residential",
            "total_area": 55.0,
            "boundary_polygon": [(0.0, 0.0), (8.0, 0.0), (8.0, 7.0), (0.0, 7.0)],
            "entrance_point": (4.0, 0.0),
            "rooms": [
                {"name": "LivingRoom_1", "type": "LivingRoom"},
                {"name": "Kitchen_1", "type": "Kitchen"},
                {"name": "Bedroom_1", "type": "Bedroom"},
                {"name": "Bedroom_2", "type": "Bedroom"},
                {"name": "Bathroom_1", "type": "Bathroom"},
                {"name": "Bathroom_2", "type": "Bathroom"},
            ],
            "planner_guidance": zoning_guidance,
            "learned_spatial_hints": {"LivingRoom_1": [0.5, 0.2]},
        },
        output_dir="outputs",
        output_prefix="test_planner_compact_zoning_baseline",
    )

    assert seen["planner_guidance_source"] == "zoning-plan"
    assert seen["learned_spatial_hints"] == {"LivingRoom_1": [0.5, 0.2]}
    assert result["planner_summary"]["source"] == "zoning-plan"
    assert result["planner_summary"]["compact_baseline_only"] is True
