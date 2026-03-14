import pytest
from core.building import Building
from core.room import Room
from geometry.polygon import (
    snap_to_grid, aspect_ratio, alignment_score, walls,
)
from geometry.bounding_box import bbox, bbox_area, merge_bboxes
from geometry.adjacency_intent import build_adjacency_intent
from geometry.zoning import _zone_maps_from_kg

def test_snap_to_grid():
    poly = [(0.07, 0.13), (5.07, 0.13), (5.07, 3.13), (0.07, 3.13)]
    snapped = snap_to_grid(poly, 0.15)
    assert snapped[0] == (0.0, 0.15)

def test_aspect_ratio():
    poly = [(0, 0), (4, 0), (4, 3), (0, 3)]
    ar = aspect_ratio(poly)
    assert 1.0 < ar < 2.0

def test_alignment_score(sample_building):
    score = alignment_score(sample_building)
    assert score > 0

def test_walls():
    poly = [(0, 0), (4, 0), (4, 3), (0, 3)]
    ws = walls(poly)
    assert len(ws) == 4

def test_bbox_operations():
    poly1 = [(0, 0), (4, 0), (4, 3), (0, 3)]
    poly2 = [(4, 0), (8, 0), (8, 3), (4, 3)]
    
    bb1 = bbox(poly1)
    assert bb1 == (0, 0, 4, 3)
    assert bbox_area(poly1) == 12.0
    
    merged = merge_bboxes([bbox(poly1), bbox(poly2)])
    assert merged == (0, 0, 8, 3)

def test_adjacency_intent():
    intents = build_adjacency_intent(room_types=["Bedroom", "Kitchen"])
    assert len(intents) > 0

def test_zoning():
    pub, svc, prv = _zone_maps_from_kg()
    assert len(pub) >= 1
    assert len(prv) >= 1
