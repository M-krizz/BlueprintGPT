import pytest
from ontology.ontology_bridge import OntologyBridge

@pytest.fixture
def bridge():
    return OntologyBridge("ontology/regulatory.owl")

def test_ontology_bridge_load(bridge):
    assert bridge.onto is not None

def test_semantic_precheck(bridge):
    spec = {
        "occupancy": "Residential", 
        "total_area": 100,
        "rooms": [{"name": "B1", "type": "Bedroom"}]
    }
    res = bridge.validate_spec_semantics(spec)
    assert res["valid"] is True
