from pathlib import Path

from ontology.ontology_bridge import OntologyBridge


def test_ontology_bridge_bootstraps_clean_file_without_bloating():
    path = Path("outputs/test_ontology_bridge/regulatory_clean.owl")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    first = OntologyBridge(str(path))
    assert first.onto is not None
    assert path.exists()
    size_after_first = path.stat().st_size
    assert 0 < size_after_first < OntologyBridge.MAX_PERSISTED_ONTOLOGY_BYTES

    second = OntologyBridge(str(path))
    assert second.onto is not None
    assert second.recovered_from_load_error is False
    assert second.load_error is None
    size_after_second = path.stat().st_size
    assert size_after_second <= int(size_after_first * 1.05)


def test_ontology_bridge_recovers_from_broken_file():
    path = Path("outputs/test_ontology_bridge/regulatory_broken.owl")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("<rdf:RDF><broken", encoding="utf-8")

    bridge = OntologyBridge(str(path))
    assert bridge.onto is not None
    assert bridge.recovered_from_load_error is True
    assert bridge.load_error is None
    assert path.stat().st_size < OntologyBridge.MAX_PERSISTED_ONTOLOGY_BYTES
    assert path.read_text(encoding="utf-8").rstrip().endswith("</rdf:RDF>")

    reloaded = OntologyBridge(str(path))
    assert reloaded.onto is not None
    assert reloaded.recovered_from_load_error is False
    assert reloaded.load_error is None
