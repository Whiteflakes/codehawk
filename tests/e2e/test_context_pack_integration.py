"""Integration test covering indexing, search, and context pack assembly."""

from codehawk.context import ContextEngine


def test_context_pack_includes_relations_and_lineage(populated_database: int, context_engine: ContextEngine):
    """Context packs should surface graph relations and commit lineage."""

    context_pack = context_engine.get_context_pack("helper", limit=5)

    assert context_pack["chunks"], "Expected search to return chunk results"
    assert any(
        rel["relation_type"] in {"imports", "calls", "inherits"}
        for rel in context_pack["relations"]
    ), "Graph relations were not persisted"
    assert context_pack["lineage"], "Lineage entries were not included"
