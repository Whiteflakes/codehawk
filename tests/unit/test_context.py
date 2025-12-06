import pytest

from codehawk.context import ContextEngine


class DummyDatabase:
    def __init__(self):
        self.relations_called = False
        self.lineage_called = False

    def get_relations_for_chunks(self, chunk_ids):
        self.relations_called = True
        return [
            {
                "source_chunk_id": 1,
                "target_chunk_id": 2,
                "relation_type": "calls",
                "metadata": {"note": "test"},
            }
        ]

    def get_lineage_for_chunks(self, chunk_ids):
        self.lineage_called = True
        return [
            {
                "chunk_id": 1,
                "file_path": "app.py",
                "commit_hash": "abc123",
                "author": "Test Author",
                "message": "Test commit",
                "timestamp": "2024-01-01T00:00:00",
                "change_type": "modified",
            }
        ]


def test_get_context_pack_includes_relations_and_lineage(monkeypatch):
    """Ensure get_context_pack enriches results with graph relations and lineage."""

    engine = ContextEngine.__new__(ContextEngine)

    mock_results = [
        {"id": 1, "content": "code A", "file_path": "app.py", "repository": "repo"},
        {"id": 2, "content": "code B", "file_path": "utils.py", "repository": "repo"},
    ]

    monkeypatch.setattr(engine, "search", lambda query, limit=5: mock_results)
    engine.db = DummyDatabase()

    context_pack = engine.get_context_pack("test query")

    assert engine.db.relations_called
    assert engine.db.lineage_called
    assert context_pack["relations"][0]["relation_type"] == "calls"
    assert context_pack["lineage"][0]["commit_hash"] == "abc123"


def test_get_context_pack_respects_include_flags(monkeypatch):
    """Relations and lineage should be omitted when flags are disabled."""

    engine = ContextEngine.__new__(ContextEngine)
    engine.db = DummyDatabase()

    mock_results = [{"id": 1, "content": "code A", "file_path": "app.py", "repository": "repo"}]
    monkeypatch.setattr(engine, "search", lambda query, limit=5: mock_results)

    context_pack = engine.get_context_pack(
        "test query", include_relations=False, include_lineage=False
    )

    assert not engine.db.relations_called
    assert not engine.db.lineage_called
    assert context_pack["relations"] == []
    assert context_pack["lineage"] == []
